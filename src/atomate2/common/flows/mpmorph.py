"""Define code agnostic MPMorph flows.

This file generalizes the MPMorph workflows of
https://github.com/materialsproject/mpmorph
originally written in atomate for VASP only to a more general
code agnostic form.

For information about the current flows, contact:
- Bryant Li (@BryantLi-BLI)
- Aaron Kaplan (@esoteric-ephemera)
- Max Gallant (@mcgalcode)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from jobflow import Flow, Maker, Response, job

# from atomate2.common.flows.eos import CommonEosMaker
from atomate2.common.jobs.eos import MPMorphPVPostProcess, _apply_strain_to_structure
from atomate2.forcefields.md import ForceFieldMDMaker
from atomate2.vasp.jobs.md import MDMaker

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from jobflow import Job
    from pymatgen.core import Composition, Structure

    from atomate2.common.jobs.eos import EOSPostProcessor


@dataclass
class EquilibriumVolumeMaker(Maker):
    """
    Equilibrate structure using NVT + EOS fitting.

    Parameters
    ----------
    name : str = "Equilibrium Volume Maker"
        Name of the flow
    md_maker : Maker
        Maker to perform NVT MD runs
    postprocessor : atomate2.common.jobs.eos.EOSPostProcessor
        Postprocessing step to fit the EOS
    min_strain : float, default = 0.5
        Minimum absolute percentage linear strain to apply to the structure
    max_attempts : int | None = 20
        Number of times to continue attempting to equilibrate the structure.
        If None, the workflow will not terminate if an equilibrated structure
        cannot be determined.
    """

    name: str = "Equilibrium Volume Maker"
    md_maker: Maker | None = None
    postprocessor: EOSPostProcessor = field(default_factory=MPMorphPVPostProcess)
    min_strain: float = 0.5
    max_attempts: int | None = 20
    energy_average_frames : int = 1

    @job
    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
        working_outputs: dict[str, Any] | None = None,
    ) -> Flow:
        """
        Run an NVT+EOS equilibration flow.

        Parameters
        ----------
        structure : Structure
            structure to equilibrate
        prev_dir : str | Path | None (default)
            path to copy files from
        working_outputs : dict or None
            contains the outputs of the flow as it recursively updates

        Returns
        -------
        .Flow, an MPMorph flow
        """
        if working_outputs is None:
            linear_strain = np.linspace(-0.2, 0.2, self.postprocessor.min_data_points)
            working_outputs = {
                "relax": {key: [] for key in ("energies", "volume", "stress", "pressure")}
            }

        else:

            working_outputs["relax"]["energy"] = [
                sum(frame[-self.energy_average_frames:])/self.energy_average_frames
                for frame in working_outputs["relax"]["energies"]
            ]

            self.postprocessor.fit(working_outputs)
            working_outputs = dict(self.postprocessor.results)
            for k in ("pressure","energy",):
                working_outputs["relax"].pop(k,None)

            if (
                working_outputs.get("V0") is None
            ):  # breaks whole flow here if EOS is not fitted properly
                return Response(output=working_outputs, stop_children=True)
            if (
                working_outputs.get("V0") <= working_outputs.get("Vmax")
                and working_outputs.get("V0") >= working_outputs.get("Vmin")
            ) or (
                self.max_attempts
                and (
                    len(working_outputs["relax"]["volume"])
                    - self.postprocessor.min_data_points
                )
                >= self.max_attempts
            ):
                # If the equilibrium volume is within the range of fitted volumes,
                # or if the maximum number of attempts has been performed, stop
                # and return structure at estimated equilibrium volume
                final_structure = structure.copy()
                final_structure.scale_lattice(working_outputs["V0"])
                return final_structure

            # Else, if the extrapolated equilibrium volume is outside the range of
            # fitted volumes, scale appropriately
            if working_outputs.get("V0") > working_outputs.get("Vmax"):
                v_ref = working_outputs["Vmax"]

            elif working_outputs.get("V0") < working_outputs.get("Vmax"):
                v_ref = working_outputs["Vmin"]

            eps_0 = (working_outputs["V0"] / v_ref) ** (1.0 / 3.0) - 1.0
            linear_strain = [np.sign(eps_0) * (abs(eps_0) + self.min_strain)]

        deformation_matrices = [np.eye(3) * (1.0 + eps) for eps in linear_strain]
        deformed_structures = _apply_strain_to_structure(
            structure, deformation_matrices
        )

        eos_jobs = []
        for index in range(len(deformation_matrices)):
            md_job = self.md_maker.make(
                structure=deformed_structures[index].final_structure,
                prev_dir=None,
            )
            md_job.name = (
                f"{self.name} {md_job.name} {len(working_outputs['relax']['volume'])+1}"
            )

            if self.energy_average_frames == 1:
                energies = [md_job.output.output.energy]
            else:
                raise ValueError("Cannot perform energy averaging just yet!")
                try:
                    energies = [frame.energy for frame in md_job.output.output.ionic_steps]
                except AttributeError:
                    energies = [frame.e_0_energy for frame in md_job.output.vasp_objects["trajectory"].frame_properties]

            working_outputs["relax"]["energies"].append(energies)
            working_outputs["relax"]["volume"].append(md_job.output.structure.volume)
            working_outputs["relax"]["stress"].append(md_job.output.output.stress)
            eos_jobs.append(md_job)

        recursive = self.make(
            structure=structure,
            prev_dir=None,
            working_outputs=working_outputs,
        )

        new_eos_flow = Flow([*eos_jobs, recursive], output=recursive.output)

        return Response(replace=new_eos_flow, output=recursive.output)


@dataclass
class MPMorphMDMaker(Maker):
    """Base MPMorph flow for amorphous solid equilibration.

    This flow uses NVT molecular dynamics to:
    (1 - optional) Determine the equilibrium volume of an amorphous
        structure via EOS fit.
    (2 - optional) Quench the equilibrium volume structure from a higher
        temperature down to a lower desired "production" temperature.
    (3) Run a production, longer-time MD run in NVT.
        The production run can be broken up into smaller steps to
        ensure the simulation does not hit wall time limits.

    Check atomate2.vasp.flows.mpmorph for MPMorphVaspMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher
    quench_maker :  SlowQuenchMaker or FastQuenchMaker or None
        SlowQuenchMaker - MDMaker that quenchs structure from high to low temperature
        FastQuenchMaker - DoubleRelaxMaker + Static that "quenchs" structure at 0K
    production_md_maker : Maker
        MDMaker to generate the production run(s)
    """

    name: str = "MP Morph md"
    convergence_md_maker: EquilibriumVolumeMaker | None = (
        None  # check logic on this line
    )
    # May need to fix next two into ForceFieldMDMakers later..)
    production_md_maker: Maker | None = None
    quench_maker: FastQuenchMaker | SlowQuenchMaker | None = None

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """
        Create a flow with MPMorph molecular dynamics (and relax+static).

        By default, production run is broken up into multiple smaller steps.
        Converegence and quench are optional and may be used to equilibrate
        the wt cell volume (useful for high temperature production runs of
        structures extracted from Materials Project) and to quench the
        structure from high to low temperature (e.g. amorphous structures).

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing series of molecular dynamics run (and relax+static).
        """
        self._post_init_update()
        flow_jobs = []

        if self.convergence_md_maker is not None:
            convergence_flow = self.convergence_md_maker.make(
                structure, prev_dir=prev_dir
            )
            flow_jobs.append(convergence_flow)

            # convergence_flow only outputs a structure
            structure = convergence_flow.output

        self.production_md_maker.name = self.name + " production run"
        production_run = self.production_md_maker.make(
            structure=structure, prev_dir=prev_dir
        )
        flow_jobs.append(production_run)

        if self.quench_maker:
            quench_flow = self.quench_maker.make(
                structure=production_run.output.structure,
                prev_dir=production_run.output.dir_name,
            )
            flow_jobs += [quench_flow]

        return Flow(
            flow_jobs,
            output=production_run.output,
            name=self.name,
        )

    def _post_init_update(self) -> None:
        """Update Maker prior to creating flow."""


@dataclass
class FastQuenchMaker(Maker):
    """Fast quench flow from high temperature to 0K structures.

    Quenches a provided structure with a single (or double)
    relaxation and a static calculation at 0K.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker :  Maker
        Relax Maker
    relax_maker2 :  Maker or None
        Relax Maker for a second relaxation; useful for tighter convergence
    static_maker : Maker
        Static Maker
    """

    name: str = "fast quench"
    relax_maker: Maker = Maker
    relax_maker2: Maker | None = None
    static_maker: Maker = Maker

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """
        Create a fast quench flow with relax and static makers.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing series of relax and static runs.
        """
        jobs: list[Job] = []

        relax1 = self.relax_maker.make(structure, prev_dir=prev_dir)
        jobs += [relax1]
        structure = relax1.output.structure
        prev_dir = relax1.output.dir_name

        if self.relax_maker2 is not None:
            relax2 = self.relax_maker2.make(structure, prev_dir=prev_dir)
            jobs += [relax2]
            structure = relax2.output.structure
            prev_dir = relax2.output.dir_name

        static = self.static_maker.make(structure, prev_dir=prev_dir)
        jobs += [static]
        return Flow(
            jobs,
            output=static.output,
            name=self.name,
        )


@dataclass
class SlowQuenchMaker(Maker):  # Works only for VASP and MLFFs
    """Slow quench from high to low temperature structures.

    Quenches a provided structure with a molecular dynamics
    run from a desired high temperature to a desired low temperature.
    Flow creates a series of MD runs that holds at a certain temperature
    and initiates the following MD run at a lower temperature (step-wise
    temperature MD runs).

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    md_maker :  Maker | None = None
        Can only be an MDMaker or ForceFieldMDMaker.
        Defaults to None. If None, will not work. #WORK IN PROGRESS.
    quench_start_temperature : float = 3000
        Starting temperature for quench; default 3000K
    quench_end_temperature : float = 500
        Ending temperature for quench; default 500K
    quench_temperature_step : float = 500
        Temperature step for quench; default 500K drop
    quench_n_steps : int = 1000
        Number of steps for quench; default 1000 steps
    descent_method : str = "stepwise"
        Descent method for quench; default "stepwise".
        Others available: "linear with hold"
    """

    name: str = "slow quench"
    md_maker: Maker | None = None
    quench_start_temperature: float = 3000
    quench_end_temperature: float = 500
    quench_temperature_step: float = 500
    quench_n_steps: int = 1000
    descent_method: Literal["stepwise", "linear with hold"] = "stepwise"

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        # TODO : main objective: modified to work with other MD codes.
        # Only works for VASP and MLFF_MD now.
        """
        Create a slow quench flow with md maker.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing series of relax and static runs.
        """
        md_jobs: list[Job] = []
        for temp in np.arange(
            self.quench_start_temperature,
            self.quench_end_temperature,
            -self.quench_temperature_step,
        ):
            prev_dir = (
                None
                if temp == self.quench_start_temperature
                else md_jobs[-1].output.dir_name
            )
            if self.descent_method == "stepwise":
                md_job = self.call_md_maker(
                    structure=structure,
                    temp=temp,
                    prev_dir=prev_dir,
                )

            elif (
                self.descent_method == "linear with hold"
            ):  # TODO: Work in Progress; needs testing
                md_job_linear = self.call_md_maker(
                    structure=structure,
                    temp=[temp, temp - self.quench_temperature_step],  # type: ignore[arg-type]
                    prev_dir=prev_dir,
                )

                md_job = self.call_md_maker(
                    structure=md_job_linear.output.structure,
                    temp=temp - self.quench_temperature_step,
                    prev_dir=md_job_linear.output.dir_name,
                )

                md_jobs.append(md_job_linear)

            md_jobs.append(md_job)

            structure = md_job.output.structure

        return Flow(
            md_jobs,
            output=md_jobs[-1].output,
            name=self.name,
        )

    def call_md_maker(
        self,
        structure: Structure,
        temp: float,
        prev_dir: str | Path | None = None,
    ) -> Flow | Job:
        """Call MD maker for slow quench.

        TODO write docstr
        """
        if not isinstance(self.md_maker, (MDMaker, ForceFieldMDMaker)):
            raise TypeError(
                "***WORK IN PROGRESS*** md_maker must be "
                "an MDMaker or ForceFieldMDMaker."
            )
        return self.md_maker.make(structure, prev_dir)


@dataclass
class AmorphousLimitMaker(Maker):
    """Create an amorphous structure and equilibrate with MPMorph.

    Creates an amorphous structure from a desired stiochiometry,
    then performs the MPMorph molecular dynamics runs on top of it.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    mpmorph_maker :  MPMorphMDMaker
        MDMaker to generate the molecular dynamics jobs specifically
        for MPMorph workflow
    """

    name: str = "Amorphous Limit Maker"
    mpmorph_maker: MPMorphMDMaker = field(default_factory=MPMorphMDMaker)

    def make(
        self,
        structure: Structure | None = None,
        composition: str | Composition | None = None,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """
        Run the amorphous structure maker and equilibration.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        composition : str or Composition
            Composition of the amorphous structure to generate.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing series of rescaled volume molecular
            dynamics runs, EOS fitted, then production run at the
            equilibirum volume.
        """
        if structure is None:
            if composition is None:
                raise ValueError("Either structure or composition must be provided.")

            from atomate2.common.jobs.structure_gen import get_random_packed

            structure = get_random_packed(composition)

        mpmorph_flow = self.mpmorph_maker.make(structure=structure, prev_dir=prev_dir)
        return Flow(
            [mpmorph_flow],
            name=self.name,
        )
