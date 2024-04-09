"""Flows adapted from MPMorph *link to origin github repo*"""  # TODO: Add link to origin github repo

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import job, Flow, Maker, Response

# from atomate2.common.flows.eos import CommonEosMaker
from atomate2.common.jobs.eos import (
    _apply_strain_to_structure,
    MPMorphPVPostProcess,
)
from atomate2.vasp.jobs.md import MDMaker
from atomate2.vasp.sets.core import MDSetGenerator
from atomate2.forcefields.md import ForceFieldMDMaker

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from atomate2.common.jobs.eos import EOSPostProcessor
    from jobflow import Job

    from pymatgen.core import Structure


@dataclass
class EquilibriumVolumeMaker(Maker):
    """
    Equilibrate structure using NVT + EOS fitting.

    Parameters
    -----------
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

    @job
    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
        working_outputs: dict[str, dict] | None = None,
    ) -> Flow:
        """
        Run an NVT+EOS equilibration flow.

        Parameters
        -----------
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
            working_outputs: dict[str, dict] = {
                "relax": {key: [] for key in ("energy", "volume", "stress", "pressure")}
            }

        else:
            self.postprocessor.fit(working_outputs)
            # print("____EOS FIT PARAMS_____")
            # print(self.postprocessor.results)
            # print("_______________________")
            working_outputs = dict(self.postprocessor.results)
            working_outputs["relax"].pop(
                "pressure", None
            )  # remove pressure from working_outputs

            if (
                working_outputs["V0"] <= working_outputs["Vmax"]
                and working_outputs["V0"] >= working_outputs["Vmin"]
            ) or (
                self.max_attempts
                and (
                    len(working_outputs["relax"]["volume"])
                    - self.postprocessor.min_data_points
                )
                >= self.max_attempts
            ):
                final_structure = structure.copy()
                final_structure.scale_lattice(working_outputs["V0"])
                return final_structure

            elif working_outputs["V0"] > working_outputs["Vmax"]:
                v_ref = working_outputs["Vmax"]

            elif working_outputs["V0"] < working_outputs["Vmin"]:
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
                f"{self.name} {md_job.name} {len(working_outputs['relax']['energy'])+1}"
            )

            working_outputs["relax"]["energy"].append(md_job.output.output.energy)
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
    """Base MPMorph flow for volume equilibration, quench, and production runs via molecular dynamics

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    (optional) for the volume followed by quench (optional) from high temperature to low temperature
    and finally a production run(s) at a given temperature. Production run is broken up into multiple
    smaller steps to ensure simulation does not hit wall time limits.

    Check atomate2.vasp.flows.mpmorph for MPMorphVaspMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher
    quench_maker :  SlowQuenchMaker or FastQuenchMaker or None
        SlowQuenchMaker - MDMaker that quenchs structure from high temperature to low temperature
        FastQuenchMaker - DoubleRelaxMaker + Static that "quenchs" structure at 0K
    production_md_maker : Maker
        MDMaker to generate the production run(s)
    """

    name: str = "MP Morph md"
    convergence_md_maker: EquilibriumVolumeMaker = None  # check logic on this line
    # May need to fix next two into ForceFieldMDMakers later..)
    production_md_maker: Maker | None = None
    quench_maker: FastQuenchMaker | SlowQuenchMaker | None = None

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ):
        """
        Create a flow with MPMorph molecular dynamics (and relax+static).

        By default, production run is broken up into multiple smaller steps. Converegence and
        quench are optional and may be used to equilibrate the unit cell volume (useful for
        high temperature production runs of structures extracted from Materials Project) and
        to quench the structure from high to low temperature (e.g. amorphous structures).

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


@dataclass
class FastQuenchMaker(Maker):
    """Fast quench flow for quenching high temperature structures to 0K

    Quench's a provided structure with a single (or double) relaxation and a static calculation at 0K.
    Adapted from MPMorph Workflow

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
    """Slow quench flow for quenching high temperature structures to low temperature

    Quench's a provided structure with a molecular dyanmics run from a desired high temperature to
    a desired low temperature. Flow creates a series of MD runs that holds at a certain temperature
    and initiates the following MD run at a lower temperature (step-wise temperature MD runs).
    Adapted from MPMorph Workflow.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    md_maker :  Maker | None = None
        Can only be an MDMaker or ForceFieldMDMaker. Defaults to None. If None, will not work. #WORK IN PROGRESS.
    quench_start_temperature : int = 3000
        Starting temperature for quench; default 3000K
    quench_end_temperature : int = 500
        Ending temperature for quench; default 500K
    quench_temperature_step : int = 500
        Temperature step for quench; default 500K drop
    quench_nsteps : int = 1000
        Number of steps for quench; default 1000 steps
    """

    name: str = "slow quench"
    md_maker: Maker | None = None
    quench_start_temperature: int = 3000
    quench_end_temperature: int = 500
    quench_temperature_step: int = 500
    quench_nsteps: int = 1000

    def make(
        self, structure: Structure, prev_dir: str | Path | None = None
    ) -> (
        Flow
    ):  # TODO : main objective: modified to work with other MD codes; Only works for VASP and MLFF_MD now.
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

        md_jobs = []
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

            md_job = self.call_md_maker(
                structure, prev_dir, temp=temp, nsteps=self.quench_nsteps
            )
            """ # Logic of call_md_maker is to substitute the following code:
            if isinstance(self.md_maker, MDMaker):
                md_job = MDMaker(
                    input_set_generator=MDSetGenerator(
                        start_temp=temp,
                        end_temp=temp,
                        nsteps=self.quench_tempature_setup["nsteps"],
                    )
                ).make(structure, prev_dir)

            elif isinstance(self.md_maker, ForceFieldMDMaker):
                self.md_maker = self.md_maker.update_kwargs(
                    update={
                        "temperature": temp,
                        "nsteps": self.quench_tempature_setup["nsteps"],
                    }
                )
                md_job = self.md_maker.make(
                    structure,
                    prev_dir,
                )
            else:
                raise ValueError(
                    "***WORK IN PROGRESS*** md_maker must be an MDMaker or ForceFieldMDMaker."
                )"""

            md_jobs.append(md_job)

            structure = md_job.output.structure

        return Flow(
            md_jobs,
            output=md_jobs[-1].output,
            name=self.name,
        )

    def call_md_maker(self, structure, prev_dir, temp, nsteps):
        if not (
            isinstance(self.md_maker, MDMaker)
            or isinstance(self.md_maker, ForceFieldMDMaker)
        ):
            raise ValueError(
                "***WORK IN PROGRESS*** md_maker must be an MDMaker or ForceFieldMDMaker."
            )
        return self.md_maker.make(structure, prev_dir)


# TODO: Amorphous structure maker, Amorphour limit, and VolumeTemperatureSweepMaker
