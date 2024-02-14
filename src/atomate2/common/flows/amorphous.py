"""Flows adapted from MPMorph *link to origin github repo*"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import job, Flow, Maker, Response
#from atomate2.common.flows.eos import CommonEosMaker
from atomate2.common.jobs.eos import (
    apply_strain_to_structure,
    MPMorphPVPostProcess,
)

from atomate2.vasp.sets.core import MDSetGenerator

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
        working_outputs: dict | None = None,
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
            linear_strain = np.linspace(-0.2,0.2,self.postprocessor.min_data_points)
            working_outputs : dict[str,dict] = {
                "relax": { key : [] for key in ("energy","volume","stress",)}
            }

        else:
            
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
        deformed_structures = apply_strain_to_structure(
            structure, deformation_matrices
        )

        eos_jobs = []
        for index in range(len(deformation_matrices)):
            md_job = self.md_maker.make(
                structure=deformed_structures.output[index],
                prev_dir=None,
            )
            md_job.name = f"{self.name} {md_job.name} {len(working_outputs['relax']['energy'])+1}"
            
            working_outputs["relax"]["energy"].append(
                md_job.output.output.energy
            )
            working_outputs["relax"]["volume"].append(
                md_job.output.structure.volume
            )
            working_outputs["relax"]["stress"].append(
                md_job.output.output.stress
            )
            eos_jobs.append(md_job)

        # The postprocessor has a .fit and .make arg that do similar things
        # The .make function is a jobflow Job and returns the dict as output
        # The .fit function is a regular function that doesn't return anything
        postprocess_job = self.postprocessor.make(working_outputs)
        postprocess_job.name = self.name + "_" + postprocess_job.name
        working_outputs = postprocess_job.output
        eos_jobs.append(postprocess_job)

        recursive = EquilibriumVolumeMaker(
            md_maker=self.md_maker,
            postprocessor=self.postprocessor,
            min_strain=self.min_strain,
            max_attempts=self.max_attempts
        ).make(
            structure=structure,
            prev_dir=None, 
            working_outputs=working_outputs,
        )

        new_eos_flow = Flow([*eos_jobs,recursive], output=working_outputs)

        return Response(replace=new_eos_flow, output=new_eos_flow.output)


@dataclass
class FastQuenchMaker(Maker):
    """TODO: docstr"""

    name: str = "fast quench"
    relax_maker: Maker = Maker
    relax_maker2: Maker | None = None
    static_maker: Maker = Maker

    def make(self, structure: Structure) -> Flow:
        """TODO: docstr"""
        relax1 = self.relax_maker.make(structure)
        if self.relax_maker2 is not None:
            relax2 = self.relax_maker2.make(relax1.output.structure)
            static = self.static_maker.make(relax2.output.structure)
            return Flow(
                [relax1, relax2, static],
                output=static.output,
                name=self.name,
            )
        static = self.static_maker.make(relax1.output.structure)
        return Flow(
            [relax1, static],
            output=static.output,
            name=self.name,
        )


@dataclass
class SlowQuenchMaker(Maker):  # Work in Progress
    """TODO: docstr"""

    name: str = "slow quench"
    md_maker: Maker = Maker  # Goal is to eventually migrate to the general Maker
    quench_tempature_setup: dict = field(
        default_factory=lambda: {"start_temp": 3000, "end_temp": 500, "temp_step": 500}
    )

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """TODO: docstr"""
        md_jobs = []
        for temp in np.arange(
            self.quench_tempature_setup["start_temp"],
            self.quench_tempature_setup["end_temp"],
            -self.quench_tempature_setup["temp_step"],
        ):  # check if this is the best way to unpack

            prev_dir = (
                None
                if temp == self.quench_tempature_setup["start_temp"]
                else md_jobs[-1].output.dir_name
            )

            md_job = self.md_maker(
                input_set_generator=MDSetGenerator(
                    start_temp=temp,
                    end_temp=temp - self.quench_tempature_setup["temp_step"],
                )
            ).make(structure, prev_dir)

            md_jobs.append(md_job)

            structure = md_job.output.structure

        return Flow(
            md_jobs,
            output=md_jobs[-1].output,
            name=self.name,
        )


@dataclass
class MPMorphMDMaker(Maker):
    """Base MPMorph flow for volume equilibration, quench, and production runs via molecular dynamics

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    (optional) for the volume followed by quench (optional) from high temperature to low temperature
    and finally a production run(s) at a given temperature. Production run is broken up into multiple
    smaller steps to ensure simulation does not hit wall time limits.

    Check atomate2.vasp.flows.amorphous for MPMorphVaspMDMaker

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
    production_md_maker: Maker = (
        Maker  # May need to fix this into ForceFieldMDMaker later..)
    )
    quench_maker: FastQuenchMaker | SlowQuenchMaker = (
        None  # May need to fix this into ForceFieldMDMaker later..)
    )

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
            convergence_flow = self.convergence_md_maker.make(structure, prev_dir)
            flow_jobs.extend(convergence_flow.jobs)

            structure = convergence_flow.output.structure
            prev_dir = convergence_flow.output.dir_name

        production_flow = self.production_md_maker.make(structure, prev_dir)

        structure = production_flow.output.structure
        prev_dir = production_flow.output.dir_name

        if self.quench_maker:
            quench_flow = self.quench_maker.make(structure, prev_dir)
            flow_jobs.extend(quench_flow.jobs)

        production_flow = self.production_md_maker.make(structure, prev_dir)
        flow_jobs.extend(production_flow.jobs)

        return Flow(
            flow_jobs,
            output=production_flow.output,
            name=self.name,
        )


# To Do: VolumeTemperatureSweepMaker
