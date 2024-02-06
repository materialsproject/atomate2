"""Flows adapted from MPMorph *link to origin github repo*"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import job, Flow, Maker, Response
from atomate2.common.flows.eos import CommonEosMaker
from atomate2.common.jobs.eos import (
    apply_strain_to_structure,
    postprocess_eos,
    extract_eos_sampling_data,
)

from atomate2.vasp.sets.core import MDSetGenerator

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Job

    from pymatgen.core import Structure


@dataclass
class EquilibriumVolumeMaker(Maker):
    name: str = "Equilibrium Volume Maker"
    eos_relax_maker: Maker | None = None
    postprocessor: Job = postprocess_eos  # TODO change to postprocess_pv_eos once ready
    extracting_job: Job = extract_eos_sampling_data
    min_strain: float = 0.5

    @job
    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
        working_outputs: dict | None = None,
    ) -> Flow:
        print("new print test")
        print(working_outputs)
        if working_outputs is None:

            eos_maker = CommonEosMaker(
                name=self.name + " initial EOS fit",
                initial_relax_maker=None,
                eos_relax_maker=self.eos_relax_maker,
                static_maker=None,
                postprocessor=self.postprocessor,
                number_of_frames=4,
            )
            eos_flow = eos_maker.make(structure=structure, prev_dir=prev_dir)

            extract_job = self.extracting_job(eos_flow.output)

            eos_jobs = [*eos_flow.jobs, extract_job]

            working_outputs = extract_job.output

        else:

            if working_outputs["V0<Vmax"] and working_outputs["V0>Vmin"]:
                final_structure = structure.copy()
                final_structure.scale_lattice(working_outputs["V0"])
                return final_structure

            elif not working_outputs["V0<Vmax"]:
                v_ref = working_outputs["Vmax"]

            elif not working_outputs["V0>Vmin"]:
                v_ref = working_outputs["Vmin"]

            eps_0 = (working_outputs["V0"] / v_ref) ** (1.0 / 3.0) - 1.0
            linear_strain = [np.sign(eps_0) * (abs(eps_0) + self.min_strain)]

            deformation_matrices = [np.eye(3) * (1 + eps) for eps in linear_strain]
            deformed_structures = apply_strain_to_structure(
                structure, deformation_matrices
            )

            eos_jobs = []
            for structure in deformed_structures:
                eos_jobs.append(
                    self.eos_relax_maker.make(
                        structure=structure,
                        prev_dir=None,
                    )
                )
                working_outputs["energies"].append(eos_jobs[-1].output.output.energy)
                working_outputs["volumes"].append(eos_jobs[-1].ouput.structure.volume)
                working_outputs["pressure"].append(
                    1 / 3 * np.trace(eos_jobs[-1].output.stress)
                )

            postprocess_job = self.postprocessor(working_outputs)
            postprocess_job.name = self.name + "_" + postprocess_job.name
            working_outputs = postprocess_job.output
            eos_jobs.append(postprocess_job)

        recursive = self.make(structure=structure, working_outputs=working_outputs)

        new_eos_flow = Flow([*eos_jobs, recursive], output=working_outputs)

        return Response(replace=new_eos_flow, output=new_eos_flow.output)


@dataclass
class FastQuenchMaker(Maker):
    name: str = "fast quench"
    relax_maker: Maker = Maker
    relax_maker2: Maker | None = None
    static_maker: Maker = Maker

    def make(self, structure: Structure) -> Flow:
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
    name: str = "slow quench"
    md_maker: Maker = Maker  # Goal is to eventually migrate to the general Maker
    quench_tempature_setup: dict = field(
        default_factory=lambda: {"start_temp": 3000, "end_temp": 500, "temp_step": 500}
    )

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
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
