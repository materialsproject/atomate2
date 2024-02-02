"""Flows adapted from MPMorph *link to origin github repo*"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker
from atomate2.common.jobs.equilibrate import EquilibrateVolumeMaker

from atomate2.vasp.jobs.md import MDMaker
from atomate2.vasp.sets.core import MDSetGenerator

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Structure


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
class SlowQuenchMaker(Maker):
    name: str = "slow quench"
    md_maker: Maker = MDMaker  # Goal is to eventually migrate to the general Maker
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
    convergence_md_maker: EquilibrateVolumeMaker | None = (
        None  # check logic on this line
    )
    quench_maker: FastQuenchMaker | SlowQuenchMaker | None = (
        None  # May need to fix this into ForceFieldMDMaker later..)
    )
    production_md_maker: Maker = Maker  # Same issue as line above

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
        # Someone please help me make these if statements more efficient... :(

        if self.convergence_md_maker is not None:
            convergence_md_job = self.convergence_md_maker.make(structure, prev_dir)

            if self.quench_maker is not None:
                quench_job = self.quench_maker.make(
                    convergence_md_job.output, convergence_md_job.output.dir_name
                )

                production_md_job = self.production_md_maker.make(
                    quench_job.output, quench_job.output.dir_name
                )

                return Flow(
                    [convergence_md_job, quench_job, production_md_job],
                    output=production_md_job.output,
                    name="MPMorph Converge-Quench-Production MD",
                )

            production_md_job = self.production_md_maker.make(
                convergence_md_job.output, convergence_md_job.output.dir_name
            )
            return Flow(
                [convergence_md_job, production_md_job],
                output=production_md_job.output,
                name="MPMorph Converge-Quench-Production MD",
            )
        if self.quench_maker is not None:
            quench_job = self.quench_maker.make(structure, prev_dir)

            production_md_job = self.production_md_maker.make(
                quench_job.output, quench_job.output.dir_name
            )

            return Flow(
                [quench_job, production_md_job],
                output=production_md_job.output,
                name="MPMorph Converge-Quench-Production MD",
            )

        production_md_job = self.production_md_maker.make(structure, prev_dir)
        return Flow(
            [production_md_job],
            output=production_md_job.output,
            name="MPMorph Converge-Quench-Production MD",
        )


# To Do: VolumeTemperatureSweepMaker
