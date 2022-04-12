"""Flows used in the calculation of defect properties."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from jobflow import Flow, Maker, job
from pymatgen.core.structure import Structure

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker
from atomate2.vasp.jobs.defect import (
    FiniteDiffMaker,
    get_ccd_from_task_docs,
    spawn_energy_curve_calcs,
)
from atomate2.vasp.schemas.defect import CCDDocument
from atomate2.vasp.sets.core import StaticSetGenerator
from atomate2.vasp.sets.defect import AtomicRelaxSetGenerator

logger = logging.getLogger(__name__)


DEFAULT_DISTORTIONS = (-1, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 1)

DEFECT_RELAX_GENERATOR = AtomicRelaxSetGenerator(use_structure_charge=True)
DEFECT_STATIC_GENERATOR = StaticSetGenerator(
    user_incar_settings={
        "ISMEAR": 0,
        "LWAVE": True,
        "SIGMA": 0.05,
    }
)


@dataclass
class ConfigurationCoordinateMaker(Maker):
    """Class to generate VASP input sets for the calculation of the configuration coordinate diagram."""

    name: str = "config. coordinate"
    relax_maker: BaseVaspMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=DEFECT_RELAX_GENERATOR,
        )
    )
    static_maker: BaseVaspMaker = field(
        default_factory=lambda: StaticMaker(input_set_generator=DEFECT_STATIC_GENERATOR)
    )
    distortions: tuple[float, ...] = DEFAULT_DISTORTIONS

    def make(
        self,
        structure: Structure,
        charge_state1: int,
        charge_state2: int,
    ):
        """
        Make a job for the calculation of the configuration coordinate diagram.

        Parameters
        ----------
        structure
            A structure.
        charge_state1
            The reference charge state of the defect.
        charge_state2
            The excited charge state of the defect

        """
        name = f"{self.name}: {structure.formula}({charge_state1}-{charge_state2})"
        # need to wrap this up in a job so that references to undone calculations can be passed in
        charged_structures = self._get_charged_structures(
            structure, charge_state1, charge_state2
        )

        relax1 = self.relax_maker.make(charged_structures.output["struct1"])
        relax2 = self.relax_maker.make(charged_structures.output["struct2"])
        relax1.append_name(f" q={charge_state1}")
        relax2.append_name(f" q={charge_state2}")

        dir1 = relax1.output.dir_name
        dir2 = relax2.output.dir_name
        struct1 = relax1.output.structure
        struct2 = relax2.output.structure

        deformations1 = spawn_energy_curve_calcs(
            struct1,
            struct2,
            distortions=self.distortions,
            static_maker=self.static_maker,
            prev_vasp_dir=dir1,
            add_name=f"q={charge_state1}",
        )

        deformations2 = spawn_energy_curve_calcs(
            struct2,
            struct1,
            distortions=self.distortions,
            static_maker=self.static_maker,
            prev_vasp_dir=dir2,
            add_name=f"q={charge_state2}",
        )

        deformations1.append_name(f" q={charge_state1}")
        deformations2.append_name(f" q={charge_state2}")

        ccd_job = get_ccd_from_task_docs(
            deformations1.output, deformations2.output, struct1, struct2
        )

        return Flow(
            jobs=[
                charged_structures,
                relax1,
                relax2,
                deformations1,
                deformations2,
                ccd_job,
            ],
            output=ccd_job.output,
            name=name,
        )

    @job(name="charge structures")
    def _get_charged_structures(self, structure, charge_state1, charge_state2):
        struct1: Structure = structure.copy()
        struct1.set_charge(charge_state1)
        struct2: Structure = structure.copy()
        struct2.set_charge(charge_state2)
        return {"struct1": struct1, "struct2": struct2}


@dataclass
class NonRadMaker(ConfigurationCoordinateMaker):
    """Class to generate workflows for the calculation of the non-radiative defect capture."""

    wswq_maker: FiniteDiffMaker = field(default_factory=lambda: FiniteDiffMaker())

    def make(
        self,
        structure: Structure,
        charge_state1: int,
        charge_state2: int,
    ):
        """Create the job for Non-Radiative defect capture.

        Make a job for the calculation of the configuration coordinate diagram.
        Also calculate the el-phon matrix elements for 1-D special phonon.

        Parameters
        ----------
        structure
            A structure.
        charge_state1
            The reference charge state of the defect.
        charge_state2
            The excited charge state of the defect

        """
        name = f"{self.name}: {structure.formula}({charge_state1}-{charge_state2})"
        flow = super().make(
            structure=structure,
            charge_state1=charge_state1,
            charge_state2=charge_state2,
        )
        ccd: CCDDocument = flow.output

        dirs0 = ccd.distorted_calcs_dirs[0]
        dirs1 = ccd.distorted_calcs_dirs[1]
        mid_index0 = len(self.distortions) // 2
        mid_index1 = len(self.distortions) // 2
        finite_diff_job1 = self.wswq_maker.make(
            ref_calc_dir=dirs0[mid_index0], distorted_calc_dirs=dirs0
        )
        finite_diff_job2 = self.wswq_maker.make(
            ref_calc_dir=dirs1[mid_index1], distorted_calc_dirs=dirs1
        )
        finite_diff_job1.append_name(f" q={charge_state1}")
        finite_diff_job2.append_name(f" q={charge_state2}")

        output = {
            charge_state1: finite_diff_job1.output,
            charge_state2: finite_diff_job2.output,
        }
        return Flow(
            jobs=[flow, finite_diff_job1, finite_diff_job2], output=output, name=name
        )
