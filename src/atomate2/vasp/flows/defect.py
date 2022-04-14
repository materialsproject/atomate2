"""Flows used in the calculation of defect properties."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

from jobflow import Flow, Maker, OutputReference, job
from pymatgen.core.structure import Structure

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker
from atomate2.vasp.jobs.defect import (
    FiniteDifferenceMaker,
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
        "KSPACING": None,
        "ENCUT": 500,
    },
    user_kpoints_settings={"reciprocal_density": 64},
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

        Returns
        -------
        Flow
            The full workflow for the calculation of the configuration coordinate diagram.
        """
        # use a more descriptive name when possible
        if not isinstance(structure, OutputReference):
            name = f"{self.name}: {structure.formula}"
            if not (
                isinstance(charge_state1, OutputReference)
                or isinstance(charge_state2, OutputReference)
            ):
                name = (
                    f"{self.name}: {structure.formula}({charge_state1}-{charge_state2})"
                )

        # need to wrap this up in a job so that references to undone calculations can be passed in
        charged_structures = get_charged_structures(
            structure, [charge_state1, charge_state2]
        )

        relax1 = self.relax_maker.make(structure=charged_structures.output[0])
        relax2 = self.relax_maker.make(structure=charged_structures.output[1])
        relax1.append_name(" q1")
        relax2.append_name(" q2")

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
            add_name="q1",
        )

        deformations2 = spawn_energy_curve_calcs(
            struct2,
            struct1,
            distortions=self.distortions,
            static_maker=self.static_maker,
            prev_vasp_dir=dir2,
            add_name="q2",
        )

        deformations1.append_name(" q1")
        deformations2.append_name(" q2")

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


@job
def get_charged_structures(structure: Structure, charges: Iterable):
    """Adding charges to structure.

    This needs to be a job so the results of other jobs can be passed in.

    Parameters
    ----------
    structure
        A structure.
    charges
        A list of charges on the structure

    Returns
    -------
    dict
        A dictionary with the two structures with the charge states added.

    """
    structs_out = [structure.copy() for _ in charges]
    for i, q in enumerate(charges):
        structs_out[i].set_charge(q)
    return structs_out


@dataclass
class NonRadiativeMaker(Maker):
    """Class to generate workflows for the calculation of the non-radiative defect capture."""

    ccd_maker: ConfigurationCoordinateMaker
    name: str = "non-radiative"
    wswq_maker: FiniteDifferenceMaker = field(
        default_factory=lambda: FiniteDifferenceMaker()
    )

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
        if not isinstance(structure, OutputReference):
            name = f"{self.name}: {structure.formula}"
            if not (
                isinstance(charge_state1, OutputReference)
                or isinstance(charge_state2, OutputReference)
            ):
                name = (
                    f"{self.name}: {structure.formula}({charge_state1}-{charge_state2})"
                )

        flow = self.ccd_maker.make(
            structure=structure,
            charge_state1=charge_state1,
            charge_state2=charge_state2,
        )
        ccd: CCDDocument = flow.output

        dirs0 = ccd.distorted_calcs_dirs[0]
        dirs1 = ccd.distorted_calcs_dirs[1]
        mid_index0 = len(self.ccd_maker.distortions) // 2
        mid_index1 = len(self.ccd_maker.distortions) // 2
        finite_diff_job1 = self.wswq_maker.make(
            ref_calc_dir=dirs0[mid_index0], distorted_calc_dirs=dirs0
        )
        finite_diff_job2 = self.wswq_maker.make(
            ref_calc_dir=dirs1[mid_index1], distorted_calc_dirs=dirs1
        )
        finite_diff_job1.append_name(" q1")
        finite_diff_job2.append_name(" q2")

        output = {
            charge_state1: finite_diff_job1.output,
            charge_state2: finite_diff_job2.output,
        }
        return Flow(
            jobs=[flow, finite_diff_job1, finite_diff_job2], output=output, name=name
        )
