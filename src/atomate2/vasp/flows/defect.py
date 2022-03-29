"""Flows used in the calculation of defect properties."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from jobflow import Flow, Maker
from pymatgen.core.structure import Structure

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker
from atomate2.vasp.jobs.defect import calculate_energy_curve, get_ccd_from_task_docs
from atomate2.vasp.sets.core import StaticSetGenerator
from atomate2.vasp.sets.defect import AtomicRelaxSetGenerator

logger = logging.getLogger(__name__)


DEFAULT_DISTORTIONS = (-1, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 1)

DEFECT_RELAX_GENERATOR = AtomicRelaxSetGenerator(use_structure_charge=True)
DEFECT_STATIC_GENERATOR = StaticSetGenerator(
    user_incar_settings={"ISMEAR": 0, "LWAVE": True, "SIGMA": 0.05}
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
    wswq: bool = False

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
        struct1: Structure = structure.copy()
        struct1.set_charge(charge_state1)
        struct2: Structure = structure.copy()
        struct2.set_charge(charge_state2)

        relax1 = self.relax_maker.make(struct1)
        relax2 = self.relax_maker.make(struct2)
        relax1.append_name(f" q={charge_state1}")
        relax2.append_name(f" q={charge_state2}")

        dir1 = relax1.output.dir_name
        dir2 = relax2.output.dir_name
        struct1 = relax1.output.structure
        struct2 = relax2.output.structure

        deformations1 = calculate_energy_curve(
            struct1,
            struct2,
            distortions=self.distortions,
            static_maker=self.static_maker,
            prev_vasp_dir=dir1,
            add_name=f"q={charge_state1}",
        )

        deformations2 = calculate_energy_curve(
            struct2,
            struct1,
            distortions=self.distortions,
            static_maker=self.static_maker,
            prev_vasp_dir=dir2,
            add_name=f"q={charge_state2}",
        )

        deformations1.append_name(f" q={charge_state1}")
        deformations2.append_name(f" q={charge_state2}")

        ccd_docs = get_ccd_from_task_docs(
            deformations1.output, deformations2.output, struct1, struct2
        )

        if self.wswq:
            raise NotImplementedError("WSWQ not implemented yet")

        return Flow(
            jobs=[relax1, relax2, deformations1, deformations2, ccd_docs],
            output=ccd_docs.output,
            name=name,
        )
