"""Flows used in the calculation of defect properties."""


import logging
from dataclasses import dataclass, field

from jobflow import Flow, Maker
from pymatgen.core.structure import Structure

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker
from atomate2.vasp.jobs.defect import calculate_energy_curve
from atomate2.vasp.sets.core import RelaxSetGenerator
from atomate2.vasp.sets.defect import AtomicRelaxSetGenerator

logger = logging.getLogger(__name__)


DEFAULT_DISTORTIONS = (-0.15 - 0.1, 0.05, 0, 0.05, 0.1, 0.15, 1)
DEFECT_RELAX_GEN = AtomicRelaxSetGenerator(use_structure_charge=True)


@dataclass
class ConfigurationCoordinateMaker(Maker):
    """Class to generate VASP input sets for the calculation of the configuration coordinate diagram."""

    name: str = "config. coordinate"
    charged_relax_maker: BaseVaspMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=RelaxSetGenerator(use_structure_charge=True)
        )
    )
    static_maker: BaseVaspMaker = field(default_factory=StaticMaker)
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
        struct1: Structure = structure.copy()
        struct1.set_charge(charge_state1)
        struct2: Structure = structure.copy()
        struct2.set_charge(charge_state2)

        relax1 = self.relax_maker.make(struct1)
        relax2 = self.relax_maker.make(struct2)

        struct1 = relax1.output.structure
        struct2 = relax2.output.structure
        static1 = self.static_maker.make(struct1)
        static2 = self.static_maker.make(struct2)
        static1_dir = static1.output.dir_name
        static2_dir = static2.output.dir_name

        deformations1 = calculate_energy_curve(
            struct1,
            struct2,
            distortions=self.distortions,
            static_maker=self.static_maker,
            prev_base_dir=static1_dir,
        )
        deformations2 = calculate_energy_curve(
            struct2,
            struct1,
            distortions=self.distortions,
            static_maker=self.static_maker,
            prev_base_dir=static2_dir,
        )
        jobs = [relax1, relax2, static1, static2, deformations1, deformations2]
        return Flow(jobs=jobs, name=name, output=[deformations1, deformations2])
