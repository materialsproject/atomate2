"""Flows used in the calculation of defect properties."""


import logging
from dataclasses import dataclass, field

from jobflow import Maker
from pymatgen.core.structure import Structure

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker
from atomate2.vasp.sets.core import RelaxSetGenerator
from atomate2.vasp.sets.defect import AtomicRelaxSetGenerator

logger = logging.getLogger(__name__)


DEFAULT_DISTORTIONS = (-0.15 - 0.1, 0.05, 0.05, 0.1, 0.15, 1)
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
        struct1 = structure.copy()
        struct1.set_charge(charge_state1)
        struct2 = structure.copy()
        struct2.set_charge(charge_state2)

        self.relax_maker.make(struct1)
        self.relax_maker.make(struct2)
