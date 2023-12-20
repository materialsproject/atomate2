"""Flow for electrode analysis."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher

from atomate2.common.jobs.electrode import get_stable_inserted_structure

if TYPE_CHECKING:
    from pymatgen.alchemy import ElementLike
    from pymatgen.core.structure import Structure
    from pymatgen.io.vasp.outputs import VolumetricData

logger = logging.getLogger(__name__)

__author__ = "Jimmy Shen"
__email__ = "jmmshn@gmail.com"


@dataclass
class ElectrodeInsertionMaker(Maker, ABC):
    """Attempt ion insertion into a structure.

    The basic unit for cation insertion is:
        [get_stable_inserted_structure]:
            (static) -> N x (chgcar analysis -> relax) -> (return best structure)

    The workflow is:
        [relax structure]
        [get_stable_inserted_structure]
        [get_stable_inserted_structure]
        [get_stable_inserted_structure]
        ... until the insertion is no longer topotactic.

    This workflow requires the users to provide the following functions:
        self.get_charge_density(task_doc: TaskDoc):
            Get the charge density of a TaskDoc output from a calculation.
        self.update_static_maker():
            Ensure that the static maker will store the desired data.

    Parameters
    ----------
    name: str
        The name of the flow created by this maker.
    relax_maker: RelaxMaker
        A maker to perform relaxation calculations.
    static_maker: Maker
        A maker to perform static calculations.
    insertions_per_step: int
        The maximum number of ion insertion sites to attempt.
    stucture_matcher: StructureMatcher
        The structure matcher to use to determine if additional insertion is needed.
    """

    name: str
    relax_maker: Maker
    static_maker: Maker
    insertions_per_step: int = 4
    structure_matcher: StructureMatcher = field(
        default_factory=lambda: StructureMatcher(
            comparator=ElementComparator(),
        )
    )

    def make(self, structure: Structure, inserted_element: ElementLike) -> Flow:
        """Make the flow.

        Parameters
        ----------
        structure: Structure to insert ion into.
        inserted_species: Species to insert.

        Returns
        -------
            Flow for ion insertion.
        """
        # First relax the structure
        relax = self.relax_maker.make(structure)
        self.update_static_maker()
        # Get the inserted structure
        inserted_structure = get_stable_inserted_structure(
            structure=relax.output.structure,
            inserted_element=inserted_element,
            structure_matcher=self.structure_matcher,
            static_maker=self.static_maker,
            relax_maker=self.relax_maker,
            get_charge_density=self.get_charge_density,
            insertions_per_step=self.insertions_per_step,
        )
        return Flow([relax, inserted_structure])

    @abstractmethod
    def get_charge_density(self, task_doc) -> VolumetricData:
        """Get the charge density of a structure.

        Args:
            structure: The structure to get the charge density of.

        Returns
        -------
            The charge density.
        """

    @abstractmethod
    def update_static_maker(self):
        """Ensure that the static maker will store the desired data."""
