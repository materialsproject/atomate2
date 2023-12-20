"""Flow for electrode analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher

from atomate2.common.jobs.electrode import get_stable_inserted_structure

if TYPE_CHECKING:
    from pymatgen.alchemy import ElementLike
    from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)


@dataclass
class ElectrodeInsertionMaker(Maker):
    """Attempt ion insertion into a structure.

    The basic unit for cation insertion is:
        [get_stable_inserted_structure]:
            (static) -> N x (chgcar analysis -> relax) -> (return best structure)

    The workflow is:
        [relax structure]
        [get_stable_inserted_structure]
        [relax structure]
        [get_stable_inserted_structure]

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

    def make(self, structure: Structure, inserted_species: ElementLike) -> Flow:
        """Make the flow.

        Args:
            structure: Structure to insert ion into.
            inserted_species: Species to insert.

        Returns
        -------
            Flow for ion insertion.
        """
        # First relax the structure
        relax = self.relax_maker.make(structure)
        structure = relax.output.structure

        # Get the inserted structure
        inserted_structure = get_stable_inserted_structure(
            structure=structure,
        )
        return Flow.from_job(inserted_structure)
