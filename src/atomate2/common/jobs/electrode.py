"""Jobs for electrode analysis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from jobflow import Maker, job

if TYPE_CHECKING:
    from pymatgen.alchemy import ElementLike
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure


logger = logging.getLogger(__name__)


@job
def get_stable_inserted_structure(
    structure: Structure,
    inserted_species: ElementLike,
    structure_matcher: StructureMatcher,
    static_maker: Maker,
    relax_maker: Maker,
    get_charge_density: Callable,
    check_static_maker: Callable = lambda: True,
    insertions_per_step: int = 4,
):
    """Attempt ion insertion.

    The basic unit for cation insertion is:
        [get_stable_inserted_structure]:
            (static) -> N x (chgcar analysis -> relax) -> (return best structure)


    Args:
        structure: The structure to insert into.
        inserted_species: The species to insert.
        structure_matcher: The structure matcher to use to determine if additional
            insertion is needed.
        static_maker: A maker to perform static calculations.
        relax_maker: A maker to perform relaxation calculations.
        insertions_per_step: The maximum number of ion insertion sites to attempt.
        use_aeccar: Whether to use the AECCAR0 and AECCAR2 files for the charge density.
            This is often necessary since the CHGCAR file has spurious effects near the
            core which often breaks the min-filter algorithms used to identify the local
            minima.
    """
