"""Module defining DFT code agnostic magnetic orderings builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from emmet.core.utils import jsanitize
from maggma.builders import Builder
from monty.serialization import MontyDecoder
from pymatgen.analysis.structure_matcher import StructureMatcher

from atomate2.common.schemas.magnetism import MagneticOrderingsDocument

if TYPE_CHECKING:
    from collections.abc import Iterator

    from maggma.core import Store


class MagneticOrderingsBuilder(Builder):
    """Builder to analyze the results of magnetic orderings calculations.

    This job will process the output documents of the calculations and return new
    documents with relevant parameters (such as the total magnetization, whether the
    ordering changed, whether the particular ordering is the ground state, etc.). This
    is especially useful for performing postprocessing of magnetic ordering
    calculations.

    Parameters
    ----------
    tasks : .Store
        Store of task documents.
    magnetic_orderings : .Store
        Store for magnetic ordering documents.
    query : dict
        Dictionary query to limit tasks to be analyzed.
    structure_match_stol : float
        Numerical site tolerance for structure equivalence. Default is 0.3.
    structure_match_ltol : float
        Numerical length tolerance for structure equivalence. Default is 0.3
    structure_match_angle_tol : float
        Numerical angle tolerance in degrees for structure equivalence. Default is 5.
    **kwargs : dict
        Keyword arguments that will be passed to the Builder init.
    """

    def __init__(
        self,
        tasks: Store,
        magnetic_orderings: Store,
        query: dict = None,
        structure_match_stol: float = 0.3,
        structure_match_ltol: float = 0.2,
        structure_match_angle_tol: float = 5,
        **kwargs,
    ) -> None:
        self.tasks = tasks
        self.magnetic_orderings = magnetic_orderings
        self.query = query or {}
        self.structure_match_stol = structure_match_stol
        self.structure_match_ltol = structure_match_ltol
        self.structure_match_angle_tol = structure_match_angle_tol

        self.kwargs = kwargs

        super().__init__(sources=[tasks], targets=[magnetic_orderings], **kwargs)

    def ensure_indexes(self) -> None:
        """Ensure indices on the tasks and magnetic orderings collections."""
        self.tasks.ensure_index("output.formula_pretty")
        self.tasks.ensure_index("last_updated")
        self.magnetic_orderings.ensure_index("last_updated")

    def get_items(self) -> Iterator[list[dict]]:
        """Get all items to process into magnetic ordering documents.

        This step does a first grouping by formula (which is fast) and then the magnetic
        orderings are grouped by parent structure.

        Yields
        ------
        list of dict
            A list of magnetic ordering relaxation or static task outputs, grouped by
            formula.
        """
        self.logger.info("Magnetic orderings builder started")
        self.logger.debug("Adding/ensuring indices...")
        self.ensure_indexes()

        criteria = dict(self.query)
        criteria.update({"metadata.ordering": {"$exists": True}})
        self.logger.info("Grouping by formula...")
        num_formulas = len(
            self.tasks.distinct("output.formula_pretty", criteria=criteria)
        )
        results = self.tasks.groupby("output.formula_pretty", criteria=criteria)

        for n_formula, (keys, docs) in enumerate(results):
            formula = keys["output"]["formula_pretty"]
            self.logger.debug(
                "Getting %s (Formula %d of %d)", formula, n_formula + 1, num_formulas
            )
            decoded_docs = MontyDecoder().process_decoded(docs)
            grouped_tasks = _group_orderings(
                decoded_docs,
                self.structure_match_ltol,
                self.structure_match_stol,
                self.structure_match_angle_tol,
            )
            n_groups = len(grouped_tasks)
            for n_group, group in enumerate(grouped_tasks):
                self.logger.debug(
                    "Found %d tasks for %s (Parent structure %d of %d)",
                    len(group),
                    formula,
                    n_group + 1,
                    n_groups,
                )
                yield group

    def process_item(self, tasks: list[dict]) -> list[MagneticOrderingsDocument]:
        """Process magnetic ordering relaxation/static calculations into documents.

        The magnetic ordering tasks will be grouped based on their parent structure
        (i.e., the structure before the magnetic ordering transformation was applied).
        See _group_orderings for more details.

        Parameters
        ----------
        tasks : list[dict]
            A list of magnetic ordering tasks grouped by same formula.

        Returns
        -------
        list of .MagneticOrderingsDocument
            A list of magnetic ordering documents (one for each unique parent
            structure).
        """
        self.logger.debug("Processing %s", tasks[0]["output"].formula_pretty)

        if not tasks:
            return []

        return jsanitize(
            MagneticOrderingsDocument.from_tasks(tasks).model_dump(),
            allow_bson=True,
        )

    def update_targets(self, items: list[MagneticOrderingsDocument]) -> None:
        """Insert new magnetic orderings into the magnetic orderings Store.

        Parameters
        ----------
        items : list of .MagneticOrderingsDocument
            A list of magnetic ordering documents to add to the database.
        """
        self.logger.info("Updating %s magnetic orderings documents", len(items))
        self.magnetic_orderings.update(items, key="ground_state_uuid")


def _group_orderings(
    tasks: list[dict], ltol: float, stol: float, angle_tol: float
) -> list[list[dict]]:
    """Group ordering tasks by their parent structure.

    This is useful for distinguishing between different polymorphs (i.e., same formula).

    Parameters
    ----------
    tasks : list[dict]
        A list of ordering tasks.
    tol : float
        Numerical tolerance for structure equivalence.

    Returns
    -------
    list[list[dict]]
        The tasks grouped by their parent structure.
    """
    tasks = [dict(task) for task in tasks]

    grouped_tasks = [[tasks[0]]]
    sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)

    for task in tasks[1:]:
        parent_structure = MontyDecoder().process_decoded(
            task["metadata"]["parent_structure"]
        )

        match = False
        for group in grouped_tasks:
            group_parent_structure = MontyDecoder().process_decoded(
                group[0]["metadata"]["parent_structure"]
            )

            #  parent structure lattice/coords may be same but in different order
            #  so we need to be more rigorous in checking equivalence
            if sm.fit(parent_structure, group_parent_structure):
                group.append(task)
                match = True
                break

        if not match:
            grouped_tasks.append([task])

    return MontyDecoder().process_decoded(grouped_tasks)
