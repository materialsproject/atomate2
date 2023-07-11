"""Module defining elastic document builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from emmet.core.utils import jsanitize
from maggma.builders import Builder
from monty.serialization import MontyDecoder

from atomate2.common.schemas.magnetism import MagneticOrderingDocument
from atomate2.vasp.schemas.magnetism import (
    MagneticOrderingOutput,
    MagneticOrderingRelaxation,
)

if TYPE_CHECKING:
    from maggma.core import Store


class MagneticOrderingBuilder(Builder):
    """
    Builder to analyze the results of magnetic orderings calculations in VASP. This job
    will process the output documents of the calculations and return new documents
    with relevant parameters (such as the total magnetization, whether the ordering
    changed, whether the particular ordering is the ground state, etc.)

    Parameters
    ----------
    tasks : .Store
        Store of task documents.
    magnetic_orderings : .Store
        Store for magnetic ordering documents.
    query : dict
        Dictionary query to limit tasks to be analyzed.
    structure_match_tol : float
        Numerical tolerance for structure equivalence.
    **kwargs
        Keyword arguments that will be passed to the Builder init.
    """

    def __init__(
        self,
        tasks: Store,
        magnetic_orderings: Store,
        query: dict = None,
        structure_match_tol: float = 1e-5,
        **kwargs,
    ):
        self.tasks = tasks
        self.magnetic_orderings = magnetic_orderings
        self.query = query if query else {}
        self.structure_match_tol = structure_match_tol
        self.kwargs = kwargs

        super().__init__(sources=[tasks], targets=[magnetic_orderings], **kwargs)

    def ensure_indexes(self):
        """Ensure indices on the tasks and magnetic orderings collections."""
        self.tasks.ensure_index("output.formula_pretty")
        self.tasks.ensure_index("last_updated")
        self.magnetic_orderings.ensure_index("last_updated")

    def get_items(self):
        """
        Get all items to process into magnetic ordering documents. This step does a
        first grouping by formula (which is fast) and then the magnetic orderings are
        grouped by parent structure.

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
        criteria.update(
            {
                "metadata.ordering": {"$exists": True},
            }
        )
        self.logger.info("Grouping by formula...")
        nformulas = len(self.tasks.distinct("output.formula_pretty", criteria=criteria))
        results = self.tasks.groupby("output.formula_pretty", criteria=criteria)
        self.logger.info("Aggregation complete")

        for n, (keys, docs) in enumerate(results):
            formula = keys["output"]["formula_pretty"]
            self.logger.debug(f"Getting {formula} (Formula {n + 1} of {nformulas})")
            decoded_docs = MontyDecoder().process_decoded(docs)
            grouped_tasks = _group_orderings(decoded_docs, self.structure_match_tol)
            n_groups = len(grouped_tasks)
            for n, group in enumerate(grouped_tasks):
                self.logger.debug(
                    f"Found {len(group)} structures for {formula} (Parent structure"
                    f" {n+1} of {n_groups})"
                )
                yield group

    def process_item(self, tasks: list[dict]) -> list[MagneticOrderingDocument]:
        """
        Process magnetic ordering relaxation/static calculations into magnetic ordering documents.

        The magnetic ordering tasks will be grouped based on their parent structure (i.e., the
        structure before the magnetic ordering transformation was applied).

        Parameters
        ----------
        tasks : list of dict
            A list of magnetic ordering tasks grouped by same formula.

        Returns
        -------
        list of .MagneticOrderingDocument
            A list of magnetic ordering documents (one for each unique parent structure).
        """
        self.logger.debug(f"Processing {tasks[0]['output'].formula_pretty}")

        if not tasks:
            return []

        parent_structure = tasks[0]["metadata"]["parent_structure"]

        relax_tasks, static_tasks = [], []
        for task in tasks:
            if task["output"].task_type.value.lower() == "structure optimization":
                relax_tasks.append(task)
            elif task["output"].task_type.value.lower() == "static":
                static_tasks.append(task)

        outputs = []
        for task in static_tasks:
            relax_output, energy_diff = None, None
            for r_task in relax_tasks:
                if r_task["uuid"] == task["metadata"]["parent_uuid"]:
                    relax_output = MagneticOrderingRelaxation.from_task_document(
                        r_task["output"],
                        uuid=r_task["uuid"],
                    )
                    break
            output = MagneticOrderingOutput.from_task_document(
                task["output"],
                uuid=task["uuid"],
            )
            if relax_output is not None:
                energy_diff = output.energy_per_atom - relax_output.energy_per_atom
            output.relax_output = relax_output
            output.energy_diff_relax_static = energy_diff
            outputs.append(output)

        doc = jsanitize(
            MagneticOrderingDocument.from_outputs(
                outputs, parent_structure=parent_structure
            ).dict(),
            allow_bson=True,
        )

        return doc

    def update_targets(self, items: list[MagneticOrderingDocument]):
        """
        Insert new magnetic orderings into the magnetic orderings Store.

        Parameters
        ----------
        items : list of .MagneticOrderingDocument
            A list of magnetic ordering documents to add to the database.
        """
        self.logger.info(f"Updating {len(items)} magnetic orderings documents")
        self.magnetic_orderings.update(items, key="ground_state_uuid")


def _group_orderings(tasks: list[dict], tol: float) -> list[list[dict]]:
    """
    Group deformation tasks by their parent structure.

    Parameters
    ----------
    tasks : list of dict
        A list of deformation tasks.
    tol : float
        Numerical tolerance for structure equivalence.

    Returns
    -------
    list of list of dict
        The tasks grouped by their parent (undeformed structure).
    """
    grouped_tasks = [[tasks[0]]]

    for task in tasks[1:]:
        parent_structure = task["metadata"]["parent_structure"]

        match = False
        for group in grouped_tasks:
            group_parent_structure = task["metadata"]["parent_structure"]

            # parent structure should really be exactly identical (from same workflow)
            lattice_match = np.allclose(
                parent_structure.lattice.matrix,
                group_parent_structure.lattice.matrix,
                atol=tol,
            )
            coords_match = np.allclose(
                parent_structure.frac_coords,
                group_parent_structure.frac_coords,
                atol=tol,
            )
            if lattice_match and coords_match:
                group.append(task)
                match = True
                break

        if not match:
            grouped_tasks.append([task])

    return grouped_tasks
