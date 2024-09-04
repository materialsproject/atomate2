"""Module defining elastic document builder."""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
from maggma.builders import Builder
from pydash import get
from pymatgen.analysis.elasticity import Deformation, Stress

from atomate2 import SETTINGS
from atomate2.common.schemas.elastic import ElasticDocument

if TYPE_CHECKING:
    from collections.abc import Generator

    from maggma.core import Store


class ElasticBuilder(Builder):
    """
    The elastic builder compiles deformation tasks into an elastic document.

    The process can be summarised as:

    1. Find all deformation documents with the same formula.
    2. Group the deformations by their parent structures.
    3. Create an ElasticDocument from the group of tasks.

    Parameters
    ----------
    tasks : .Store
        Store of task documents.
    elasticity : .Store
        Store for final elastic documents.
    query : dict
        Dictionary query to limit tasks to be analyzed.
    sympec : float
        Symmetry precision for desymmetrising deformations.
    fitting_method : str
        The method used to fit the elastic tensor. See pymatgen for more details on the
        methods themselves. The options are:
        - "finite_difference" (note this is required if fitting a 3rd order tensor)
        - "independent"
        - "pseudoinverse"
    structure_match_tol : float
        Numerical tolerance for structure equivalence.
    **kwargs
        Keyword arguments that will be passed to the Builder init.
    """

    def __init__(
        self,
        tasks: Store,
        elasticity: Store,
        query: dict = None,
        symprec: float = SETTINGS.SYMPREC,
        fitting_method: str = SETTINGS.ELASTIC_FITTING_METHOD,
        structure_match_tol: float = 1e-5,
        **kwargs,
    ) -> None:
        self.tasks = tasks
        self.elasticity = elasticity
        self.query = query or {}
        self.kwargs = kwargs
        self.symprec = symprec
        self.fitting_method = fitting_method
        self.structure_match_tol = structure_match_tol

        super().__init__(sources=[tasks], targets=[elasticity], **kwargs)

    def ensure_indexes(self) -> None:
        """Ensure indices on the tasks and elasticity collections."""
        self.tasks.ensure_index("output.formula_pretty")
        self.tasks.ensure_index("last_updated")
        self.elasticity.ensure_index("fitting_data.uuids.0")
        self.elasticity.ensure_index("last_updated")

    def get_items(self) -> Generator:
        """Get all items to process into elastic documents.

        Yields
        ------
        list of dict
            A list of deformation tasks aggregated by formula and containing the
            required data to generate elasticity documents.
        """
        self.logger.info("Elastic builder started")
        self.logger.debug("Adding indices")
        self.ensure_indexes()

        # query for deformations
        qry = dict(self.query) | {
            "output.transformations.history.0.@class": "DeformationTransformation",
            "output.orig_inputs.NSW": {"$gt": 1},
            "output.orig_inputs.ISIF": {"$gt": 2},
        }
        return_props = [
            "uuid",
            "output.transformations",
            "output.output.stress",
            "output.formula_pretty",
            "output.dir_name",
        ]

        self.logger.info("Starting aggregation")
        n_formulas = len(self.tasks.distinct("output.formula_pretty", criteria=qry))
        results = self.tasks.groupby(
            "output.formula_pretty", criteria=qry, properties=return_props
        )
        self.logger.info("Aggregation complete")

        for idx, (keys, docs) in enumerate(results):
            formula = keys["output"]["formula_pretty"]
            self.logger.debug(f"Getting {formula} ({idx + 1} of {n_formulas})")
            yield docs

    def process_item(self, tasks: list[dict]) -> list[ElasticDocument]:
        """
        Process deformation tasks into elasticity documents.

        The deformation tasks will be grouped based on their parent structure (i.e., the
        structure before the deformation was applied).

        Parameters
        ----------
        tasks : list of dict
            A list of deformation task, all with the same formula.

        Returns
        -------
        list of .ElasticDocument
            A list of elastic documents for each unique parent structure.
        """
        self.logger.debug(f"Processing {tasks[0]['output']['formula_pretty']}")

        if not tasks:
            return []

        # group deformations by parent structure
        grouped = _group_deformations(tasks, self.structure_match_tol)

        elastic_docs = []
        for group in grouped:
            elastic_doc = _get_elastic_document(
                group, self.symprec, self.fitting_method
            )
            elastic_docs.append(elastic_doc)

        return elastic_docs

    def update_targets(self, items: list[ElasticDocument]) -> None:
        """
        Insert new elastic documents into the elasticity store.

        Parameters
        ----------
        items : list of .ElasticDocument
            A list of elasticity documents.
        """
        _items = chain.from_iterable(filter(bool, items))

        if len(list(_items)) > 0:
            self.logger.info(f"Updating {len(list(_items))} elastic documents")
            self.elasticity.update(_items, key="fitting_data.uuids.0")
        else:
            self.logger.info("No items to update")


def _group_deformations(tasks: list[dict], tol: float) -> list[list[dict]]:
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
        orig_structure = get(task, "output.transformations.history.0.input_structure")

        match = False
        for group in grouped_tasks:
            group_orig_structure = get(
                group[0], "output.transformations.history.0.input_structure"
            )

            # strict but fast structure matching, the structures should be identical
            lattice_match = np.allclose(
                orig_structure.lattice.matrix,
                group_orig_structure.lattice.matrix,
                atol=tol,
            )
            coords_match = np.allclose(
                orig_structure.frac_coords, group_orig_structure.frac_coords, atol=tol
            )
            if lattice_match and coords_match:
                group.append(task)
                match = True
                break

        if not match:
            # no match; start a new group
            grouped_tasks.append([task])

    return grouped_tasks


def _get_elastic_document(
    tasks: list[dict],
    symprec: float,
    fitting_method: str,
) -> ElasticDocument:
    """
    Turn a list of deformation tasks into an elastic document.

    Parameters
    ----------
    tasks : list of dict
        A list of deformation tasks.
    symprec : float
        Symmetry precision for deriving symmetry equivalent deformations. If
        ``symprec=None``, then no symmetry operations will be applied.
    fitting_method : str
        The method used to fit the elastic tensor. See pymatgen for more details on the
        methods themselves. The options are:
        - "finite_difference" (note this is required if fitting a 3rd order tensor)
        - "independent"
        - "pseudoinverse"

    Returns
    -------
    ElasticDocument
        An elastic document.
    """
    structure = get(tasks[0], "output.transformations.history.0.input_structure")

    stresses = []
    deformations = []
    uuids = []
    job_dirs = []
    for doc in tasks:
        deformation = get(doc, "output.transformations.history.0.deformation")
        stress = get(doc, "output.output.stress")

        deformations.append(Deformation(deformation))
        stresses.append(Stress(stress))
        uuids.append(doc["uuid"])
        job_dirs.append(doc["output"]["dir_name"])

    return ElasticDocument.from_stresses(
        structure,
        stresses,
        deformations,
        uuids,
        job_dirs,
        fitting_method=fitting_method,
        symprec=symprec,
    )
