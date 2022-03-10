"""Schemas for defect documents."""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, List, Tuple

import numpy as np
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry

from atomate2.vasp.schemas.task import TaskDocument

logger = logging.getLogger(__name__)

__all__ = [
    "CCDTaskDocument",
]


class CCDTaskDocument(BaseModel):
    """Configuration-coordiante definition of configuration-coordinate diagram."""

    q1: int = Field(None, description="Charge state 1.")
    q2: int = Field(None, description="Charge state 2.")
    structure1: Structure = Field(
        None, description="The structure of defect (supercell) in charge state (q2)."
    )
    structure2: Structure = Field(
        None, description="The structure of defect (supercell) in charge state (q2)."
    )

    distortions1: List[float] = Field(
        None,
        description="The distortions of the defect (supercell) in charge state (q1).",
    )
    distortions2: List[float] = Field(
        None,
        description="The distortions of the defect (supercell) in charge state (q2).",
    )

    energies1: List[float] = Field(
        None, description="The energies of the defect (supercell) in charge state (q1)."
    )
    energies2: List[float] = Field(
        None, description="The energies of the defect (supercell) in charge state (q2)."
    )

    distorted_calcs_dirs: List[List[str]] = Field(
        None,
        description="Directories of distorted calculations, stored as seperate lists for each charge state",
    )

    @classmethod
    def from_distorted_calcs(
        cls,
        distortion1_tasks: Iterable[TaskDocument],
        distortion2_tasks: Iterable[TaskDocument],
        structure1: Structure,
        structure2: Structure,
    ):
        """
        Create a CCDTaskDocument from a list of distorted calculations.

        Parameters
        ----------
        structure1
            The structure of defect (supercell) in charge state (q1).
        structure2
            The structure of defect (supercell) in charge state (q2).
        distortion1_tasks
            List of distorted calculations for charge state 1.
        distortion2_tasks
            List of distorted calculations for charge state 2.

        """

        def get_ent(task: TaskDocument):
            return ComputedStructureEntry(
                structure=task.structure,
                energy=task.energy,
                data={"dir_name": task.dir_name},
            )

        entries1 = [get_ent(task) for task in distortion1_tasks]
        entries2 = [get_ent(task) for task in distortion2_tasks]

        return cls.from_entries(entries1, entries2, structure1, structure2)

    @classmethod
    def from_entries(
        cls,
        entries1: List[ComputedStructureEntry],
        entries2: List[ComputedStructureEntry],
        structure1: Structure | None = None,
        structure2: Structure | None = None,
    ):
        """
        Create a CCDTaskDocument from a list of distorted calculations.

        Parameters
        ----------
        entries1
            List of distorted calculations for charge state 1.
        entries2
            List of distorted calculations for charge state 2.
        structure1
            The structure of defect (supercell) in charge state (q1).
        structure2
            The structure of defect (supercell) in charge state (q2).

        """

        def dQ_entries(e1, e2):
            """Get the displacement between two entries."""
            return get_dQ(e1.structure, e2.structure)

        # if the structures are not provided, use the structures with the lowest energy
        if structure1 is None:
            ent1 = min(entries1, key=lambda e: e.energy_per_atom)
            structure1 = ent1.structure
        else:
            ent1 = min(entries1, key=lambda e: get_dQ(structure1, e.structure))

        if structure2 is None:
            ent2 = min(entries2, key=lambda e: e.energy_per_atom)
            structure2 = ent2.structure
        else:
            ent2 = min(entries2, key=lambda e: get_dQ(e.structure, structure2))

        s_entries1, distortions1 = sort_pos_dist(entries1, ent1, ent2, dist=dQ_entries)
        s_entries2, distortions2 = sort_pos_dist(entries2, ent1, ent2, dist=dQ_entries)

        energies1 = [entry.energy for entry in s_entries1]
        energies2 = [entry.energy for entry in s_entries2]

        dir_names = []
        if ent1.data.get("dir_name") is not None:
            dir_names.append([e.data["dir_name"] for e in s_entries1])
            dir_names.append([e.data["dir_name"] for e in s_entries1])

        obj = cls(
            q1=structure1.charge,
            q2=structure2.charge,
            structure1=structure1,
            structure2=structure2,
            distortions1=distortions1,
            distortions2=distortions2,
            energies1=energies1,
            energies2=energies2,
        )

        if dir_names:
            obj.distorted_calcs_dirs = dir_names

        return obj


def sort_pos_dist(
    list_in: List[Any], s1: Any, s2: Any, dist: Callable
) -> Tuple[List[Any], List[float]]:
    """
    Sort a list defined when we can only compute a positive-definite distance.

    Sometimes, we can only compute a positive-definite distance between two objects.
    (Ex. Displacement between two structures).
    In these cases, standard sorting algorithms will not work.
    Here, we accept two reference points to give some sense of direction.
    We then sort the list based on the distance between the reference points.
    Note: this only works if the list falls on a line of some sort

    Parameters
    ----------
    list_in
        The list to sort.
    s1
        The first reference point.
    s2
        The second reference point.
    dist
        The distance function.

    Returns
    -------
    List[Any]
        The sorted list.
    """
    d1 = [dist(s, s1) for s in list_in]
    d2 = [dist(s, s2) for s in list_in]
    D0 = dist(s1, s2)

    d_vs_s = []
    for q1, q2, s in zip(d1, d2, list_in):
        sign = +1
        if q1 < q2 and q2 > D0:
            sign = -1
        d_vs_s.append((sign * q1, s))
    d_vs_s.sort()
    return [s for _, s in d_vs_s], [d for d, _ in d_vs_s]


def get_dQ(ref: Structure, distorted: Structure) -> float:
    """
    Calculate dQ from the initial and final structures.

    Parameters
    ----------
    ground : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the ground (final) state
    excited : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the excited (initial) state

    Returns
    -------
    float
        the dQ value (amu^{1/2} Angstrom)
    """
    return np.sqrt(
        np.sum(
            list(
                map(
                    lambda x: x[0].distance(x[1]) ** 2 * x[0].specie.atomic_mass,
                    zip(ref, distorted),
                )
            )
        )
    )
