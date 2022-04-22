"""Schemas for defect documents."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, List, Tuple, Type

import numpy as np
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.outputs import WSWQ

from atomate2.vasp.schemas.task import TaskDocument

logger = logging.getLogger(__name__)

__all__ = ["CCDDocument", "WSWQ", "FiniteDifferenceDocument"]


class FiniteDifferenceDocument(BaseModel):
    """Collection of computed WSWQ objects using a single ref WAVECAR and a list of distorted WAVECARs."""

    wswqs: List[WSWQ]

    dir_name: str = Field(
        None, description="Directory where the WSWQ calculations are performed"
    )
    ref_dir: str = Field(
        None, description="Directory where the reference W(0) wavefunction comes from"
    )
    distorted_dirs: List[str] = Field(
        None,
        description="List of directories where the distorted W(Q) wavefunctions come from",
    )

    @classmethod
    def from_directory(
        cls, directory: str | Path, **kwargs
    ) -> FiniteDifferenceDocument:
        """
        Read the FintieDiff file.

        Parameters
        ----------
        directory : str | Path
            Path to the FintieDiff directory.
        ref_dir : str
            Directory where the reference W(0) wavefunction comes from.
        distorted_dirs : List[str]
            List of directories where the distorted W(Q) wavefunctions come from.

        Returns
        -------
        FintieDiffDocument
            FintieDiffDocument object.
        """
        wswq_dir = Path(directory)
        files = list(Path(wswq_dir).glob("WSWQ.[0-9]*"))
        ordered_files = sorted(files, key=lambda x: int(x.name.split(".")[1]))
        wswq_documents = []
        for f in ordered_files:
            wswq_documents.append(WSWQ.from_file(f))

        return cls(wswqs=wswq_documents, dir_name=str(wswq_dir), **kwargs)


class CCDDocument(BaseModel):
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

    static_dirs1: List[str] = Field(
        None,
        description="Directories of distorted calculations for the defect (supercell) in charge state (q1).",
    )

    static_dirs2: List[str] = Field(
        None,
        description="Directories of distorted calculations for the defect (supercell) in charge state (q2).",
    )

    static_uuids1: List[str] = Field(
        None,
        description="UUIDs of distorted calculations for the defect (supercell) in charge state (q1).",
    )

    static_uuids2: List[str] = Field(
        None,
        description="UUIDs of distorted calculations for the defect (supercell) in charge state (q2).",
    )

    relaxed_calc1: tuple[int, str | None] = Field(
        None,
        description="The (index, directory name) of the static calculation in `energies1` that corresponds to "
        "the relaxed charge state (q1).",
    )

    relaxed_calc2: tuple[int, str] = Field(
        None,
        description="The (index, directory name) of the static calculation in `energies1` that corresponds to "
        "the relaxed charge state (q2).",
    )

    @classmethod
    def from_task_outputs(
        cls,
        structures1: List[Structure],
        structures2: List[Structure],
        energies1: List[float],
        energies2: List[float],
        static_dirs1: List[str],
        static_dirs2: List[str],
        static_uuids1: List[str],
        static_uuids2: List[str],
        relaxed_uuid1: str,
        relaxed_uuid2: str,
    ):
        """Create a CCDDocument from a lists of structures, energies from completed static calculations.

        The directories and the UUIDs of the static calculations are also provided as separate lists and zipped together.

        Parameters
        ----------
        structure1
            The structure of defect (supercell) in charge state (q1).
        structure2
            The structure of defect (supercell) in charge state (q2).
        energies1
            The energies of the defect (supercell) in charge state (q1).
        energies2
            The energies of the defect (supercell) in charge state (q2).
        static_dirs1
            Directories of distorted calculations for the defect (supercell) in charge state (q1).
        static_dirs2
            Directories of distorted calculations for the defect (supercell) in charge state (q2).
        static_uuids1
            UUIDs of distorted calculations for the defect (supercell) in charge state (q1).
        static_uuids2
            UUIDs of distorted calculations for the defect (supercell) in charge state (q2).
        relaxed_uuid1
            UUID of relaxed calculation in charge state (q1).
        relaxed_uuid2
            UUID of relaxed calculation in charge state (q2).
        """

        def get_ent(struct, energy, dir_name, uuid):
            return ComputedStructureEntry(
                structure=struct,
                energy=energy,
                data={"dir_name": dir_name, "uuid": uuid},
            )

        entries1 = [
            get_ent(s, e, d, u)
            for s, e, d, u in zip(structures1, energies1, static_dirs1, static_uuids1)
        ]
        entries2 = [
            get_ent(s, e, d, u)
            for s, e, d, u in zip(structures2, energies2, static_dirs2, static_uuids2)
        ]

        return cls.from_entries(entries1, entries2, relaxed_uuid1, relaxed_uuid2)

    @classmethod
    def from_entries(
        cls: Type[CCDDocument],
        entries1: List[ComputedStructureEntry],
        entries2: List[ComputedStructureEntry],
        relaxed_uuid1: str | None = None,
        relaxed_uuid2: str | None = None,
    ) -> CCDDocument:
        """
        Create a CCDTaskDocument from a list of distorted calculations.

        Parameters
        ----------
        entries1
            List of distorted calculations for charge state (q1).
        entries2
            List of distorted calculations for charge state (q2)
        relaxed_uuid1
            UUID of relaxed calculation in charge state (q1).
        relaxed_uuid1
            UUID of relaxed calculation in charge state (q2).

        """

        def find_entry(entries, uuid) -> tuple[int, ComputedStructureEntry]:
            """Find the entry with the given given UUID."""
            for itr, entry in enumerate(entries):
                if entry.data["uuid"] == uuid:
                    return itr, entry
            raise ValueError(f"Could not find entry with UUID: {uuid}")

        def dQ_entries(e1, e2):
            """Get the displacement between two entries."""
            return get_dQ(e1.structure, e2.structure)

        # ensure the "dir_name" is provided for each entry
        if any(e.data.get("dir_name", None) is None for e in entries1 + entries2):
            raise ValueError("[dir_name] must be provided for all entries.")

        if any(e.data.get("uuid", None) is None for e in entries1 + entries2):
            raise ValueError("[uuid] must be provided for all entries.")

        idx1, ent_r1 = find_entry(entries1, relaxed_uuid1)
        idx2, ent_r2 = find_entry(entries2, relaxed_uuid2)

        s_entries1, distortions1 = sort_pos_dist(
            entries1, ent_r1, ent_r2, dist=dQ_entries
        )
        s_entries2, distortions2 = sort_pos_dist(
            entries2, ent_r1, ent_r2, dist=dQ_entries
        )

        energies1 = [entry.energy for entry in s_entries1]
        energies2 = [entry.energy for entry in s_entries2]

        sdirs1 = [e.data["dir_name"] for e in s_entries1]
        sdirs2 = [e.data["dir_name"] for e in s_entries2]

        obj = cls(
            q1=ent_r1.structure.charge,
            q2=ent_r2.structure.charge,
            structure1=ent_r1.structure,
            structure2=ent_r2.structure,
            distortions1=distortions1,
            distortions2=distortions2,
            energies1=energies1,
            energies2=energies2,
            static_dirs1=sdirs1,
            static_dirs2=sdirs2,
            relaxed_calc1=(idx1, ent_r1.data["dir_name"]),
            relaxed_calc2=(idx2, ent_r2.data["dir_name"]),
        )

        return obj

    def get_taskdocs(self):
        """Get the distorted task documents."""

        def remove_host_name(dir_name):
            return dir_name.split(":")[-1]

        return [
            [
                TaskDocument.from_directory(remove_host_name(dir_name))
                for dir_name in self.static_dirs1
            ],
            [
                TaskDocument.from_directory(remove_host_name(dir_name))
                for dir_name in self.static_dirs2
            ],
        ]


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
    List[float]
        The signed distances to the reference point (s1).
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


def find_entry_with_dir_name(entries, dir_name):
    """Find the entry with the given dir_name."""
    for entry in entries:
        if entry.data["dir_name"] == dir_name:
            return entry
    raise ValueError(f"Could not find entry with dir_name {dir_name}")
