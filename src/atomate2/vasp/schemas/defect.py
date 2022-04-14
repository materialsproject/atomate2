"""Schemas for defect documents."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, List, Tuple

import numpy as np
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.outputs import WSWQ

from atomate2.vasp.schemas.task import TaskDocument

logger = logging.getLogger(__name__)

__all__ = ["CCDDocument", "WSWQDocument", "FiniteDifferenceDocument"]


class WSWQDocument(BaseModel):
    """WSWQ document schema."""

    nspin: int = Field(None, description="Number of spins channels")
    nkpoints: int = Field(None, description="Number of k-points")
    nbands: int = Field(None, description="Number of bands")
    matrix_elements: List[List[List[List[float]]]] = Field(
        None,
        description="Array of of real numbers representing the matrix element |<W(0)|S|W(Q)>|\n"
        "Since complex numbers are not JSON serializable, we store the absolute values",
    )

    dir0: str = Field(
        None, description="Directory where the W(0) wavefunction comes from"
    )
    uuid0: str = Field(None, description="UUID of the W(0) calculation")

    dir1: str = Field(
        None, description="Directory where the W(Q) wavefunction comes from"
    )
    uuid1: str = Field(None, description="UUID of the W(Q) calculation")

    @classmethod
    def from_file(cls, filename: str | Path, **kwargs) -> WSWQDocument:
        """
        Read the WSWQ file.

        Parameters
        ----------
        filename : str
            Path to the WSWQ file.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        WSWQDocument
            WSWQDocument object.
        """
        fname = str(filename)
        wswq = WSWQ.from_file(fname)
        return cls.from_wswq(wswq, **kwargs)

    @classmethod
    def from_wswq(cls, wswq: WSWQ, **kwargs) -> WSWQDocument:
        """
        Read the WSWQ file.

        Parameters
        ----------
        wswq : WSWQ
            WSWQ object.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        WSWQDocument
            WSWQDocument object.
        """
        # TODO make the pymatgen code automatically determine if the object is complex or absolute value
        data = np.abs(wswq.data)
        return cls(
            nspin=wswq.nspin,
            nkpoints=wswq.nkpoints,
            nbands=wswq.nbands,
            data=data.tolist(),
            **kwargs,
        )

    def to_wswq(self) -> WSWQ:
        """
        Convert to WSWQ object.

        Returns
        -------
        WSWQ
            WSWQ object.
        """
        return WSWQ(
            nspin=self.nspin,
            nkpoints=self.nkpoints,
            nbands=self.nbands,
            data=np.array(self.matrix_elements),
        )


class FiniteDifferenceDocument(BaseModel):
    """Collection of computed WSWQDocuments using a single ref WAVECAR and a list of distorted WAVECARs."""

    wswq_documents: List[WSWQDocument]
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
            wswq_documents.append(WSWQDocument.from_file(f))

        return cls(wswq_documents=wswq_documents, dir_name=str(wswq_dir), **kwargs)


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

    static_dirs1: List[List[str]] = Field(
        None,
        description="Directories of distorted calculations for the defect (supercell) in charge state (q1).",
    )

    static_dirs2: List[List[str]] = Field(
        None,
        description="Directories of distorted calculations for the defect (supercell) in charge state (q2).",
    )

    static_uuids1: List[List[str]] = Field(
        None,
        description="UUIDs of distorted calculations for the defect (supercell) in charge state (q1).",
    )

    static_uuids2: List[List[str]] = Field(
        None,
        description="UUIDs of distorted calculations for the defect (supercell) in charge state (q2).",
    )

    relaxed_calc_dir1: str = Field(
        None,
        description="Directory of relaxed calculation in charge state (q1).",
    )

    relaxed_calc_dir2: str = Field(
        None,
        description="Directory of relaxed calculation in charge state (q2).",
    )

    @classmethod
    def from_struct_en(
        cls,
        structures1: List[Structure],
        structures2: List[Structure],
        energies1: List[float],
        energies2: List[float],
        dir_names1: List[str],
        dir_names2: List[str],
        static_uuids1: List[str],
        static_uuids2: List[str],
        relaxed_uuid1: str,
        relaxed_uuid2: str,
    ):
        """
        Create a CCDDocument from a list of distorted calculations.

        Parameters
        ----------
        structure1
            The structure of defect (supercell) in charge state (q1).
        structure2
            The structure of defect (supercell) in charge state (q2).
        distortion1_calcs
            List of distorted calculations for charge state 1.
        distortion2_calcs
            List of distorted calculations for charge state 2.

        """

        def get_ent(struct, energy, dir_name, uuid):
            return ComputedStructureEntry(
                structure=struct,
                energy=energy,
                data={"dir_name": dir_name, "uuid": uuid},
            )

        entries1 = [
            get_ent(s, e, d, u)
            for s, e, d, u in zip(structures1, energies1, dir_names1, static_uuids1)
        ]
        entries2 = [
            get_ent(s, e, d, u)
            for s, e, d, u in zip(structures2, energies2, dir_names2, static_uuids2)
        ]

        return cls.from_entries(entries1, entries2, relaxed_uuid1, relaxed_uuid2)

    @classmethod
    def from_entries(
        cls,
        entries1: List[ComputedStructureEntry],
        entries2: List[ComputedStructureEntry],
        relaxed_uuid1: str | None = None,
        relaxed_uuid2: str | None = None,
    ):
        """
        Create a CCDTaskDocument from a list of distorted calculations.

        Parameters
        ----------
        entries1
            List of distorted calculations for charge state (q1).
        entries2
            List of distorted calculations for charge state (q2)
        relaxed_dir1
            Directory of relaxed calculation in charge state (q1).
        relaxed_dir2
            Directory of relaxed calculation in charge state (q2).

        """

        def find_entry(entries, uuid):
            """Find the entry with the given given UUID."""
            for entry in entries:
                if entry.data["uuid"] == uuid:
                    return entry
            raise ValueError(f"Could not find entry with UUID: {uuid}")

        def dQ_entries(e1, e2):
            """Get the displacement between two entries."""
            return get_dQ(e1.structure, e2.structure)

        # ensure the "dir_name" is provided for each entry
        if any(e.data.get("dir_name", None) is None for e in entries1):
            raise ValueError("dir_name must be provided for all entries.")

        ent_r1 = find_entry(entries1, relaxed_uuid1)
        ent_r2 = find_entry(entries2, relaxed_uuid2)

        s_entries1, distortions1 = sort_pos_dist(
            entries1, ent_r1, ent_r2, dist=dQ_entries
        )
        s_entries2, distortions2 = sort_pos_dist(
            entries2, ent_r1, ent_r2, dist=dQ_entries
        )

        energies1 = [entry.energy for entry in s_entries1]
        energies2 = [entry.energy for entry in s_entries2]

        dir_names = []
        if ent_r1.data.get("dir_name") is not None:
            dir_names.append([e.data["dir_name"] for e in s_entries1])
            dir_names.append([e.data["dir_name"] for e in s_entries1])

        obj = cls(
            q1=ent_r1.structure.charge,
            q2=ent_r2.structure.charge,
            structure1=ent_r1.structure,
            structure2=ent_r2.structure,
            distortions1=distortions1,
            distortions2=distortions2,
            energies1=energies1,
            energies2=energies2,
            distorted_calcs_dirs=dir_names,
            relaxed_calc_dir1=ent_r1.data["dir_name"],
            relaxed_calc_dir2=ent_r2.data["dir_name"],
        )

        return obj

    def get_taskdocs(self):
        """Get the distorted task documents."""

        def remove_host_name(dir_name):
            return dir_name.split(":")[-1]

        return [
            [
                TaskDocument.from_directory(remove_host_name(dir_name))
                for dir_name in self.distorted_calcs_dirs[0]
            ],
            [
                TaskDocument.from_directory(remove_host_name(dir_name))
                for dir_name in self.distorted_calcs_dirs[1]
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
