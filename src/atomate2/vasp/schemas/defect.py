"""Schemas for defect documents."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterable, List, Tuple

import numpy as np
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.outputs import WSWQ

from atomate2.vasp.schemas.task import TaskDocument

logger = logging.getLogger(__name__)

__all__ = [
    "CCDDocument",
]


class WSWQDocument(BaseModel):
    nspin: int = Field(None, description="Number of spins channels")
    nkpoints: int = Field(None, description="Number of k-points")
    nbands: int = Field(None, description="Number of bands")
    data: List[List[List[List[complex]]]] = Field(
        None,
        description="2D array of of complex numbers representing the <W(0)|S|W(Q)>",
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
        return cls(
            nspin=wswq.nspin,
            nkpoints=wswq.nkpoints,
            nbands=wswq.nbands,
            data=wswq.data.tolist(),
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
            data=np.array(self.data),
        )

    class Config:
        arbitrary_types_allowed = True


class FiniteDiffDocument(BaseModel):
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
    def from_directory(cls, directory: str | Path, **kwargs) -> FiniteDiffDocument:
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

    distorted_calcs_dirs: List[List[str]] = Field(
        None,
        description="Directories of distorted calculations, stored as seperate lists for each charge state",
    )

    @classmethod
    def from_distorted_calcs(
        cls,
        distortion1_calcs: Iterable[TaskDocument],
        distortion2_calcs: Iterable[TaskDocument],
        structure1: Structure,
        structure2: Structure,
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

        def get_ent(task: TaskDocument):
            return ComputedStructureEntry(
                structure=task.output.structure,
                energy=task.output.energy,
                data={"dir_name": task.dir_name},
            )

        entries1 = [get_ent(task) for task in distortion1_calcs]
        entries2 = [get_ent(task) for task in distortion2_calcs]

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
