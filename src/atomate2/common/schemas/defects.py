"""General schemas for defect workflow outputs."""

import logging
from collections.abc import Sequence
from itertools import starmap
from typing import Any, Callable, Optional, Union

import numpy as np
from emmet.core.tasks import TaskDoc
from pydantic import BaseModel, Field
from pymatgen.analysis.defects.core import Defect
from pymatgen.analysis.defects.thermo import DefectEntry, FormationEnergyDiagram
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from typing_extensions import Self

logger = logging.getLogger(__name__)


class FormationEnergyDiagramDocument(BaseModel):
    """A document for storing a formation energy diagram.

    Basically a pydantic version of the `FormationEnergyDiagram` dataclass with some
    additional data fields. The `pd_entries` field is now optional since the workflow
    will not necessarily have all the entries in the phase diagram computed.
    """

    bulk_entry: Optional[ComputedStructureEntry] = Field(
        None,
        description="The ComputedEntry representing the bulk structure.",
    )

    defect_entries: Optional[list[DefectEntry]] = Field(
        None,
        description="The defect entries for the formation energy diagram.",
    )

    pd_entries: Optional[list[ComputedEntry]] = Field(
        None,
        description="The entries used to construct the phase diagram.",
    )

    vbm: Optional[float] = Field(
        None,
        description="The VBM of the pristine supercell calculation.",
    )

    band_gap: Optional[float] = Field(
        None,
        description="The band gap of the pristine supercell calculation.",
    )

    inc_inf_values: Optional[bool] = Field(
        None,
        description="Whether or not to include infinite values in the diagram.",
    )

    defect: Optional[Defect] = Field(
        None,
        description="The defect for which the diagram is being calculated.",
    )

    bulk_sc_dir: Optional[str] = Field(
        None,
        description="The directory name of the pristine supercell calculation.",
    )

    defect_sc_dirs: Optional[dict[int, str]] = Field(
        None,
        description="The directory names of the charged defect calculations.",
    )

    dielectric: Optional[Union[float, list[list[float]]]] = Field(
        None,
        description="The dielectric constant or tensor, can be used to compute "
        "finite-size corrections.",
    )

    @classmethod
    def from_formation_energy_diagram(
        cls,
        fed: FormationEnergyDiagram,
        **kwargs,
    ) -> Self:
        """Create a document from a `FormationEnergyDiagram` object.

        Args:
            fed: The `FormationEnergyDiagram` object.
            kwargs: Additional keyword arguments to pass to the document.
        """
        defect = fed.defect_entries[0].defect
        return cls(
            defect=defect,
            bulk_entry=fed.bulk_entry,
            defect_entries=fed.defect_entries,
            vbm=fed.vbm,
            band_gap=fed.band_gap,
            pd_entries=fed.pd_entries,
            inc_inf_values=fed.inc_inf_values,
            **kwargs,
        )

    def as_formation_energy_diagram(
        self,
        pd_entries: Optional[list[ComputedEntry]] = None,
    ) -> FormationEnergyDiagram:
        """Create a `FormationEnergyDiagram` object from the document.

        Since the `pd_entries` field is optional, this method allows the user
        to pass in the phase diagram entries if they are not stored in this document.

        Args:
            pd_entries: The entries used to construct the phase diagram. If None,
            the `pd_entries` field of the document will be used.
        """
        if pd_entries is None:
            pd_entries = self.pd_entries
        return FormationEnergyDiagram(
            bulk_entry=self.bulk_entry,
            defect_entries=self.defect_entries,
            vbm=self.vbm,
            band_gap=self.band_gap,
            pd_entries=pd_entries,
            inc_inf_values=self.inc_inf_values,
        )


class CCDDocument(BaseModel):
    """Configuration-coordinate definition of configuration-coordinate diagram."""

    q1: Optional[int] = Field(None, description="Charge state 1.")
    q2: Optional[int] = Field(None, description="Charge state 2.")
    structure1: Optional[Structure] = Field(
        None,
        description="The structure of defect (supercell) in charge state (q2).",
    )
    structure2: Optional[Structure] = Field(
        None,
        description="The structure of defect (supercell) in charge state (q2).",
    )

    distortions1: Optional[list[float]] = Field(
        None,
        description="The distortions of the defect (supercell) in charge state (q1).",
    )
    distortions2: Optional[list[float]] = Field(
        None,
        description="The distortions of the defect (supercell) in charge state (q2).",
    )
    energies1: Optional[list[float]] = Field(
        None,
        description="The energies of the defect (supercell) in charge state (q1).",
    )
    energies2: Optional[list[float]] = Field(
        None,
        description="The energies of the defect (supercell) in charge state (q2).",
    )

    static_dirs1: Optional[list[str]] = Field(
        None,
        description="Directories of distorted calculations for the defect (supercell) "
        "in charge state (q1).",
    )

    static_dirs2: Optional[list[str]] = Field(
        None,
        description="Directories of distorted calculations for the defect (supercell) "
        "in charge state (q2).",
    )

    static_uuids1: Optional[list[str]] = Field(
        None,
        description="UUIDs of distorted calculations for the defect (supercell) in "
        "charge state (q1).",
    )

    static_uuids2: Optional[list[str]] = Field(
        None,
        description="UUIDs of distorted calculations for the defect (supercell) in "
        "charge state (q2).",
    )

    relaxed_index1: Optional[int] = Field(
        None,
        description="The index of the static calculation in that corresponds to the "
        "relaxed charge state (q1).",
    )

    relaxed_index2: Optional[int] = Field(
        None,
        description="The index of the static calculation in that corresponds to the "
        "relaxed charge state (q2).",
    )

    @classmethod
    def from_task_outputs(
        cls,
        structures1: list[Structure],
        structures2: list[Structure],
        energies1: list[float],
        energies2: list[float],
        static_dirs1: list[str],
        static_dirs2: list[str],
        static_uuids1: list[str],
        static_uuids2: list[str],
        relaxed_uuid1: str,
        relaxed_uuid2: str,
    ) -> Self:
        """Create a CCDDocument from a lists of structures and energies.

        The directories and the UUIDs of the static calculations are also provided as
        separate lists and zipped together.

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
            Directories of distorted calculations for the defect (supercell) in charge
            state (q1).
        static_dirs2
            Directories of distorted calculations for the defect (supercell) in charge
            state (q2).
        static_uuids1
            UUIDs of distorted calculations for the defect (supercell) in charge
            state (q1).
        static_uuids2
            UUIDs of distorted calculations for the defect (supercell) in charge
            state (q2).
        relaxed_uuid1
            UUID of relaxed calculation in charge state (q1).
        relaxed_uuid2
            UUID of relaxed calculation in charge state (q2).
        """

        def get_cs_entry(
            struct: Structure, energy: float, dir_name: str, uuid: str
        ) -> ComputedStructureEntry:
            data = {"dir_name": dir_name, "uuid": uuid}
            return ComputedStructureEntry(structure=struct, energy=energy, data=data)

        entries1 = list(
            starmap(
                get_cs_entry, zip(structures1, energies1, static_dirs1, static_uuids1)
            )
        )
        entries2 = list(
            starmap(
                get_cs_entry, zip(structures2, energies2, static_dirs2, static_uuids2)
            )
        )

        return cls.from_entries(entries1, entries2, relaxed_uuid1, relaxed_uuid2)

    @classmethod
    def from_entries(
        cls,
        entries1: list[ComputedStructureEntry],
        entries2: list[ComputedStructureEntry],
        relaxed_uuid1: Optional[str] = None,
        relaxed_uuid2: Optional[str] = None,
    ) -> Self:
        """Create a CCDTaskDocument from a list of distorted calculations.

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

        def find_entry(
            entries: Sequence[ComputedStructureEntry], uuid: str
        ) -> tuple[int, ComputedStructureEntry]:
            """Find the entry with the given UUID."""
            for idx, entry in enumerate(entries):
                if entry.data["uuid"] == uuid:
                    return idx, entry
            raise ValueError(f"Could not find entry with UUID: {uuid}")

        def dQ_entries(e1: ComputedStructureEntry, e2: ComputedStructureEntry) -> float:  # noqa: N802
            """Get the displacement between two entries."""
            return get_dQ(e1.structure, e2.structure)

        # ensure the "dir_name" is provided for each entry
        if any(entry.data.get("dir_name") is None for entry in entries1 + entries2):
            raise ValueError("[dir_name] must be provided for all entries.")

        if any(entry.data.get("uuid") is None for entry in entries1 + entries2):
            raise ValueError("[uuid] must be provided for all entries.")

        idx1, ent_r1 = find_entry(entries1, relaxed_uuid1)
        idx2, ent_r2 = find_entry(entries2, relaxed_uuid2)

        s_entries1, distortions1 = sort_pos_dist(
            entries1,
            ent_r1,
            ent_r2,
            dist=dQ_entries,
        )
        s_entries2, distortions2 = sort_pos_dist(
            entries2,
            ent_r1,
            ent_r2,
            dist=dQ_entries,
        )

        energies1 = [entry.energy for entry in s_entries1]
        energies2 = [entry.energy for entry in s_entries2]

        sdirs1 = [e.data["dir_name"] for e in s_entries1]
        sdirs2 = [e.data["dir_name"] for e in s_entries2]

        return cls(
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
            relaxed_index1=idx1,
            relaxed_index2=idx2,
        )

    def get_taskdocs(self) -> tuple[list[TaskDoc], list[TaskDoc]]:
        """Get the distorted task documents."""

        def remove_host_name(dir_name: str) -> str:
            return dir_name.split(":")[-1]

        static1_task_docs = [
            TaskDoc.from_directory(remove_host_name(dir_name))
            for dir_name in self.static_dirs1
        ]
        static2_task_docs = [
            TaskDoc.from_directory(remove_host_name(dir_name))
            for dir_name in self.static_dirs2
        ]

        return static1_task_docs, static2_task_docs


def sort_pos_dist(
    list_in: list[Any],
    s1: Any,
    s2: Any,
    dist: Callable,
) -> tuple[list[Any], list[float]]:
    """
    Sort a list defined when we can only compute a positive-definite distance.

    Sometimes, we can only compute a positive-definite distance between two objects.
    (E.g., the displacement between two structures). In these cases, standard
    sorting algorithms will not work. Here, we accept two reference points to give
    some sense of direction. We then sort the list based on the distance between the
    reference points. Note: this only works if the list falls on a line of some sort

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
    d0 = dist(s1, s2)

    d_vs_s = []
    for q1, q2, s in zip(d1, d2, list_in):
        sign = +1
        if q1 < q2 and q2 > d0:
            sign = -1
        d_vs_s.append((sign * q1, s))
    d_vs_s.sort()
    return [s for _, s in d_vs_s], [d for d, _ in d_vs_s]


def get_dQ(ref: Structure, distorted: Structure) -> float:  # noqa: N802
    """
    Calculate dQ from the initial and final structures.

    Parameters
    ----------
    ground : pymatgen.core.structure.Structure
        A pymatgen structure corresponding to the ground (final) state.
    excited : pymatgen.core.structure.Structure
        A pymatgen structure corresponding to the excited (initial) state.

    Returns
    -------
    float
        The dQ value (amu^{1/2} Angstrom).
    """
    return np.sqrt(
        np.sum(
            [
                x[0].distance(x[1]) ** 2 * x[0].specie.atomic_mass
                for x in zip(ref, distorted)
            ],
        ),
    )
