"""Schemas for MD documents."""

from emmet.core.structure import StructureMetadata
from pydantic import Field
from pymatgen.core import Structure


class MultiMDOutput(StructureMetadata):
    """Output of a MultiMD Flow."""

    structure: Structure = Field("Final structure of the last step of the flow")
    vasp_dir: str = Field("Path to the last vasp folder of the flow")
    traj_ids: list[str] = Field("List of uuids of the MD calculations in the flow")
    full_traj_ids: list[str] = Field(
        "List of uuids of the MD calculations in the flow and in previous linked flows"
    )
