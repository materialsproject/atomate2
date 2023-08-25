"""Schemas for MD documents."""

from typing import List

from pydantic import BaseModel, Field
from pymatgen.core import Structure


class MultiMDOutput(BaseModel):
    """Output of a MultiMD Flow."""

    structure: Structure = Field("Final structure of the last step of the flow")
    vasp_dir: str = Field("Path to the last vasp folder of the flow")
    traj_ids: List[str] = Field("List of uuids of the MD calculations in the flow")
