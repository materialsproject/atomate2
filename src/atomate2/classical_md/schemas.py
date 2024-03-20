from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List

import pymatgen
from emmet.core.vasp.task_valid import TaskState
from monty.json import MSONable
from openff import toolkit as tk
from openff.interchange import Interchange
from pydantic import BaseModel, field_validator, confloat, PositiveInt, Field

# from pydantic.dataclasses import dataclass
from pymatgen.analysis.graphs import MoleculeGraph
from dataclasses import dataclass


@dataclass
class MoleculeSpec(MSONable):
    """
    A molecule schema to be output by OpenMMGenerators.
    """

    name: str
    count: int
    formal_charge: int
    charge_method: str
    openff_mol: tk.Molecule


class ClassicalMDTaskDocument(BaseModel, extra="allow"):
    """Definition of the OpenMM task document."""

    tags: Optional[List[str]] = Field(
        [], title="tag", description="Metadata tagged to a given task."
    )
    dir_name: Optional[str] = Field(
        None, description="The directory for this VASP task"
    )
    state: Optional[TaskState] = Field(None, description="State of this calculation")

    calcs_reversed: Optional[List] = Field(
        None,
        title="Calcs reversed data",
        description="Detailed data for each VASP calculation contributing to the task document.",
    )

    interchange: Optional[dict] = Field(
        None, description="Final output structure from the task"
    )

    molecule_specs: Optional[List[MoleculeSpec]] = Field(
        None, description="Molecules within the box."
    )

    forcefield: Optional[str | List[str]] = Field(None, description="forcefield")

    task_type: Optional[str] = Field(None, description="The type of calculation.")

    # task_label: Optional[str] = Field(None, description="A description of the task")
    # TODO: where does task_label get added
    # additional_json

    last_updated: Optional[datetime] = Field(
        None,
        description="Timestamp for the most recent calculation for this task document",
    )
