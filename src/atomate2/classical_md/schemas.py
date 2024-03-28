"""Schemas for classical MD package."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from emmet.core.vasp.task_valid import TaskState
from monty.json import MSONable
from openff import toolkit as tk
from pydantic import BaseModel, Field


@dataclass
class MoleculeSpec(MSONable):
    """A molecule schema to be output by OpenMMGenerators."""

    name: str
    count: int
    formal_charge: int
    charge_method: str
    openff_mol: tk.Molecule  # a tk.Molecule object serialized with to_dict


class ClassicalMDTaskDocument(BaseModel, extra="allow"):  # type: ignore[call-arg]
    """Definition of the OpenMM task document."""

    tags: Optional[list[str]] = Field(
        [], title="tag", description="Metadata tagged to a given task."
    )
    dir_name: Optional[str] = Field(
        None, description="The directory for this VASP task"
    )
    state: Optional[TaskState] = Field(None, description="State of this calculation")

    calcs_reversed: Optional[list] = Field(
        None,
        title="Calcs reversed data",
        description="Detailed data for each VASP calculation contributing to "
        "the task document.",
    )

    interchange: Optional[str] = Field(
        None, description="Final output structure from the task"
    )

    molecule_specs: Optional[list[MoleculeSpec]] = Field(
        None, description="Molecules within the box."
    )

    forcefield: Optional[str] = Field(None, description="forcefield")

    task_type: Optional[str] = Field(None, description="The type of calculation.")

    # task_label: Optional[str] = Field(None, description="A description of the task")
    # TODO: where does task_label get added

    last_updated: Optional[datetime] = Field(
        None,
        description="Timestamp for the most recent calculation for this task document",
    )
