from typing import Optional, Dict, Callable, List
from datetime import datetime
from pathlib import Path

from jobflow import job
from emmet.core.vasp.task_valid import TaskState
from atomate2.classical_md.openmm.schemas.molecules import InputMoleculeSpec
from atomate2.classical_md.openff_utils import process_mol_specs

from openff.toolkit import ForceField
from openff.interchange import Interchange
from openff.interchange.components._packmol import pack_box
from openff.units import unit

from pydantic import BaseModel, Field


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

    interchange: Optional[Interchange] = Field(
        None, description="Final output structure from the task"
    )

    molecule_specs: Optional[List] = Field(
        None, description="Molecules within the box."
    )

    forcefield: Optional[str | List[str]] = Field(None, description="forcefield")

    task_type: Optional[str] = Field(None, description="The type of calculation.")

    task_label: Optional[str] = Field(None, description="A description of the task")

    # additional_json

    last_updated: Optional[datetime] = Field(
        None,
        description="Timestamp for the most recent calculation for this task document",
    )


def openff_job(method: Callable):
    """

    Parameters
    ----------
    method : callable
        A BaseOpenMMMaker.make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate OpenMM jobs.
    """
    # todo: add data keyword argument to specify where to write bigger files like trajectory files
    return job(method, output_schema=ClassicalMDTaskDocument)


@openff_job
def generate_interchange(
    input_molecule_dicts: List[InputMoleculeSpec | dict],
    mass_density: float,
    force_field: str | Path | List[str | Path],
    charge_method: str,  # enum, add default
    pack_box_kwargs: Optional[Dict] = None,
):

    molecule_specs = process_mol_specs(input_molecule_dicts, charge_method)

    topology = pack_box(
        molecules=[spec["openff_mol"] for spec in molecule_specs],
        number_of_copies=[spec["count"] for spec in molecule_specs],
        mass_density=mass_density * unit.grams / unit.milliliter,
        **pack_box_kwargs
    )

    # add force_field logic and string logic

    interchange = Interchange.from_smirnoff(
        force_field=ForceField(force_field),
        topology=topology,
        charge_from_molecules=[spec["openff_mol"] for spec in molecule_specs],
        allow_nonintegral_charges=True,
    )

    if not isinstance(force_field, list):
        force_field = [force_field]

    ff_list = [ff.name if isinstance(ff, Path) else ff for ff in force_field]
    force_field_names = ff_list if len(force_field) > 1 else ff_list[0]

    task_doc = ClassicalMDTaskDocument(
        state=TaskState.SUCCESS,
        interchange=interchange,
        molecule_specs=molecule_specs,
        forcefield=force_field_names,
    )

    return task_doc
