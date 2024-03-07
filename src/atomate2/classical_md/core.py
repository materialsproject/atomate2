from typing import Optional, Dict, Callable, List
from pathlib import Path

from jobflow import job
from emmet.core.vasp.task_valid import TaskState

from atomate2.classical_md.schemas import InputMoleculeSpec
from atomate2.classical_md.utils import process_mol_specs

from openff.toolkit import ForceField
from openff.interchange import Interchange
from openff.interchange.components._packmol import pack_box
from openff.units import unit

from atomate2.classical_md.schemas import ClassicalMDTaskDocument


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
