from typing import Optional, Dict, Callable, List
from pathlib import Path
import copy

from jobflow import job
from emmet.core.vasp.task_valid import TaskState

from atomate2.classical_md.schemas import MoleculeSpec
from atomate2.classical_md.utils import (
    merge_specs_by_name_and_smile,
    create_mol_spec,
)

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
    mol_specs: List[MoleculeSpec] | List[dict],
    mass_density: float,
    force_field: str = "openff_unconstrained-2.1.1.offxml",
    pack_box_kwargs: Dict = {},
):
    if all(isinstance(spec, dict) for spec in mol_specs):
        mol_specs = [create_mol_spec(**spec) for spec in mol_specs]
    assert all(
        isinstance(spec, MoleculeSpec) for spec in mol_specs
    ), "mol_specs must be a list of dicts or MoleculeSpec"

    mol_specs = copy.deepcopy(mol_specs)
    mol_specs.sort(key=lambda x: x.openff_mol.to_smiles() + x.name)

    topology = pack_box(
        molecules=[spec.openff_mol for spec in mol_specs],
        number_of_copies=[spec.count for spec in mol_specs],
        mass_density=mass_density * unit.grams / unit.milliliter,
        **pack_box_kwargs
    )

    # TODO: document this
    mol_specs = merge_specs_by_name_and_smile(mol_specs)

    # TODO: ForceField doesn't currently support iterables, fix this
    # force_field: str | Path | List[str | Path] = "openff_unconstrained-2.1.1.offxml",

    # valid FFs: https://github.com/openforcefield/openff-forcefields
    interchange = Interchange.from_smirnoff(
        force_field=ForceField(force_field),
        topology=topology,
        charge_from_molecules=[spec.openff_mol for spec in mol_specs],
        allow_nonintegral_charges=True,
    )

    # currently not needed because ForceField isn't correctly supporting iterables

    # coerce force_field to a str or list of str
    # if not isinstance(force_field, list):
    #     force_field = [force_field]
    # ff_list = [ff.name if isinstance(ff, Path) else ff for ff in force_field]
    # force_field_names = ff_list if len(force_field) > 1 else ff_list[0]

    interchange_json = interchange.json()

    task_doc = ClassicalMDTaskDocument(
        state=TaskState.SUCCESS,
        interchange=interchange_json,
        molecule_specs=mol_specs,
        forcefield=force_field,
    )

    return task_doc
