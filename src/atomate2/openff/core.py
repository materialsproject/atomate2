"""Core jobs for classical MD module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import openff.toolkit as tk
from emmet.core.openff import ClassicalMDTaskDocument, MoleculeSpec
from emmet.core.vasp.task_valid import TaskState
from jobflow import Response, job
from openff.interchange import Interchange
from openff.interchange.components._packmol import pack_box
from openff.toolkit import ForceField
from openff.units import unit

from atomate2.openff.utils import create_mol_spec_list, merge_specs_by_name_and_smiles

if TYPE_CHECKING:
    from collections.abc import Callable


def openff_job(method: Callable) -> job:
    """Decorate the ``make`` method of ClassicalMD job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.Job` that configures common
    settings for all ClassicalMD jobs. Namely, configures the output schema to be a
    :obj:`.ClassicalMDTaskDocument`.

    Any makers that return classical md jobs (not flows) should decorate the ``make``
    method with @openff_job. For example:

    .. code-block:: python

        class MyClassicalMDMaker(BaseOpenMMMaker):
            @openff_job
            def make(structure):
                # code to run OpenMM job.
                pass

    Parameters
    ----------
    method : callable
        A BaseVaspMaker.make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate jobs.
    """
    return job(
        method,
        output_schema=ClassicalMDTaskDocument,
        data=["interchange", "traj_blob"],
    )


@openff_job
def generate_interchange(
    input_mol_specs: list[MoleculeSpec | dict],
    mass_density: float,
    force_field: str = "openff_unconstrained-2.1.1.offxml",
    pack_box_kwargs: dict = None,
    tags: list[str] = None,
) -> Response:
    """Generate an OpenFF Interchange object from a list of molecule specifications.

    This function takes a list of molecule specifications (either as
    MoleculeSpec objects or dictionaries), a target mass density, and
    optional force field and box packing parameters. It processes the molecule
    specifications, packs them into a box using the specified mass density, and
    creates an OpenFF Interchange object using the specified force field.

    If you'd like to have multiple distinct input geometries, you
    can pass multiple mol_specs with the same name and SMILES string.
    After packing the box, they will be merged into a single mol_spec
    and treated as a single component in the resulting system.

    Parameters
    ----------
    input_mol_specs : List[Union[MoleculeSpec, dict]]
        A list of molecule specifications, either as MoleculeSpec objects or
        dictionaries that can be passed to `create_mol_spec` to create
        MoleculeSpec objects. See the `create_mol_spec` function
        for details on the expected format of the dictionaries.
    mass_density : float
        The target mass density for packing the molecules into
        a box, kg/L.
    force_field : str, optional
        The name of the force field to use for creating the
        Interchange object. This is passed directly to openff.toolkit.ForceField.
        Default is "openff_unconstrained-2.1.1.offxml".
    pack_box_kwargs : Dict, optional
        Additional keyword arguments to pass to the
        toolkit.interchange.components._packmol.pack_box. Default is an empty dict.
    tags : List[str], optional
        A list of tags to attach to the task document.

    Returns
    -------
    ClassicalMDTaskDocument
        A task document containing the generated OpenFF Interchange
        object, molecule specifications, and force field information.

    Notes
    -----
    - The function assumes that all dictionaries in the mol_specs list can be used to
    create valid MoleculeSpec objects.
    - The function sorts the molecule specifications based on their SMILES string
    and name before packing the box.
    - The function uses the merge_specs_by_name_and_smiles function to merge molecule
    specifications with the same name and SMILES string.
    - The function currently does not support passing a list of force fields due to
    limitations in the OpenFF Toolkit.
    """
    mol_specs = create_mol_spec_list(input_mol_specs)

    pack_box_kwargs = pack_box_kwargs or {}
    topology = pack_box(
        molecules=[tk.Molecule.from_json(spec.openff_mol) for spec in mol_specs],
        number_of_copies=[spec.count for spec in mol_specs],
        mass_density=mass_density * unit.grams / unit.milliliter,
        **pack_box_kwargs,
    )

    mol_specs = merge_specs_by_name_and_smiles(mol_specs)

    # TODO: ForceField doesn't currently support iterables, fix this
    # force_field: str | Path | List[str | Path] = "openff_unconstrained-2.1.1.offxml",

    # valid FFs: https://github.com/openforcefield/openff-forcefields
    ff_object = ForceField(force_field)

    interchange = Interchange.from_smirnoff(
        force_field=ff_object,
        topology=topology,
        charge_from_molecules=[
            tk.Molecule.from_json(spec.openff_mol) for spec in mol_specs
        ],
        allow_nonintegral_charges=True,
    )

    # currently not needed because ForceField isn't correctly supporting iterables
    # coerce force_field to a str or list of str
    # if not isinstance(force_field, list):
    #     force_field = [force_field]
    # ff_list = [ff.name if isinstance(ff, Path) else ff for ff in force_field]
    # force_field_names = ff_list if len(force_field) > 1 else ff_list[0]

    interchange_json = interchange.json()

    dir_name = Path.cwd()

    task_doc = ClassicalMDTaskDocument(
        dir_name=str(dir_name),
        state=TaskState.SUCCESS,
        interchange=interchange_json,
        mol_specs=mol_specs,
        force_field=force_field,
        tags=tags,
    )

    with open(dir_name / "taskdoc.json", "w") as file:
        file.write(task_doc.json())

    return Response(output=task_doc)
