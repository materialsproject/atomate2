"""Utilities for working with the OPLS forcefield in OpenMM."""

from __future__ import annotations

import copy
import io
from pathlib import Path

import numpy as np
import openff.toolkit as tk
from emmet.core.openff import MoleculeSpec
from emmet.core.openmm import OpenMMInterchange, OpenMMTaskDocument
from emmet.core.vasp.task_valid import TaskState
from jobflow import Response
from openff.interchange.components._packmol import pack_box
from openff.units import unit
from openmm import Context, LangevinMiddleIntegrator, XmlSerializer
from openmm.app.pdbfile import PDBFile
from openmm.unit import kelvin, picoseconds
from pymatgen.io.openff import get_atom_map

from atomate2.openff.utils import create_mol_spec, merge_specs_by_name_and_smiles
from atomate2.openmm.jobs.base import openmm_job
from atomate2.openmm.utils import XMLMoleculeFF, create_system_from_xml


@openmm_job
def generate_openmm_interchange(
    input_mol_specs: list[MoleculeSpec | dict],
    mass_density: float,
    ff_xmls: list[str | Path],
    xml_method_and_scaling: tuple[str, float] = None,
    pack_box_kwargs: dict = None,
    tags: list[str] = None,
) -> Response:
    """Generate an OpenMM Interchange object from a list of molecule specifications.

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
    ff_xmls : List[str]
        A list of force field XML strings. The order of the XML strings
        must match the order of the input_mol_specs.
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
    """
    # TODO: warn if using unsupported properties

    mol_specs = []
    for spec in input_mol_specs:
        if isinstance(spec, dict):
            mol_specs.append(create_mol_spec(**spec))
        elif isinstance(spec, MoleculeSpec):
            mol_specs.append(copy.deepcopy(spec))
        else:
            raise TypeError("mol_specs must be a list of dicts or MoleculeSpec")

    xml_mols = [XMLMoleculeFF.from_file(xml) for xml in ff_xmls]
    if len(mol_specs) != len(xml_mols):
        raise ValueError(
            "The number of molecule specifications and XML files must match."
        )

    for mol_spec, xml_mol in zip(mol_specs, xml_mols):
        openff_mol = tk.Molecule.from_json(mol_spec.openff_mol)
        xml_openff_mol = xml_mol.to_openff_molecule()
        is_isomorphic, atom_map = get_atom_map(openff_mol, xml_openff_mol)
        if not is_isomorphic:
            raise ValueError(
                "The mol_specs and ff_xmls must index identical molecules."
            )
        if xml_method_and_scaling:
            charge_method, charge_scaling = xml_method_and_scaling
            mol_spec.charge_method = charge_method
            mol_spec.charge_scaling = charge_scaling
            openff_mol.partial_charges = xml_openff_mol.partial_charges
            mol_spec.openff_mol = openff_mol.to_json()
        else:
            xml_mol.assign_partial_charges(openff_mol)

    mol_specs.sort(
        key=lambda x: tk.Molecule.from_json(x.openff_mol).to_smiles() + x.name
    )
    mol_specs = merge_specs_by_name_and_smiles(mol_specs)

    pack_box_kwargs = pack_box_kwargs or {}
    topology = pack_box(
        molecules=[tk.Molecule.from_json(spec.openff_mol) for spec in mol_specs],
        number_of_copies=[spec.count for spec in mol_specs],
        mass_density=mass_density * unit.grams / unit.milliliter,
        **pack_box_kwargs,
    )

    system = create_system_from_xml(topology, xml_mols)

    # these values don't actually matter because integrator is only
    # used to generate the state
    integrator = LangevinMiddleIntegrator(
        298 * kelvin, 1 / picoseconds, 1 * picoseconds
    )
    context = Context(system, integrator)
    context.setPositions(topology.get_positions().magnitude / 10)
    state = context.getState(getPositions=True)

    with io.StringIO() as s:
        PDBFile.writeFile(
            topology.to_openmm(), np.zeros(shape=(topology.n_atoms, 3)), file=s
        )
        s.seek(0)
        pdb = s.read()

    interchange = OpenMMInterchange(
        system=XmlSerializer.serialize(system),
        state=XmlSerializer.serialize(state),
        topology=pdb,
    )

    # TODO: fix all jsons
    interchange_json = interchange.json()
    interchange_bytes = interchange_json.encode("utf-8")

    dir_name = Path.cwd()

    task_doc = OpenMMTaskDocument(
        dir_name=str(dir_name),
        state=TaskState.SUCCESS,
        interchange=interchange_bytes,
        interchange_meta=mol_specs,
        force_field="opls",  # TODO: change to flexible value
        tags=tags,
    )

    with open(dir_name / "taskdoc.json", "w") as file:
        file.write(task_doc.json())

    return Response(output=task_doc)
