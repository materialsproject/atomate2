"""Utilities for working with the OPLS forcefield in OpenMM."""

from __future__ import annotations

import copy
import io
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.etree.ElementTree import tostring

import numpy as np
from emmet.core.openff import MoleculeSpec
from emmet.core.openmm import OpenMMInterchange, OpenMMTaskDocument
from emmet.core.vasp.task_valid import TaskState
from jobflow import Response
from openmm import Context, LangevinMiddleIntegrator, System, XmlSerializer
from openmm.app import PME, ForceField
from openmm.app.pdbfile import PDBFile
from openmm.unit import kelvin, picoseconds
from pymatgen.core import Element
from pymatgen.io.openff import get_atom_map

from atomate2.openff.utils import create_mol_spec, merge_specs_by_name_and_smiles
from atomate2.openmm.jobs.base import openmm_job

try:
    import openff.toolkit as tk
    from openff.interchange.components._packmol import pack_box
    from openff.units import unit
except ImportError as e:
    raise ImportError(
        "Using the atomate2.openmm.generate "
        "module requires the openff-toolkit package."
    ) from e


class XMLMoleculeFF:
    """A class for manipulating XML files representing OpenMM-compatible forcefields."""

    def __init__(self, xml_string: str) -> None:
        """Create an XMLMoleculeFF object from a string version of the XML file."""
        self.tree = ET.parse(io.StringIO(xml_string))  # noqa: S314

        root = self.tree.getroot()
        canonical_order = {}
        for i, atom in enumerate(root.findall(".//Residues/Residue/Atom")):
            canonical_order[atom.attrib["type"]] = i

        non_to_res_map = {}
        for i, atom in enumerate(root.findall(".//NonbondedForce/Atom")):
            non_to_res_map[i] = canonical_order[atom.attrib["type"]]
            # self._res_to_non.append(canonical_order[atom.attrib["type"]])
        # invert map, change to list
        self._res_to_non = [
            k for k, v in sorted(non_to_res_map.items(), key=lambda item: item[1])
        ]
        self._non_to_res = list(non_to_res_map.values())

    def __str__(self) -> str:
        """Return the a string version of the XML file."""
        return tostring(self.tree.getroot(), encoding="unicode")

    def increment_types(self, increment: str) -> None:
        """Increment the type names in the XMLMoleculeFF object.

        This method is needed because LigParGen will reuse type names
        in XML files, then causing an error in OpenMM. We differentiate
        the types with this method.
        """
        root_type = [
            (".//AtomTypes/Type", "name"),
            (".//AtomTypes/Type", "class"),
            (".//Residues/Residue/Atom", "type"),
            (".//HarmonicBondForce/Bond", "class"),
            (".//HarmonicAngleForce/Angle", "class"),
            (".//PeriodicTorsionForce/Proper", "class"),
            (".//PeriodicTorsionForce/Improper", "class"),
            (".//NonbondedForce/Atom", "type"),
        ]
        for xpath, type_stub in root_type:
            for element in self.tree.getroot().findall(xpath):
                for key in element.attrib:
                    if type_stub in key:
                        element.attrib[key] += increment

    def to_openff_molecule(self) -> tk.Molecule:
        """Convert the XMLMoleculeFF to an openff_toolkit Molecule."""
        if sum(self.partial_charges) > 1e-3:
            # TODO: update message
            warnings.warn("Formal charges not considered.", stacklevel=1)

        p_table = {e.symbol: e.number for e in Element}
        openff_mol = tk.Molecule()
        for atom in self.tree.getroot().findall(".//Residues/Residue/Atom"):
            symbol = re.match(r"^[A-Za-z]+", atom.attrib["name"]).group()
            atomic_number = p_table[symbol]
            openff_mol.add_atom(atomic_number, formal_charge=0, is_aromatic=False)

        for bond in self.tree.getroot().findall(".//Residues/Residue/Bond"):
            openff_mol.add_bond(
                int(bond.attrib["from"]),
                int(bond.attrib["to"]),
                bond_order=1,
                is_aromatic=False,
            )

        openff_mol.partial_charges = self.partial_charges * unit.elementary_charge

        return openff_mol

    @property
    def partial_charges(self) -> np.ndarray:
        """Get the partial charges from the XMLMoleculeFF object."""
        atoms = self.tree.getroot().findall(".//NonbondedForce/Atom")
        charges = np.array([float(atom.attrib["charge"]) for atom in atoms])
        return charges[self._res_to_non]

    @partial_charges.setter
    def partial_charges(self, partial_charges: np.ndarray) -> None:
        for i, atom in enumerate(self.tree.getroot().findall(".//NonbondedForce/Atom")):
            charge = partial_charges[self._non_to_res[i]]
            atom.attrib["charge"] = str(charge)

    def assign_partial_charges(self, mol_or_method: tk.Molecule | str) -> None:
        """Assign partial charges to the XMLMoleculeFF object.

        Parameters
        ----------
        mol_or_method : Union[tk.Molecule, str]
            If a molecule is provided, it must have partial charges assigned.
            If a string is provided, openff_toolkit.Molecule.assign_partial_charges
            will be used to generate the partial charges.
        """
        if isinstance(mol_or_method, str):
            openff_mol = self.to_openff_molecule()
            openff_mol.assign_partial_charges(mol_or_method)
            mol_or_method = openff_mol
        self_mol = self.to_openff_molecule()
        _isomorphic, atom_map = get_atom_map(mol_or_method, self_mol)
        mol_charges = mol_or_method.partial_charges[list(atom_map.values())].magnitude
        self.partial_charges = mol_charges

    def to_file(self, file: str | Path) -> None:
        """Write the XMLMoleculeFF object to an XML file."""
        self.tree.write(file, encoding="utf-8")

    @classmethod
    def from_file(cls, file: str | Path) -> XMLMoleculeFF:
        """Create an XMLMoleculeFF object from an XML file."""
        with open(file) as f:
            xml_str = f.read()
        return cls(xml_str)


def create_system_from_xml(
    topology: tk.Topology,
    xml_mols: list[XMLMoleculeFF],
) -> System:
    """Create an OpenMM system from a list of molecule specifications and XML files."""
    io_files = []
    for i, xml in enumerate(xml_mols):
        xml_copy = copy.deepcopy(xml)
        xml_copy.increment_types(f"_{i}")
        io_files.append(io.StringIO(str(xml_copy)))

    ff = ForceField(io_files[0])
    for i, xml in enumerate(io_files[1:]):  # type: ignore[assignment]
        ff.loadFile(xml, resname_prefix=f"_{i + 1}")

    return ff.createSystem(topology.to_openmm(), nonbondedMethod=PME)


@openmm_job
def generate_openmm_interchange(
    input_mol_specs: list[MoleculeSpec | dict],
    mass_density: float,
    ff_xmls: list[str],
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
        A list of force field XML strings, these should be the raw text
        of the XML files. The order of the XML strings
        must match the order of the input_mol_specs.
    xml_method_and_scaling : Tuple[str, float], optional
        A tuple containing the charge method and scaling factor to use for
        the partial charges in the xml. If this is not set, partial charges
        will be generated by openff toolkit.
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
    mol_specs = []
    for spec in input_mol_specs:
        if isinstance(spec, dict):
            mol_specs.append(create_mol_spec(**spec))
        elif isinstance(spec, MoleculeSpec):
            mol_specs.append(copy.deepcopy(spec))
        else:
            raise TypeError(
                f"item in mol_specs is a {type(spec)}, but mol_specs "
                f"must be a list of dicts or MoleculeSpec"
            )

    xml_mols = [XMLMoleculeFF(xml) for xml in ff_xmls]
    if len(mol_specs) != len(xml_mols):
        raise ValueError(
            "The number of molecule specifications and XML files must match."
        )

    for mol_spec, xml_mol in zip(mol_specs, xml_mols, strict=True):
        openff_mol = tk.Molecule.from_json(mol_spec.openff_mol)
        xml_openff_mol = xml_mol.to_openff_molecule()
        is_isomorphic, _atom_map = get_atom_map(openff_mol, xml_openff_mol)
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

    dir_name = Path.cwd()

    task_doc = OpenMMTaskDocument(
        dir_name=str(dir_name),
        state=TaskState.SUCCESS,
        interchange=interchange_json,
        mol_specs=mol_specs,
        force_field="opls",  # TODO: change to flexible value
        tags=tags,
    )

    with open(dir_name / "taskdoc.json", "w") as file:
        file.write(task_doc.json())

    return Response(output=task_doc)
