import copy
from pathlib import Path

import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import Element
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender

import openff.toolkit as tk

from typing import Dict, List, Tuple, Union
import numpy as np
from openmm.unit import elementary_charge, angstrom

from pint import Quantity

from atomate2.classical_md.schemas import MoleculeSpec


def molgraph_to_openff_mol(molgraph: MoleculeGraph) -> tk.Molecule:
    """
    Convert a Pymatgen MoleculeGraph to an OpenFF Molecule.

    If partial charges, formal charges, and aromaticity are present in site properties
    they will be mapped onto atoms. If bond order and bond aromaticity are present in
    edge weights and edge properties they will be mapped onto bonds.

    Args:
        molgraph: PyMatGen MoleculeGraph

    Returns:
        openff_mol: OpenFF Molecule
    """
    # create empty openff_mol and prepare a periodic table
    p_table = {str(el): el.Z for el in Element}
    openff_mol = tk.topology.Molecule()

    # set atom properties
    partial_charges = []
    # TODO: should assert that there is only one molecule
    for i_node in range(len(molgraph.graph.nodes)):
        node = molgraph.graph.nodes[i_node]
        atomic_number = (
            node.get("atomic_number")
            or p_table[molgraph.molecule[i_node].species_string]
        )

        # put formal charge on first atom if there is none present
        formal_charge = node.get("formal_charge")
        if formal_charge is None:
            formal_charge = (i_node == 0) * molgraph.molecule.charge * elementary_charge

        # assume not aromatic if no info present
        is_aromatic = node.get("is_aromatic") or False

        openff_mol.add_atom(atomic_number, formal_charge, is_aromatic=is_aromatic)

        # add to partial charge array
        partial_charge = node.get("partial_charge")
        if isinstance(partial_charge, Quantity):
            partial_charge = partial_charge.magnitude
        partial_charges.append(partial_charge)

    charge_array = np.array(partial_charges)
    if np.not_equal(charge_array, None).all():
        openff_mol.partial_charges = charge_array * elementary_charge

    # set edge properties, default to single bond and assume not aromatic
    for i_node, j, bond_data in molgraph.graph.edges(data=True):
        bond_order = bond_data.get("bond_order") or 1
        is_aromatic = bond_data.get("is_aromatic") or False
        openff_mol.add_bond(i_node, j, bond_order, is_aromatic=is_aromatic)

    openff_mol.add_conformer(molgraph.molecule.cart_coords * angstrom)
    return openff_mol


def get_atom_map(inferred_mol, openff_mol) -> Tuple[bool, Dict[int, int]]:
    """
    Get a mapping between two openff Molecules.
    """
    # do not apply formal charge restrictions
    kwargs = dict(
        return_atom_map=True,
        formal_charge_matching=False,
    )
    isomorphic, atom_map = tk.topology.Molecule.are_isomorphic(
        openff_mol, inferred_mol, **kwargs
    )
    if isomorphic:
        return True, atom_map
    # relax stereochemistry restrictions
    kwargs["atom_stereochemistry_matching"] = False
    kwargs["bond_stereochemistry_matching"] = False
    isomorphic, atom_map = tk.topology.Molecule.are_isomorphic(
        openff_mol, inferred_mol, **kwargs
    )
    if isomorphic:
        print(
            f"stereochemistry ignored when matching inferred"
            f"mol: {openff_mol} to {inferred_mol}"
        )
        return True, atom_map
    # relax bond order restrictions
    kwargs["bond_order_matching"] = False
    isomorphic, atom_map = tk.topology.Molecule.are_isomorphic(
        openff_mol, inferred_mol, **kwargs
    )
    if isomorphic:
        print(
            f"stereochemistry ignored when matching inferred"
            f"mol: {openff_mol} to {inferred_mol}"
        )
        print(
            f"bond_order restrictions ignored when matching inferred"
            f"mol: {openff_mol} to {inferred_mol}"
        )
        return True, atom_map
    return False, {}


def infer_openff_mol(
    mol_geometry: pymatgen.core.Molecule,
) -> tk.Molecule:
    """
    Infer an OpenFF molecule from a pymatgen Molecule.

    Args:
        mol_geometry: A pymatgen Molecule

    Returns:
        an OpenFF Molecule

    """
    molgraph = MoleculeGraph.with_local_env_strategy(mol_geometry, OpenBabelNN())
    molgraph = metal_edge_extender(molgraph)
    inferred_mol = molgraph_to_openff_mol(molgraph)
    return inferred_mol


def add_conformer(openff_mol: tk.Molecule, geometry: pymatgen.core.Molecule):
    """
    Adds conformers to an OpenFF Molecule based on the provided geometries or generates conformers up to the specified maximum number.

    Parameters
    ----------
    openff_mol : openff.toolkit.topology.Molecule
        The OpenFF Molecule to add conformers to.
    geometry : Geometry:
        A list of Geometry objects containing the coordinates of the conformers to be added.
    max_conformers : int
        The maximum number of conformers to be generated if no geometries are provided.

    Returns
    -------
    openff.toolkit.topology.Molecule, Dict[int, int]
        A tuple containing the OpenFF Molecule with added conformers and a dictionary representing the atom mapping between the input and output molecules.


    """
    # TODO: test this
    if geometry:
        # for geometry in geometries:
        inferred_mol = infer_openff_mol(geometry)
        is_isomorphic, atom_map = get_atom_map(inferred_mol, openff_mol)
        if not is_isomorphic:
            raise ValueError(
                f"An isomorphism cannot be found between smile {openff_mol.to_smiles()} "
                f"and the provided molecule {geometry}."
            )
        new_mol = pymatgen.core.Molecule.from_sites(
            [geometry.sites[i] for i in atom_map.values()]
        )
        openff_mol.add_conformer(new_mol.cart_coords * angstrom)
    else:
        atom_map = {i: i for i in range(openff_mol.n_atoms)}
        openff_mol.generate_conformers(n_conformers=1)
    return openff_mol, atom_map


def assign_partial_charges(
    openff_mol: tk.Molecule,
    atom_map: Dict[int, int],
    charge_method: str,
    partial_charges: Union[None, List[float]],
):
    """
    Assigns partial charges to an OpenFF Molecule using the provided partial charges or a specified charge method.

    Parameters
    ----------
    openff_mol : openff.toolkit.topology.Molecule
        The OpenFF Molecule to assign partial charges to.
    partial_charges : List[float]
        A list of partial charges to be assigned to the molecule, or None to use the charge method.
    charge_method : str
        The charge method to be used if partial charges are not provided.
    atom_map : Dict[int, int]
        A dictionary representing the atom mapping between the input and output molecules.

    Returns
    -------
    openff.toolkit.topology.Molecule
        The OpenFF Molecule with assigned partial charges.

    """
    # TODO: test this
    # assign partial charges
    if partial_charges is not None:
        partial_charges = np.array(partial_charges)
        openff_mol.partial_charges = partial_charges[list(atom_map.values())] * elementary_charge  # type: ignore
    elif openff_mol.n_atoms == 1:
        openff_mol.partial_charges = (
            np.array([openff_mol.total_charge.magnitude]) * elementary_charge
        )
    else:
        openff_mol.assign_partial_charges(charge_method)
    return openff_mol


def create_openff_mol(
    smile: str,
    geometry: Union[pymatgen.core.Molecule, str, Path] = None,
    charge_scaling: float = 1,
    partial_charges: List[float] = None,
    backup_charge_method: str = "am1bcc",
):
    if isinstance(geometry, (str, Path)):
        geometry = pymatgen.core.Molecule.from_file(str(geometry))

    if partial_charges is not None:
        if geometry is None:
            raise ValueError("geometries must be set if partial_charges is set")
        if len(partial_charges) != len(geometry):
            raise ValueError(
                "partial charges must have same length & order as geometry"
            )

    openff_mol = tk.Molecule.from_smiles(smile, allow_undefined_stereo=True)

    # add conformer
    openff_mol, atom_map = add_conformer(openff_mol, geometry)
    # assign partial charges
    openff_mol = assign_partial_charges(
        openff_mol, atom_map, backup_charge_method, partial_charges
    )
    openff_mol.partial_charges *= charge_scaling

    return openff_mol


def create_mol_spec(
    smile: str,
    count: int,
    name: str = None,
    charge_scaling: float = 1,
    charge_method: str = None,
    geometry: Union[pymatgen.core.Molecule, str, Path] = None,
    partial_charges: List[float] = None,
) -> MoleculeSpec:
    """
    Processes a list of input molecular specifications, generating conformers, assigning partial charges, and creating output molecular specifications.

    Parameters
    ----------
    smile: str
        The SMILES string of the molecule.
    count: int
        The number of molecules to be created.
    name: str
        The name of the molecule.
    charge_scaling: float
        The scaling factor to be applied to the partial charges.
    charge_method: str
        The default charge method to be used if not specified in the input molecular specifications.
    geometry: Union[pymatgen.core.Molecule, str, Path]
        The geometry of the molecule.
    partial_charges: List[float]
        A list of partial charges to be assigned to the molecule, or None to use the charge method.

    Returns
    -------
    List[MoleculeSpec]
        A list of dictionaries containing processed molecular specifications.
    """
    # TODO: test this

    if charge_method is None:
        if partial_charges:
            charge_method = "custom"
        else:
            charge_method = "am1bcc"

    openff_mol = create_openff_mol(
        smile,
        geometry,
        charge_scaling,
        partial_charges,
        charge_method,
    )

    # create mol_spec
    return MoleculeSpec(
        name=(name or smile),
        count=count,
        formal_charge=int(
            np.sum(openff_mol.partial_charges.magnitude) / charge_scaling
        ),
        charge_method=charge_method,
        openff_mol=openff_mol,
    )


def merge_specs_by_name_and_smile(mol_specs: List[MoleculeSpec]) -> List[MoleculeSpec]:
    """
    Merges molecular specifications with the same name and smile into a
    single molecular specification.

    Parameters
    ----------
    mol_specs : List[MoleculeSpec]
        A list of molecular specifications.

    Returns
    -------
    List[MoleculeSpec]
        A list of molecular specifications with unique names and smiles.

    """
    mol_specs = copy.deepcopy(mol_specs)
    merged_spec_dict = {}
    for spec in mol_specs:
        key = (spec.openff_mol.to_smiles(), spec.name)
        if key in merged_spec_dict:
            merged_spec_dict[key].count += spec.count
        else:
            merged_spec_dict[key] = spec
    return list(merged_spec_dict.values())
