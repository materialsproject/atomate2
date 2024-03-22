"""Utility functions for classical md subpackage."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import openff.toolkit as tk
import pymatgen
from openmm.unit import angstrom, elementary_charge
from pint import Quantity
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender
from pymatgen.core import Element

from atomate2.classical_md.schemas import MoleculeSpec


def molgraph_to_openff_mol(molgraph: MoleculeGraph) -> tk.Molecule:
    """
    Convert a Pymatgen MoleculeGraph to an OpenFF Molecule.

    Maps partial charges, formal charges, and aromaticity
    from site properties onto atoms. Maps bond order and bond
    aromaticity from edge weights and edge properties onto bonds.

    Args:
        molgraph (MoleculeGraph): The Pymatgen MoleculeGraph to be converted.

    Returns
    -------
        tk.Molecule: The converted OpenFF Molecule.
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


def get_atom_map(
    inferred_mol: tk.Molecule, openff_mol: tk.Molecule
) -> tuple[bool, dict[int, int]]:
    """
    Compute an atom mapping between two OpenFF Molecules.

    Attempts to find an isomorphism between the molecules, considering various matching
    criteria such as formal charges, stereochemistry, and bond orders. Returns the atom
    mapping if an isomorphism is found, otherwise returns an empty mapping.

    Args:
        inferred_mol (tk.Molecule): The first OpenFF Molecule.
        openff_mol (tk.Molecule): The second OpenFF Molecule.

    Returns
    -------
        Tuple[bool, Dict[int, int]]: A tuple containing a boolean indicating if an
            isomorphism was found and a dictionary representing the atom mapping.
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
        return True, atom_map
    # relax bond order restrictions
    kwargs["bond_order_matching"] = False
    isomorphic, atom_map = tk.topology.Molecule.are_isomorphic(
        openff_mol, inferred_mol, **kwargs
    )
    if isomorphic:
        return True, atom_map
    return False, {}


def infer_openff_mol(
    mol_geometry: pymatgen.core.Molecule,
) -> tk.Molecule:
    """
    Infers an OpenFF Molecule from a Pymatgen Molecule.

    Constructs a MoleculeGraph from the Pymatgen Molecule using the OpenBabelNN local
    environment strategy and extends metal edges. Converts the resulting MoleculeGraph
    to an OpenFF Molecule using molgraph_to_openff_mol.

    Args:
        mol_geometry (pymatgen.core.Molecule): The Pymatgen Molecule to infer from.

    Returns
    -------
        tk.Molecule: The inferred OpenFF Molecule.
    """
    molgraph = MoleculeGraph.with_local_env_strategy(mol_geometry, OpenBabelNN())
    molgraph = metal_edge_extender(molgraph)
    return molgraph_to_openff_mol(molgraph)


def add_conformer(
    openff_mol: tk.Molecule, geometry: pymatgen.core.Molecule | None
) -> tuple[tk.Molecule, dict[int, int]]:
    """
    Add conformers to an OpenFF Molecule based on the provided geometry.

    If a geometry is provided, infers an OpenFF Molecule from it,
    finds an atom mapping between the inferred molecule and the
    input molecule, and adds the conformer coordinates to the input
    molecule. If no geometry is provided, generates a single conformer.

    Args:
        openff_mol (tk.Molecule): The OpenFF Molecule to add conformers to.
        geometry (pymatgen.core.Molecule): The geometry to use for adding conformers.

    Returns
    -------
        Tuple[tk.Molecule, Dict[int, int]]: A tuple containing
            the updated OpenFF Molecule with added conformers
            and a dictionary representing the atom mapping.
    """
    # TODO: test this
    if geometry:
        # for geometry in geometries:
        inferred_mol = infer_openff_mol(geometry)
        is_isomorphic, atom_map = get_atom_map(inferred_mol, openff_mol)
        if not is_isomorphic:
            raise ValueError(
                f"An isomorphism cannot be found between smile {openff_mol.to_smiles()}"
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
    atom_map: dict[int, int],
    charge_method: str,
    partial_charges: None | list[float],
) -> tk.Molecule:
    """
    Assign partial charges to an OpenFF Molecule.

    If partial charges are provided, assigns them to the molecule
    based on the atom mapping. If the molecule has only one atom,
    assigns the total charge as the partial charge. Otherwise,
    assigns partial charges using the specified charge method.

    Args:
        openff_mol (tk.Molecule): The OpenFF Molecule to assign partial charges to.
        atom_map (Dict[int, int]): A dictionary representing the atom mapping.
        charge_method (str): The charge method to use if partial charges are
            not provided.
        partial_charges (Union[None, List[float]]): A list of partial charges to
            assign or None to use the charge method.

    Returns
    -------
        tk.Molecule: The OpenFF Molecule with assigned partial charges.

    """
    # TODO: test this
    # assign partial charges
    if partial_charges is not None:
        partial_charges = np.array(partial_charges)
        chargs = partial_charges[list(atom_map.values())]  # type: ignore[call-overload]
        openff_mol.partial_charges = chargs * elementary_charge
    elif openff_mol.n_atoms == 1:
        openff_mol.partial_charges = (
            np.array([openff_mol.total_charge.magnitude]) * elementary_charge
        )
    else:
        openff_mol.assign_partial_charges(charge_method)
    return openff_mol


def create_openff_mol(
    smile: str,
    geometry: pymatgen.core.Molecule | str | Path | None = None,
    charge_scaling: float = 1,
    partial_charges: list[float] | None = None,
    backup_charge_method: str = "am1bcc",
) -> tk.Molecule:
    """
    Create an OpenFF Molecule from a SMILES string and optional geometry.

    Constructs an OpenFF Molecule from the provided SMILES
    string, adds conformers based on the provided geometry (if
    any), assigns partial charges using the specified method
    or provided partial charges, and applies charge scaling.

    Args:
        smile (str): The SMILES string of the molecule.
        geometry (Union[pymatgen.core.Molecule, str, Path], optional): The geometry to
        use for adding conformers. Can be a Pymatgen Molecule, file path, or None.
        charge_scaling (float, optional): The scaling factor for partial charges.
            Default is 1.
        partial_charges (List[float], optional): A list of partial charges to assign,
        or None to use the charge method.
        backup_charge_method (str, optional): The backup charge method to use if
            partial charges are not provided. Default is "am1bcc".

    Returns
    -------
        tk.Molecule: The created OpenFF Molecule.
    """
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
    geometry: pymatgen.core.Molecule | str | Path = None,
    partial_charges: list[float] = None,
) -> MoleculeSpec:
    """
    Create a MoleculeSpec from a SMILES string and other parameters.

    Constructs an OpenFF Molecule using create_openff_mol and creates a MoleculeSpec
    with the specified parameters.

    Args:
    smile (str): The SMILES string of the molecule.
    count (int): The number of molecules to create.
    name (str, optional): The name of the molecule. If not provided, defaults to
        the SMILES string.
    charge_scaling (float, optional): The scaling factor for partial charges.
        Default is 1.
    charge_method (str, optional): The charge method to use if partial charges
        are not provided. If not specified, defaults to "custom" if partial charges
        are provided, else "am1bcc".
    geometry (Union[pymatgen.core.Molecule, str, Path], optional): The geometry to
        use for adding conformers. Can be a Pymatgen Molecule, file path, or None.
    partial_charges (List[float], optional): A list of partial charges to assign, or
        None to use the charge method.

    Returns
    -------
        MoleculeSpec: The created MoleculeSpec
    """
    # TODO: test this

    if charge_method is None:
        charge_method = "custom" if partial_charges else "am1bcc"

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


def merge_specs_by_name_and_smile(mol_specs: list[MoleculeSpec]) -> list[MoleculeSpec]:
    """
    Merge MoleculeSpecs with the same name and SMILES string.

    Groups MoleculeSpecs by their name and SMILES string, and merges the counts of specs
    with matching name and SMILES. Returns a list of unique MoleculeSpecs.

    Args:
        mol_specs (List[MoleculeSpec]): A list of MoleculeSpecs to merge.

    Returns
    -------
        List[MoleculeSpec]: A list of merged MoleculeSpecs with unique name and SMILES
        combinations.
    """
    mol_specs = copy.deepcopy(mol_specs)
    merged_spec_dict: dict[tuple[str, str], MoleculeSpec] = {}
    for spec in mol_specs:
        key = (spec.openff_mol.to_smiles(), spec.name)
        if key in merged_spec_dict:
            merged_spec_dict[key].count += spec.count
        else:
            merged_spec_dict[key] = spec
    return list(merged_spec_dict.values())


def create_list_summing_to(total_sum: int, n_pieces: int) -> list:
    """
    Create a NumPy array with n_pieces elements that sum up to total_sum.

    Divides total_sum by n_pieces to determine the base value for each element.
    Distributes the remainder evenly among the elements.

    Args:
        total_sum (int): The desired sum of the array elements.
        n_pieces (int): The number of elements in the array.

    Returns
    -------
        numpy.ndarray: A 1D NumPy array with n_pieces elements summing up to total_sum.
    """
    div, mod = total_sum // n_pieces, total_sum % n_pieces
    return [div + 1] * mod + [div] * (n_pieces - mod)
