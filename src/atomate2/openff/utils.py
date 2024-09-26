"""Utility functions for classical md subpackage."""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Literal

import numpy as np
import openff.toolkit as tk
from emmet.core.openff import MoleculeSpec
from pymatgen.core import Element, Molecule
from pymatgen.io.openff import create_openff_mol

if TYPE_CHECKING:
    import pathlib


def create_mol_spec(
    smiles: str,
    count: int,
    name: str = None,
    charge_scaling: float = 1,
    charge_method: str = None,
    geometry: Molecule | str | pathlib.Path = None,
    partial_charges: list[float] = None,
) -> MoleculeSpec:
    """Create a MoleculeSpec from a SMILES string and other parameters.

    Constructs an OpenFF Molecule using create_openff_mol and creates a MoleculeSpec
    with the specified parameters.

    Parameters
    ----------
    smiles : str
        The SMILES string of the molecule.
    count : int
        The number of molecules to create.
    name : str, optional
        The name of the molecule. If not provided, defaults to the SMILES string.
    charge_scaling : float, optional
        The scaling factor for partial charges. Default is 1.
    charge_method : str, optional
        The charge method to use if partial charges are not provided. If not specified,
        defaults to "custom" if partial charges are provided, else "am1bcc".
    geometry : Union[pymatgen.core.Molecule, str, Path], optional
        The geometry to use for adding conformers. Can be a Pymatgen Molecule, file path
        or None.
    partial_charges : List[float], optional
        A list of partial charges to assign, or None to use the charge method.

    Returns
    -------
    MoleculeSpec
        The created MoleculeSpec
    """
    if charge_method is None:
        charge_method = "custom" if partial_charges is not None else "am1bcc"

    openff_mol = create_openff_mol(
        smiles,
        geometry,
        charge_scaling,
        partial_charges,
        charge_method,
    )

    # create mol_spec
    return MoleculeSpec(
        name=(name or smiles),
        count=count,
        charge_scaling=charge_scaling,
        charge_method=charge_method,
        openff_mol=openff_mol.to_json(),
    )


def create_mol_spec_list(
    input_mol_specs: list[MoleculeSpec | dict],
) -> list[MoleculeSpec]:
    """
    Coerce and sort a MoleculeSpecs and dicts to MoleculeSpecs.

    Will sort alphabetically based on concatenated smiles and name.

    Parameters
    ----------
    input_mol_specs : list[dict | MoleculeSpec]
        List of dicts or MoleculeSpecs to coerce and sort.

    Returns
    -------
    List[MoleculeSpec]
        List of MoleculeSpecs sorted by smiles and name.
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

    mol_specs.sort(
        key=lambda x: tk.Molecule.from_json(x.openff_mol).to_smiles() + x.name
    )

    return mol_specs


def merge_specs_by_name_and_smiles(mol_specs: list[MoleculeSpec]) -> list[MoleculeSpec]:
    """Merge MoleculeSpecs with the same name and SMILES string.

    Groups MoleculeSpecs by their name and SMILES string, and merges the counts of specs
    with matching name and SMILES. Returns a list of unique MoleculeSpecs.

    Parameters
    ----------
    mol_specs : List[MoleculeSpec]
        A list of MoleculeSpecs to merge.

    Returns
    -------
    List[MoleculeSpec]
        A list of merged MoleculeSpecs with unique name and SMILES combinations.
    """
    mol_specs = copy.deepcopy(mol_specs)
    merged_spec_dict: dict[tuple[str, str], MoleculeSpec] = {}
    for spec in mol_specs:
        key = (tk.Molecule.from_json(spec.openff_mol).to_smiles(), spec.name)
        if key in merged_spec_dict:
            merged_spec_dict[key].count += spec.count
        else:
            merged_spec_dict[key] = spec
    return list(merged_spec_dict.values())


def calculate_elyte_composition(
    solvents: dict[str, float],
    salts: dict[str, float],
    solvent_densities: dict = None,
    solvent_ratio_dimension: Literal["mass", "volume"] = "mass",
) -> dict[str, float]:
    """Calculate the normalized mass ratios of an electrolyte solution.

    Parameters
    ----------
    solvents : dict
        Dictionary of solvent SMILES strings and their relative unit fraction.
    salts : dict
        Dictionary of salt SMILES strings and their molarities.
    solvent_densities : dict
        Dictionary of solvent SMILES strings and their densities (g/ml).
    solvent_ratio_dimension: optional, str
        Whether the solvents are included with a ratio of "mass" or "volume"

    Returns
    -------
    dict
        A dictionary containing the normalized mass ratios of molecules in
        the electrolyte solution.
    """
    # Check if all solvents have corresponding densities
    solvent_densities = solvent_densities or {}
    if set(solvents) > set(solvent_densities):
        raise ValueError("solvent_densities must contain densities for all solvents.")

    # convert masses to volumes so we can normalize volume
    if solvent_ratio_dimension == "mass":
        solvents = {
            smile: mass / solvent_densities[smile] for smile, mass in solvents.items()
        }

    # normalize volume ratios
    total_vol = sum(solvents.values())
    solvent_volumes = {smile: vol / total_vol for smile, vol in solvents.items()}

    # Convert volume ratios to mass ratios using solvent densities
    mass_ratio = {
        smile: vol * solvent_densities[smile] for smile, vol in solvent_volumes.items()
    }

    # Calculate the molecular weights of the solvent
    masses = {el.Z: el.atomic_mass for el in Element}
    salt_mws = {}
    for smile in salts:
        mol = tk.Molecule.from_smiles(smile, allow_undefined_stereo=True)
        salt_mws[smile] = sum(masses[atom.atomic_number] for atom in mol.atoms)

    # Convert salt mole ratios to mass ratios
    salt_mass_ratio = {
        salt: molarity * salt_mws[salt] / 1000 for salt, molarity in salts.items()
    }

    # Combine solvent and salt mass ratios
    combined_mass_ratio = {**mass_ratio, **salt_mass_ratio}

    # Calculate the total mass
    total_mass = sum(combined_mass_ratio.values())

    # Normalize the mass ratios
    return {species: mass / total_mass for species, mass in combined_mass_ratio.items()}


def counts_from_masses(species: dict[str, float], n_mol: int) -> dict[str, float]:
    """Calculate the number of mols needed to yield a given mass ratio.

    Parameters
    ----------
    species : list of str
        Dictionary of species SMILES strings and their relative mass fractions.
    n_mol : float
        Total number of mols. Returned array will sum to near n_mol.


    Returns
    -------
    numpy.ndarray
        n_mols: Number of each SMILES needed for the given mass ratio.
    """
    masses = {el.Z: el.atomic_mass for el in Element}

    mol_weights = []
    for smile in species:
        mol = tk.Molecule.from_smiles(smile, allow_undefined_stereo=True)
        mol_weights.append(sum(masses[atom.atomic_number] for atom in mol.atoms))

    mol_ratio = np.array(list(species.values())) / np.array(mol_weights)
    mol_ratio /= sum(mol_ratio)
    return {
        smile: int(np.round(ratio * n_mol))
        for smile, ratio in zip(species, mol_ratio, strict=True)
    }


def counts_from_box_size(
    species: dict[str, float], side_length: float, density: float = 0.8
) -> dict[str, float]:
    """Calculate the number of molecules needed to fill a box.

    Parameters
    ----------
    species : dict of str, float
        Dictionary of species SMILES strings and their relative mass fractions.
    side_length : int
        Side length of the cubic simulation box in nm.
    density : int, optional
        Density of the system in g/cm^3. Default is 1 g/cm^3.

    Returns
    -------
    dict of str, float
        Number of each species needed to fill the box with the given density.
    """
    masses = {el.Z: el.atomic_mass for el in Element}

    na = 6.02214076e23  # Avogadro's number
    volume = (side_length * 1e-7) ** 3  # Convert from nm3 to cm^3
    total_mass = volume * density  # grams

    # Calculate molecular weights
    mol_weights = []
    for smile in species:
        mol = tk.Molecule.from_smiles(smile, allow_undefined_stereo=True)
        mol_weights.append(sum(masses[atom.atomic_number] for atom in mol.atoms))
    mean_mw = np.mean(mol_weights)
    n_mol = (total_mass / mean_mw) * na

    # Calculate the number of moles needed for each species
    mol_ratio = np.array(list(species.values())) / np.array(mol_weights)
    mol_ratio /= sum(mol_ratio)

    # Convert moles to number of molecules
    return {
        smile: int(np.round(ratio * n_mol))
        for smile, ratio in zip(species.keys(), mol_ratio, strict=True)
    }


def create_mol_dicts(
    counts: dict[str, float],
    ion_charge_scaling: float,
    name_lookup: dict[str, str] = None,
    xyz_charge_lookup: dict[str, tuple] = None,
) -> list[dict]:
    """Create lists of mol specs from just counts. Still rudimentary."""
    spec_dicts = []
    for smile, count in counts.items():
        spec_dict = {
            "smile": smile,
            "count": count,
            "name": name_lookup.get(smile, smile),
        }
        if re.search(r"[+-]", smile):
            spec_dict["charge_scaling"] = ion_charge_scaling
        xyz_charge = xyz_charge_lookup.get(smile)
        if xyz_charge is not None:
            spec_dict["geometry"] = xyz_charge[0]
            spec_dict["partial_charges"] = xyz_charge[1]
            spec_dict["charge_method"] = "RESP"
        spec_dicts.append(spec_dict)
    return spec_dicts
