import pytest

import openff.toolkit as tk
import numpy as np

import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph

from atomate2.classical_md.utils import (
    molgraph_to_openff_mol,
    infer_openff_mol,
    get_atom_map,
    add_conformer,
    assign_partial_charges,
    create_openff_mol,
    create_mol_spec,
    merge_specs_by_name_and_smile,
)
from atomate2.classical_md.schemas import MoleculeSpec


@pytest.fixture
def mol_files(classical_md_data):
    geo_dir = classical_md_data / "molecule_charge_files"
    return {
        "CCO_xyz": str(geo_dir / "CCO.xyz"),
        "CCO_charges": str(geo_dir / "CCO.npy"),
        "FEC_r_xyz": str(geo_dir / "FEC-r.xyz"),
        "FEC_s_xyz": str(geo_dir / "FEC-s.xyz"),
        "FEC_charges": str(geo_dir / "FEC.npy"),
        "PF6_xyz": str(geo_dir / "PF6.xyz"),
        "PF6_charges": str(geo_dir / "PF6.npy"),
        "Li_charges": str(geo_dir / "Li.npy"),
        "Li_xyz": str(geo_dir / "Li.xyz"),
    }


def test_molgraph_to_openff_pf6(mol_files):
    """transform a water MoleculeGraph to a OpenFF water molecule"""
    pf6_mol = pymatgen.core.Molecule.from_file(mol_files["PF6_xyz"])
    pf6_mol.set_charge_and_spin(charge=-1)
    pf6_molgraph = MoleculeGraph.with_edges(
        pf6_mol,
        {
            (0, 1): {"weight": 1},
            (0, 2): {"weight": 1},
            (0, 3): {"weight": 1},
            (0, 4): {"weight": 1},
            (0, 5): {"weight": 1},
            (0, 6): {"weight": 1},
        },
    )

    pf6_openff_1 = tk.Molecule.from_smiles("F[P-](F)(F)(F)(F)F")

    pf6_openff_2 = molgraph_to_openff_mol(pf6_molgraph)
    assert pf6_openff_1 == pf6_openff_2


def test_molgraph_to_openff_cco(mol_files):
    from pymatgen.analysis.local_env import OpenBabelNN

    cco_pmg = pymatgen.core.Molecule.from_file(mol_files["CCO_xyz"])
    cco_molgraph = MoleculeGraph.with_local_env_strategy(cco_pmg, OpenBabelNN())

    cco_openff_1 = molgraph_to_openff_mol(cco_molgraph)

    cco_openff_2 = tk.Molecule.from_smiles("CCO")
    cco_openff_2.assign_partial_charges("mmff94")

    assert cco_openff_1 == cco_openff_2


@pytest.mark.parametrize(
    "xyz_path, smile, map_values",
    [
        ("CCO_xyz", "CCO", [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ("FEC_r_xyz", "O=C1OC[C@@H](F)O1", [0, 1, 2, 3, 4, 6, 7, 9, 8, 5]),
        ("FEC_s_xyz", "O=C1OC[C@H](F)O1", [0, 1, 2, 3, 4, 6, 7, 9, 8, 5]),
        ("PF6_xyz", "F[P-](F)(F)(F)(F)F", [1, 0, 2, 3, 4, 5, 6]),
    ],
)
def test_get_atom_map(xyz_path, smile, map_values, mol_files):
    mol = pymatgen.core.Molecule.from_file(mol_files[xyz_path])
    inferred_mol = infer_openff_mol(mol)
    openff_mol = tk.Molecule.from_smiles(smile)
    isomorphic, atom_map = get_atom_map(inferred_mol, openff_mol)
    assert isomorphic
    assert map_values == list(atom_map.values())


@pytest.mark.parametrize(
    "xyz_path, n_atoms, n_bonds",
    [
        ("CCO_xyz", 9, 8),
        ("FEC_r_xyz", 10, 10),
        ("FEC_s_xyz", 10, 10),
        ("PF6_xyz", 7, 6),
    ],
)
def test_infer_openff_mol(xyz_path, n_atoms, n_bonds, mol_files):
    mol = pymatgen.core.Molecule.from_file(mol_files[xyz_path])
    openff_mol = infer_openff_mol(mol)
    assert isinstance(openff_mol, tk.Molecule)
    assert openff_mol.n_atoms == n_atoms
    assert openff_mol.n_bonds == n_bonds


def test_add_conformer(mol_files):
    openff_mol = tk.Molecule.from_smiles("CCO")
    geometry = pymatgen.core.Molecule.from_file(mol_files["CCO_xyz"])
    openff_mol, atom_map = add_conformer(openff_mol, geometry)
    assert openff_mol.n_conformers == 1
    assert list(atom_map.values()) == list(range(openff_mol.n_atoms))


def test_assign_partial_charges(mol_files):
    openff_mol = tk.Molecule.from_smiles("CCO")
    geometry = pymatgen.core.Molecule.from_file(mol_files["CCO_xyz"])
    openff_mol, atom_map = add_conformer(openff_mol, geometry)
    partial_charges = np.load(mol_files["CCO_charges"])
    openff_mol = assign_partial_charges(openff_mol, atom_map, "am1bcc", partial_charges)
    assert np.allclose(openff_mol.partial_charges.magnitude, partial_charges)


def test_create_openff_mol(mol_files):
    smile = "CCO"
    geometry = mol_files["CCO_xyz"]
    partial_charges = np.load(mol_files["CCO_charges"])
    openff_mol = create_openff_mol(smile, geometry, 1.0, partial_charges, "am1bcc")
    assert isinstance(openff_mol, tk.Molecule)
    assert openff_mol.n_atoms == 9
    assert openff_mol.n_bonds == 8
    assert np.allclose(openff_mol.partial_charges.magnitude, partial_charges)


def test_create_mol_spec(mol_files):
    smile, count, name, geometry = "CCO", 10, "ethanol", mol_files["CCO_xyz"]
    partial_charges = np.load(mol_files["CCO_charges"])
    mol_spec = create_mol_spec(
        smile, count, name, 1.0, "am1bcc", geometry, partial_charges
    )
    assert isinstance(mol_spec, MoleculeSpec)
    assert mol_spec.name == name
    assert mol_spec.count == count
    assert mol_spec.formal_charge == int(np.sum(partial_charges))
    assert mol_spec.charge_method == "am1bcc"
    assert isinstance(mol_spec.openff_mol, tk.Molecule)


def test_merge_specs_by_name_and_smile(mol_files):
    smile1, count1, name1, geometry1 = "CCO", 5, "ethanol", mol_files["CCO_xyz"]
    partial_charges1 = np.load(mol_files["CCO_charges"])
    mol_spec1 = create_mol_spec(
        smile1, count1, name1, 1.0, "am1bcc", geometry1, partial_charges1
    )

    smile2, count2, name2, geometry2 = "CCO", 8, "ethanol", mol_files["CCO_xyz"]
    partial_charges2 = np.load(mol_files["CCO_charges"])
    mol_spec2 = create_mol_spec(
        smile2, count2, name2, 1.0, "am1bcc", geometry2, partial_charges2
    )

    mol_specs = [mol_spec1, mol_spec2]
    merged_specs = merge_specs_by_name_and_smile(mol_specs)
    assert len(merged_specs) == 1
    assert merged_specs[0].count == count1 + count2

    mol_specs[1].name = "ethanol2"
    merged_specs = merge_specs_by_name_and_smile(mol_specs)
    assert len(merged_specs) == 2
    assert merged_specs[0].name == name1
    assert merged_specs[0].count == count1
    assert merged_specs[1].name == "ethanol2"
    assert merged_specs[1].count == count2
