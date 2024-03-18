import pytest
from pathlib import Path

import openff.toolkit as tk

import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph

from atomate2.classical_md.utils import (
    molgraph_to_openff_mol,
    infer_openff_mol,
    get_atom_map,
)

mol_files = Path("../test_data/classical_md/molecule_charge_files/")

CCO_xyz = str(mol_files / "CCO.xyz")
CCO_charges = str(mol_files / "CCO.npy")
FEC_r_xyz = str(mol_files / "FEC-r.xyz")
FEC_s_xyz = str(mol_files / "FEC-s.xyz")
FEC_charges = str(mol_files / "FEC.npy")
PF6_xyz = str(mol_files / "PF6.xyz")
PF6_charges = str(mol_files / "PF6.npy")
Li_charges = str(mol_files / "Li.npy")
Li_xyz = str(mol_files / "Li.xyz")


def test_molgraph_to_openff_pf6():
    """transform a water MoleculeGraph to a OpenFF water molecule"""
    pf6_mol = pymatgen.core.Molecule.from_file(PF6_xyz)
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


def test_molgraph_to_openff_cco():
    from pymatgen.analysis.local_env import OpenBabelNN

    cco_pmg = pymatgen.core.Molecule.from_file(CCO_xyz)
    cco_molgraph = MoleculeGraph.with_local_env_strategy(cco_pmg, OpenBabelNN())

    cco_openff_1 = molgraph_to_openff_mol(cco_molgraph)

    cco_openff_2 = tk.Molecule.from_smiles("CCO")
    cco_openff_2.assign_partial_charges("mmff94")

    assert cco_openff_1 == cco_openff_2


@pytest.mark.parametrize(
    "xyz_path, smile, map_values",
    [
        (CCO_xyz, "CCO", [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        (FEC_r_xyz, "O=C1OC[C@@H](F)O1", [0, 1, 2, 3, 4, 6, 7, 9, 8, 5]),
        (FEC_s_xyz, "O=C1OC[C@H](F)O1", [0, 1, 2, 3, 4, 6, 7, 9, 8, 5]),
        (PF6_xyz, "F[P-](F)(F)(F)(F)F", [1, 0, 2, 3, 4, 5, 6]),
    ],
)
def test_get_atom_map(xyz_path, smile, map_values):
    mol = pymatgen.core.Molecule.from_file(xyz_path)
    inferred_mol = infer_openff_mol(mol)
    openff_mol = tk.Molecule.from_smiles(smile)
    isomorphic, atom_map = get_atom_map(inferred_mol, openff_mol)
    assert isomorphic
    assert map_values == list(atom_map.values())


@pytest.mark.parametrize(
    "xyz_path, n_atoms, n_bonds",
    [
        (CCO_xyz, 9, 8),
        (FEC_r_xyz, 10, 10),
        (FEC_s_xyz, 10, 10),
        (PF6_xyz, 7, 6),
    ],
)
def test_infer_openff_mol(xyz_path, n_atoms, n_bonds):
    mol = pymatgen.core.Molecule.from_file(xyz_path)
    openff_mol = infer_openff_mol(mol)
    assert isinstance(openff_mol, tk.Molecule)
    assert openff_mol.n_atoms == n_atoms
    assert openff_mol.n_bonds == n_bonds


def test_add_conformer():
    return


def test_assign_partial_charges():
    return


def test_create_openff_mol():
    return


def test_create_mol_spec():
    return


def test_merge_specs_by_name_and_smile():
    return
