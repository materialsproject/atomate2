import numpy as np
import pymatgen
import pytest
from emmet.core.openff import MoleculeSpec
from pymatgen.analysis.graphs import MoleculeGraph

from atomate2.openff.utils import (
    counts_from_box_size,
    counts_from_masses,
    create_mol_spec,
    merge_specs_by_name_and_smiles,
)

pytest.importorskip("openff.toolkit")
import openff.toolkit as tk
from openff.interchange import Interchange
from openff.toolkit.topology import Topology
from openff.toolkit.topology.molecule import Molecule
from openff.units import Quantity
from pymatgen.io.openff import (
    add_conformer,
    assign_partial_charges,
    create_openff_mol,
    get_atom_map,
    infer_openff_mol,
    mol_graph_to_openff_mol,
)


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

    pf6_openff_2 = mol_graph_to_openff_mol(pf6_molgraph)
    assert pf6_openff_1 == pf6_openff_2


def test_molgraph_to_openff_cco(mol_files):
    from pymatgen.analysis.local_env import OpenBabelNN

    cco_pmg = pymatgen.core.Molecule.from_file(mol_files["CCO_xyz"])
    cco_molgraph = MoleculeGraph.with_local_env_strategy(cco_pmg, OpenBabelNN())

    cco_openff_1 = mol_graph_to_openff_mol(cco_molgraph)

    cco_openff_2 = tk.Molecule.from_smiles("CCO")
    cco_openff_2.assign_partial_charges("mmff94")

    assert cco_openff_1 == cco_openff_2


@pytest.mark.parametrize(
    "xyz_path, smiles, map_values",
    [
        ("CCO_xyz", "CCO", [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ("FEC_r_xyz", "O=C1OC[C@@H](F)O1", [0, 1, 2, 3, 4, 6, 7, 9, 8, 5]),
        ("FEC_s_xyz", "O=C1OC[C@H](F)O1", [0, 1, 2, 3, 4, 6, 7, 9, 8, 5]),
        ("PF6_xyz", "F[P-](F)(F)(F)(F)F", [1, 0, 2, 3, 4, 5, 6]),
    ],
)
def test_get_atom_map(xyz_path, smiles, map_values, mol_files):
    mol = pymatgen.core.Molecule.from_file(mol_files[xyz_path])
    inferred_mol = infer_openff_mol(mol)
    openff_mol = tk.Molecule.from_smiles(smiles)
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
    smiles = "CCO"
    geometry = mol_files["CCO_xyz"]
    partial_charges = np.load(mol_files["CCO_charges"])
    openff_mol = create_openff_mol(smiles, geometry, 1.0, partial_charges, "am1bcc")
    assert isinstance(openff_mol, tk.Molecule)
    assert openff_mol.n_atoms == 9
    assert openff_mol.n_bonds == 8
    assert np.allclose(openff_mol.partial_charges.magnitude, partial_charges)


def test_create_mol_spec(mol_files):
    smiles, count, name, geometry = "CCO", 10, "ethanol", mol_files["CCO_xyz"]
    partial_charges = np.load(mol_files["CCO_charges"])
    mol_spec = create_mol_spec(
        smiles, count, name, 1.0, "am1bcc", geometry, partial_charges
    )
    assert isinstance(mol_spec, MoleculeSpec)
    assert mol_spec.name == name
    assert mol_spec.count == count
    assert mol_spec.charge_scaling == 1.0
    assert mol_spec.charge_method == "am1bcc"
    assert isinstance(tk.Molecule.from_json(mol_spec.openff_mol), tk.Molecule)


def test_merge_specs_by_name_and_smiles(mol_files):
    smiles1, count1, name1, geometry1 = "CCO", 5, "ethanol", mol_files["CCO_xyz"]
    partial_charges1 = np.load(mol_files["CCO_charges"])
    mol_spec1 = create_mol_spec(
        smiles1, count1, name1, 1.0, "am1bcc", geometry1, partial_charges1
    )

    smiles2, count2, name2, geometry2 = "CCO", 8, "ethanol", mol_files["CCO_xyz"]
    partial_charges2 = np.load(mol_files["CCO_charges"])
    mol_spec2 = create_mol_spec(
        smiles2, count2, name2, 1.0, "am1bcc", geometry2, partial_charges2
    )

    mol_specs = [mol_spec1, mol_spec2]
    merged_specs = merge_specs_by_name_and_smiles(mol_specs)
    assert len(merged_specs) == 1
    assert merged_specs[0].count == count1 + count2

    mol_specs[1].name = "ethanol2"
    merged_specs = merge_specs_by_name_and_smiles(mol_specs)
    assert len(merged_specs) == 2
    assert merged_specs[0].name == name1
    assert merged_specs[0].count == count1
    assert merged_specs[1].name == "ethanol2"
    assert merged_specs[1].count == count2


def test_openff_mol_as_from_monty_dict():
    mol = Molecule.from_smiles("CCO")
    mol_dict = mol.as_dict()
    reconstructed_mol = Molecule.from_dict(mol_dict)

    assert mol.to_smiles() == reconstructed_mol.to_smiles()
    assert mol.n_atoms == reconstructed_mol.n_atoms
    assert mol.n_bonds == reconstructed_mol.n_bonds
    assert mol.n_angles == reconstructed_mol.n_angles
    assert mol.n_propers == reconstructed_mol.n_propers
    assert mol.n_impropers == reconstructed_mol.n_impropers


def test_openff_topology_as_from_monty_dict():
    topology = Topology.from_molecules([Molecule.from_smiles("CCO")])
    topology_dict = topology.as_dict()
    reconstructed_topology = Topology.from_dict(topology_dict)

    assert topology.n_molecules == reconstructed_topology.n_molecules
    assert topology.n_atoms == reconstructed_topology.n_atoms
    assert topology.n_bonds == reconstructed_topology.n_bonds
    assert topology.box_vectors == reconstructed_topology.box_vectors


def test_openff_interchange_as_from_monty_dict(interchange):
    # interchange = Interchange.from_smirnoff("openff-2.0.0.offxml", "CCO")
    interchange_dict = interchange.as_dict()
    reconstructed_interchange = Interchange.from_dict(interchange_dict)

    assert np.all(interchange.positions == reconstructed_interchange.positions)
    assert np.all(interchange.velocities == reconstructed_interchange.velocities)
    assert np.all(interchange.box == reconstructed_interchange.box)

    assert interchange.mdconfig == reconstructed_interchange.mdconfig

    topology = interchange.topology
    reconstructed_topology = reconstructed_interchange.topology

    assert topology.n_molecules == reconstructed_topology.n_molecules
    assert topology.n_atoms == reconstructed_topology.n_atoms
    assert topology.n_bonds == reconstructed_topology.n_bonds
    assert np.all(topology.box_vectors == reconstructed_topology.box_vectors)


def test_openff_quantity_as_from_monty_dict():
    quantity = Quantity(1.0, "kilocalorie / mole")
    quantity_dict = quantity.as_dict()
    reconstructed_quantity = Quantity.from_dict(quantity_dict)

    assert quantity.magnitude == reconstructed_quantity.magnitude
    assert quantity.units == reconstructed_quantity.units
    assert quantity == reconstructed_quantity


def test_calculate_elyte_composition():
    from atomate2.openff.utils import calculate_elyte_composition, counts_from_masses

    vol_ratio = {"O": 0.5, "CCO": 0.5}
    salts = {"[Li+]": 1.0, "F[P-](F)(F)(F)(F)F": 1.0}
    solvent_densities = {"O": 1.0, "CCO": 0.8}

    comp_dict = calculate_elyte_composition(
        vol_ratio, salts, solvent_densities, "volume"
    )
    counts = counts_from_masses(comp_dict, 100)
    assert sum(counts.values()) == 100

    mol_ratio = {
        "[Li+]": 0.00616,
        "F[P-](F)(F)(F)(F)F": 0.128,
        "C1COC(=O)O1": 0.245,  # EC
        "CCOC(=O)OC": 0.462,  # EMC
        "CC#N": 0.130,
        "FC1COC(=O)O1": 0.028,
    }
    counts2 = counts_from_masses(mol_ratio, 1000)
    assert np.allclose(sum(counts2.values()), 1000, atol=5)


def test_counts_calculators():
    mass_fractions = {"O": 0.5, "CCO": 0.5}

    counts_size = counts_from_box_size(mass_fractions, 3)
    counts_number = counts_from_masses(mass_fractions, 324)

    assert 200 < sum(counts_size.values()) < 500

    assert counts_size == counts_number
