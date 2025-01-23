import numpy as np
import pytest
from emmet.core.openmm import OpenMMInterchange
from jobflow import Flow
from openmm import XmlSerializer

from atomate2.openff.utils import create_mol_spec
from atomate2.openmm.jobs import EnergyMinimizationMaker
from atomate2.openmm.jobs.base import BaseOpenMMMaker
from atomate2.openmm.jobs.generate import (
    XMLMoleculeFF,
    create_system_from_xml,
    generate_openmm_interchange,
)

pytest.importorskip("openff.toolkit")
import openff.toolkit as tk
from openff.interchange.components._packmol import pack_box
from openff.units import unit


def test_create_system_from_xml(openmm_data):
    # load strings of xml files into dict
    ff_xmls = [
        XMLMoleculeFF.from_file(openmm_data / "opls_xml_files" / "CCO.xml"),
        XMLMoleculeFF.from_file(openmm_data / "opls_xml_files" / "CO.xml"),
    ]

    # uncomment to regenerate data
    # download_opls_xml("CCO", opls_xmls / "CCO.xml")
    # download_opls_xml("CO", opls_xmls / "CO.xml")

    mol_specs = [
        {"smiles": "CCO", "count": 10},
        {"smiles": "CO", "count": 20},
    ]

    topology = pack_box(
        molecules=[tk.Molecule.from_smiles(spec["smiles"]) for spec in mol_specs],
        number_of_copies=[spec["count"] for spec in mol_specs],
        mass_density=0.8 * unit.grams / unit.milliliter,
    )

    create_system_from_xml(topology, ff_xmls)


def test_xml_molecule_from_file(openmm_data):
    xml_mol = XMLMoleculeFF.from_file(openmm_data / "opls_xml_files" / "CO.xml")

    assert isinstance(str(xml_mol), str)
    assert len(str(xml_mol)) > 100


def test_to_openff_molecule(openmm_data):
    xml_mol = XMLMoleculeFF.from_file(openmm_data / "opls_xml_files" / "CO.xml")

    mol = xml_mol.to_openff_molecule()
    assert len(mol.atoms) == 6
    assert len(mol.bonds) == 5


def test_assign_partial_charges_w_mol(openmm_data):
    xml_mol = XMLMoleculeFF.from_file(openmm_data / "opls_xml_files" / "CO.xml")

    openff_mol = tk.Molecule()

    atom_c00 = openff_mol.add_atom(6, 0, is_aromatic=False)
    atom_h02 = openff_mol.add_atom(1, 0, is_aromatic=False)
    atom_h03 = openff_mol.add_atom(1, 0, is_aromatic=False)
    atom_h04 = openff_mol.add_atom(1, 0, is_aromatic=False)
    atom_h05 = openff_mol.add_atom(1, 0, is_aromatic=False)
    atom_o01 = openff_mol.add_atom(8, 0, is_aromatic=False)

    openff_mol.add_bond(atom_c00, atom_o01, bond_order=1, is_aromatic=False)
    openff_mol.add_bond(atom_c00, atom_h02, bond_order=1, is_aromatic=False)
    openff_mol.add_bond(atom_c00, atom_h03, bond_order=1, is_aromatic=False)
    openff_mol.add_bond(atom_c00, atom_h04, bond_order=1, is_aromatic=False)
    openff_mol.add_bond(atom_o01, atom_h05, bond_order=1, is_aromatic=False)

    openff_mol.assign_partial_charges("mmff94")

    xml_mol.assign_partial_charges(openff_mol)
    assert xml_mol.partial_charges[0] > 0.2  # C
    assert xml_mol.partial_charges[1] < -0.3  # O
    assert xml_mol.partial_charges[5] > 0.1  # alcohol H


def test_assign_partial_charges_w_method(openmm_data):
    xml_mol = XMLMoleculeFF.from_file(openmm_data / "opls_xml_files" / "CO.xml")
    xml_mol.assign_partial_charges("mmff94")
    assert xml_mol.partial_charges[0] > 0.2  # C
    assert xml_mol.partial_charges[1] < -0.3  # O
    assert xml_mol.partial_charges[5] > 0.1  # alcohol H


def test_generate_openmm_interchange(openmm_data, run_job):
    mol_specs = [
        create_mol_spec("CCO", 10, name="ethanol", charge_method="mmff94"),
        create_mol_spec("CO", 300, name="water", charge_method="mmff94"),
    ]

    ff_xmls = [
        (openmm_data / "opls_xml_files" / "CCO.xml").read_text(),
        (openmm_data / "opls_xml_files" / "CO.xml").read_text(),
    ]

    job = generate_openmm_interchange(
        mol_specs, 1.0, ff_xmls, xml_method_and_scaling=("cm1a-lbcc", 1.14)
    )
    task_doc = run_job(job)
    molecule_specs = task_doc.mol_specs
    assert len(molecule_specs) == 2
    assert molecule_specs[0].name == "ethanol"
    assert molecule_specs[0].count == 10
    assert molecule_specs[1].name == "water"
    assert molecule_specs[1].count == 300
    co = tk.Molecule.from_json(molecule_specs[1].openff_mol)

    assert np.allclose(
        co.partial_charges.magnitude,
        np.array([-0.0492, -0.5873, 0.0768, 0.0768, 0.0768, 0.4061]),  # from file
    )


def test_make_from_prev(openmm_data, run_job):
    mol_specs = [
        create_mol_spec("CCO", 10, name="ethanol", charge_method="mmff94"),
        create_mol_spec("CO", 300, name="water", charge_method="mmff94"),
    ]

    ff_xmls = [
        (openmm_data / "opls_xml_files" / "CCO.xml").read_text(),
        (openmm_data / "opls_xml_files" / "CO.xml").read_text(),
    ]
    inter_job = generate_openmm_interchange(mol_specs, 1, ff_xmls)

    # Create an instance of BaseOpenMMMaker
    maker = BaseOpenMMMaker(n_steps=10)

    # monkey patch to allow running the test without openmm
    def do_nothing(self, sim, dir_name):
        pass

    BaseOpenMMMaker.run_openmm = do_nothing

    # Call the make method
    base_job = maker.make(
        inter_job.output.interchange, prev_dir=inter_job.output.dir_name
    )

    task_doc = run_job(Flow([inter_job, base_job]))

    assert task_doc.mol_specs is not None


def test_evolve_simulation(openmm_data, run_job):
    mol_specs = [
        create_mol_spec("CCO", 10, name="ethanol", charge_method="mmff94"),
        create_mol_spec("CO", 300, name="water", charge_method="mmff94"),
    ]

    ff_xmls = [
        (openmm_data / "opls_xml_files" / "CCO.xml").read_text(),
        (openmm_data / "opls_xml_files" / "CO.xml").read_text(),
    ]
    inter_job = generate_openmm_interchange(mol_specs, 1, ff_xmls)

    task_doc = run_job(inter_job)

    # test that opls charges are not being used
    co = tk.Molecule.from_json(task_doc.mol_specs[1].openff_mol)
    assert not np.allclose(
        co.partial_charges.magnitude,
        np.array([-0.5873, -0.0492, 0.0768, 0.0768, 0.4061, 0.0768]),  # from file
    )

    interchange_str = task_doc.interchange
    interchange = OpenMMInterchange.parse_raw(interchange_str)

    initial_state = XmlSerializer.deserialize(interchange.state)
    initial_position = initial_state.getPositions(asNumpy=True)

    maker = EnergyMinimizationMaker(max_iterations=1)
    min_job = maker.make(interchange)

    task_doc2 = run_job(min_job)
    interchange_str2 = task_doc2.interchange
    interchange2 = OpenMMInterchange.parse_raw(interchange_str2)

    final_state = XmlSerializer.deserialize(interchange2.state)
    final_position = final_state.getPositions(asNumpy=True)

    assert not (final_position == initial_position).all()
