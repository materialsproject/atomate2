import numpy as np
import openff.toolkit as tk
from emmet.core.openmm import OpenMMInterchange
from jobflow import Flow
from openmm import XmlSerializer

from atomate2.openff.utils import create_mol_spec
from atomate2.openmm.jobs import EnergyMinimizationMaker
from atomate2.openmm.jobs.base import BaseOpenMMMaker
from atomate2.openmm.jobs.generate import generate_openmm_interchange


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
    molecule_specs = task_doc.interchange_meta
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
    def do_nothing(self, sim):
        pass

    BaseOpenMMMaker.run_openmm = do_nothing

    # Call the make method
    base_job = maker.make(
        inter_job.output.interchange, prev_dir=inter_job.output.dir_name
    )

    task_doc = run_job(Flow([inter_job, base_job]))

    assert task_doc.interchange_meta is not None


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
    co = tk.Molecule.from_json(task_doc.interchange_meta[1].openff_mol)
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
