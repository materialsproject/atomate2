import pytest
from emmet.core.openff import ClassicalMDTaskDocument, MoleculeSpec

from atomate2.openff.core import generate_interchange

pytest.importorskip("openff.toolkit")
from openff.interchange import Interchange


def test_generate_interchange(mol_specs_small, run_job):
    mass_density = 1
    force_field = "openff_unconstrained-2.1.1.offxml"
    mol_specs = mol_specs_small

    job = generate_interchange(mol_specs, mass_density, force_field)
    task_doc = run_job(job)

    assert isinstance(task_doc, ClassicalMDTaskDocument)
    assert task_doc.force_field == force_field

    interchange = Interchange.parse_raw(task_doc.interchange)
    assert isinstance(interchange, Interchange)

    topology = interchange.topology
    assert topology.n_molecules == 30
    assert topology.n_atoms == 150
    assert topology.n_bonds == 120

    molecule_specs = task_doc.mol_specs
    assert len(molecule_specs) == 2
    assert all(isinstance(spec, MoleculeSpec) for spec in molecule_specs)
    assert molecule_specs[0].name == "ethanol"
    assert molecule_specs[0].count == 10
    assert molecule_specs[1].name == "water"
    assert molecule_specs[1].count == 20

    # TODO: debug issue with ForceField accepting iterables of FFs

    # Test with mol_specs as a list of dicts
    mol_specs_dicts = [
        {"smiles": "CCO", "count": 10, "name": "ethanol", "charge_method": "mmff94"},
        {"smiles": "O", "count": 20, "name": "water", "charge_method": "mmff94"},
    ]
    job = generate_interchange(mol_specs_dicts, mass_density, force_field)
    task_doc = run_job(job)
    molecule_specs = task_doc.mol_specs
    assert len(molecule_specs) == 2
    assert molecule_specs[0].name == "ethanol"
    assert molecule_specs[0].count == 10
    assert molecule_specs[1].name == "water"
    assert molecule_specs[1].count == 20


def test_generate_interchange_salt(mol_specs_salt, run_job):
    mass_density = 1
    force_field = "openff_unconstrained-2.1.1.offxml"
    mol_specs = mol_specs_salt

    job = generate_interchange(mol_specs, mass_density, force_field)
    task_doc = run_job(job)

    molecule_specs = task_doc.mol_specs
    assert len(molecule_specs) == 4
    assert molecule_specs[1].name == "ethanol"
    assert molecule_specs[1].count == 10
    assert molecule_specs[2].name == "water"
    assert molecule_specs[2].count == 20
