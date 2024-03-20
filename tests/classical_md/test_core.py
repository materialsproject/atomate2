from openff.interchange import Interchange

from atomate2.classical_md.core import generate_interchange
from atomate2.classical_md.schemas import MoleculeSpec, ClassicalMDTaskDocument


def test_generate_interchange(mol_specs_small, run_job):
    mass_density = 1
    force_field = "openff_unconstrained-2.1.1.offxml"
    mol_specs = mol_specs_small

    job = generate_interchange(mol_specs, mass_density, force_field)
    task_doc = run_job(job)

    assert isinstance(task_doc, ClassicalMDTaskDocument)
    assert task_doc.forcefield == force_field

    interchange = Interchange(**task_doc.interchange)
    assert isinstance(interchange, Interchange)

    topology = interchange.topology
    assert topology.n_molecules == 30
    assert topology.n_atoms == 150
    assert topology.n_bonds == 120

    molecule_specs = task_doc.molecule_specs
    assert len(molecule_specs) == 2
    assert all(isinstance(spec, MoleculeSpec) for spec in molecule_specs)
    assert molecule_specs[0].name == "ethanol"
    assert molecule_specs[0].count == 10
    assert molecule_specs[1].name == "water"
    assert molecule_specs[1].count == 20

    # TODO: debug issue with ForceField accepting iterables of FFs

    # Test with mol_specs as a list of dicts
    mol_specs_dicts = [
        {"smile": "CCO", "count": 10, "name": "ethanol"},
        {"smile": "O", "count": 20, "name": "water"},
    ]
    job = generate_interchange(mol_specs_dicts, mass_density, force_field)
    task_doc = run_job(job)
    assert len(task_doc.molecule_specs) == 2
    assert task_doc.molecule_specs[0].name == "ethanol"
    assert task_doc.molecule_specs[0].count == 10
    assert task_doc.molecule_specs[1].name == "water"
    assert task_doc.molecule_specs[1].count == 20