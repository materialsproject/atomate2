from jobflow import run_locally

from atomate2.openmm.jobs.openmm_set_maker import OpenMMSetMaker

from maggma.stores import MemoryStore


def test_openmm_set_maker(job_store):
    input_mol_dicts = [
        {"smile": "O", "count": 200},
        {"smile": "CCO", "count": 20},
    ]

    set_maker = OpenMMSetMaker()

    set_job = set_maker.make(input_mol_dicts=input_mol_dicts, density=1)

    responses = run_locally(set_job, store=job_store)

    print("hi")
