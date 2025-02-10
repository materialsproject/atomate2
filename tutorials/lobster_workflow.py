#!/usr/bin/env python
# The first lines are needed to ensure that we can mock VASP and LOBSTER runs.


from mock_lobster import mock_lobster
from mock_vasp import TEST_DIR, mock_vasp

ref_paths = {
    "relax 1": "Si_lobster_uniform/relax_1",
    "relax 2": "Si_lobster_uniform/relax_2",
    "static": "Si_lobster_uniform/static",
    "non-scf uniform": "Si_lobster_uniform/non-scf_uniform",
}
ref_paths_lobster = {
    "lobster_run_0": "Si_lobster/lobster_0",
}


# We first load a structure that we want to analyze with bonding analysis.


from jobflow import JobStore, run_locally
from maggma.stores import MemoryStore
from pymatgen.core import Structure

from atomate2.vasp.flows.lobster import LobsterMaker, VaspLobsterMaker
from atomate2.vasp.powerups import update_user_incar_settings

job_store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
si_structure = Structure.from_file(TEST_DIR / "structures" / "Si.cif")


# Then, we initialize a workflow:


job = VaspLobsterMaker(
    lobster_maker=LobsterMaker(
        task_document_kwargs={"analyze_outputs": False},
        user_lobsterin_settings={
            "COHPstartEnergy": -5.0,
            "COHPEndEnergy": 5.0,
            "cohpGenerator": "from 0.1 to 3.0 orbitalwise",
        },
    ),
    delete_wavecars=True,
).make(si_structure)
job = update_user_incar_settings(job, {"NPAR": 4})

# run the flow or job and ensure that it finished running successfully


# We then run this workflow locally to show-case the capabilities. In real-life, you would omit the `mock*` parts.


with mock_vasp(ref_paths) as mf:
    with mock_lobster(ref_paths_lobster) as mf2:
        responses = run_locally(
            job, store=job_store, create_folders=True, ensure_success=True
        )
