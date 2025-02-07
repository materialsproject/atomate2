from atomate2.vasp.flows.lobster import VaspLobsterMaker
from pymatgen.core.structure import Structure
from jobflow_remote import submit_flow, set_run_config
from atomate2.vasp.powerups import update_user_incar_settings
from atomate2.vasp.powerups import update_vasp_custodian_handlers

structure = Structure(
    lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
    species=["Mg", "O"],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)

lobster = VaspLobsterMaker().make(structure)

resources = {"nodes": 3, "partition": "micro", "time": "00:55:00", "ntasks": 144}

resources_lobster = {"nodes": 1, "partition": "micro", "time": "02:55:00", "ntasks": 48}
lobster = set_run_config(lobster, name_filter="lobster", resources=resources_lobster)

lobster = update_user_incar_settings(lobster, {"NPAR": 4})
print(submit_flow(lobster, worker="my_worker", resources=resources, project="my_project"))
