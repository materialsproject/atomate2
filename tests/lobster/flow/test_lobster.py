import numpy as np
from pymatgen.core.structure import Structure
from atomate2.vasp.flows.lobster import LobsterMaker
from atomate2.lobster.schemas import LobsterTaskDocument
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.dos import LobsterCompleteDos
from pymatgen.io.lobster import Lobsterin
# from atomate2.vasp.powerups import (
#        update_user_incar_settings,
#    )

def test_lobster_wf(mock_vasp, mock_lobster, clean_dir):
    from jobflow import run_locally

    #structure = Structure(
    #    lattice=[[4.34162192, 0.0, 2.50663673], [1.44720731, 4.09332126, 2.50663673], [0.0, 0.0, 5.01327346]],
    #    species=["Ba", "Te"],
    #    coords=[[-0.0, -0.0, -0.0], [0.5, 0.5, 0.5]],
    #)

    structure = Structure(lattice=[[3.422015, 0.0, 1.975702],[1.140671, 3.226306, 1.975702],[0.0, 0.0, 3.951402]],
                          species=["Na", "Cl"],
                          coords=[[-0.0, -0.0, -0.0], [0.5, 0.5, 0.5]])


    # mapping from job name to directory containing test files
    ref_paths = {
        "relax 1": "NaCl_static_relax_lobs/relax_1",
        "relax 2": "NaCl_static_relax_lobs/relax_2",
        "additional_static": "NaCl_static_relax_lobs/additonal_static",
        "static_run": "NaCl_static_relax_lobs/static_run",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "relax 1": {"incar_settings": ["NSW", "ISMEAR"]},
        "relax 2": {"incar_settings": ["NSW", "ISMEAR"]},
        "additional_static": {"incar_settings": ["NSW", "ISMEAR"]},
        "static_run": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    ref_paths_lobster = {
        "lobster_run_0": "NaCl_lobster_run_0",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_lobster_kwargs = {
        "lobster_run_0": {"lobsterin_settings": ["basisfunctions"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)
    mock_lobster(ref_paths_lobster)

    # !!! Generate job
    job = LobsterMaker(user_lobsterin_settings={'LSODOS':True},
                      additional_outputs=['DOSCAR.LSO.lobster']).make(structure=structure)

    #job = update_user_incar_settings(job, {"KSPACING": None}, name_filter="additional_static")


    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # !!! validation on the outputs
    #assert isinstance(responses[job.jobs[-1].uuid][1].output, LobsterTaskDocument)
