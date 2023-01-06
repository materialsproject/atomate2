import numpy as np
from pymatgen.core.structure import Structure
from atomate2.vasp.flows.lobster import LobsterMaker

from atomate2.lobster.schemas import LobsterTaskDocument
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.dos import LobsterCompleteDos
from pymatgen.io.lobster import Lobsterin
from atomate2.vasp.powerups import (
        update_user_incar_settings,
        update_user_kpoints_settings
    )
from atomate2.vasp.jobs.lobster import (VaspLobsterMaker,
                                        get_basis_infos,update_user_incar_settings_job,
                                        get_lobster_jobs,)
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import StaticMaker, RelaxMaker
from atomate2.vasp.sets.core import StaticSetGenerator

    #assert isinstance(responses["lobster_run_0"], LobsterTaskDocument)

def test_lobstermaker(mock_vasp,mock_lobster,clean_dir):
    from jobflow import run_locally

    structure = Structure(lattice=[[3.422015, 0.0, 1.975702], [1.140671, 3.226306, 1.975702], [0.0, 0.0, 3.951402]],
                          species=["Na", "Cl"],
                          coords=[[-0.0, -0.0, -0.0], [0.5, 0.5, 0.5]])

    # mapping from job name to directory containing test files
    ref_paths = {
        #"relax 1": "NaCl_static_relax_lobs/relax_1",
        #"relax 2": "NaCl_static_relax_lobs/relax_2",
        #"additional_static_run": "NaCl_static_relax_lobs/additional_static",
        "static_run": "NaCl_static_relax_lobs/static_run",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
       # "relax 1": {"incar_settings": ["NSW", "ISMEAR"]},
       # "relax 2": {"incar_settings": ["NSW", "ISMEAR"]},
       # "additional_static_run": {"incar_settings": ["NSW", "ISMEAR"]},
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
    mock_lobster(ref_paths_lobster, fake_run_lobster_kwargs)

    # !!! Generate job
    job = LobsterMaker(user_lobsterin_settings={'LSODOS': True} ,bulk_relax_maker=None, additional_static_run_maker=None,
                       additional_outputs=['DOSCAR.LSO.lobster'],
                       delete_all_wavecars=False).make(structure=structure)

    # job = update_user_incar_settings(job, {"KSPACING": None}, name_filter="additional_static")

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)
    print(responses)

    assert isinstance(responses[job.jobs[-2].uuid][1].output, LobsterTaskDocument)
