from jobflow import run_locally
from pymatgen.core.structure import Structure

from atomate2.lobster.schemas import LobsterTaskDocument
from atomate2.vasp.flows.lobster import LobsterMaker
from atomate2.vasp.powerups import (
    update_user_incar_settings,
)


def test_lobstermaker(mock_vasp, mock_lobster, clean_dir, memory_jobstore):
    # mapping from job name to directory containing test files
    ref_paths = {
        "preconvergence run": "Si_lobster/preconvergence_run",
        "relax 1": "Si_lobster/relax_1",
        "relax 2": "Si_lobster/relax_2",
        "static_run": "Si_lobster/static_run",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "relax 1": {"incar_settings": ["NSW", "ISMEAR"]},
        "relax 2": {"incar_settings": ["NSW", "ISMEAR"]},
        "preconvergence run": {
            "incar_settings": ["NSW", "ISMEAR", "LWAVE", "ISYM"],
        },
        "static_run": {
            "incar_settings": ["NSW", "LWAVE", "ISMEAR", "ISYM", "NBANDS"],
            "check_inputs": ["poscar", "potcar", "kpoints", "incar", "wavecar"],
        },
    }

    ref_paths_lobster = {
        "lobster_run_0": "Si_lobster/lobster_0",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_lobster_kwargs = {
        "lobster_run_0": {"lobsterin_settings": ["basisfunctions"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)
    mock_lobster(ref_paths_lobster, fake_run_lobster_kwargs)

    si_structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )
    job = LobsterMaker(
        user_lobsterin_settings={
            "COHPstartEnergy": -5.0,
            "COHPEndEnergy": 5.0,
            "cohpGenerator": "from 0.1 to 3.0 orbitalwise",
        },
        delete_all_wavecars=False,
    ).make(si_structure)
    job = update_user_incar_settings(job, {"NPAR": 4})

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        job, store=memory_jobstore, create_folders=True, ensure_success=True
    )

    assert isinstance(
        responses[job.jobs[-1].uuid][1]
        .replace.output["lobster_task_documents"][0]
        .resolve(memory_jobstore),
        LobsterTaskDocument,
    )


def test_lobstermaker_delete(mock_vasp, mock_lobster, clean_dir, memory_jobstore):
    # mapping from job name to directory containing test files
    ref_paths = {
        "preconvergence run": "Si_lobster/preconvergence_run",
        "relax 1": "Si_lobster/relax_1",
        "relax 2": "Si_lobster/relax_2",
        "static_run": "Si_lobster/static_run",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "relax 1": {"incar_settings": ["NSW", "ISMEAR"]},
        "relax 2": {"incar_settings": ["NSW", "ISMEAR"]},
        "preconvergence run": {
            "incar_settings": ["NSW", "ISMEAR", "LWAVE", "ISYM"],
        },
        "static_run": {
            "incar_settings": ["NSW", "LWAVE", "ISMEAR", "ISYM", "NBANDS"],
            "check_inputs": ["poscar", "potcar", "kpoints", "incar", "wavecar"],
        },
    }

    ref_paths_lobster = {
        "lobster_run_0": "Si_lobster/lobster_0",
        "delete_lobster_wavecar": "Si_lobster/lobster_0",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_lobster_kwargs = {
        "lobster_run_0": {"lobsterin_settings": ["basisfunctions"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)
    mock_lobster(ref_paths_lobster, fake_run_lobster_kwargs)

    si_structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )
    job = LobsterMaker(
        user_lobsterin_settings={
            "COHPstartEnergy": -5.0,
            "COHPEndEnergy": 5.0,
            "cohpGenerator": "from 0.1 to 3.0 orbitalwise",
        },
        delete_all_wavecars=True,
    ).make(si_structure)
    job = update_user_incar_settings(job, {"NPAR": 4})

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        job, store=memory_jobstore, create_folders=True, ensure_success=True
    )

    assert isinstance(
        responses[job.jobs[-2].uuid][1]
        .replace.output["lobster_task_documents"][0]
        .resolve(memory_jobstore),
        LobsterTaskDocument,
    )
