from jobflow import run_locally
from pymatgen.core.structure import Structure

from atomate2.lobster.jobs import LobsterMaker
from atomate2.lobster.schemas import LobsterTaskDocument
from atomate2.vasp.flows.lobster import VaspLobsterMaker
from atomate2.vasp.flows.mp import MPVaspLobsterMaker
from atomate2.vasp.jobs.lobster import LobsterStaticMaker
from atomate2.vasp.powerups import update_user_incar_settings


def test_lobster_uniform_maker(
    mock_vasp, mock_lobster, clean_dir, memory_jobstore, si_structure: Structure
):
    # mapping from job name to directory containing test files
    ref_paths = {
        "relax 1": "Si_lobster_uniform/relax_1",
        "relax 2": "Si_lobster_uniform/relax_2",
        "static": "Si_lobster_uniform/static",
        "non-scf uniform": "Si_lobster_uniform/non-scf_uniform",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "relax 1": {"incar_settings": ["NSW", "ISMEAR"]},
        "relax 2": {"incar_settings": ["NSW", "ISMEAR"]},
        "static": {
            "incar_settings": [
                "NSW",
                "LWAVE",
                "ISMEAR",
                "ISYM",
                "NBANDS",
                "ISPIN",
                "LCHARG",
            ],
            # TODO restore POSCAR input checking e.g. when next updating test files
            "check_inputs": ["potcar", "kpoints", "incar"],
        },
        "non-scf uniform": {
            "incar_settings": [
                "NSW",
                "LWAVE",
                "ISMEAR",
                "ISYM",
                "NBANDS",
                "ISPIN",
                "ICHARG",
            ],
            # TODO restore POSCAR input checking e.g. when next updating test files
            "check_inputs": ["potcar", "kpoints", "incar"],
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

    job = VaspLobsterMaker(
        lobster_maker=LobsterMaker(
            task_document_kwargs={
                "calc_quality_kwargs": {"potcar_symbols": ["Si"], "n_bins": 10},
                "add_coxxcar_to_task_document": True,
            },
            user_lobsterin_settings={
                "COHPstartEnergy": -5.0,
                "COHPEndEnergy": 5.0,
                "cohpGenerator": "from 0.1 to 3.0 orbitalwise",
            },
        ),
        delete_wavecars=False,
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

    for key, value in (
        responses[job.jobs[-1].uuid][1]
        .replace.output["lobster_task_documents"][0]
        .resolve(memory_jobstore)
        .dict()
        .items()
    ):
        if key in ("lso_dos", "band_overlaps"):
            assert value is None
        else:
            assert value is not None


def test_lobstermaker(
    mock_vasp, mock_lobster, clean_dir, memory_jobstore, si_structure: Structure
):
    # mapping from job name to directory containing test files
    ref_paths = {
        "relax 1": "Si_lobster/relax_1",
        "relax 2": "Si_lobster/relax_2",
        "static_run": "Si_lobster/static_run",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "relax 1": {"incar_settings": ["NSW", "ISMEAR"]},
        "relax 2": {"incar_settings": ["NSW", "ISMEAR"]},
        "static_run": {
            "incar_settings": ["NSW", "LWAVE", "ISMEAR", "ISYM", "NBANDS", "ISPIN"],
            # TODO restore POSCAR input checking e.g. when next updating test files
            "check_inputs": ["potcar", "kpoints", "incar"],
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

    job = VaspLobsterMaker(
        lobster_static_maker=LobsterStaticMaker(),
        lobster_maker=LobsterMaker(
            task_document_kwargs={
                "calc_quality_kwargs": {"potcar_symbols": ["Si"], "n_bins": 10},
                "add_coxxcar_to_task_document": True,
            },
            user_lobsterin_settings={
                "COHPstartEnergy": -5.0,
                "COHPEndEnergy": 5.0,
                "cohpGenerator": "from 0.1 to 3.0 orbitalwise",
            },
        ),
        delete_wavecars=False,
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

    for key, value in (
        responses[job.jobs[-1].uuid][1]
        .replace.output["lobster_task_documents"][0]
        .resolve(memory_jobstore)
        .dict()
        .items()
    ):
        if key in ("lso_dos", "band_overlaps"):
            assert value is None
        else:
            assert value is not None


def test_lobstermaker_delete(
    mock_vasp, mock_lobster, clean_dir, memory_jobstore, si_structure: Structure
):
    # mapping from job name to directory containing test files
    ref_paths = {
        "relax 1": "Si_lobster/relax_1",
        "relax 2": "Si_lobster/relax_2",
        "static_run": "Si_lobster/static_run",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "relax 1": {"incar_settings": ["NSW", "ISMEAR"]},
        "relax 2": {"incar_settings": ["NSW", "ISMEAR"]},
        "static_run": {
            "incar_settings": ["NSW", "LWAVE", "ISMEAR", "ISYM", "NBANDS", "ISPIN"],
            # TODO restore POSCAR input checking e.g. when next updating test files
            "check_inputs": ["potcar", "kpoints", "incar"],
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

    job = VaspLobsterMaker(
        lobster_static_maker=LobsterStaticMaker(),
        lobster_maker=LobsterMaker(
            task_document_kwargs={
                "calc_quality_kwargs": {"potcar_symbols": ["Si"], "n_bins": 10},
            },
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
    responses = run_locally(
        job, store=memory_jobstore, create_folders=True, ensure_success=True
    )

    assert isinstance(
        responses[job.jobs[-2].uuid][1]
        .replace.output["lobster_task_documents"][0]
        .resolve(memory_jobstore),
        LobsterTaskDocument,
    )


def test_mp_vasp_lobstermaker(
    mock_vasp, mock_lobster, clean_dir, memory_jobstore, vasp_test_dir
):
    # mapping from job name to directory containing test files
    ref_paths = {
        "MP GGA relax 1": "Fe_lobster_mp/GGA_relax_1",
        "MP GGA relax 2": "Fe_lobster_mp/GGA_relax_2",
        "MP GGA static": "Fe_lobster_mp/GGA_static",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "MP GGA relax 1": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["poscar", "potcar", "kpoints", "incar"],
        },
        "MP GGA relax 2": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["poscar", "potcar", "kpoints", "incar"],
        },
        "MP GGA static": {
            "incar_settings": ["NSW", "LWAVE", "ISMEAR", "ISYM", "NBANDS", "ISPIN"],
            "check_inputs": ["poscar", "potcar", "kpoints", "incar"],
        },
    }

    ref_paths_lobster = {
        "lobster_run_0": "Fe_lobster/lobster_0",
        "delete_lobster_wavecar": "Fe_lobster/lobster_0",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_lobster_kwargs = {
        "lobster_run_0": {"lobsterin_settings": ["basisfunctions"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)
    mock_lobster(ref_paths_lobster, fake_run_lobster_kwargs)

    fe_structure = Structure(
        lattice=[
            [2.33614509e00, 1.11670000e-04, -8.25822930e-01],
            [-1.16807798e00, 2.02304724e00, -8.26082200e-01],
            [1.17007387e00, 2.02730500e00, 3.31032796e00],
        ],
        species=["Fe", "Fe"],
        coords=[
            [5.0000002e-01, 5.0000008e-01, 4.9999999e-01],
            [9.9999998e-01, 9.9999992e-01, 1.0000000e-08],
        ],
    )

    job = MPVaspLobsterMaker(
        lobster_maker=LobsterMaker(
            task_document_kwargs={
                "calc_quality_kwargs": {"potcar_symbols": ["Fe_pv"], "n_bins": 10},
                "save_computational_data_jsons": False,
                "save_cba_jsons": False,
                "add_coxxcar_to_task_document": False,
            },
            user_lobsterin_settings={
                "COHPstartEnergy": -5.0,
                "COHPEndEnergy": 5.0,
                "cohpGenerator": "from 0.1 to 3.0 orbitalwise",
            },
        ),
    ).make(fe_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        job, create_folders=True, ensure_success=True, store=memory_jobstore
    )

    task_doc = (
        responses[job.jobs[-2].uuid][1]
        .replace.output["lobster_task_documents"][0]
        .resolve(memory_jobstore)
    )

    assert isinstance(task_doc, LobsterTaskDocument)
