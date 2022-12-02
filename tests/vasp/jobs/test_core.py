from pytest import approx


def test_static_maker(mock_vasp, clean_dir, si_structure):
    import jobflow
    from jobflow import run_locally

    from atomate2.vasp.jobs.core import StaticMaker
    from atomate2.vasp.schemas.task import TaskDocument

    dstore = jobflow.SETTINGS.JOB_STORE.additional_stores["data"]

    # mapping from job name to directory containing test files
    ref_paths = {"static": "Si_band_structure/static"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "static": {"incar_settings": ["EDIFFG", "IBRION", "ISMEAR", "LREAL", "NSW"]}
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate job
    job = StaticMaker(task_document_kwargs={"store_volumetric_data": ["chgcar"]}).make(
        si_structure
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDocument)
    assert output1.output.energy == approx(-10.85037078)

    with dstore as s:
        doc = s.query_one({"job_uuid": job.uuid})
        dd = doc["data"]
        assert dd["@class"] == "Chgcar"


def test_relax_maker(mock_vasp, clean_dir, si_structure):
    from jobflow import run_locally

    from atomate2.vasp.jobs.core import RelaxMaker
    from atomate2.vasp.schemas.task import TaskDocument

    # mapping from job name to directory containing test files
    ref_paths = {"relax": "Si_double_relax/relax_1"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {"relax": {"incar_settings": ["EDIFFG", "ISMEAR", "NSW"]}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate job
    job = RelaxMaker().make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDocument)
    assert output1.output.energy == approx(-10.85043620)
    assert len(output1.calcs_reversed[0].output.ionic_steps) == 1
    assert output1.input.parameters["NSW"] > 1


def test_dielectric(mock_vasp, clean_dir, si_structure):
    import numpy as np
    from jobflow import run_locally

    from atomate2.vasp.jobs.core import DielectricMaker

    # mapping from job name to directory containing test files
    ref_paths = {"dielectric": "Si_dielectric"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {"dielectric": {"incar_settings": ["IBRION", "NSW"]}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # Generate dielectric flow
    job = DielectricMaker().make(si_structure)
    job.maker.input_set_generator.user_incar_settings["KSPACING"] = 0.5

    # Run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # Additional validation on the outputs of the job
    output1 = responses[job.uuid][1].output
    assert np.allclose(
        output1.calcs_reversed[0].output.epsilon_static,
        [[11.41539467, 0, 0], [0, 11.41539963, 0], [0, 0, 11.41539866]],
        atol=0.01,
    )
    assert np.allclose(
        output1.calcs_reversed[0].output.epsilon_ionic,
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        atol=0.01,
    )


def test_hse_relax(mock_vasp, clean_dir, si_structure):
    from jobflow import run_locally

    from atomate2.vasp.jobs.core import HSERelaxMaker
    from atomate2.vasp.schemas.task import TaskDocument

    # mapping from job name to directory containing test files
    ref_paths = {"hse relax": "Si_hse_relax"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {"hse relax": {"incar_settings": ["ISMEAR", "NSW"]}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate job
    job = HSERelaxMaker().make(si_structure)
    job.maker.input_set_generator.user_incar_settings["KSPACING"] = 0.4

    # Run the job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validation on the output of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDocument)
    assert output1.output.energy == approx(-12.5326576)
    assert len(output1.calcs_reversed[0].output.ionic_steps) == 3
    assert output1.input.parameters["NSW"] > 1


def test_hse_static_maker(mock_vasp, clean_dir, si_structure):
    from jobflow import run_locally

    from atomate2.vasp.jobs.core import HSEStaticMaker
    from atomate2.vasp.schemas.task import TaskDocument

    # mapping from job name to directory containing test files
    ref_paths = {"hse static": "Si_hse_band_structure/hse_static"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "hse static": {"incar_settings": ["ISMEAR", "LREAL", "NSW"]}
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate job
    job = HSEStaticMaker().make(si_structure)
    job.maker.input_set_generator.user_incar_settings["KSPACING"] = 0.4

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDocument)
    assert output1.output.energy == approx(-12.52887403)


def test_transmuter(mock_vasp, clean_dir, si_structure):
    import numpy as np
    from jobflow import run_locally

    from atomate2.vasp.jobs.core import TransmuterMaker
    from atomate2.vasp.schemas.task import TaskDocument

    # mapping from job name to directory containing test files
    ref_paths = {"transmuter": "Si_transmuter"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {"transmuter": {"incar_settings": ["ISMEAR", "NSW"]}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate transmuter job
    job = TransmuterMaker(
        transformations=["SupercellTransformation"],
        transformation_params=[{"scaling_matrix": ((1, 0, 0), (0, 1, 0), (0, 0, 2))}],
    ).make(si_structure)
    job.maker.input_set_generator.user_incar_settings["KSPACING"] = 0.5

    # run the job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDocument)
    assert output1.output.energy == approx(-21.33231747)
    assert output1.transformations["history"][0]["scaling_matrix"] == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 2],
    ]
    np.testing.assert_allclose(
        output1.structure.lattice.abc, [3.866974, 3.866975, 7.733949]
    )


def test_molecular_dynamics(mock_vasp, clean_dir, si_structure):
    import pytest
    from jobflow import run_locally

    from atomate2.vasp.jobs.core import MDMaker
    from atomate2.vasp.schemas.calculation import IonicStep, VaspObject
    from atomate2.vasp.schemas.task import TaskDocument

    # mapping from job name to directory containing test files
    ref_paths = {"molecular dynamics": "Si_molecular_dynamics/molecular_dynamics"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "molecular dynamics": {
            "incar_settings": [
                "IBRION",
                "TBEN",
                "TEND",
                "NSW",
                "POTIM",
                "MDALGO",
                "ISIF",
            ]
        }
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate job
    job = MDMaker().make(si_structure)
    NSW = 3
    job.maker.input_set_generator.user_incar_settings["NSW"] = NSW

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validation on the output

    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDocument)
    assert output1.output.energy == pytest.approx(-11.46520398)

    # check ionic steps stored as pymatgen Trajectory
    assert output1.calcs_reversed[0].output.ionic_steps is None
    traj = output1.vasp_objects[VaspObject.TRAJECTORY]
    assert len(traj.frame_properties) == NSW
    # simply check a frame property can be converted to an IonicStep
    for frame in traj.frame_properties:
        IonicStep(**frame)
