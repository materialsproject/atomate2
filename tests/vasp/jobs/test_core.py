import jobflow
import numpy as np
from emmet.core.tasks import TaskDoc
from jobflow import run_locally
from numpy.testing import assert_allclose
from pytest import approx

from atomate2.vasp.jobs.core import (
    DielectricMaker,
    HSERelaxMaker,
    HSEStaticMaker,
    RelaxMaker,
    StaticMaker,
    TransmuterMaker,
)


def test_static_maker(mock_vasp, clean_dir, si_structure):
    job_store = jobflow.SETTINGS.JOB_STORE

    # mapping from job name to directory containing test files
    ref_paths = {"static": "Si_band_structure/static"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "static": {"incar_settings": ["EDIFFG", "IBRION", "ISMEAR", "LREAL", "NSW"]}
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate job
    job = StaticMaker(task_document_kwargs={"store_volumetric_data": ("chgcar",)}).make(
        si_structure
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDoc)
    assert output1.output.energy == approx(-10.85037078)

    with job_store.additional_stores["data"] as store:
        doc = store.query_one({"job_uuid": job.uuid})
    assert doc["data"]["@class"] == "Chgcar"


def test_relax_maker(mock_vasp, clean_dir, si_structure):
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

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDoc)
    assert output1.output.energy == approx(-10.85043620)
    assert len(output1.calcs_reversed[0].output.ionic_steps) == 1
    assert output1.input.parameters["NSW"] > 1


def test_dielectric(mock_vasp, clean_dir, si_structure):
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
    assert_allclose(
        output1.calcs_reversed[0].output.epsilon_static,
        [[11.41539467, 0, 0], [0, 11.41539963, 0], [0, 0, 11.41539866]],
        atol=0.01,
    )
    assert_allclose(
        output1.calcs_reversed[0].output.epsilon_ionic,
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        atol=0.01,
    )


def test_hse_relax(mock_vasp, clean_dir, si_structure):
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
    assert isinstance(output1, TaskDoc)
    assert output1.output.energy == approx(-12.5326576)
    assert len(output1.calcs_reversed[0].output.ionic_steps) == 3
    assert output1.input.parameters["NSW"] > 1


def test_hse_static_maker(mock_vasp, clean_dir, si_structure):
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

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDoc)
    assert output1.output.energy == approx(-12.52887403)


def test_transmuter(mock_vasp, clean_dir, si_structure):
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
    assert isinstance(output1, TaskDoc)
    assert output1.output.energy == approx(-21.33231747)
    assert output1.transformations["history"][0]["scaling_matrix"] == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 2],
    ]
    np.testing.assert_allclose(
        output1.structure.lattice.abc, [3.866974, 3.866975, 7.733949]
    )
