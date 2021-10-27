import pytest


def test_static_maker(mock_vasp, clean_dir, si_structure):
    from jobflow import run_locally

    from atomate2.vasp.schemas.task import TaskDocument

    # mapping from job name to directory containing test files
    ref_paths = {"static": "Si_static"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {"static": {"incar_settings": ["NSW", "ISMEAR"]}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate job
    from atomate2.vasp.jobs.core import StaticMaker

    job = StaticMaker().make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDocument)
    assert pytest.approx(output1.output.energy, -10.85037078)
