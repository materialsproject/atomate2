from atomate2.vasp.schemas.task import TaskDocument


def test_static_maker(mock_vasp, clean_dir, si_structure):
    from jobflow import run_locally

    from atomate2.vasp.jobs.core import StaticMaker

    ref_paths = {"static": "Si_static"}
    fake_run_vasp_kwargs = {"static": {"incar_settings": ["NSW", "ISMEAR"]}}
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    job = StaticMaker().make(si_structure)
    responses = run_locally(job, create_folders=True)

    assert isinstance(responses[job.uuid][0].output, TaskDocument)
