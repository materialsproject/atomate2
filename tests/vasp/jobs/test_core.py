from atomate2.vasp.schemas.task import TaskDocument


def test_static_maker(mock_vasp, clean_dir):
    from jobflow import run_locally
    from pymatgen.core import Structure

    from atomate2.vasp.jobs.core import StaticMaker

    ref_paths = {"static": "Si_static"}
    fake_run_vasp_kwargs = {"static": {"incar_settings": ["NSW", "ISMEAR"]}}
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    structure = Structure(
        [3, 0, 0, 0, 3, 0, 0, 0, 3], ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    job = StaticMaker().make(structure)
    responses = run_locally(job, create_folders=True)

    print(responses)

    assert isinstance(responses[job.uuid][0].output, TaskDocument)
