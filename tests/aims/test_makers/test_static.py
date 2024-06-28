"""Test various makers"""

import os

import pytest

cwd = os.getcwd()


def test_static_maker(si, tmp_path, mock_aims, species_dir):
    from jobflow import run_locally
    from pymatgen.io.aims.sets.core import StaticSetGenerator

    from atomate2.aims.jobs.core import StaticMaker
    from atomate2.aims.schemas.task import AimsTaskDoc

    # mapping from job name to directory containing test files
    ref_paths = {"base": "static-si"}

    # settings passed to fake_run_aims; adjust these to check for certain input settings
    fake_run_aims_kwargs = {}

    # automatically use fake FHI-aims
    mock_aims(ref_paths, fake_run_aims_kwargs)

    parameters = {
        "k_grid": [2, 2, 2],
        "species_dir": (species_dir / "light").as_posix(),
    }
    # generate job
    maker = StaticMaker(input_set_generator=StaticSetGenerator(user_params=parameters))
    maker.name = "base"
    job = maker.make(si)

    # run the flow or job and ensure that it finished running successfully
    os.chdir(tmp_path)
    responses = run_locally(job, create_folders=True, ensure_success=True)
    os.chdir(cwd)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, AimsTaskDoc)
    assert output1.output.energy == pytest.approx(-15800.099740991)
