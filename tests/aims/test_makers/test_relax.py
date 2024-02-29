import os

import pytest
from jobflow import run_locally

from atomate2.aims.jobs.core import RelaxMaker
from atomate2.aims.schemas.task import AimsTaskDoc

cwd = os.getcwd()


def test_base_maker(tmp_path, species_dir, mock_aims, si):
    # mapping from job name to directory containing test files
    ref_paths = {"relax_si": "relax-si"}

    # settings passed to fake_run_aims; adjust these to check for certain input settings
    fake_run_aims_kwargs = {}

    # automatically use fake FHI-aims
    mock_aims(ref_paths, fake_run_aims_kwargs)

    parameters = {
        "k_grid": [2, 2, 2],
        "species_dir": (species_dir / "light").as_posix(),
    }
    # generate job
    maker = RelaxMaker.full_relaxation(user_params=parameters)
    maker.name = "relax_si"
    job = maker.make(si)

    # run the flow or job and ensure that it finished running successfully
    os.chdir(tmp_path)
    responses = run_locally(job, create_folders=True, ensure_success=True)
    os.chdir(cwd)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, AimsTaskDoc)
    assert output1.output.energy == pytest.approx(-15800.2255448846)


def test_relax_fixed_cell_maker(tmp_path, species_dir, mock_aims, si):
    # mapping from job name to directory containing test files
    ref_paths = {"relax_fixed_cell_si": "relax-fixed-cell-si"}

    # settings passed to fake_run_aims; adjust these to check for certain input settings
    fake_run_aims_kwargs = {}

    # automatically use fake FHI-aims
    mock_aims(ref_paths, fake_run_aims_kwargs)

    parameters = {
        "k_grid": [2, 2, 2],
        "species_dir": (species_dir / "light").as_posix(),
    }
    # generate job
    maker = RelaxMaker.fixed_cell_relaxation(user_params=parameters)
    maker.name = "relax_fixed_cell_si"
    structure = si.copy()
    structure.frac_coords[0, 0] += 0.25
    job = maker.make(structure)

    # run the flow or job and ensure that it finished running successfully
    os.chdir(tmp_path)
    responses = run_locally(job, create_folders=True, ensure_success=True)
    os.chdir(cwd)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, AimsTaskDoc)
    assert output1.output.energy == pytest.approx(-15800.099741042)
