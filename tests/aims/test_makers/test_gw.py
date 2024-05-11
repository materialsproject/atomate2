"""A test for GW job maker."""

import pytest
from jobflow import run_locally
from pymatgen.io.aims.sets.bs import GWSetGenerator

from atomate2.aims.jobs.core import GWMaker
from atomate2.aims.schemas.task import AimsTaskDoc


def test_gw_maker_molecule(tmp_dir, species_dir, mock_aims, o2):
    # mapping from job name to directory containing test files
    ref_paths = {"gw_o2": "gw-o2"}

    # settings passed to fake_run_aims; adjust these to check for certain input settings
    fake_run_aims_kwargs = {}

    # automatically use fake FHI-aims
    mock_aims(ref_paths, fake_run_aims_kwargs)

    parameters = {
        "k_grid": [2, 2, 2],
        "species_dir": (species_dir / "light").as_posix(),
    }
    # generate job
    maker = GWMaker(input_set_generator=GWSetGenerator(user_params=parameters))
    maker.name = "gw_o2"
    job = maker.make(o2)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validation the outputs of the job (maybe add gw energy levels as well)
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, AimsTaskDoc)
    assert output1.output.energy == pytest.approx(-4092.0702534)
