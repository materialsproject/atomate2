import pytest
from unittest.mock import patch
from jobflow import run_locally

from atomate2.jdftx.jobs.core import SinglePointMaker
from atomate2.jdftx.sets.core import SinglePoint_SetGenerator
from atomate2.jdftx.jobs.base import BaseJdftxMaker
from atomate2.jdftx.schemas.task import TaskDoc


def test_sp_maker(mock_jdftx, si_structure, mock_cwd, mock_filenames):

    ref_paths = {"single_point": "sp_test"}

    fake_run_jdftx_kwargs = {}

    mock_jdftx(ref_paths, fake_run_jdftx_kwargs)

    maker = SinglePointMaker(input_set_generator=SinglePoint_SetGenerator())

    job = maker.make(si_structure)


    responses = run_locally(job, create_folders=True, ensure_success=True)
    # output1 = responses[job.uuid][1].output
    # assert isinstance(output1, TaskDoc)



