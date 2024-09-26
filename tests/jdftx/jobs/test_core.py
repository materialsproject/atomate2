import pytest
from unittest.mock import patch
from jobflow import run_locally

from atomate2.jdftx.jobs.core import SinglePointMaker, IonicMinMaker, LatticeMinMaker
from atomate2.jdftx.sets.core import SinglePointSetGenerator, IonicMinSetGenerator, LatticeMinSetGenerator
from atomate2.jdftx.jobs.base import BaseJdftxMaker
from atomate2.jdftx.schemas.task import TaskDoc

@pytest.mark.parametrize("mock_cwd", ["sp_test"], indirect=True)
def test_sp_maker(mock_jdftx, si_structure, mock_cwd, mock_filenames, clean_dir):

    ref_paths = {"single_point": "sp_test"}

    fake_run_jdftx_kwargs = {}

    mock_jdftx(ref_paths, fake_run_jdftx_kwargs)

    maker = SinglePointMaker(input_set_generator=SinglePointSetGenerator())

    job = maker.make(si_structure)

    responses = run_locally(job, create_folders=True, ensure_success=True)
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDoc)


@pytest.mark.parametrize("mock_cwd", ["ionicmin_test"], indirect=True)
def test_ionicmin_maker(mock_jdftx, si_structure, mock_cwd, mock_filenames, clean_dir):

    ref_paths = {"ionic_min": "ionicmin_test"}

    fake_run_jdftx_kwargs = {}

    mock_jdftx(ref_paths, fake_run_jdftx_kwargs)

    maker = IonicMinMaker(input_set_generator=IonicMinSetGenerator())

    job = maker.make(si_structure)

    responses = run_locally(job, create_folders=True, ensure_success=True)
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDoc)

@pytest.mark.parametrize("mock_cwd", ["latticemin_test"], indirect=True)
def test_latticemin_maker(mock_jdftx, si_structure, mock_cwd, mock_filenames, clean_dir):

    ref_paths = {"lattice_min": "latticemin_test"}

    fake_run_jdftx_kwargs = {}

    mock_jdftx(ref_paths, fake_run_jdftx_kwargs)

    maker = LatticeMinMaker(input_set_generator=LatticeMinSetGenerator())

    job = maker.make(si_structure)

    responses = run_locally(job, create_folders=True, ensure_success=True)
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDoc)