from atomate2.classical_md.openmm.jobs.core import (
    EnergyMinimizationMaker,
    NPTMaker,
    NVTMaker,
    TempChangeMaker,
)
from jobflow import run_locally

import numpy as np


def test_energy_minimization_maker(interchange, tmp_path):

    temp_dir = tmp_path / "test_output"
    temp_dir.mkdir()

    maker = EnergyMinimizationMaker(max_iterations=1)

    base_job = maker.make(interchange, output_dir=temp_dir)
    response_dict = run_locally(base_job, ensure_success=True)
    task_doc = response_dict[base_job.uuid][1].output

    assert np.any(task_doc.interchange["positions"] != interchange.positions)


def test_npt_maker(interchange, tmp_path):
    temp_dir = tmp_path / "test_output"
    temp_dir.mkdir()

    maker = NPTMaker(steps=10, pressure=0.1, pressure_update_frequency=1)

    base_job = maker.make(interchange, output_dir=temp_dir)
    response_dict = run_locally(base_job, ensure_success=True)
    task_doc = response_dict[base_job.uuid][1].output

    # test that coordinates and box size has changed
    assert np.any(task_doc.interchange["positions"] != interchange.positions)
    assert np.any(task_doc.interchange["box"] != interchange.box)


def test_nvt_maker(interchange, tmp_path):
    temp_dir = tmp_path / "test_output"
    temp_dir.mkdir()

    maker = NVTMaker(steps=10)

    base_job = maker.make(interchange, output_dir=temp_dir)
    response_dict = run_locally(base_job, ensure_success=True)
    task_doc = response_dict[base_job.uuid][1].output

    # test that coordinates have changed
    assert np.any(task_doc.interchange["positions"] != interchange.positions)


def test_temp_change_maker(interchange, tmp_path):
    temp_dir = tmp_path / "test_output"
    temp_dir.mkdir()

    maker = TempChangeMaker(steps=10, temperature=310, temp_steps=10)

    base_job = maker.make(interchange, output_dir=temp_dir)
    response_dict = run_locally(base_job, ensure_success=True)
    task_doc = response_dict[base_job.uuid][1].output

    # test that coordinates have changed and starting temperature is present and correct
    assert np.any(task_doc.interchange["positions"] != interchange.positions)
    assert task_doc.calcs_reversed[0].input.temperature == 310
    assert task_doc.calcs_reversed[0].input.starting_temperature == 298
