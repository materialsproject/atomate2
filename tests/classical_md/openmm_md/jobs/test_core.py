from atomate2.classical_md.openmm.jobs.core import (
    EnergyMinimizationMaker,
    NPTMaker,
    NVTMaker,
    TempChangeMaker,
)
from jobflow import run_locally

import numpy as np


def test_energy_minimization_maker(interchange, temp_dir, run_job):

    maker = EnergyMinimizationMaker(max_iterations=1)
    base_job = maker.make(interchange, output_dir=temp_dir)
    task_doc = run_job(base_job)

    assert np.any(task_doc.interchange["positions"] != interchange.positions)


def test_npt_maker(interchange, temp_dir, run_job):

    maker = NPTMaker(steps=10, pressure=0.1, pressure_update_frequency=1)
    base_job = maker.make(interchange, output_dir=temp_dir)
    task_doc = run_job(base_job)

    # test that coordinates and box size has changed
    assert np.any(task_doc.interchange["positions"] != interchange.positions)
    assert np.any(task_doc.interchange["box"] != interchange.box)


def test_nvt_maker(interchange, temp_dir, run_job):

    maker = NVTMaker(steps=10)
    base_job = maker.make(interchange, output_dir=temp_dir)
    task_doc = run_job(base_job)

    # test that coordinates have changed
    assert np.any(task_doc.interchange["positions"] != interchange.positions)


def test_temp_change_maker(interchange, temp_dir, run_job):

    maker = TempChangeMaker(steps=10, temperature=310, temp_steps=10)
    base_job = maker.make(interchange, output_dir=temp_dir)
    task_doc = run_job(base_job)

    # test that coordinates have changed and starting temperature is present and correct
    assert np.any(task_doc.interchange["positions"] != interchange.positions)
    assert task_doc.calcs_reversed[0].input.temperature == 310
    assert task_doc.calcs_reversed[0].input.starting_temperature == 298
