from pathlib import Path

import numpy as np
from emmet.core.openmm import OpenMMInterchange
from openmm import XmlSerializer

from atomate2.openmm.jobs import (
    EnergyMinimizationMaker,
    NPTMaker,
    NVTMaker,
    TempChangeMaker,
)


def test_energy_minimization_maker(interchange, run_job):
    state = XmlSerializer.deserialize(interchange.state)
    start_positions = state.getPositions(asNumpy=True)

    maker = EnergyMinimizationMaker(max_iterations=1)
    base_job = maker.make(interchange)
    task_doc = run_job(base_job)

    new_interchange = OpenMMInterchange.model_validate_json(task_doc.interchange)
    new_state = XmlSerializer.deserialize(new_interchange.state)
    new_positions = new_state.getPositions(asNumpy=True)

    assert not np.all(new_positions == start_positions)
    assert (Path(task_doc.calcs_reversed[0].output.dir_name) / "state.csv").exists()


def test_npt_maker(interchange, run_job):
    state = XmlSerializer.deserialize(interchange.state)
    start_positions = state.getPositions(asNumpy=True)
    start_box = state.getPeriodicBoxVectors()

    maker = NPTMaker(n_steps=10, pressure=0.1, pressure_update_frequency=1)
    base_job = maker.make(interchange)
    task_doc = run_job(base_job)

    new_interchange = OpenMMInterchange.model_validate_json(task_doc.interchange)
    new_state = XmlSerializer.deserialize(new_interchange.state)
    new_positions = new_state.getPositions(asNumpy=True)
    new_box = new_state.getPeriodicBoxVectors()

    # test that coordinates and box size has changed
    assert not np.all(new_positions == start_positions)
    assert not np.all(new_box == start_box)


def test_nvt_maker(interchange, run_job):
    state = XmlSerializer.deserialize(interchange.state)
    start_positions = state.getPositions(asNumpy=True)

    maker = NVTMaker(n_steps=10, state_interval=1)
    base_job = maker.make(interchange)
    task_doc = run_job(base_job)

    new_interchange = OpenMMInterchange.model_validate_json(task_doc.interchange)
    new_state = XmlSerializer.deserialize(new_interchange.state)
    new_positions = new_state.getPositions(asNumpy=True)

    # test that coordinates have changed
    assert not np.all(new_positions == start_positions)

    # Test length of state attributes in calculation output
    calc_output = task_doc.calcs_reversed[0].output
    assert len(calc_output.steps_reported) == 10

    # Test that the state interval is respected
    assert calc_output.steps_reported == list(range(1, 11))


def test_temp_change_maker(interchange, run_job):
    state = XmlSerializer.deserialize(interchange.state)
    start_positions = state.getPositions(asNumpy=True)

    maker = TempChangeMaker(n_steps=10, temperature=310, temp_steps=10)
    base_job = maker.make(interchange)
    task_doc = run_job(base_job)

    new_interchange = OpenMMInterchange.model_validate_json(task_doc.interchange)
    new_state = XmlSerializer.deserialize(new_interchange.state)
    new_positions = new_state.getPositions(asNumpy=True)

    # test that coordinates have changed
    assert not np.all(new_positions == start_positions)

    # test that temperature was updated correctly in the input
    assert task_doc.calcs_reversed[0].input.temperature == 310
    assert task_doc.calcs_reversed[0].input.starting_temperature == 298
