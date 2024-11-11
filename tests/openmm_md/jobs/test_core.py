from pathlib import Path

import numpy as np
from emmet.core.openmm import OpenMMInterchange, OpenMMTaskDocument
from monty.serialization import loadfn
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

    maker = NVTMaker(n_steps=10, state_interval=1, traj_interval=5)
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


def test_trajectory_reporter_json(interchange, tmp_path, run_job):
    """Test that the trajectory reporter can be serialized to JSON."""
    # Create simulation using NVTMaker
    maker = NVTMaker(
        temperature=300,
        friction_coefficient=1.0,
        step_size=0.002,
        platform_name="CPU",
        traj_interval=1,
        n_steps=3,
        traj_file_type="json",
    )

    job = maker.make(interchange)
    task_doc = run_job(job)

    # Test serialization/deserialization
    json_str = task_doc.model_dump_json()
    new_doc = OpenMMTaskDocument.model_validate_json(json_str)

    # Verify trajectory data survived the round trip
    calc_output = new_doc.calcs_reversed[0].output
    traj_file = Path(calc_output.dir_name) / calc_output.traj_file
    traj = loadfn(traj_file)

    assert len(traj) == 3
    assert traj.coords.max() < traj.lattice.max()
    assert "kinetic_energy" in traj.frame_properties[0]

    # Check that trajectory file was written
    assert (tmp_path / "trajectory.json").exists()
