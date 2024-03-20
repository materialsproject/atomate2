import copy

import numpy as np
from jobflow import Job, run_locally

from atomate2.classical_md.openmm.jobs.base import BaseOpenMMMaker
from openmm.app import DCDReporter, StateDataReporter, Simulation
from openmm.openmm import LangevinMiddleIntegrator
from openmm.unit import kelvin, picoseconds

from atomate2.classical_md.schemas import ClassicalMDTaskDocument
from atomate2.classical_md.openmm.schemas.tasks import (
    Calculation,
    CalculationInput,
)


def test_add_reporters(interchange, temp_dir):
    maker = BaseOpenMMMaker(dcd_interval=100, state_interval=50, wrap_dcd=True)
    sim = maker.create_simulation(interchange)
    dir_name = temp_dir / "test_output"
    dir_name.mkdir()

    maker.add_reporters(sim, dir_name)

    assert len(sim.reporters) == 2
    assert isinstance(sim.reporters[0], DCDReporter)
    next_dcd = sim.reporters[0].describeNextReport(sim)
    assert next_dcd[0] == 100  # steps until next report
    assert next_dcd[5] is True  # enforce periodic boundaries
    assert isinstance(sim.reporters[1], StateDataReporter)
    next_state = sim.reporters[1].describeNextReport(sim)
    assert next_state[0] == 50  # steps until next report


def test_resolve_attr():
    maker = BaseOpenMMMaker(temperature=301, friction_coefficient=2)
    prev_task = ClassicalMDTaskDocument(
        calcs_reversed=[Calculation(input=CalculationInput(step_size=0.002))]
    )

    assert maker.resolve_attr("temperature") == 301
    assert maker.resolve_attr("friction_coefficient") == 2
    assert maker.resolve_attr("step_size", prev_task) == 0.002
    assert maker.resolve_attr("platform_name") == "CPU"


def test_create_integrator():
    maker = BaseOpenMMMaker(temperature=300, friction_coefficient=2, step_size=0.002)
    integrator = maker.create_integrator()

    assert isinstance(integrator, LangevinMiddleIntegrator)
    assert integrator.getTemperature() == 300 * kelvin
    assert integrator.getFriction() == 2 / picoseconds
    assert integrator.getStepSize() == 0.002 * picoseconds


def test_create_simulation(interchange):
    maker = BaseOpenMMMaker()

    sim = maker.create_simulation(interchange)

    assert isinstance(sim, Simulation)
    assert isinstance(sim.integrator, LangevinMiddleIntegrator)
    assert sim.context.getPlatform().getName() == "CPU"


def test_update_interchange(interchange):
    interchange = copy.deepcopy(interchange)
    maker = BaseOpenMMMaker(wrap_dcd=True)
    sim = maker.create_simulation(interchange)
    start_positions = interchange.positions
    start_velocities = interchange.velocities
    start_box = interchange.box

    # Run the simulation for a few steps
    sim.step(1)

    maker.update_interchange(interchange, sim, None)

    assert interchange.positions.shape == start_positions.shape
    assert interchange.velocities.shape == (1170, 3)

    assert np.any(interchange.positions != start_positions)
    assert np.any(interchange.velocities != start_velocities)
    assert np.all(interchange.box == start_box)


def test_create_task_doc(interchange, temp_dir):
    maker = BaseOpenMMMaker(steps=1000, temperature=300)
    dir_name = temp_dir / "test_output"
    dir_name.mkdir()

    task_doc = maker.create_task_doc(interchange, elapsed_time=10.5, dir_name=dir_name)

    assert isinstance(task_doc, ClassicalMDTaskDocument)
    assert task_doc.dir_name == str(dir_name)
    assert task_doc.state == "successful"
    assert len(task_doc.calcs_reversed) == 1
    assert task_doc.calcs_reversed[0].input.steps == 1000
    assert task_doc.calcs_reversed[0].input.temperature == 300
    assert task_doc.calcs_reversed[0].output.elapsed_time == 10.5
    assert task_doc.interchange == interchange


def test_make(interchange, temp_dir, run_job):

    # Create an instance of BaseOpenMMMaker
    maker = BaseOpenMMMaker(
        steps=1000,
        step_size=0.002,
        platform_name="CPU",
        state_interval=100,
        dcd_interval=50,
        temperature=300,
        friction_coefficient=1,
    )

    # monkey patch to allow running the test without openmm
    BaseOpenMMMaker.run_openmm = lambda self, sim: None

    # Call the make method
    base_job = maker.make(interchange, output_dir=temp_dir)
    assert isinstance(base_job, Job)

    task_doc = run_job(base_job)

    # Assert the specific values in the task document
    assert isinstance(task_doc, ClassicalMDTaskDocument)
    assert task_doc.state == "successful"
    assert task_doc.dir_name == str(temp_dir)
    assert task_doc.interchange == interchange
    assert len(task_doc.calcs_reversed) == 1

    # Assert the calculation details
    calc = task_doc.calcs_reversed[0]
    assert calc.dir_name == str(temp_dir)
    assert calc.has_openmm_completed is True
    assert calc.input.steps == 1000
    assert calc.input.step_size == 0.002
    assert calc.input.platform_name == "CPU"
    assert calc.input.state_interval == 100
    assert calc.input.dcd_interval == 50
    assert calc.input.temperature == 300
    assert calc.input.friction_coefficient == 1
    assert calc.output is not None
    assert calc.completed_at is not None
    assert calc.task_name == "base openmm job"
    assert calc.calc_type == "BaseOpenMMMaker"
