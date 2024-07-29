import copy

import numpy as np
import pytest
from emmet.core.classical_md import ClassicalMDTaskDocument
from emmet.core.classical_md.openmm import Calculation, CalculationInput
from jobflow import Flow, Job
from mdareporter import MDAReporter
from openmm.app import Simulation, StateDataReporter
from openmm.openmm import LangevinMiddleIntegrator
from openmm.unit import kelvin, picoseconds

from atomate2.classical_md.core import generate_interchange
from atomate2.classical_md.openmm.jobs.base import BaseOpenMMMaker


def test_add_reporters(interchange, temp_dir):
    maker = BaseOpenMMMaker(
        traj_interval=100, state_interval=50, wrap_traj=True, n_steps=1
    )
    sim = maker._create_simulation(interchange)  # noqa: SLF001
    dir_name = temp_dir / "test_output"
    dir_name.mkdir()

    maker._add_reporters(sim, dir_name)  # noqa: SLF001

    assert len(sim.reporters) == 2
    assert isinstance(sim.reporters[0], MDAReporter)
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

    assert maker._resolve_attr("temperature") == 301  # noqa: SLF001
    assert maker._resolve_attr("friction_coefficient") == 2  # noqa: SLF001
    assert maker._resolve_attr("step_size", prev_task) == 0.002  # noqa: SLF001
    assert maker._resolve_attr("platform_name") == "CPU"  # noqa: SLF001


def test_create_integrator():
    maker = BaseOpenMMMaker(temperature=300, friction_coefficient=2, step_size=0.002)
    integrator = maker._create_integrator()  # noqa: SLF001

    assert isinstance(integrator, LangevinMiddleIntegrator)
    assert integrator.getTemperature() == 300 * kelvin
    assert integrator.getFriction() == 2 / picoseconds
    assert integrator.getStepSize() == 0.002 * picoseconds


def test_create_simulation(interchange):
    maker = BaseOpenMMMaker()

    sim = maker._create_simulation(interchange)  # noqa: SLF001

    assert isinstance(sim, Simulation)
    assert isinstance(sim.integrator, LangevinMiddleIntegrator)
    assert sim.context.getPlatform().getName() == "CPU"


def test_update_interchange(interchange):
    interchange = copy.deepcopy(interchange)
    maker = BaseOpenMMMaker(wrap_traj=True)
    sim = maker._create_simulation(interchange)  # noqa: SLF001
    start_positions = interchange.positions
    start_velocities = interchange.velocities
    start_box = interchange.box

    # Run the simulation for one step
    sim.step(1)

    maker._update_interchange(interchange, sim, None)  # noqa: SLF001

    assert interchange.positions.shape == start_positions.shape
    assert interchange.velocities.shape == (1170, 3)

    assert np.any(interchange.positions != start_positions)
    assert np.any(interchange.velocities != start_velocities)
    assert np.all(interchange.box == start_box)


def test_create_task_doc(interchange, temp_dir):
    maker = BaseOpenMMMaker(n_steps=1000, temperature=300)
    dir_name = temp_dir / "test_output"
    dir_name.mkdir()

    task_doc = maker._create_task_doc(  # noqa: SLF001
        interchange,
        elapsed_time=10.5,
        dir_name=dir_name,
    )

    assert isinstance(task_doc, ClassicalMDTaskDocument)
    assert task_doc.dir_name == str(dir_name)
    assert task_doc.state == "successful"
    assert len(task_doc.calcs_reversed) == 1
    assert task_doc.calcs_reversed[0].input.n_steps == 1000
    assert task_doc.calcs_reversed[0].input.temperature == 300
    assert task_doc.calcs_reversed[0].output.elapsed_time == 10.5


def test_make(interchange, temp_dir, run_job):
    # Create an instance of BaseOpenMMMaker
    maker = BaseOpenMMMaker(
        n_steps=1000,
        step_size=0.002,
        platform_name="CPU",
        state_interval=100,
        traj_interval=50,
        temperature=300,
        friction_coefficient=1,
    )

    # monkey patch to allow running the test without openmm

    def do_nothing(self, sim):
        pass

    BaseOpenMMMaker.run_openmm = do_nothing

    # Call the make method
    base_job = maker.make(interchange)
    assert isinstance(base_job, Job)

    task_doc = run_job(base_job)

    # Assert the specific values in the task document
    assert isinstance(task_doc, ClassicalMDTaskDocument)
    assert task_doc.state == "successful"
    # assert task_doc.dir_name == str(temp_dir)
    assert len(task_doc.calcs_reversed) == 1

    # Assert the calculation details
    calc = task_doc.calcs_reversed[0]
    # assert calc.dir_name == str(temp_dir)
    assert calc.has_openmm_completed is True
    assert calc.input.n_steps == 1000
    assert calc.input.step_size == 0.002
    assert calc.input.platform_name == "CPU"
    assert calc.input.state_interval == 100
    assert calc.input.traj_interval == 50
    assert calc.input.temperature == 300
    assert calc.input.friction_coefficient == 1
    assert calc.output is not None
    assert calc.completed_at is not None
    assert calc.task_name == "base openmm job"
    assert calc.calc_type == "BaseOpenMMMaker"


def test_make_w_velocities(interchange, run_job):
    # monkey patch to allow running the test without openmm
    def do_nothing(self, sim):
        pass

    BaseOpenMMMaker.run_openmm = do_nothing

    maker1 = BaseOpenMMMaker(
        n_steps=1000,
        report_velocities=True,
    )

    with pytest.raises(RuntimeError):
        run_job(maker1.make(interchange))
        # run_job(base_job)

    maker2 = BaseOpenMMMaker(
        n_steps=1000,
        report_velocities=True,
        traj_file_type="h5md",
    )

    base_job = maker2.make(interchange)
    task_doc = run_job(base_job)

    # Assert the calculation details
    calc = task_doc.calcs_reversed[0]
    assert calc.input.report_velocities is True


def test_make_from_prev(run_job):
    mol_specs_dicts = [
        {"smiles": "CCO", "count": 50, "name": "ethanol", "charge_method": "mmff94"},
        {"smiles": "O", "count": 300, "name": "water", "charge_method": "mmff94"},
    ]
    inter_job = generate_interchange(mol_specs_dicts, 1)

    # Create an instance of BaseOpenMMMaker
    maker = BaseOpenMMMaker(n_steps=10)

    # monkey patch to allow running the test without openmm
    def do_nothing(self, sim):
        pass

    BaseOpenMMMaker.run_openmm = do_nothing

    # Call the make method
    base_job = maker.make(inter_job.output.interchange, prev_task=inter_job.output)

    run_job(Flow([inter_job, base_job]))