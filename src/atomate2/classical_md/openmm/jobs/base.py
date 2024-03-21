import copy
import time
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

from pathlib import Path
from jobflow import Maker, Response, Job

from atomate2.classical_md.core import openff_job

from atomate2.classical_md.openmm.schemas.tasks import (
    Calculation,
    CalculationInput,
    CalculationOutput,
    OpenMMTaskDocument,
)

from openff.interchange import Interchange

from openmm import Platform, LangevinMiddleIntegrator
from openmm.unit import kelvin, picoseconds
from openmm.app import DCDReporter, StateDataReporter
from openmm.app.simulation import Simulation


OPENMM_MAKER_DEFAULTS = {
    "step_size": 0.001,
    "platform_name": "CPU",
    "platform_properties": {},
    "state_interval": 1000,
    "dcd_interval": 10000,
    "wrap_dcd": False,
    "temperature": 298,
    "friction_coefficient": 1,
}


@dataclass
class BaseOpenMMMaker(Maker):
    """
    platform_kwargs : Optional[dict]
        platform and platform_properties passed to OpenMMSet.get_simulation.
        If no arguments provided defaults to CPU.
    dcd_reporter_interval : Optional[int]
        DCD reporter interval. If DCD reporter is not desired, set keyword argument to 0.
    state_reporter_interval : Optional[int]
        State reporter interval. If state reporter is not desired, set keyword argument to 0.
    wrap_dcd : bool
        Set to True to turn on periodic boundary conditions for the trajectory file or False
        to turn off periodic boundary conditions.
    """

    name: str = "base openmm job"
    steps: Optional[int] = field(default=None)
    step_size: Optional[float] = field(default=None)
    platform_name: Optional[str] = field(default=None)
    platform_properties: Optional[dict] = field(default=None)
    state_interval: Optional[int] = field(default=None)
    dcd_interval: Optional[int] = field(default=None)
    wrap_dcd: Optional[bool] = field(default=None)
    temperature: Optional[float] = field(default=None)
    friction_coefficient: Optional[float] = field(default=None)

    @openff_job
    def make(
        self,
        interchange: Interchange | str,
        prev_task: Optional[OpenMMTaskDocument] = None,
        output_dir: Optional[str | Path] = None,
    ) -> Job:
        """
        OpenMM Job Maker where each Job consist of three major steps:

        1. Setup an OpenMM Simulation - _setup_base_openmm_task(...)
        2. Run an OpenMM Simulation based task - run_openmm(...)
        3. Close the OpenMM task - _close_base_openmm_task(...)

        The setup and closing logic should be broadly applicable to all OpenMM Jobs. The specifics
        of each OpenMM Job is to be defined by classes that are derived from BaseOpenMMMaker by
        implementing the run_openmm(...) method. If custom setup and closing log is desired, this
        is achieved by BaseOpenMMMaker dervied classes overriding _setup_base_openmm_task(...) and
        _close_base_openmm_task(...) methods.

        Parameters
        ----------
        interchange : Interchange
            An Interchange object containing the molecular mechanics data.
        prev_task : Optional[OpenMMTaskDocument]
            The previous task document. If not provided, a new task document will be created.
        output_dir : Optional[str | Path]
        """
        # this is needed because interchange is currently using pydantic.v1
        if not isinstance(interchange, Interchange):
            interchange = Interchange.parse_raw(interchange)
        else:
            interchange = copy.deepcopy(interchange)

        # TODO: Define output_dir if as a temporary directory if not provided?
        dir_name = Path(output_dir)

        sim = self.create_simulation(interchange, prev_task)

        self.add_reporters(sim, dir_name, prev_task)

        # Run the simulation
        start = time.time()
        self.run_openmm(sim)
        elapsed_time = time.time() - start

        self.update_interchange(interchange, sim, prev_task)

        del sim

        # could consider writing out simulation details to directory

        task_doc = self.create_task_doc(interchange, elapsed_time, dir_name, prev_task)

        return Response(output=task_doc)

    def add_reporters(
        self,
        sim: Simulation,
        dir_name: Path,
        prev_task: Optional[OpenMMTaskDocument] = None,
    ):

        # add dcd reporter
        dcd_interval = self.resolve_attr("dcd_interval", prev_task)
        if dcd_interval > 0:
            dcd_reporter = DCDReporter(
                file=str(dir_name / "trajectory_dcd"),
                reportInterval=dcd_interval,
                enforcePeriodicBox=self.resolve_attr("wrap_dcd", prev_task),
            )
            sim.reporters.append(dcd_reporter)

        # add state reporter
        state_interval = self.resolve_attr("state_interval", prev_task)
        if state_interval > 0:
            state_reporter = StateDataReporter(
                file=str(dir_name / "state_csv"),
                reportInterval=state_interval,
                step=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
            )
            sim.reporters.append(state_reporter)

    def run_openmm(self, simulation: Simulation):
        """
        Abstract method for holding the task specific logic to be ran by each job. Setting up of the Simulation
        is handled by _setup_base_openmm_task which passes a Simulation object to this method. This method will
        run an OpenMM task and return a TaskDetails object which is then passed to _close_base_openmm_task that
        handles closing the simulation.

        Parameters
        ----------
        simulation : Simulation
            OpenMM Simulation constructed from an OpenMMSet and possibly state and trajectory reporters added.

        Returns
        -------
        TaskDetails

        """
        raise NotImplementedError(
            "`run_openmm` should be implemented by each child class."
        )

    def resolve_attr(self, attr: str, prev_task: Optional[OpenMMTaskDocument] = None):
        prev_task = prev_task or OpenMMTaskDocument()

        # retrieve previous CalculationInput through multiple Optional fields
        if prev_task.calcs_reversed:
            prev_input = prev_task.calcs_reversed[0].input
        else:
            prev_input = None

        if getattr(self, attr, None) is not None:
            attr_value = getattr(self, attr)
        elif getattr(prev_input, attr, None) is not None:
            attr_value = getattr(prev_input, attr)
        else:
            attr_value = OPENMM_MAKER_DEFAULTS.get(attr, None)

        setattr(self, attr, attr_value)
        return getattr(self, attr)

    def create_integrator(
        self,
        prev_task: Optional[OpenMMTaskDocument] = None,
    ):
        return LangevinMiddleIntegrator(
            self.resolve_attr("temperature", prev_task) * kelvin,
            self.resolve_attr("friction_coefficient", prev_task) / picoseconds,
            self.resolve_attr("step_size", prev_task) * picoseconds,
        )

    def create_simulation(
        self,
        interchange: Interchange,
        prev_task: Optional[OpenMMTaskDocument] = None,
    ):
        # get integrator from string?
        # openmm_integrator = getattr(openmm, self.integrator)

        integrator = self.create_integrator(prev_task)
        platform = Platform.getPlatformByName(
            self.resolve_attr("platform_name", prev_task)
        )
        platform_properties = self.resolve_attr("platform_properties", prev_task)

        sim = interchange.to_openmm_simulation(
            integrator,
            platform=platform,
            platformProperties=platform_properties,
        )
        return sim

    def update_interchange(self, interchange, sim, prev_task):
        state = sim.context.getState(
            getPositions=True,
            getVelocities=True,
            enforcePeriodicBox=self.resolve_attr("wrap_dcd", prev_task),
        )
        interchange.positions = state.getPositions(asNumpy=True)
        interchange.velocities = state.getVelocities(asNumpy=True)
        interchange.box = state.getPeriodicBoxVectors(asNumpy=True)

    def create_task_doc(
        self,
        interchange: Interchange,
        elapsed_time: Optional[float] = None,
        dir_name: Optional[Path] = None,
        prev_task: Optional[OpenMMTaskDocument] = None,
    ) -> OpenMMTaskDocument:

        maker_attrs = copy.deepcopy(vars(self))
        job_name = maker_attrs.pop("name")

        prev_calcs = getattr(prev_task, "calcs_reversed", None) or []
        n_prev_steps = sum(calc.input.steps for calc in prev_calcs)

        calc = Calculation(
            dir_name=str(dir_name),
            has_openmm_completed=True,
            input=CalculationInput(**maker_attrs),
            output=CalculationOutput.from_directory(
                dir_name, elapsed_time, n_prev_steps
            ),
            completed_at=str(datetime.now()),
            task_name=job_name,
            calc_type=self.__class__.__name__,  # TODO: will this return the right name?
        )

        prev_task = prev_task or OpenMMTaskDocument()

        interchange_json = interchange.json()

        return OpenMMTaskDocument(
            tags=None,  # TODO: where do tags come from?
            dir_name=str(dir_name),
            state="successful",
            calcs_reversed=[calc] + (prev_task.calcs_reversed or []),
            interchange=interchange_json,
            molecule_specs=prev_task.molecule_specs,
            forcefield=prev_task.forcefield,
            task_name=calc.task_name,
            task_type="test",
            last_updated=datetime.now(),
        )
