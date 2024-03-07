from typing import Callable
from jobflow import job, Maker, Job
from dataclasses import dataclass, field
from pathlib import Path

from atomate2.classical_md.schemas import ClassicalMDTaskDocument
from atomate2.openmm.schemas.openmm_task_document import OpenMMTaskDocument
from atomate2.openmm.schemas.physical_state import PhysicalState
from atomate2.openmm.schemas.task_details import TaskDetails
from atomate2.openmm.schemas.dcd_reports import DCDReports
from atomate2.openmm.schemas.state_reports import StateReports
from atomate2.openmm.schemas.calculation_input import CalculationInput
from atomate2.openmm.schemas.calculation_output import CalculationOutput
from atomate2.openmm.constants import OpenMMConstants

# from atomate2.openmm.logger import logger
from openmm import Platform, Context
from typing import Union, Optional
from openmm.app import DCDReporter, StateDataReporter, PDBReporter
from openmm.app.simulation import Simulation
from tempfile import TemporaryDirectory, NamedTemporaryFile
import copy
import os

from openff.interchange import Interchange

from atomate2.classical_md.common import openff_job


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
    platform_name: Optional[dict] = field(default_factory=dict)
    platform_properties: Optional[dict] = field(default_factory=dict)
    state_interval: Optional[int] = field(default=1000)
    dcd_interval: Optional[int] = field(default=10000)
    wrap_dcd: Optional[bool] = False

    @openff_job
    def make(
        self,
        interchange: Interchange,
        prev_task: Optional[ClassicalMDTaskDocument] = None,
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
        input_set : OpenMMSet
            pymatgen.io.openmm OpenMMSet object instance.
        output_dir : Optional[Union[str, Path]]
            Path to directory for writing state and DCD trajectory files. This could be a temp or
            persistent directory.
        """
        # Define output_dir if as a temporary directory if not provided
        # temp_dir = (
        #     None  # Define potential pointer to temporary directory to keep in scope
        # )
        # if output_dir is None:
        #     temp_dir = TemporaryDirectory()
        #     output_dir = temp_dir.name
        #     output_dir = Path(output_dir)
        # else:
        #     output_dir = Path(output_dir)
        #     output_dir.mkdir(parents=True, exist_ok=True)

        output_dir = Path()
        integrator = None

        sim = self.make_simulation(interchange, integrator)

        self.add_reporters(sim, output_dir)

        # Run the simulation
        self.run_openmm(sim)

        self.update_interchange(interchange, sim)

        return self.create_task_doc(interchange)

        # Close the simulation
        input_set = self._close_base_openmm_task(
            sim, input_set, sim.context, task_details, output_dir
        )

        return input_set

    def add_reporters(self, sim: Simulation, output_dir: Path):

        # add dcd reporter
        if self.dcd_interval > 0:
            dcd_reporter = DCDReporter(
                file=str(output_dir / "trajectory_dcd"),
                reportInterval=self.dcd_interval,
            )
            sim.reporters.append(dcd_reporter)

        # add state reporter
        if self.state_interval > 0:
            state_reporter = StateDataReporter(
                file=str(output_dir / "state_txt"),
                reportInterval=self.state_interval,
                step=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
            )
            sim.reporters.append(state_reporter)

    def make_simulation(self, interchange, integrator):
        platform_name = self.platform_name or "CPU"
        platform = Platform.getPlatformByName(platform_name)
        sim = interchange.to_openmm_simulation(
            integrator,
            platform=platform,
            platformProperties=self.platform_properties,
        )
        return sim

    def _setup_base_openmm_task(
        self, input_set: OpenMMSet, output_dir: Path
    ) -> Simulation:
        """
        Initializes an OpenMM Simulation. Classes derived from BaseOpenMMMaker define the run_openmm method
        and implement the specifics of an OpenMM task.

        Parameters
        ----------
        input_set : OpenMMSet
            OpenMM set for initializing an OpenMM Simulation.
        output_dir : Optional[Union[str, Path]]
            Path to directory for writing state and DCD trajectory files. This could be a temp or
            persistent directory.

        Returns
        -------
        sim : Simulation
            OpenMM Simulation from OpenMMSet.
        """

        # Setup compute platform and get a Simulation
        platform_name = self.platform_kwargs.get("platform")
        platform_name = platform_name if platform_name is not None else "CPU"
        platform_props = self.platform_kwargs.get("platform_properties")
        platform = Platform.getPlatformByName(platform_name)

        sim = input_set.get_simulation(
            platform=platform, platformProperties=platform_props
        )

        # Add reporters
        if self.dcd_interval > 0:
            dcd_file_name = os.path.join(
                output_dir, OpenMMConstants.TRAJECTORY_DCD_FILE_NAME.value
            )
            dcd_reporter = DCDReporter(
                file=dcd_file_name,
                reportInterval=self.dcd_interval,
            )
            # logger.info(f"Created DCDReporter that will report to {dcd_file_name}")
            sim.reporters.append(dcd_reporter)
        if self.state_interval > 0:
            state_file_name = os.path.join(
                output_dir, OpenMMConstants.STATE_REPORT_CSV_FILE_NAME.value
            )
            state_reporter = StateDataReporter(
                file=state_file_name,
                reportInterval=self.state_interval,
                step=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
            )
            # logger.info(f"Created DCDReporter that will report to {state_file_name}")
            sim.reporters.append(state_reporter)

        return sim

    def update_interchange(self, interchange, sim):
        state = sim.context.getState(
            getPositions=True,
            getVelocities=True,
            enforcePeriodicBox=self.wrap_dcd,
        )
        interchange.positions = state.getPositions(asNumpy=True)
        interchange.velocities = state.getVelocities(asNumpy=True)

    def run_openmm(self, simulation: Simulation) -> TaskDetails:
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
            "run_openmm should be implemented by each class that derives from BaseOpenMMMaker."
        )

    def create_task_doc(
        self,
        interchange=None,
        output_dir=None,
        prev_task=None,
    ):
        return

    def _close_base_openmm_task(
        self,
        sim: Simulation,
        input_set: OpenMMSet,
        context: Context,
        task_details: TaskDetails,
        output_dir: Path,
    ):

        # Create an output OpenMMSet for CalculationOutput
        output_set = copy.deepcopy(input_set)
        state = StateInput(
            context.getState(
                getPositions=True,
                getVelocities=True,
                enforcePeriodicBox=True,
            )
        )
        # overwrite input set topology with topology from simulation
        with NamedTemporaryFile(suffix=".pdb") as tmp:
            pdb_reporter = PDBReporter(tmp.name, 1)
            pdb_reporter.report(sim, sim.context.getState(getPositions=True))
            topology = TopologyInput.from_file(tmp.name)

        integrator = IntegratorInput(sim.context.getIntegrator())

        output_set[output_set.state_file] = state
        output_set[output_set.topology_file] = topology
        output_set[output_set.integrator_file] = integrator

        output_set.write_input(output_dir)

        # Grab StateDataReporter and DCDReporter if present on simulation reporters
        state_reports, dcd_reports = None, None
        if self.state_interval > 0:  # could check if file is present instead
            # todo: what happens when state_reporter_interval > 0, but nothing has been
            # reported, for example, Simulation.step was not called. Look at TaskDetails
            # for logic flow?
            state_file_name = os.path.join(output_dir, "state_csv")
            state_reports = StateReports.from_state_file(state_file_name)
        if self.dcd_interval > 0:  # could check if file is present instead
            dcd_file_name = os.path.join(output_dir, "trajectory_dcd")
            dcd_reports = DCDReports(
                location=dcd_file_name, report_interval=self.dcd_interval
            )

        calculation_input = CalculationInput(
            input_set=input_set, physical_state=PhysicalState.from_input_set(input_set)
        )
        calculation_output = CalculationOutput(
            input_set=output_set,
            physical_state=PhysicalState.from_input_set(output_set),
            state_reports=state_reports,
            dcd_reports=dcd_reports,
            task_details=task_details,
        )

        task_doc = OpenMMTaskDocument(
            input_set=output_set,
            physical_state=PhysicalState.from_input_set(output_set),
            calculation_input=calculation_input,
            calculation_output=calculation_output,
        )

        return task_doc
