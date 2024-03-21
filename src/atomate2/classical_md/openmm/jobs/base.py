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
from openmm import Integrator


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
    Base class for OpenMM simulation makers.

    This class provides a foundation for creating OpenMM simulation makers. It includes
    common attributes and methods for setting up, running, and closing OpenMM simulations.
    Subclasses can override the run_openmm method to define specific simulation logic.

    In general, any missing values will be taken from the previous task, if possible, and
    the default values defined in atomate2.classical_md.openmm.OPENMM_MAKER_DEFAULTS,
     if not.

    Attributes:
        name (str): The name of the OpenMM job.
        steps (Optional[int]): The number of simulation steps to run.
        step_size (Optional[float]): The size of each simulation step (picoseconds).
        platform_name (Optional[str]): The name of the OpenMM platform to use, passed to
            Interchange.to_openmm_simulation.
        platform_properties (Optional[dict]): Properties for the OpenMM platform, passed to
            Interchange.to_openmm_simulation.
        state_interval (Optional[int]): The interval for saving simulation state. To record
            no state, set to 0.
        dcd_interval (Optional[int]): The interval for saving DCD frames. To record no DCD,
            set to 0.
        wrap_dcd (Optional[bool]): Whether to wrap DCD coordinates.
        temperature (Optional[float]): The simulation temperature (kelvin).
        friction_coefficient (Optional[float]): The friction coefficient for the integrator
            (inverse picoseconds).
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
    ) -> Response:
        """
        Run an OpenMM calculation.

        This method sets up an OpenMM simulation, runs the simulation based on the specific job
        logic defined in run_openmm, and closes the simulation. It returns a response containing
        the output task document.

        Args:
        interchange (Interchange | str): An Interchange object or a JSON string of
            the Interchange object.
        prev_task (Optional[OpenMMTaskDocument]): The previous task document.
        output_dir (Optional[str | Path]): The directory to save the output files.
            Resolution order is output_dir > prev_task.dir_name > Path.cwd(). Will
            create a directory if needed.

        Returns:
            Response: A response object containing the output task document.
        """
        # this is needed because interchange is currently using pydantic.v1
        if not isinstance(interchange, Interchange):
            interchange = Interchange.parse_raw(interchange)
        else:
            interchange = copy.deepcopy(interchange)

        dir_name = Path(
            output_dir or getattr(prev_task, "dir_name", None) or Path.cwd()
        )
        dir_name.mkdir(exist_ok=True, parents=True)

        sim = self._create_simulation(interchange, prev_task)

        self._add_reporters(sim, dir_name, prev_task)

        # Run the simulation
        start = time.time()
        self.run_openmm(sim)
        elapsed_time = time.time() - start

        self._update_interchange(interchange, sim, prev_task)

        del sim

        task_doc = self._create_task_doc(interchange, elapsed_time, dir_name, prev_task)

        # write out task_doc json to output dir
        with open(dir_name / "taskdoc_json", "w") as file:
            file.write(task_doc.json())

        return Response(output=task_doc)

    def _add_reporters(
        self,
        sim: Simulation,
        dir_name: Path,
        prev_task: Optional[OpenMMTaskDocument] = None,
    ):
        """
        Adds reporters to the OpenMM simulation.

        This method adds DCD and state reporters to the OpenMM simulation based on the specified
        intervals and settings.

        Args:
            sim (Simulation): The OpenMM simulation object.
            dir_name (Path): The directory to save the reporter output files.
            prev_task (Optional[OpenMMTaskDocument]): The previous task document.

        """

        has_steps = self._resolve_attr("steps", prev_task) > 0
        # add dcd reporter
        dcd_interval = self._resolve_attr("dcd_interval", prev_task)
        if has_steps & (dcd_interval > 0):
            dcd_reporter = DCDReporter(
                file=str(dir_name / "trajectory_dcd"),
                reportInterval=dcd_interval,
                enforcePeriodicBox=self._resolve_attr("wrap_dcd", prev_task),
            )
            sim.reporters.append(dcd_reporter)

        # add state reporter
        state_interval = self._resolve_attr("state_interval", prev_task)
        if has_steps & (state_interval > 0):
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
                append=(dir_name / "state_csv").exists(),
            )
            sim.reporters.append(state_reporter)

    def run_openmm(self, simulation: Simulation):
        """
        Abstract method for running the OpenMM simulation.

        This method should be implemented by subclasses to define the specific simulation logic.
        It takes an OpenMM simulation object and evolves the simulation.

        Args:
            simulation (Simulation): The OpenMM simulation object.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError(
            "`run_openmm` should be implemented by each child class."
        )

    def _resolve_attr(
        self,
        attr: str,
        prev_task: Optional[OpenMMTaskDocument] = None,
        add_defaults: Optional[dict] = None,
    ):
        """
        Resolves an attribute and set its value.

        This method retrieves the value of an attribute from the current maker, previous task
        input, or a default value (in that order of priority). It sets the attribute on the
        current maker and returns the resolved value.

        Default values are defined in `OPENMM_MAKER_DEFAULTS`.

        Args:
            attr (str): The name of the attribute to resolve.
            prev_task (Optional[OpenMMTaskDocument]): The previous task document.
            add_defaults (Optional[dict]): Additional default values to use,
                overrides `OPENMM_MAKER_DEFAULTS`.

        Returns:
            The resolved attribute value.
        """
        prev_task = prev_task or OpenMMTaskDocument()

        # retrieve previous CalculationInput through multiple Optional fields
        if prev_task.calcs_reversed:
            prev_input = prev_task.calcs_reversed[0].input
        else:
            prev_input = None

        defaults = {**OPENMM_MAKER_DEFAULTS, **(add_defaults or {})}

        if getattr(self, attr, None) is not None:
            attr_value = getattr(self, attr)
        elif getattr(prev_input, attr, None) is not None:
            attr_value = getattr(prev_input, attr)
        else:
            attr_value = defaults.get(attr, None)

        setattr(self, attr, attr_value)
        return getattr(self, attr)

    def _create_integrator(
        self,
        prev_task: Optional[OpenMMTaskDocument] = None,
    ) -> Integrator:
        """
        Creates an OpenMM integrator.

        This method creates a Langevin middle integrator based on the resolved temperature,
        friction coefficient, and step size.

        Args:
            prev_task (Optional[OpenMMTaskDocument]): The previous task document.

        Returns:
            LangevinMiddleIntegrator: The created OpenMM integrator.
        """
        return LangevinMiddleIntegrator(
            self._resolve_attr("temperature", prev_task) * kelvin,
            self._resolve_attr("friction_coefficient", prev_task) / picoseconds,
            self._resolve_attr("step_size", prev_task) * picoseconds,
        )

    def _create_simulation(
        self,
        interchange: Interchange,
        prev_task: Optional[OpenMMTaskDocument] = None,
    ):
        """
        Creates an OpenMM simulation.

        This method creates an OpenMM simulation using the provided Interchange object,
        the get_integrator method, and the platform and platform_properties attributes.

        Args:
            interchange (Interchange): The Interchange object containing the MD data.
            prev_task (Optional[OpenMMTaskDocument]): The previous task document.

        Returns:
            Simulation: The created OpenMM simulation object.
        """
        integrator = self._create_integrator(prev_task)
        platform = Platform.getPlatformByName(
            self._resolve_attr("platform_name", prev_task)
        )
        platform_properties = self._resolve_attr("platform_properties", prev_task)

        sim = interchange.to_openmm_simulation(
            integrator,
            platform=platform,
            platformProperties=platform_properties,
        )
        return sim

    def _update_interchange(self, interchange, sim, prev_task):
        """
        Updates the Interchange object with the current simulation state.

        This method updates the positions, velocities, and box vectors of the
        Interchange object based on the current state of the OpenMM simulation.

        Args:
        interchange (Interchange): The Interchange object to update.
        sim (Simulation): The OpenMM simulation object.
        prev_task (OpenMMTaskDocument): The previous task document.

        """
        state = sim.context.getState(
            getPositions=True,
            getVelocities=True,
            enforcePeriodicBox=self._resolve_attr("wrap_dcd", prev_task),
        )
        interchange.positions = state.getPositions(asNumpy=True)
        interchange.velocities = state.getVelocities(asNumpy=True)
        interchange.box = state.getPeriodicBoxVectors(asNumpy=True)

    def _create_task_doc(
        self,
        interchange: Interchange,
        elapsed_time: Optional[float] = None,
        dir_name: Optional[Path] = None,
        prev_task: Optional[OpenMMTaskDocument] = None,
    ) -> OpenMMTaskDocument:
        """
        Creates a task document for the OpenMM job.

        This method creates an OpenMMTaskDocument based on the current maker attributes, previous
        task document, and simulation results.

        Args:
            interchange (Interchange): The updated Interchange object.
            elapsed_time (Optional[float]): The elapsed time of the simulation. Default is None.
            dir_name (Optional[Path]): The directory where the output files are saved.
            Default is None.
            prev_task (Optional[OpenMMTaskDocument]): The previous task document. Default is None.

        Returns:
            OpenMMTaskDocument: The created task document.
        """

        maker_attrs = copy.deepcopy(vars(self))
        job_name = maker_attrs.pop("name")

        calc = Calculation(
            dir_name=str(dir_name),
            has_openmm_completed=True,
            input=CalculationInput(**maker_attrs),
            output=CalculationOutput.from_directory(
                dir_name,
                elapsed_time,
                self._resolve_attr("steps", prev_task),
                self._resolve_attr("state_interval", prev_task),
            ),
            completed_at=str(datetime.now()),
            task_name=job_name,
            calc_type=self.__class__.__name__,
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
