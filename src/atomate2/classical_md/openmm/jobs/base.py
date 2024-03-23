"""The base class for OpenMM simulation makers."""

from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from jobflow import Maker, Response
from mdtraj.reporters.hdf5reporter import HDF5Reporter
from openff.interchange import Interchange
from openmm import Integrator, LangevinMiddleIntegrator, Platform
from openmm.app import DCDReporter, StateDataReporter
from openmm.unit import kelvin, picoseconds

from atomate2.classical_md.core import openff_job
from atomate2.classical_md.openmm.schemas.tasks import (
    Calculation,
    CalculationInput,
    CalculationOutput,
    OpenMMTaskDocument,
)

if TYPE_CHECKING:
    from openmm.app.simulation import Simulation

OPENMM_MAKER_DEFAULTS = {
    "step_size": 0.001,
    "temperature": 298,
    "friction_coefficient": 1,
    "platform_name": "CPU",
    "platform_properties": {},
    "state_file_name": "state",
    "state_interval": 1000,
    "trajectory_file_name": "trajectory",
    "trajectory_interval": 10000,
    "wrap_trajectory": False,
    "trajectory_file_type": "dcd",
    "report_velocities": False,
}


@dataclass
class BaseOpenMMMaker(Maker):
    """Base class for OpenMM simulation makers.

    This class provides a foundation for creating OpenMM simulation
    makers. It includes common attributes and methods for setting up,
    running, and closing OpenMM simulations. Subclasses can override
    the run_openmm method to define specific simulation logic.

    In general, any missing values will be taken from the
    previous task, if possible, and the default values defined in
    atomate2.classical_md.openmm.OPENMM_MAKER_DEFAULTS, if not.

    Attributes
    ----------
    name : str
        The name of the OpenMM job.
    n_steps : Optional[int]
        The number of simulation steps to run.
    step_size : Optional[float]
        The size of each simulation step (picoseconds).
    temperature : Optional[float]
        The simulation temperature (kelvin).
    friction_coefficient : Optional[float]
        The friction coefficient for the
        integrator (inverse picoseconds).
    platform_name : Optional[str]
        The name of the OpenMM platform to use, passed to
        Interchange.to_openmm_simulation.
    platform_properties : Optional[dict]
        Properties for the OpenMM platform,
        passed to Interchange.to_openmm_simulation.
    state_file_name : Optional[str]
        The name of the state file to save.
    state_interval : Optional[int]
        The interval for saving simulation state.
        To record no state, set to 0.
    trajectory_file_name : Optional[str]
        The name of the trajectory file to save.
    trajectory_interval : Optional[int]
        The interval for saving trajectory frames. To record
        no trajectory, set to 0.
    wrap_trajectory : Optional[bool]
        Whether to wrap trajectory coordinates.
    trajectory_file_type : Optional[str]
        The type of trajectory file to save. Options are "dcd" and "hdf5".
    report_velocities : Optional[bool]
        Whether to report velocities in the trajectory file.
    """

    name: str = "base openmm job"
    n_steps: int | None = field(default=None)
    step_size: float | None = field(default=None)
    temperature: float | None = field(default=None)
    friction_coefficient: float | None = field(default=None)
    platform_name: str | None = field(default=None)
    platform_properties: dict | None = field(default=None)
    state_file_name: str | None = field(default=None)
    state_interval: int | None = field(default=None)
    trajectory_file_name: str | None = field(default=None)
    trajectory_interval: int | None = field(default=None)
    wrap_trajectory: bool | None = field(default=None)
    trajectory_file_type: str | None = field(default=None)
    report_velocities: bool | None = field(default=None)

    @openff_job
    def make(
        self,
        interchange: Interchange | str,
        prev_task: OpenMMTaskDocument | None = None,
        output_dir: str | Path | None = None,
    ) -> Response:
        """Run an OpenMM calculation.

        This method sets up an OpenMM simulation, runs the simulation based
        on the specific job logic defined in run_openmm, and closes the
        simulation. It returns a response containing the output task document.

        Parameters
        ----------
        interchange : Union[Interchange, str]
            An Interchange object or a JSON string of the Interchange object.
        prev_task : Optional[OpenMMTaskDocument]
            The previous task document.
        output_dir : Optional[Union[str, Path]]
            The directory to save the output files.
            Resolution order is output_dir > prev_task.dir_name > Path.cwd(). Will
            create a directory if needed.

        Returns
        -------
        Response
            A response object containing the output task document.
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
        prev_task: OpenMMTaskDocument | None = None,
    ) -> None:
        """Add reporters to the OpenMM simulation.

        This method adds DCD and state reporters to the OpenMM
        simulation based on the specified intervals and settings.

        Parameters
        ----------
        sim : Simulation
            The OpenMM simulation object.
        dir_name : Path
            The directory to save the reporter output files.
        prev_task : Optional[OpenMMTaskDocument]
            The previous task document.
        """
        has_steps = self._resolve_attr("n_steps", prev_task) > 0
        # add trajectory reporter
        traj_interval = self._resolve_attr("trajectory_interval", prev_task)
        traj_file_name = self._resolve_attr("trajectory_file_name", prev_task)
        traj_file_type = self._resolve_attr("trajectory_file_type", prev_task)
        report_velocities = self._resolve_attr("report_velocities", prev_task)
        if has_steps & (traj_interval > 0):
            traj_file = dir_name / f"{traj_file_name}_{traj_file_type}"
            if traj_file_type == "dcd":
                if report_velocities:
                    raise ValueError(
                        "DCDReporter does not support reporting velocities. If you'd"
                        "like to report velocities, switch to HDF5 format."
                    )
                dcd_reporter = DCDReporter(
                    file=str(traj_file),
                    reportInterval=traj_interval,
                    enforcePeriodicBox=self._resolve_attr("wrap_trajectory", prev_task),
                )
                sim.reporters.append(dcd_reporter)
            elif traj_file_type == "hdf5":
                # sadly, our hdf5 implementation cannot append
                if traj_file.exists():
                    # logic to increment count on file name
                    re_match = re.search(rf"(\d*)_{traj_file_type}", traj_file.name)
                    position = re_match.start(1)
                    new_count = int(re_match.group(1) or 1) + 1
                    new_name = f"{traj_file_name[:position]}{new_count}"
                    self.trajectory_file_name = new_name
                    traj_file = dir_name / f"{new_name}_{traj_file_type}"

                hdf5_reporter = HDF5Reporter(
                    file=str(traj_file),
                    reportInterval=traj_interval,
                    cell=True,
                    velocities=self._resolve_attr("report_velocities", prev_task),
                    enforcePeriodicBox=self._resolve_attr("wrap_trajectory", prev_task),
                )
                sim.reporters.append(hdf5_reporter)
            else:
                raise ValueError(f"Unsupported trajectory file type: {traj_file_type}")

        # add state reporter
        state_interval = self._resolve_attr("state_interval", prev_task)
        state_file_name = self._resolve_attr("state_file_name", prev_task)
        if has_steps & (state_interval > 0):
            state_reporter = StateDataReporter(
                file=f"{dir_name / state_file_name}_csv",
                reportInterval=state_interval,
                step=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                append=(dir_name / (state_file_name + "_csv")).exists(),
            )
            sim.reporters.append(state_reporter)

    def run_openmm(self, simulation: Simulation) -> NoReturn:
        """Abstract method for running the OpenMM simulation.

        This method should be implemented by subclasses to
        define the specific simulation logic. It takes an
        OpenMM simulation object and evolves the simulation.

        Parameters
        ----------
        simulation : Simulation
            The OpenMM simulation object.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError(
            "`run_openmm` should be implemented by each child class."
        )

    def _resolve_attr(
        self,
        attr: str,
        prev_task: OpenMMTaskDocument | None = None,
        add_defaults: dict | None = None,
    ) -> Any:
        """Resolve an attribute and set its value.

        This method retrieves the value of an attribute from the current maker,
        previous task input, or a default value (in that order of priority). It
        sets the attribute on the current maker and returns the resolved value.

        Default values are defined in `OPENMM_MAKER_DEFAULTS`.

        Parameters
        ----------
        attr : str
            The name of the attribute to resolve.
        prev_task : Optional[OpenMMTaskDocument]
            The previous task document.
        add_defaults : Optional[dict]
            Additional default values to use,
            overrides `OPENMM_MAKER_DEFAULTS`.

        Returns
        -------
        Any
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
            attr_value = defaults.get(attr)

        setattr(self, attr, attr_value)
        return getattr(self, attr)

    def _create_integrator(
        self,
        prev_task: OpenMMTaskDocument | None = None,
    ) -> Integrator:
        """Create an OpenMM integrator.

        This method creates a Langevin middle integrator based on the
        resolved temperature, friction coefficient, and step size.

        Parameters
        ----------
        prev_task : Optional[OpenMMTaskDocument]
            The previous task document.

        Returns
        -------
        LangevinMiddleIntegrator
            The created OpenMM integrator.
        """
        return LangevinMiddleIntegrator(
            self._resolve_attr("temperature", prev_task) * kelvin,
            self._resolve_attr("friction_coefficient", prev_task) / picoseconds,
            self._resolve_attr("step_size", prev_task) * picoseconds,
        )

    def _create_simulation(
        self,
        interchange: Interchange,
        prev_task: OpenMMTaskDocument | None = None,
    ) -> Simulation:
        """Create an OpenMM simulation.

        This method creates an OpenMM simulation using the provided Interchange object,
        the get_integrator method, and the platform and platform_properties attributes.

        Parameters
        ----------
        interchange : Interchange
            The Interchange object containing the MD data.
        prev_task : Optional[OpenMMTaskDocument]
            The previous task document.

        Returns
        -------
        Simulation
            The created OpenMM simulation object.
        """
        integrator = self._create_integrator(prev_task)
        platform = Platform.getPlatformByName(
            self._resolve_attr("platform_name", prev_task)
        )
        platform_properties = self._resolve_attr("platform_properties", prev_task)

        return interchange.to_openmm_simulation(
            integrator,
            platform=platform,
            platformProperties=platform_properties,
        )

    def _update_interchange(
        self,
        interchange: Interchange,
        sim: Simulation,
        prev_task: OpenMMTaskDocument | None = None,
    ) -> None:
        """Update the Interchange object with the current simulation state.

        This method updates the positions, velocities, and box vectors of the
        Interchange object based on the current state of the OpenMM simulation.

        Parameters
        ----------
        interchange : Interchange
            The Interchange object to update.
        sim : Simulation
            The OpenMM simulation object.
        prev_task : Optional[OpenMMTaskDocument]
            The previous task document.
        """
        state = sim.context.getState(
            getPositions=True,
            getVelocities=True,
            enforcePeriodicBox=self._resolve_attr("wrap_trajectory", prev_task),
        )
        interchange.positions = state.getPositions(asNumpy=True)
        interchange.velocities = state.getVelocities(asNumpy=True)
        interchange.box = state.getPeriodicBoxVectors(asNumpy=True)

    def _create_task_doc(
        self,
        interchange: Interchange,
        elapsed_time: float | None = None,
        dir_name: Path | None = None,
        prev_task: OpenMMTaskDocument | None = None,
    ) -> OpenMMTaskDocument:
        """Create a task document for the OpenMM job.

        This method creates an OpenMMTaskDocument based on the current
        maker attributes, previous task document, and simulation results.

        Parameters
        ----------
        interchange : Interchange
            The updated Interchange object.
        elapsed_time : Optional[float]
            The elapsed time of the simulation. Default is None.
        dir_name : Optional[Path]
            The directory where the output files are saved.
            Default is None.
        prev_task : Optional[OpenMMTaskDocument]
            The previous task document. Default is None.

        Returns
        -------
        OpenMMTaskDocument
            The created task document.
        """
        maker_attrs = copy.deepcopy(dict(vars(self)))
        job_name = maker_attrs.pop("name")
        state_file_name = self._resolve_attr("state_file_name", prev_task)
        traj_file_name = self._resolve_attr("trajectory_file_name", prev_task)
        traj_file_type = self._resolve_attr("trajectory_file_type", prev_task)
        calc = Calculation(
            dir_name=str(dir_name),
            has_openmm_completed=True,
            input=CalculationInput(**maker_attrs),
            output=CalculationOutput.from_directory(
                dir_name,
                f"{state_file_name}_csv",
                f"{traj_file_name}_{traj_file_type}",
                elapsed_time,
                self._resolve_attr("n_steps", prev_task),
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
