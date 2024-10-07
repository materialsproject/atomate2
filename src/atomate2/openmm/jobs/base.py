"""The base class for OpenMM simulation makers."""

from __future__ import annotations

import copy
import json
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from emmet.core.openmm import (
    Calculation,
    CalculationInput,
    CalculationOutput,
    OpenMMInterchange,
    OpenMMTaskDocument,
)
from jobflow import Maker, Response, job
from mdareporter.mdareporter import MDAReporter
from monty.json import MontyDecoder, MontyEncoder
from openmm import Integrator, LangevinMiddleIntegrator, Platform, XmlSerializer
from openmm.app import StateDataReporter
from openmm.unit import angstrom, kelvin, picoseconds
from pymatgen.core import Structure

from atomate2.openmm.utils import increment_name, task_reports

if TYPE_CHECKING:
    from collections.abc import Callable

    from openmm.app.simulation import Simulation


try:
    # so we can load OpenMM Interchange created by openmmml
    import openmmml
except ImportError:
    openmmml = None

try:
    from openff.interchange import Interchange
except ImportError:

    class Interchange:  # type: ignore[no-redef]
        """Dummy class for failed imports of Interchange."""

        def model_validate(self, _: str) -> None:
            """Parse raw is the first method called on the Interchange object."""
            raise ImportError(
                "openff-interchange must be installed for OpenMM makers to"
                "to support OpenFF Interchange objects."
            )


OPENMM_MAKER_DEFAULTS = {
    "step_size": 0.001,
    "temperature": 298,
    "friction_coefficient": 1,
    "platform_name": "CPU",
    "platform_properties": {},
    "state_interval": 1000,
    "state_file_name": "state",
    "traj_interval": 10000,
    "wrap_traj": False,
    "report_velocities": False,
    "traj_file_name": "trajectory",
    "traj_file_type": "dcd",
    "embed_traj": False,
    "save_structure": False,
}


def openmm_job(method: Callable) -> job:
    """Decorate the ``make`` method of ClassicalMD job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.Job` that configures common
    settings for all ClassicalMD jobs. Namely, configures the output schema to be a
    :obj:`.ClassicalMDTaskDocument`.

    Any makers that return classical md jobs (not flows) should decorate the ``make``
    method with @openff_job. For example:

    .. code-block:: python

        class MyClassicalMDMaker(BaseOpenMMMaker):
            @openff_job
            def make(structure):
                # code to run OpenMM job.
                pass

    Parameters
    ----------
    method : callable
        A BaseVaspMaker.make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate jobs.
    """
    return job(
        method,
        output_schema=OpenMMTaskDocument,
        data=["interchange", "traj_blob"],
    )


@dataclass
class BaseOpenMMMaker(Maker):
    """Base class for OpenMM simulation makers.

    This class provides a foundation for creating OpenMM simulation
    makers. It includes common attributes and methods for setting up,
    running, and closing OpenMM simulations. Subclasses can override
    the run_openmm method to define specific simulation logic.

    In general, any missing values will be taken from the
    previous task, if possible, and the default values defined in
    atomate2.openmm.OPENMM_MAKER_DEFAULTS, if not.

    Attributes
    ----------
    name : str
        The name of the OpenMM job.
    tags : Optional[List[str]]
        Tags for the OpenMM job.
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
    state_interval : Optional[int]
        The interval for saving simulation state.
        To record no state, set to 0.
    state_file_name : Optional[str]
        The name of the state file to save.
    traj_interval : Optional[int]
        The interval for saving trajectory frames. To record
        no trajectory, set to 0.
    wrap_traj : Optional[bool]
        Whether to wrap trajectory coordinates.
    report_velocities : Optional[bool]
        Whether to report velocities in the trajectory file.
    traj_file_name : Optional[str]
        The name of the trajectory file to save.
    traj_file_type : Optional[str]
        The type of trajectory file to save. Supports any output format
        supported by MDAnalysis.
    embed_traj : Optional[bool]
        Whether to embed the trajectory in the task document.
    save_structure : Optional[bool]
        Whether to save the final structure in the task document.
    """

    name: str = "base openmm job"
    tags: list[str] | None = field(default=None)
    n_steps: int | None = field(default=None)
    step_size: float | None = field(default=None)
    temperature: float | None = field(default=None)
    friction_coefficient: float | None = field(default=None)
    platform_name: str | None = field(default=None)
    platform_properties: dict | None = field(default=None)
    state_interval: int | None = field(default=None)
    state_file_name: str | None = field(default=None)
    traj_interval: int | None = field(default=None)
    wrap_traj: bool | None = field(default=None)
    report_velocities: bool | None = field(default=None)
    traj_file_name: str | None = field(default=None)
    traj_file_type: str | None = field(default=None)
    embed_traj: bool | None = field(default=None)
    save_structure: bool | None = field(default=None)

    @openmm_job
    def make(
        self,
        interchange: Interchange | OpenMMInterchange | str,
        prev_dir: str | None = None,
    ) -> Response:
        """Run an OpenMM calculation.

        This method sets up an OpenMM simulation, runs the simulation based
        on the specific job logic defined in run_openmm, and closes the
        simulation. It returns a response containing the output task document.

        Parameters
        ----------
        interchange : Union[Interchange, OpenMMInterchange, str]
            An Interchange object, OpenMMInterchange object or byte encoded equivalent.
        prev_task : Optional[str]
            The directory of the previous task.

        Returns
        -------
        Response
            A response object containing the output task document.
        """
        if prev_dir:
            with open(Path(prev_dir) / "taskdoc.json") as file:
                task_dict = json.load(file, cls=MontyDecoder)
                prev_task = OpenMMTaskDocument.model_validate(task_dict)
        else:
            prev_task = None

        interchange = self._load_interchange(interchange)

        dir_name = Path.cwd()

        sim = self._create_simulation(interchange, prev_task)

        self._add_reporters(sim, dir_name, prev_task)

        # Run the simulation
        start = time.time()
        self.run_openmm(sim, dir_name)
        elapsed_time = time.time() - start

        self._update_interchange(interchange, sim, prev_task)

        structure = self._create_structure(sim, prev_task)

        task_doc = self._create_task_doc(
            interchange, structure, elapsed_time, dir_name, prev_task
        )

        # leaving the MDAReporter makes the builders fail
        for _ in range(len(sim.reporters)):
            reporter = sim.reporters.pop()
            del reporter
        del sim

        # write out task_doc json to output dir
        with open(dir_name / "taskdoc.json", "w") as file:
            json.dump(task_doc.model_dump(), file, cls=MontyEncoder)

        return Response(output=task_doc)

    def _load_interchange(
        self, interchange: Interchange | OpenMMInterchange | str
    ) -> Interchange | OpenMMInterchange:
        """Load an Interchange object from a JSON string or bytes.

        This method loads an Interchange object from a JSON string or bytes
        representation. It is used to convert interchange data to an Interchange
        object for use in the OpenMM simulation.

        Parameters
        ----------
        interchange : Union[Interchange, str]
            An Interchange object or a JSON string of the Interchange object.

        Returns
        -------
        Interchange
            The loaded Interchange object.
        """
        if isinstance(interchange, str):
            try:
                interchange = Interchange.parse_raw(interchange)
            except:  # noqa: E722
                # parse with openmm instead
                interchange = OpenMMInterchange.model_validate_json(interchange)
        else:
            interchange = copy.deepcopy(interchange)
        return interchange

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
        traj_interval = self._resolve_attr("traj_interval", prev_task)
        traj_file_name = self._resolve_attr("traj_file_name", prev_task)
        traj_file_type = self._resolve_attr("traj_file_type", prev_task)
        report_velocities = self._resolve_attr("report_velocities", prev_task)
        wrap_traj = self._resolve_attr("wrap_traj", prev_task)

        if has_steps & (traj_interval > 0):
            writer_kwargs = {}
            # these are the only file types that support velocities
            if traj_file_type in ["h5md", "nc", "ncdf"]:
                writer_kwargs["velocities"] = report_velocities
                writer_kwargs["forces"] = False
            elif report_velocities and traj_file_type != "trr":
                raise ValueError(
                    f"File type {traj_file_type} does not support velocities as"
                    f"of MDAnalysis 2.7.0. Select another file type"
                    f"or do not attempt to report velocities."
                )

            traj_file = dir_name / f"{traj_file_name}.{traj_file_type}"

            if traj_file.exists() and task_reports(prev_task, "traj"):
                self.traj_file_name = increment_name(traj_file_name)

            # TODO: MDA 2.7.0 has a bug that prevents velocity reporting
            #  this is a stop gap measure before MDA 2.8.0 is released
            kwargs = dict(
                file=str(dir_name / f"{self.traj_file_name}.{traj_file_type}"),
                reportInterval=traj_interval,
                enforcePeriodicBox=wrap_traj,
            )
            if report_velocities:
                # assert package version

                kwargs["writer_kwargs"] = writer_kwargs
                warnings.warn(
                    "Reporting velocities is only supported with the"
                    "development version of MDAnalysis, >= 2.8.0, "
                    "proceed with caution.",
                    stacklevel=1,
                )
            traj_reporter = MDAReporter(**kwargs)

            sim.reporters.append(traj_reporter)

        # add state reporter
        state_interval = self._resolve_attr("state_interval", prev_task)
        state_file_name = self._resolve_attr("state_file_name", prev_task)
        if has_steps & (state_interval > 0):
            state_file = dir_name / f"{state_file_name}.csv"
            if state_file.exists() and task_reports(prev_task, "state"):
                self.state_file_name = increment_name(state_file_name)

            state_reporter = StateDataReporter(
                file=f"{dir_name / self.state_file_name}.csv",
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

    def run_openmm(self, sim: Simulation, dir_name: Path) -> NoReturn:
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
        interchange: Interchange | OpenMMInterchange,
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
        interchange: Interchange | OpenMMInterchange,
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
            enforcePeriodicBox=self._resolve_attr("wrap_traj", prev_task),
        )
        if isinstance(interchange, Interchange):
            interchange.positions = state.getPositions(asNumpy=True)
            interchange.velocities = state.getVelocities(asNumpy=True)
            interchange.box = state.getPeriodicBoxVectors(asNumpy=True)
        elif isinstance(interchange, OpenMMInterchange):
            interchange.state = XmlSerializer.serialize(state)
        else:
            raise TypeError(
                f"Interchange must be an Interchange or "
                f"OpenMMInterchange object, got {type(interchange).__name__}"
            )

    def _create_structure(
        self, sim: Simulation, prev_task: OpenMMTaskDocument | None = None
    ) -> Structure | None:
        """Create a pymatgen Structure from the OpenMM simulation.

        Parameters
        ----------
        sim : Simulation
            The OpenMM simulation object.
        """
        if not self._resolve_attr("save_structure", prev_task):
            return None

        state = sim.context.getState(
            getPositions=True,
            getVelocities=True,
            enforcePeriodicBox=self._resolve_attr("wrap_traj", prev_task),
        )

        return Structure(
            lattice=state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(angstrom),
            species=[atom.element.symbol for atom in sim.topology.atoms()],
            coords=state.getPositions(asNumpy=True).value_in_unit(angstrom),
            coords_are_cartesian=True,
        )

    def _create_task_doc(
        self,
        interchange: Interchange | OpenMMInterchange,
        structure: Structure | None,
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
        structure : Structure
            The final structure of the simulation.
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
        tags = maker_attrs.pop("tags")
        state_file_name = self._resolve_attr("state_file_name", prev_task)
        traj_file_name = self._resolve_attr("traj_file_name", prev_task)
        traj_file_type = self._resolve_attr("traj_file_type", prev_task)
        calc = Calculation(
            dir_name=str(dir_name),
            has_openmm_completed=True,
            input=CalculationInput(**maker_attrs),
            output=CalculationOutput.from_directory(
                dir_name,
                f"{state_file_name}.csv",
                f"{traj_file_name}.{traj_file_type}",
                elapsed_time=elapsed_time,
                embed_traj=self._resolve_attr("embed_traj", prev_task),
            ),
            completed_at=str(datetime.now(tz=timezone.utc)),
            task_name=job_name,
            calc_type=type(self).__name__,
        )

        prev_task = prev_task or OpenMMTaskDocument()

        if isinstance(interchange, Interchange):
            interchange_json = interchange.json()
        else:
            interchange_json = interchange.model_dump_json()

        return OpenMMTaskDocument(
            tags=tags,
            dir_name=str(dir_name),
            state="successful",
            calcs_reversed=[calc],
            interchange=interchange_json,
            mol_specs=prev_task.mol_specs,
            structure=structure,
            force_field=prev_task.force_field,
            task_name=calc.task_name,
            task_type="test",
            last_updated=datetime.now(tz=timezone.utc),
        )
