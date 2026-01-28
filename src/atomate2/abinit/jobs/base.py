"""Definition of base ABINIT job maker."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import jobflow
from jobflow import Maker, Response, job

from atomate2 import SETTINGS
from atomate2.abinit.files import write_abinit_input_set
from atomate2.abinit.run import run_abinit
from atomate2.abinit.schemas.calculation import TaskState
from atomate2.abinit.schemas.outfiles import AbinitStoredFile
from atomate2.abinit.schemas.task import AbinitTaskDoc
from atomate2.abinit.utils.common import UnconvergedError
from atomate2.abinit.utils.history import JobHistory

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from abipy.flowtk.events import AbinitCriticalWarning
    from pymatgen.core.structure import Structure

    from atomate2.abinit.sets.base import AbinitInputGenerator

logger = logging.getLogger(__name__)

__all__ = ["BaseAbinitMaker", "JobSetupVars", "abinit_job", "setup_job"]


class JobSetupVars(NamedTuple):
    """
    Configuration variables for setting up an ABINIT job.

    Attributes
    ----------
    start_time : float
        Unix timestamp when the job started.
    history : JobHistory
        Job history tracking restarts and previous runs.
    workdir : str
        Working directory path for the job.
    abipy_manager : None
        Manager for AbiPy operations. Currently disabled.
    wall_time : int or None
        Maximum wall time in seconds, or None if no limit.
    """

    start_time: float
    history: JobHistory
    workdir: str
    abipy_manager: None
    wall_time: int | None


def setup_job(
    structure: Structure | None,
    prev_outputs: str | Path | list[str] | None,
    restart_from: str | Path | list[str] | None,
    history: JobHistory | None,
    wall_time: int | None,
) -> JobSetupVars:
    """
    Set up an ABINIT job with configuration and logging.

    This function initializes the job environment, including creating the
    job history, setting up the working directory, and configuring logging.

    Parameters
    ----------
    structure : Structure or None
        A pymatgen Structure object. At least one of structure, prev_outputs,
        or restart_from must be provided.
    prev_outputs : str or Path or list[str] or None
        Path(s) to previous calculation directories to use as inputs.
    restart_from : str or Path or list[str] or None
        Path(s) to previous calculation directories to restart from. If
        provided and history is not None, a restart will be logged.
    history : JobHistory or None
        Job history object. If None, a new JobHistory is created.
    wall_time : int or None
        Maximum wall time in seconds for the job.

    Returns
    -------
    JobSetupVars
        Configuration variables for the job including start time, history,
        working directory, and wall time.

    Raises
    ------
    RuntimeError
        If none of structure, prev_outputs, or restart_from are provided.
    """
    start_time = time.time()

    if structure is None and prev_outputs is None and restart_from is None:
        raise RuntimeError(
            "At least one of structure, prev_outputs or restart_from should be defined."
        )

    if history is None:
        # First time the job is created
        history = JobHistory()
    elif restart_from is not None:
        # Log restart only for automatic restarts (unconverged jobs),
        # not for manual restarts with different parameters
        history.log_restart()

    workdir = os.getcwd()
    history.log_start(workdir=workdir, start_time=start_time)

    # Configure logging to atomate2_abinit.log for all ABINIT-related loggers
    log_handler = logging.FileHandler("atomate2_abinit.log")
    log_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logging.getLogger("pymatgen.io.abinit").addHandler(log_handler)
    logging.getLogger("abipy").addHandler(log_handler)
    logging.getLogger("atomate2").addHandler(log_handler)

    # AbiPy manager currently disabled (needed for autoparal, not yet supported)
    abipy_manager = None

    return JobSetupVars(
        start_time=start_time,
        history=history,
        workdir=workdir,
        abipy_manager=abipy_manager,
        wall_time=wall_time,
    )


_DATA_OBJECTS = [  # either str (TaskDoc fields) or MSONable class
    AbinitStoredFile,
]


def abinit_job(method: Callable) -> job:
    """
    Decorate the ``make`` method of ABINIT job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.job` that configures common
    settings for all abinit jobs. For example, it ensures that large data objects
    (band structures, density of states, DDB, etc) are all stored in the
    atomate2 data store. It also configures the output schema to be an Abinit
    :obj:`.TaskDocument`.

    Any makers that return Abinit jobs (not flows) should decorate the ``make`` method
    with @abinit_job. For example:

    .. code-block:: python

        class MyAbinitMaker(BaseAbinitMaker):
            @abinit_job
            def make(structure):
                # code to run abinit job.
                pass

    Parameters
    ----------
    method : callable
        A BaseAbinitMaker.make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate Abinit jobs.
    """
    return job(method, data=_DATA_OBJECTS, output_schema=AbinitTaskDoc)


@dataclass
class BaseAbinitMaker(Maker):
    """
    Base ABINIT job maker.

    This is the base class for all ABINIT job makers. It provides common
    functionality for running ABINIT calculations including input generation,
    execution, output parsing, and automatic restart logic.

    Parameters
    ----------
    input_set_generator : AbinitInputGenerator
        Generator for creating ABINIT input files.
    name : str
        The job name. Default is "base abinit job".
    wall_time : int or None
        Maximum wall time in seconds for the job. If None, no limit is set.
    run_abinit_kwargs : dict
        Additional keyword arguments passed to :obj:`.run_abinit`.
    task_document_kwargs : dict[str, Any]
        Additional keyword arguments passed to :obj:`.TaskDoc.from_directory`.
    stop_jobflow_on_failure : bool
        If True, stop the entire jobflow when this job fails after maximum
        restarts. Default is False.
    """

    input_set_generator: AbinitInputGenerator
    name: str = "base abinit job"
    wall_time: int | None = None
    run_abinit_kwargs: dict[str, Any] = field(default_factory=dict)
    task_document_kwargs: dict[str, Any] = field(default_factory=dict)
    handle_unsuccessful: str = SETTINGS.ABINIT_HANDLE_UNSUCCESSFUL

    # class variables
    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = ()

    def __post_init__(self) -> None:
        """
        Initialize critical events list from class variable.

        This converts the class-level CRITICAL_EVENTS tuple to an instance
        list that can be modified per-instance if needed.
        """
        self.critical_events = list(self.CRITICAL_EVENTS)

    @property
    def calc_type(self) -> str:
        """Get the type of calculation for this maker."""
        return self.input_set_generator.calc_type

    @abinit_job
    def make(
        self,
        structure: Structure | None = None,
        prev_outputs: str | Path | list[str] | None = None,
        restart_from: str | Path | list[str] | None = None,
        history: JobHistory | None = None,
    ) -> jobflow.Job:
        """
        Create an ABINIT job.

        Parameters
        ----------
        structure : Structure or None
            A pymatgen Structure object. At least one of structure,
            prev_outputs, or restart_from must be provided.
        prev_outputs : str or Path or list[str] or None
            Path(s) to previous calculation directories. Used to chain
            calculations where outputs from previous jobs are needed as
            inputs (e.g., using a previous DEN file).
        restart_from : str or Path or list[str] or None
            Path(s) to previous calculation directories to restart from.
            Used when continuing an unconverged calculation with the same
            parameters.
        history : JobHistory or None
            A JobHistory object tracking previous runs and restarts.

        Returns
        -------
        Response
            A jobflow Response containing an AbinitTaskDoc and potential
            restart jobs if the calculation did not converge.
        """
        # Setup job and get general job configuration
        config = setup_job(
            structure=structure,
            prev_outputs=prev_outputs,
            restart_from=restart_from,
            history=history,
            wall_time=self.wall_time,
        )

        # Write abinit input set
        write_abinit_input_set(
            structure=structure,
            input_set_generator=self.input_set_generator,
            prev_outputs=prev_outputs,
            restart_from=restart_from,
            directory=config.workdir,
        )

        # Run abinit
        run_abinit(
            wall_time=config.wall_time,
            start_time=config.start_time,
            **self.run_abinit_kwargs,
        )

        # Parse ABINIT outputs
        task_doc = AbinitTaskDoc.from_directory(
            Path.cwd(),
            additional_fields={"history_dirs": config.history.prev_dirs},
            **self.task_document_kwargs,
        )
        task_doc.task_label = self.name
        if len(task_doc.event_report.filter_types(self.critical_events)) > 0:
            task_doc = task_doc.model_copy(update={"state": TaskState.UNCONVERGED})
            task_doc.calcs_reversed[-1] = task_doc.calcs_reversed[-1].model_copy(
                update={"has_abinit_completed": TaskState.UNCONVERGED}
            )

        return self.get_response(
            task_document=task_doc,
            history=config.history,
            max_restarts=SETTINGS.ABINIT_MAX_RESTARTS,
            prev_outputs=prev_outputs,
        )

    def get_response(
        self,
        task_document: AbinitTaskDoc,
        history: JobHistory,
        max_restarts: int = 5,
        prev_outputs: str | tuple | list | Path | None = None,
    ) -> Response:
        """
        Generate a response with restart logic for unconverged calculations.

        This method examines the task document state and determines whether
        to return the result as-is, create a restart job, or stop the workflow
        if maximum restarts are exceeded.

        Parameters
        ----------
        task_document : AbinitTaskDoc
            The task document from the completed calculation.
        history : JobHistory
            Job history tracking previous runs and restarts.
        max_restarts : int
            Maximum number of restart attempts. Default is 5.
        prev_outputs : str or tuple or list or Path or None
            Path(s) to pass to restart jobs for chaining calculations.

        Returns
        -------
        Response
            A jobflow Response. Contains the task document and either:
            - Nothing else if calculation succeeded
            - A replacement job if restarting
            - stop_children/stop_jobflow flags if max restarts exceeded
                depending on the value of handle_unsuccessful
        """
        if task_document.state == TaskState.SUCCESS:
            return Response(
                output=task_document,
            )

        if history.run_number > max_restarts:
            unconverged_error = UnconvergedError(
                self,
                msg=f"Unconverged after {history.run_number} runs.",
                abinit_input=task_document.input.abinit_input,
                history=history,
            )
            # Max restarts exceeded - action depends on self.handle_unsuccessful
            if self.handle_unsuccessful == "error":
                raise unconverged_error
            if self.handle_unsuccessful == "stop_flow":
                return Response(
                    output=task_document,
                    stop_children=True,
                    stop_jobflow=True,
                    stored_data={"error": unconverged_error},
                )
            if self.handle_unsuccessful == "stop_children":
                return Response(
                    output=task_document,
                    stop_children=True,
                    stop_jobflow=False,
                    stored_data={"error": unconverged_error},
                )
            if self.handle_unsuccessful == "continue":
                return Response(
                    output=task_document,
                    stop_children=False,
                    stop_jobflow=False,
                    stored_data={"error": unconverged_error},
                )
            raise RuntimeError(f"Unknown option for {self.handle_unsuccessful=}")

        logger.info("Getting restart job.")

        new_job = self.make(
            structure=task_document.structure,
            restart_from=task_document.dir_name,
            prev_outputs=prev_outputs,
            history=history,
        )

        return Response(
            output=task_document,
            replace=new_job,
        )
