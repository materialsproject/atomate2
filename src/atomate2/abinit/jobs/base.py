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
from atomate2.abinit.schemas.task import AbinitTaskDoc
from atomate2.abinit.utils.common import UnconvergedError
from atomate2.abinit.utils.history import JobHistory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from abipy.flowtk.events import AbinitCriticalWarning
    from pymatgen.core.structure import Structure

    from atomate2.abinit.sets.base import AbinitInputGenerator

logger = logging.getLogger(__name__)

__all__ = ["BaseAbinitMaker"]


class JobSetupVars(NamedTuple):
    start_time: float
    history: JobHistory
    workdir: str
    abipy_manager: None  # To change in the future
    wall_time: int | None


def setup_job(
    structure: Structure,
    prev_outputs: str | Path | list[str] | None,
    restart_from: str | Path | list[str] | None,
    history: JobHistory,
    wall_time: int | None,
) -> JobSetupVars:
    """Set up job."""
    # Get the start time.
    start_time = time.time()

    if structure is None and prev_outputs is None and restart_from is None:
        raise RuntimeError(
            "At least one of structure, prev_outputs or "
            "restart_from should be defined."
        )

    if history is None:
        # Supposedly the first time the job is created
        history = JobHistory()
    elif restart_from is not None:
        # We want to log the restart only if the restart_from is due to
        # an automatic restart, not a restart from e.g. another scf or relax
        # with different parameters.
        history.log_restart()

    # Set up working directory
    workdir = os.getcwd()

    # Log information about the start of the job
    history.log_start(workdir=workdir, start_time=start_time)

    # Set up logging
    log_handler = logging.FileHandler("atomate2_abinit.log")
    log_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logging.getLogger("pymatgen.io.abinit").addHandler(log_handler)
    logging.getLogger("abipy").addHandler(log_handler)
    logging.getLogger("atomate2").addHandler(log_handler)

    # Load the atomate settings for abinit to get configuration parameters
    # TODO: how to allow for tuned parameters on a per-job basis ?
    #  (similar to fw_spec-passed settings)
    abipy_manager = None  # Currently disabled as it is needed for autoparal,
    # which is not yet supported
    # abipy_manager = get_abipy_manager(SETTINGS)

    # set walltime, if possible
    # TODO: see in set_walltime, where to put this walltime_command
    # wall_time = wall_time
    return JobSetupVars(
        start_time=start_time,
        history=history,
        workdir=workdir,
        abipy_manager=abipy_manager,
        wall_time=wall_time,
    )


@dataclass
class BaseAbinitMaker(Maker):
    """
    Base ABINIT job maker.

    Parameters
    ----------
    input_set_generator : AbinitInputGenerator
        Input generator to be used.
    name : str
        The job name.
    wall_time : int
        The wall time for the job.
    run_abinit_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_abinit`.
    task_document_kwargs : dict[str, Any]
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    """

    input_set_generator: AbinitInputGenerator
    name: str = "base abinit job"
    wall_time: int | None = None
    run_abinit_kwargs: dict[str, Any] = field(default_factory=dict)
    task_document_kwargs: dict[str, Any] = field(default_factory=dict)

    # class variables
    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = ()

    def __post_init__(self) -> None:
        """Process post-init configuration."""
        self.critical_events = list(self.CRITICAL_EVENTS)

    @property
    def calc_type(self) -> str:
        """Get the type of calculation for this maker."""
        return self.input_set_generator.calc_type

    @job
    def make(
        self,
        structure: Structure | None = None,
        prev_outputs: str | Path | list[str] | None = None,
        restart_from: str | Path | list[str] | None = None,
        history: JobHistory | None = None,
    ) -> jobflow.Job:
        """Get an ABINIT jobflow.Job.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_outputs : TODO: add description from sets.base
        restart_from : TODO: add description from sets.base
        history : JobHistory
            A JobHistory object containing the history of this job.
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

        # parse Abinit outputs

        task_doc = AbinitTaskDoc.from_directory(
            Path.cwd(),
            **self.task_document_kwargs,
        )
        task_doc.task_label = self.name
        if len(task_doc.event_report.filter_types(self.critical_events)) > 0:
            task_doc = task_doc.model_copy(update={"state": TaskState.UNCONVERGED})
            task_doc.calcs_reversed[-1] = task_doc.calcs_reversed[-1].model_copy(
                update={"has_abinit_completed": TaskState.UNCONVERGED}
            )  # optional I think

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
        """Get new job to restart abinit calculation."""
        if task_document.state == TaskState.SUCCESS:
            return Response(
                output=task_document,
            )

        if history.run_number > max_restarts:
            # TODO: check here if we should stop jobflow or children or
            #  if we should throw an error.
            unconverged_error = UnconvergedError(
                self,
                msg=f"Unconverged after {history.run_number} runs.",
                abinit_input=task_document.input.abinit_input,
                history=history,
            )
            return Response(
                output=task_document,
                stop_children=True,
                stop_jobflow=False,
                stored_data={"error": unconverged_error},
            )

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
