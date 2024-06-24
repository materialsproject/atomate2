"""Merge DDB jobs for merging DDB files from ABINIT calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import jobflow
from jobflow import Maker, Response, job

from atomate2 import SETTINGS
from atomate2.abinit.files import write_mrgddb_input_set
from atomate2.abinit.jobs.base import setup_job
from atomate2.abinit.run import run_mrgddb
from atomate2.abinit.schemas.calculation import TaskState
from atomate2.abinit.schemas.mrgddb import MrgddbTaskDoc
from atomate2.abinit.sets.mrgddb import MrgddbInputGenerator, MrgddbSetGenerator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from abipy.flowtk.events import AbinitCriticalWarning

    from atomate2.abinit.utils.history import JobHistory

logger = logging.getLogger(__name__)

__all__ = [
    "MrgddbMaker",
]


@dataclass
class MrgddbMaker(Maker):
    """Maker to create a job with a merge of DDB files from ABINIT.

    Parameters
    ----------
    name : str
        The job name.
    """

    # VT need to remove the following because of the @property below
    # _calc_type: str = "mrgddb_merge"
    # would have been okay in a child class with @dataclass
    name: str = "Merge DDB"
    input_set_generator: MrgddbInputGenerator = field(
        default_factory=MrgddbSetGenerator
    )
    # input_set_generator: MrgddbInputGenerator = MrgddbSetGenerator()
    wall_time: int | None = None

    # TODO: is there a critical events for this?
    # CRITICAL_EVENTS: ClassVar[Sequence[str]] = ("NscfConvergenceWarning",)
    # class variables
    # CRITICAL_EVENTS: ClassVar[Sequence[str]] = ("ScfConvergenceWarning",)
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
        prev_outputs: list[str] | None = None,
        history: JobHistory | None = None,
    ) -> jobflow.Flow | jobflow.Job:
        """
        Return a MRGDDB jobflow.Job.

        Parameters
        ----------
        prev_outputs : TODO: add description from sets.base
        history : JobHistory
            A JobHistory object containing the history of this job.
        """
        # Flatten the list of previous outputs dir
        prev_outputs = [item for sublist in prev_outputs for item in sublist]

        # Setup job and get general job configuration
        config = setup_job(
            structure=None,
            prev_outputs=prev_outputs,
            restart_from=None,
            history=history,
            wall_time=self.wall_time,
        )

        # Write mrgddb input set
        write_mrgddb_input_set(
            input_set_generator=self.input_set_generator,
            prev_outputs=prev_outputs,
            directory=config.workdir,
        )

        # Run mrgddb
        run_mrgddb(
            wall_time=config.wall_time,
            start_time=config.start_time,
        )

        # parse Mrgddb DDB output
        task_doc = MrgddbTaskDoc.from_directory(
            Path.cwd(),
            # **self.task_document_kwargs,
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
        task_document: MrgddbTaskDoc,
        history: JobHistory,
        max_restarts: int = 5,
        prev_outputs: str | tuple | list | Path | None = None,
    ) -> Response:
        """Get new job to restart mrgddb calculation."""
        if task_document.state == TaskState.SUCCESS:
            return Response(
                output=task_document,
            )

        # if history.run_number > max_restarts:
        #    # TODO: check here if we should stop jobflow or children or
        #    #  if we should throw an error.
        #    unconverged_error = UnconvergedError(
        #        self,
        #        msg=f"Unconverged after {history.run_number} runs.",
        #        mrgddb_input=task_document.mrgddb_input,
        #        history=history,
        #    )
        #    return Response(
        #        output=task_document,
        #        stop_children=True,
        #        stop_jobflow=True,
        #        stored_data={"error": unconverged_error},
        #    )

        logger.info("Getting restart job.")

        new_job = self.make(
            structure=task_document.structure,
            prev_outputs=prev_outputs,
            history=history,
        )

        return Response(
            output=task_document,
            replace=new_job,
        )
