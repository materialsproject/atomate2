"""Merge DDB jobs for merging DDB files from ABINIT calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import jobflow
import numpy as np
from jobflow import Maker, Response, job

from atomate2 import SETTINGS
from atomate2.abinit.files import write_anaddb_input_set
from atomate2.abinit.jobs.base import setup_job
from atomate2.abinit.run import run_anaddb
from atomate2.abinit.schemas.anaddb import AnaddbTaskDoc
from atomate2.abinit.schemas.calculation import TaskState
from atomate2.abinit.sets.anaddb import (
    AnaddbDfptDteInputGenerator,
    AnaddbInputGenerator,
)

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

    from atomate2.abinit.utils.history import JobHistory

logger = logging.getLogger(__name__)

__all__ = ["AnaddbMaker", "AnaddbDfptDteMaker"]


@dataclass
class AnaddbMaker(Maker):
    """Maker to create a job to analyze a DDB file with the utility anaddb.

    Parameters
    ----------
    name : str
        The job name.
    """

    name: str = "Anaddb"
    input_set_generator: AnaddbInputGenerator = field(
        default_factory=AnaddbInputGenerator
    )
    wall_time: int | None = None

    @property
    def calc_type(self) -> str:
        """Get the type of calculation for this maker."""
        return self.input_set_generator.calc_type

    @job
    def make(
        self,
        structure: Structure,
        prev_outputs: str | list[str] | None = None,
        history: JobHistory | None = None,
    ) -> jobflow.Flow | jobflow.Job:
        """
        Return an AnaDDB jobflow.Job.

        Parameters
        ----------
        prev_outputs : TODO: add description from sets.base
        history : JobHistory
            A JobHistory object containing the history of this job.
        """

        # Setup job and get general job configuration
        config = setup_job(
            structure=None,
            prev_outputs=prev_outputs,
            restart_from=None,
            history=history,
            wall_time=self.wall_time,
        )

        # Write anaddb input set
        write_anaddb_input_set(
            structure=structure,
            input_set_generator=self.input_set_generator,
            prev_outputs=prev_outputs,
            directory=config.workdir,
        )

        # Run anaddb
        run_anaddb(
            wall_time=config.wall_time,
            start_time=config.start_time,
        )

        # parse Anaddb DDB output
        task_doc = AnaddbTaskDoc.from_directory(
            Path.cwd(),
            # **self.task_document_kwargs,
        )
        task_doc.task_label = self.name

        return self.get_response(
            task_document=task_doc,
            history=config.history,
            max_restarts=SETTINGS.ABINIT_MAX_RESTARTS,
            prev_outputs=prev_outputs,
        )

    def get_response(
        self,
        task_document: AnaddbTaskDoc,
        history: JobHistory,
        max_restarts: int = 5,
        prev_outputs: str | tuple | list | Path | None = None,
    ) -> Response:
        """Get new job to restart anaddb calculation."""
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
        #        anaddb_input=task_document.anaddb_input,
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


@dataclass
class AnaddbDfptDteMaker(AnaddbMaker):
    """Maker to get info from DFPT calculations (with DTE) from a merged DDB file.

    Parameters
    ----------
    name : str
        The job name.
    """

    name: str = "Anaddb"
    input_set_generator: AnaddbInputGenerator = field(
        default_factory=AnaddbDfptDteInputGenerator
    )
