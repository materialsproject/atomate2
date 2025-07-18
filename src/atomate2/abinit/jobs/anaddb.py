"""Merge DDB jobs for merging DDB files from ABINIT calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import jobflow
from jobflow import Maker, Response, job

from atomate2.abinit.files import write_anaddb_input_set
from atomate2.abinit.jobs.base import setup_job
from atomate2.abinit.run import run_anaddb
from atomate2.abinit.schemas.anaddb import AnaddbTaskDoc
from atomate2.abinit.sets.anaddb import (
    AnaddbDfptDteInputGenerator,
    AnaddbInputGenerator,
)

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

    from atomate2.abinit.utils.history import JobHistory

logger = logging.getLogger(__name__)

__all__ = ["AnaddbDfptDteMaker", "AnaddbMaker"]


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

    @job(output_schema=AnaddbTaskDoc)
    def make(
        self,
        structure: Structure,
        prev_outputs: str | list[str] | None = None,
        history: JobHistory | None = None,
    ) -> jobflow.Response:
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

        return Response(output=task_doc)


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
