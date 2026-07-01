"""MRGDDB jobs for merging derivative database files from ABINIT DFPT calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import jobflow
import numpy as np
from jobflow import Maker, Response, job

from atomate2.abinit.files import write_mrgddb_input_set
from atomate2.abinit.jobs.base import setup_job
from atomate2.abinit.run import run_mrgddb
from atomate2.abinit.schemas.mrgddb import MrgddbTaskDoc
from atomate2.abinit.schemas.outfiles import AbinitStoredFile
from atomate2.abinit.sets.mrgddb import MrgddbInputGenerator

if TYPE_CHECKING:
    from collections.abc import Callable

    from atomate2.abinit.utils.history import JobHistory

logger = logging.getLogger(__name__)

__all__ = [
    "MrgddbMaker",
    "mrgddb_job",
]

_MRGDDB_DATA_OBJECTS = [
    AbinitStoredFile,
]


def mrgddb_job(method: Callable) -> job:
    """
    Decorate the ``make`` method of MRGDDB job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.job` that configures
    common settings for all MRGDDB jobs. It ensures that large data objects
    (merged DDB files) are stored in the atomate2 data store and configures
    the output schema to be a :obj:`.MrgddbTaskDoc`.

    Any makers that return MRGDDB jobs (not flows) should decorate the ``make``
    method with @mrgddb_job. For example:

    .. code-block:: python

        class MyMrgddbMaker(MrgddbMaker):
            @mrgddb_job
            def make(prev_outputs):
                # code to run mrgddb job.
                pass

    Parameters
    ----------
    method : callable
        A MrgddbMaker.make method. This should not be specified directly and
        is implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate MRGDDB jobs.
    """
    return job(method, data=_MRGDDB_DATA_OBJECTS, output_schema=MrgddbTaskDoc)


@dataclass
class MrgddbMaker(Maker):
    """
    Maker to create jobs for merging DDB files using MRGDDB.

    MRGDDB (MeRGe Derivative DataBase) is an ABINIT utility that merges
    multiple derivative database (DDB) files from different DFPT calculations
    into a single DDB file. This is typically needed before running ANADDB
    analysis.

    Parameters
    ----------
    name : str
        The job name. Default is "Merge DDB".
    input_set_generator : MrgddbInputGenerator
        Generator for MRGDDB input files. Defaults to MrgddbInputGenerator.
    """

    name: str = "Merge DDB"
    input_set_generator: MrgddbInputGenerator = field(
        default_factory=MrgddbInputGenerator
    )

    @property
    def calc_type(self) -> str:
        """Get the type of calculation for this maker."""
        return self.input_set_generator.calc_type

    @mrgddb_job
    def make(
        self,
        prev_outputs: list[str] | None = None,
        history: JobHistory | None = None,
    ) -> jobflow.Response:
        """
        Create a MRGDDB job to merge DDB files.

        Parameters
        ----------
        prev_outputs : list[str] or None
            List of paths to previous calculation directories containing DDB
            files to merge. Can be a nested list, which will be flattened.
            Must be provided to merge DDB files.
        history : JobHistory or None
            A JobHistory object containing the history of previous jobs in
            the workflow.

        Returns
        -------
        Response
            A jobflow Response containing a MrgddbTaskDoc with the merged
            DDB file information.
        """
        # Flatten nested list of previous output directories
        prev_outputs = list(np.hstack(prev_outputs))

        # Setup job and get general job configuration
        config = setup_job(
            structure=None,
            prev_outputs=prev_outputs,
            restart_from=None,
            history=history,
            wall_time=None,
        )

        # Write mrgddb input set
        write_mrgddb_input_set(
            input_set_generator=self.input_set_generator,
            prev_outputs=prev_outputs,
            directory=config.workdir,
        )

        # Run mrgddb
        run_mrgddb(
            start_time=config.start_time,
        )

        # Parse MRGDDB output
        task_doc = MrgddbTaskDoc.from_directory(Path.cwd())
        task_doc.task_label = self.name

        return Response(
            output=task_doc,
        )
