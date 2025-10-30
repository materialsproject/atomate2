"""MRGDV jobs for merging first-order potential files from ABINIT DFPT calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import jobflow
import numpy as np
from jobflow import Maker, Response, job

from atomate2.abinit.files import write_mrgdv_input_set
from atomate2.abinit.jobs.base import setup_job
from atomate2.abinit.run import run_mrgdv
from atomate2.abinit.schemas.mrgdvdb import MrgdvdbTaskDoc
from atomate2.abinit.schemas.outfiles import AbinitStoredFile
from atomate2.abinit.sets.mrgdvdb import MrgdvInputGenerator

if TYPE_CHECKING:
    from collections.abc import Callable

    from atomate2.abinit.utils.history import JobHistory

logger = logging.getLogger(__name__)

__all__ = [
    "MrgdvMaker",
    "mrgdv_job",
]

_MRGDV_DATA_OBJECTS = [
    AbinitStoredFile,
]


def mrgdv_job(method: Callable) -> job:
    """
    Decorate the ``make`` method of MRGDV job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.job` that configures
    common settings for all MRGDV jobs. It ensures that large data objects
    (merged POT files) are stored in the atomate2 data store and configures
    the output schema to be a :obj:`.MrgdvdbTaskDoc`.

    Any makers that return MRGDV jobs (not flows) should decorate the ``make``
    method with @mrgdv_job. For example:

    .. code-block:: python

        class MyMrgdvMaker(MrgdvMaker):
            @mrgdv_job
            def make(prev_outputs):
                # code to run mrgdv job.
                pass

    Parameters
    ----------
    method : callable
        A MrgdvMaker.make method. This should not be specified directly and
        is implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate MRGDV jobs.
    """
    return job(method, data=_MRGDV_DATA_OBJECTS, output_schema=MrgdvdbTaskDoc)


@dataclass
class MrgdvMaker(Maker):
    """
    Maker to create jobs for merging POT files using MRGDV.

    MRGDV (MeRGe Derivative of V) is an ABINIT utility that merges multiple
    first-order potential (POT) files from different DFPT calculations into
    a single DVDB file. This is typically needed for non-linear response
    calculations or for efficient storage of first-order potentials.

    Parameters
    ----------
    name : str
        The job name. Default is "Merge DVDB".
    input_set_generator : MrgdvInputGenerator
        Generator for MRGDV input files. Defaults to MrgdvInputGenerator.
    """

    name: str = "Merge DVDB"
    input_set_generator: MrgdvInputGenerator = field(
        default_factory=MrgdvInputGenerator
    )

    @property
    def calc_type(self) -> str:
        """Get the type of calculation for this maker."""
        return self.input_set_generator.calc_type

    @mrgdv_job
    def make(
        self,
        prev_outputs: list[str] | None = None,
        history: JobHistory | None = None,
    ) -> jobflow.Response:
        """
        Create a MRGDV job to merge POT files.

        Parameters
        ----------
        prev_outputs : list[str] or None
            List of paths to previous calculation directories containing POT
            files to merge. Can be a nested list, which will be flattened.
            Must be provided to merge POT files.
        history : JobHistory or None
            A JobHistory object containing the history of previous jobs in
            the workflow.

        Returns
        -------
        Response
            A jobflow Response containing a MrgdvdbTaskDoc with the merged
            DVDB file information.
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

        # Write mrgdv input set
        write_mrgdv_input_set(
            input_set_generator=self.input_set_generator,
            prev_outputs=prev_outputs,
            directory=config.workdir,
        )

        # Run mrgdv
        run_mrgdv(
            start_time=config.start_time,
        )

        # Parse MRGDV output
        task_doc = MrgdvdbTaskDoc.from_directory(Path.cwd())
        task_doc.task_label = self.name

        return Response(
            output=task_doc,
        )
