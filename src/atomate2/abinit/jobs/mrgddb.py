"""Merge DDB jobs for merging DDB files from ABINIT calculations."""

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
    from typing import Callable

    from atomate2.abinit.utils.history import JobHistory

logger = logging.getLogger(__name__)

__all__ = [
    "MrgddbMaker",
]

_MRGDDB_DATA_OBJECTS = [
    AbinitStoredFile,
]


def mrgddb_job(method: Callable) -> job:
    """
    Decorate the ``make`` method of mrgddb job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.job` that configures common
    settings for all mrgddb jobs. For example, it ensures that large data objects
    (band structures, density of states, Cubes, etc) are all stored in the
    atomate2 data store. It also configures the output schema to be a mrgddb
    :obj:`.TaskDocument`.

    Any makers that return mrgddb jobs (not flows) should decorate the ``make`` method
    with @mrgddb_job. For example:

    .. code-block:: python

        class MyMrgddbMaker(MrgddbMaker):
            @mrgddb_job
            def make(structure):
                # code to run mrgddb job.
                pass

    Parameters
    ----------
    method : callable
        A BaseMrgddbMaker.make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate mrgddb jobs.
    """
    return job(method, data=_MRGDDB_DATA_OBJECTS, output_schema=MrgddbTaskDoc)


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
        default_factory=MrgddbInputGenerator
    )
    # input_set_generator: MrgddbInputGenerator = MrgddbInputGenerator()
    wall_time: int | None = None

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
        Return a MRGDDB jobflow.Job.

        Parameters
        ----------
        prev_outputs : TODO: add description from sets.base
        history : JobHistory
            A JobHistory object containing the history of this job.
        """
        # Flatten the list of previous outputs dir
        # prev_outputs = [item for sublist in prev_outputs for item in sublist]
        prev_outputs = list(np.hstack(prev_outputs))

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

        return Response(
            output=task_doc,
        )
