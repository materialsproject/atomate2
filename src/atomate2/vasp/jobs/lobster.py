"""Module defining amset jobs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Maker, Response, job
from monty.serialization import loadfn
from monty.shutil import gzip_dir

__all__ = ["LobsterMaker"]

logger = logging.getLogger(__name__)


@dataclass
class LobsterMaker(Maker):
    """
    Lobster job maker.

    Parameters
    ----------
    name : str
        Name of jobs produced by this maker.
    resubmit : bool
        Could this be interesting?
    task_document_kwargs : dict
        Keyword arguments passed to :obj:`.LobsterTaskDocument.from_directory`.
    """

    name: str = "lobster"
    resubmit: bool = False
    task_document_kwargs: dict = field(default_factory=dict)

    @job(output_schema=LobsterTaskDocument)
    def make(
        self,
        settings: dict,
        #prev_lobster_dir: str | Path = None, # needed?
        wavefunction_dir: str | Path = None,
    ):
        """
        Run an AMSET calculation.

        Parameters
        ----------
        settings : dict
            Amset settings.
        wavefunction_dir : str or Path
            A directory containing a VASP computation including WAVECAR
            # could be extended to other codes as well

        """
        copy_lobster_files(wavefunction_dir)

        # write amset settings
        write_lobster_settings(settings)

        # run amset
        logger.info("Running LOBSTER")
        run_lobster()


        # what checks might be useful? we have validators in custodian already



        # parse amset outputs
        task_doc = LosterTaskDocument.from_directory(
            Path.cwd(), **self.task_document_kwargs
        )

        # gzip folder
        gzip_dir(".")

        # handle resubmission for non-converged calculations
        # not sure what do here or if needed!


        return Response(output=task_doc)
