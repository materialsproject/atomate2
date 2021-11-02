"""Module defining amset jobs."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Union

from jobflow import Maker, Response, job
from monty.serialization import loadfn
from monty.shutil import gzip_dir

from atomate2.amset.files import copy_amset_files, write_amset_settings
from atomate2.amset.run import check_converged, run_amset
from atomate2.amset.schemas import AmsetTaskDocument

__all__ = ["AmsetMaker"]

logger = logging.getLogger(__name__)


@dataclass
class AmsetMaker(Maker):
    """
    AMSET job maker.

    Parameters
    ----------
    name
        Name of jobs produced by this maker.
    resubmit
        Whether to resubmit an new calculation with a denser interpolation factor if the
        transport results are not converged. Note, checking for convergence requires
        a previous AMSET directory.
    task_document_kwargs
        Keyword arguments passed to :obj:`.AmsetTaskDocument.from_directory`.
    """

    name: str = "amset"
    resubmit: bool = False
    task_document_kwargs: dict = field(default_factory=dict)

    @job(output_schema=AmsetTaskDocument)
    def make(
        self,
        settings: Dict,
        prev_amset_dir: Union[str, Path] = None,
        wavefunction_dir: Union[str, Path] = None,
        deformation_dir: Union[str, Path] = None,
    ):
        """
        Run an AMSET calculation.

        Parameters
        ----------
        settings
            Amset settings.
        prev_amset_dir
            A previous AMSET calculation directory to copy output files from. The
            previous directory is also used to check for transport convergence.
        wavefunction_dir
            A directory containing a wavefunction.h5 file.
        deformation_dir
            A directory containing a deformation.h5 file.
        """
        # copy previous inputs
        from_prev = prev_amset_dir is not None
        if prev_amset_dir is not None:
            copy_amset_files(prev_amset_dir)

        # write amset settings
        write_amset_settings(settings, from_prev=from_prev)

        # run amset
        logger.info("Running AMSET")
        transport_data = run_amset()[0].transport

        converged = None
        if self.resubmit:
            prev_transport_file = Path("transport.prev.json")
            if not prev_transport_file.exists():
                logger.info("No previous transport calculations found.")
                converged = False
            else:
                converged = check_converged(transport_data, loadfn(prev_transport_file))

        if "include_mesh" not in self.task_document_kwargs:
            self.task_document_kwargs["include_mesh"] = converged is not False

        # parse amset outputs
        task_doc = AmsetTaskDocument.from_directory(
            Path.cwd(), **self.task_document_kwargs
        )
        task_doc.task_label = self.name
        task_doc.converged = converged

        # gzip folder
        gzip_dir(".")

        return Response(output=task_doc)
