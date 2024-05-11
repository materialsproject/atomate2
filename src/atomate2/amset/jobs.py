"""Module defining amset jobs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Maker, Response, job
from monty.serialization import loadfn
from monty.shutil import gzip_dir

from atomate2.amset.files import copy_amset_files, write_amset_settings
from atomate2.amset.run import check_converged, run_amset
from atomate2.amset.schemas import AmsetTaskDocument

logger = logging.getLogger(__name__)


@dataclass
class AmsetMaker(Maker):
    """
    AMSET job maker.

    Parameters
    ----------
    name : str
        Name of jobs produced by this maker.
    resubmit : bool
        Whether to resubmit an new calculation with a denser interpolation factor if the
        transport results are not converged. Note, checking for convergence requires
        a previous AMSET directory.
    task_document_kwargs : dict
        Keyword arguments passed to :obj:`.AmsetTaskDocument.from_directory`.
    """

    name: str = "amset"
    resubmit: bool = False
    task_document_kwargs: dict = field(default_factory=dict)

    @job(output_schema=AmsetTaskDocument, data=["transport", "mesh"])
    def make(
        self,
        settings: dict,
        prev_dir: str | Path = None,
        wavefunction_dir: str | Path = None,
        deformation_dir: str | Path = None,
        bandstructure_dir: str | Path = None,
    ) -> Response:
        """Run an AMSET calculation.

        Parameters
        ----------
        settings : dict
            Amset settings.
        prev_dir : str or Path
            A previous AMSET calculation directory to copy output files from. The
            previous directory is also used to check for transport convergence.
        wavefunction_dir : str or Path
            A directory containing a wavefunction.h5 file.
        deformation_dir : str or Path
            A directory containing a deformation.h5 file.
        bandstructure_dir : str or Path
            A directory containing the dense band structure file (vasprun.xml or
            band_structure_data.json).
        """
        # copy previous inputs
        from_prev = prev_dir is not None
        if prev_dir is not None:
            copy_amset_files(prev_dir)
        else:
            if bandstructure_dir is None:
                raise ValueError("Either prev_dir or bandstructure_dir must be set")

            copy_amset_files(bandstructure_dir)

            if deformation_dir is not None:
                copy_amset_files(deformation_dir)

            if wavefunction_dir is not None:
                copy_amset_files(wavefunction_dir)

        # write amset settings
        write_amset_settings(settings, from_prev=from_prev)

        # run amset
        logger.info("Running AMSET")
        run_amset()

        converged = None
        if self.resubmit:
            prev_transport_file = Path("transport.prev.json")
            if not prev_transport_file.exists():
                logger.info("No previous transport calculations found.")
                converged = False
            else:
                transport_data = loadfn(next(Path().glob("transport_*.json")))
                converged = check_converged(transport_data, loadfn(prev_transport_file))

        self.task_document_kwargs.setdefault("include_mesh", converged is not False)

        # parse amset outputs
        task_doc = AmsetTaskDocument.from_directory(
            Path.cwd(), **self.task_document_kwargs
        )
        task_doc.converged = converged

        # gzip folder
        gzip_dir(".")

        # handle resubmission for non-converged calculations
        replace = None
        if self.resubmit and not converged:
            replace = self.make(
                {"interpolation_factor": settings.get("interpolation_factor", 10) + 5},
                prev_dir=task_doc.dir_name,
            )

        return Response(output=task_doc, replace=replace)
