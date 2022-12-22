"""Module defining amset jobs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Maker, Response, job
from monty.serialization import loadfn
from monty.shutil import gzip_dir
from pymatgen.io.lobster import Lobsterin
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.dos import LobsterCompleteDos
from atomate2.lobster.files import copy_lobster_files, write_lobster_settings
from atomate2.lobster.schemas import LobsterTaskDocument
from atomate2.lobster.run import run_lobster

__all__ = ["PureLobsterMaker"]

logger = logging.getLogger(__name__)


@dataclass
class PureLobsterMaker(Maker):
    """
    LOBSTER job maker.

    Parameters
    ----------
    name : str
        Name of jobs produced by this maker.
    resubmit : bool
        Maybe useful.
    task_document_kwargs : dict
        Keyword arguments passed to :obj:`.LobsterTaskDocument.from_directory`.
    """

    name: str = "lobster"
    resubmit: bool = False
    task_document_kwargs: dict = field(default_factory=dict)

    @job(output_schema=LobsterTaskDocument, data=[CompleteCohp, LobsterCompleteDos])
    def make(
        self,
        wavefunction_dir: str | Path = None,
        basis_dict: dict = None,
        user_lobsterin_settings: dict = None,
        additional_outputs: list[str] = (None,)
        # something for the basis
    ):
        """
        Run an LOBSTER calculation.

        Parameters
        ----------
        wavefunction_dir : str or Path
            A directory containing a WAVEFUNCTION and other outputs needed for Lobster

        """
        # copy previous inputs # VASP for example
        copy_lobster_files(wavefunction_dir)

        # write lobster settings
        lobsterin = Lobsterin.standard_calculations_from_vasp_files(
            "POSCAR", "INCAR", dict_for_basis=basis_dict
        )
        if user_lobsterin_settings:
            for key, parameter in user_lobsterin_settings.items():
                lobsterin[key] = parameter

        lobsterin.write_lobsterin("lobsterin")
        # run lobster
        logger.info("Running LOBSTER")
        run_lobster()

        # converged = None

        # gzip folder
        gzip_dir(".")

        # parse amset outputs
        task_doc = LobsterTaskDocument.from_directory(
            Path.cwd(), **self.task_document_kwargs, additional_fields=additional_outputs
        )
        # task_doc.converged = converged

        return Response(output=task_doc)
