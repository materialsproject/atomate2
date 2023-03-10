"""Module defining lobster jobs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Maker, job
from monty.shutil import gzip_dir
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.dos import LobsterCompleteDos
from pymatgen.io.lobster import Lobsterin

from atomate2.lobster.files import copy_lobster_files
from atomate2.lobster.run import run_lobster
from atomate2.lobster.schemas import LobsterTaskDocument

__all__ = ["PureLobsterMaker"]

logger = logging.getLogger(__name__)


@dataclass
class PureLobsterMaker(Maker):
    """
    LOBSTER job maker.
    The maker copies the DFT output files
    necessary for the LOBSTER run.
    It will create all lobsterin files, run LOBSTER,
    zip the outputs and parse the LOBSTER outputs.

    Parameters
    ----------
    name : str
        Name of jobs produced by this maker.
    resubmit : bool
        Maybe useful.
    task_document_kwargs : dict
        Keyword arguments passed to :obj:`.LobsterTaskDocument.from_directory`.
    user_lobsterin_settings: dict
        Dict including additional information on the Lobster settings.
    additional_outputs: list[str]
        A list including additional output files.
    calculation_type: str
        Type of calculation for the Lobster run.
    """

    name: str = "lobster"
    resubmit: bool = False
    task_document_kwargs: dict = field(default_factory=dict)
    user_lobsterin_settings: dict | None = None
    additional_outputs: list | None = None
    calculation_type: str = "standard"

    @job(output_schema=LobsterTaskDocument, data=[CompleteCohp, LobsterCompleteDos])
    def make(
        self,
        wavefunction_dir: str | Path = None,
        basis_dict: dict | None = None,
    ):
        """
        Run a LOBSTER calculation.

        Parameters
        ----------
        wavefunction_dir : str or Path
            A directory containing a WAVEFUNCTION and other outputs needed for Lobster
        basis_dict: dict
            A dict including information on the basis set
        """
        # copy previous inputs # VASP for example
        copy_lobster_files(wavefunction_dir)

        # write lobster settings
        lobsterin = Lobsterin.standard_calculations_from_vasp_files(
            "POSCAR", "INCAR", dict_for_basis=basis_dict, option=self.calculation_type
        )

        if self.user_lobsterin_settings:
            for key, parameter in self.user_lobsterin_settings.items():
                # basis function can only be changed with the help of a yaml file
                if key != "basisfunctions":
                    lobsterin[key] = parameter

        lobsterin.write_lobsterin("lobsterin")
        # run lobster
        logger.info("Running LOBSTER")
        run_lobster()

        # gzip folder
        gzip_dir(".")

        # parse lobster outputs
        task_doc = LobsterTaskDocument.from_directory(
            Path.cwd(),
            **self.task_document_kwargs,
            additional_fields=self.additional_outputs,
        )
        return task_doc
