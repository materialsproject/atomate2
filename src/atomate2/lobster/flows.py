"""Module defining lobster jobs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Maker, job, Flow
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.dos import LobsterCompleteDos
from pymatgen.io.lobster import Bandoverlaps, Icohplist, Lobsterin

from atomate2 import SETTINGS
from atomate2.common.files import gzip_output_folder
from atomate2.lobster.files import (
    LOBSTEROUTPUT_FILES,
    VASP_OUTPUT_FILES,
    copy_lobster_files,
)
from atomate2.lobster.jobs import LobsterMaker, retrieve_relevant_bonds
from atomate2.lobster.run import run_lobster
from atomate2.lobster.schemas import LobsterTaskDocument


logger = logging.getLogger(__name__)


_FILES_TO_ZIP = [*LOBSTEROUTPUT_FILES, "lobsterin", *VASP_OUTPUT_FILES]



@dataclass
class AdvancedLobsterMaker(Maker):
    """
    LOBSTER job maker with additional speedup.

    1. The maker copies DFT output files necessary for the LOBSTER run.
    2. It will create all lobsterin files, run LOBSTER several times,
        zip the outputs and parse the LOBSTER outputs. In this step, the COHP/COBI/COOP
        curves will only be written with a limited accuracy.
    3. After an analysis of the most important bonds, COHP/COBI/COOP curves for those will
        be computed with high accuracy.

    In the future, this workflow could also benefit from further symmetry considerations.

    Parameters
    ----------
    name : str
        Name of jobs produced by this maker.
    task_document_kwargs : dict
        Keyword arguments passed to :obj:`.LobsterTaskDocument.from_directory`.
    user_lobsterin_settings : dict
        Dict including additional information on the Lobster settings.
    run_lobster_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_lobster`.
    calculation_type : str
        Type of calculation for the Lobster run that will get passed to
        :obj:`.Lobsterin.standard_calculations_from_vasp_files`.
    """
    name: str = "lobster"
    lobster_maker_1: LobsterMaker=field(default_factory=lambda:LobsterMaker(user_lobsterin_settings={"cohpsteps":1}))
    lobster_maker_2: LobsterMaker=field(default_factory=LobsterMaker)

    def make(
            self,
            wavefunction_dir: str | Path = None,
            basis_dict: dict | None = None,
    ) -> LobsterTaskDocument:
        """Run a LOBSTER calculation.

        Parameters
        ----------
        wavefunction_dir : str or Path
            A directory containing a WAVEFUNCTION and other outputs needed for Lobster
        basis_dict: dict
            A dict including information on the basis set
        """
        # TODO: why is calc quality failing?
        jobs=[]
        lobster_1=self.lobster_maker_1.make(wavefunction_dir, basis_dict)
        jobs.append(lobster_1)

        # code to postprocess the lobster data and identify relevant bonds
        # switch between cation, anion modes and potentially different ways to indentify the bonds
        cohp_between_dict=retrieve_relevant_bonds(lobster_1.output.lobsterpy_data)

        # think about how to modify the input dict during the run time
        lobster_2 = self.lobster_maker_2.make(wavefunction_dir, basis_dict, cohp_between_dict)
        jobs.append(lobster_2)
        return Flow(jobs=jobs, output=lobster_2.output)