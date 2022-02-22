"""Definition of base ABINIT job maker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union, List
import pseudo_dojo

import jobflow
from jobflow import Maker, job
from pymatgen.core.structure import Structure
from pymatgen.io.abinit.pseudos import PseudoTable

__all__ = ["BaseAbinitMaker"]
pseudo_dojo.OfficialDojoTable

@dataclass
class BaseAbinitMaker(Maker):
    """
    Base ABINIT job maker.

    Parameters
    ----------
    name : str
        The job name.
    """

    name: str = "base abinit job"
    pseudos: Union[List[str], ] = PseudoTable()

    @job
    def make(
        self, structure: Structure, prev_dir: str | Path | None = None
    ) -> Union[jobflow.Flow, jobflow.Job]:
        """
        Return an ABINIT jobflow.Job.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous ABINIT calculation directory.
        """
        return self.run(structure=structure, prev_dir=prev_dir)

    def run(self, structure: Structure, prev_dir: str | Path | None = None):
        """
        Run the actual job.

        Must be overridden with a concrete implementation.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous ABINIT calculation directory.
        """
        raise NotImplementedError
