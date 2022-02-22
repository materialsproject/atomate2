"""Core jobs for running ABINIT calculations."""

from __future__ import annotations

from dataclasses import dataclass
from atomate2.abinit.jobs.base import BaseAbinitMaker
import logging
from pymatgen.core.structure import Structure
from pathlib import Path
from atomate2.abinit.run import run_abinit
from abipy.abio.factories import scf_input


logger = logging.getLogger(__name__)

__all__ = ["BaseAbinitMaker"]


@dataclass
class ScfMaker(BaseAbinitMaker):
    """
    Maker to create ABINIT scf jobs.

    Parameters
    ----------
    name : str
        The job name.
    """

    name: str = "scf"

    def run(self, structure: Structure, prev_dir: str | Path | None = None):
        """
        Run the scf calculation.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous ABINIT calculation directory.
        """


        abinit_input = scf_input(structure)
        run_abinit()
        raise NotImplementedError
