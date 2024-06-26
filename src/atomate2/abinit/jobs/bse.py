"""Core jobs for running ABINIT calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import numpy as np
from jobflow import Response, job

from atomate2.abinit.jobs.base import BaseAbinitMaker
from atomate2.abinit.sets.bse import BSEmdfSetGenerator, BSEscrSetGenerator

logger = logging.getLogger(__name__)

__all__ = ["BSEmdfMaker", "BSEscrMaker"]


@dataclass
class BSEscrMaker(BaseAbinitMaker):
    """Maker to create non SCF calculations."""

    calc_type: str = "bse_scr"
    name: str = "BSE scr calculation"

    input_set_generator: BSEscrSetGenerator = field(
        default_factory=BSEscrSetGenerator
    )

    @job
    def make(
        self,
        structure: Structure | None = None,
        prev_outputs: str | list[str] | None = None,
        mdf_epsinf: float | None = None, 
        mbpt_sciss: float = 0.0,
        bs_loband: float = 0.0, 
        nband: float = 0.0,
        bs_freq_mesh: list[float] | None = None, 
    ) -> Job:
        """
        Run a non-scf ABINIT job.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        mode : str
            Type of band structure calculation. Options are:
            - "line": Full band structure along symmetry lines.
            - "uniform": Uniform mesh band structure.
        """

        if len(prev_outputs)!=2:
            raise RuntimeError("Need previous SCF and SCREENING calculations")

        self.input_set_generator.factory_kwargs = {"mbpt_sciss": mbpt_sciss,
                                                   "bs_loband": bs_loband,
                                                   "nband": nband,
                                                   "mdf_epsinf": mdf_epsinf,
                                                   "bs_freq_mesh": bs_freq_mesh
                                                  }

        return super().make.original(
            self,
            structure=structure,
            prev_outputs=prev_outputs,
        )

@dataclass
class BSEmdfMaker(BaseAbinitMaker):
    """Maker to create non SCF calculations."""

    calc_type: str = "bse_mdf"
    name: str = "BSE mdf calculation"

    input_set_generator: BSEmdfSetGenerator = field(
        default_factory=BSEmdfSetGenerator
    )

    @job
    def make(
        self,
        structure: Structure | None = None,
        prev_outputs: str | list[str] | None = None,
        mdf_epsinf: float | None = None, 
        mbpt_sciss: float = 0.0,
        bs_loband: float = 0.0, 
        nband: float = 0.0,
        bs_freq_mesh: list[float] | None = None, 
    ) -> Job:
        """
        Run a non-scf ABINIT job.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        mode : str
            Type of band structure calculation. Options are:
            - "line": Full band structure along symmetry lines.
            - "uniform": Uniform mesh band structure.
        """
        if mdf_epsinf==None:
            raise RuntimeError("Need a value of mdf_epsinf")
        if len(prev_outputs)!=1:
            raise RuntimeError("Need previous SCF calculation")

        self.input_set_generator.factory_kwargs = {"mbpt_sciss": mbpt_sciss,
                                                   "bs_loband": bs_loband,
                                                   "nband": nband,
                                                   "mdf_epsinf": mdf_epsinf,   
                                                   "bs_freq_mesh": bs_freq_mesh
                                                  }

        return super().make.original(
            self,
            structure=structure,
            prev_outputs=prev_outputs,
        )
