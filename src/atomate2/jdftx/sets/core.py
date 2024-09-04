from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.jdftx.sets.base import _BASE_JDFTX_SET, JdftxInputGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Outcar, Vasprun


logger = logging.getLogger(__name__)


@dataclass
class RelaxSetGenerator(JdftxInputGenerator):
class RelaxSetGenerator(JdftxInputGenerator):
    """Class to generate VASP relaxation input sets."""

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """Get updates to the INCAR for a relaxation job.

        Parameters
        ----------
        structure
            A structure.
        prev_incar
            An incar from a previous calculation.
        bandgap
            The band gap.
        vasprun
            A vasprun from a previous calculation.
        outcar
            An outcar from a previous calculation.

        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        return {"NSW": 99, "LCHARG": False, "ISIF": 3, "IBRION": 2}


@dataclass
class BEASTSetGenerator(JdftxInputGenerator):
    default_settings: dict = field(
        default_factory=lambda: {
            **_BASE_JDFTX_SET,
            "elec-initial-magnetization": {"M": 5, "constrain": False},
            "fluid": {"type": "LinearPCM"},
            "pcm-variant": "CANDLE",
            "fluid-solvent": {"name": "H2O"},
            "fluid-cation": {"name": "Na+", "concentration": 0.5},
            "fluid-anion": {"name": "F-", "concentration": 0.5},
        }
    )
