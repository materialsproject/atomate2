from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from pymatgen.core.periodic_table import Element

from atomate2.jdftx.sets.base import JdftxInputGenerator

if TYPE_CHECKING:
    from emmet.core.math import Vector3D
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Outcar, Vasprun


logger = logging.getLogger(__name__)

@dataclass
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