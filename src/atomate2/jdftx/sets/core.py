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