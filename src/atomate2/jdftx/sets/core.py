"""Module defining core JDFTx input set generators."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from atomate2.jdftx.sets.base import _BASE_JDFTX_SET, JdftxInputGenerator

logger = logging.getLogger(__name__)


@dataclass
class SinglePointSetGenerator(JdftxInputGenerator):
    """Class to generate JDFTx input sets that follow BEAST convention."""

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


@dataclass
class IonicMinSetGenerator(JdftxInputGenerator):
    """Class to generate JDFTx relax sets."""

    default_settings: dict = field(
        default_factory=lambda: {
            **_BASE_JDFTX_SET,
            "elec-initial-magnetization": {"M": 5, "constrain": False},
            "fluid": {"type": "LinearPCM"},
            "pcm-variant": "CANDLE",
            "fluid-solvent": {"name": "H2O"},
            "fluid-cation": {"name": "Na+", "concentration": 0.5},
            "fluid-anion": {"name": "F-", "concentration": 0.5},
            "ionic-minimize": {"nIterations": 100},
        }
    )


@dataclass
class LatticeMinSetGenerator(JdftxInputGenerator):
    """Class to generate JDFTx lattice minimization sets."""

    default_settings: dict = field(
        default_factory=lambda: {
            **_BASE_JDFTX_SET,
            "elec-initial-magnetization": {"M": 5, "constrain": False},
            "fluid": {"type": "LinearPCM"},
            "pcm-variant": "CANDLE",
            "fluid-solvent": {"name": "H2O"},
            "fluid-cation": {"name": "Na+", "concentration": 0.5},
            "fluid-anion": {"name": "F-", "concentration": 0.5},
            "lattice-minimize": {"nIterations": 100},
            "latt-move-scale": {"s0": 1, "s1": 1, "s2": 1},
        }
    )
