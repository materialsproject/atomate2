"""Tier presets collection.

This module aggregates all tier presets from different categories into
a single TIER_PRESETS dictionary for easy access.
"""

from __future__ import annotations

from typing import Any

from .defect import DEFECT_PRESETS
from .electrocatalysis import ELECTROCATALYSIS_PRESETS
from .magnetic import MAGNETIC_PRESETS
from .molecular import MOLECULAR_PRESETS
from .optical import OPTICAL_PRESETS
from .performance import PERFORMANCE_PRESETS
from .phonon import PHONON_PRESETS
from .structural import STRUCTURAL_PRESETS
from .surface import SURFACE_PRESETS
from .two_dimension import TWO_DIMENSION_PRESETS

# Aggregate all presets into a single dictionary
TIER_PRESETS: dict[str, dict[str, Any]] = {}

# Add all preset categories
TIER_PRESETS.update(STRUCTURAL_PRESETS)
TIER_PRESETS.update(SURFACE_PRESETS)
TIER_PRESETS.update(MOLECULAR_PRESETS)
TIER_PRESETS.update(MAGNETIC_PRESETS)
TIER_PRESETS.update(PHONON_PRESETS)
TIER_PRESETS.update(OPTICAL_PRESETS)
TIER_PRESETS.update(PERFORMANCE_PRESETS)
TIER_PRESETS.update(TWO_DIMENSION_PRESETS)
TIER_PRESETS.update(DEFECT_PRESETS)
TIER_PRESETS.update(ELECTROCATALYSIS_PRESETS)

__all__ = [
    "TIER_PRESETS",
    "STRUCTURAL_PRESETS",
    "SURFACE_PRESETS",
    "MOLECULAR_PRESETS",
    "MAGNETIC_PRESETS",
    "PHONON_PRESETS",
    "OPTICAL_PRESETS",
    "PERFORMANCE_PRESETS",
    "TWO_DIMENSION_PRESETS",
    "DEFECT_PRESETS",
    "ELECTROCATALYSIS_PRESETS",
]
