"""Default parameters for each tier level.

This module defines the base parameter sets for the four tier levels:
basic, intermediate, advanced, and expert. These serve as starting points
that can be overridden by specific presets.
"""

from __future__ import annotations

from typing import Any

# ==============================================================================
# TIER-LEVEL DEFAULT PARAMETERS
# ==============================================================================

TIER_DEFAULTS: dict[str, dict[str, Any]] = {
    "dirty": {
        "PAO.BasisSize": "SZ",
        "a2s_kpts": [1, 1, 1],
        "Mesh.Cutoff": "50 Ry",
    },
    "basic": {
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [3, 3, 3],
        "Mesh.Cutoff": "150 Ry",
    },
    "intermediate": {
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [6, 6, 6],
        "Mesh.Cutoff": "200 Ry",
    },
    "advanced": {
        "PAO.BasisSize": "TZP",
        "a2s_kpts": [6, 6, 6],
        "Mesh.Cutoff": "300 Ry",
    },
    "expert": {
        "PAO.BasisSize": "TZP",
        "a2s_kpts": [8, 8, 8],
        "Mesh.Cutoff": "400 Ry",
    },
    "ultra": {
        "PAO.BasisSize": "TZDP",
        "a2s_kpts": [10, 10, 10],
        "Mesh.Cutoff": "800 Ry",
    },
}
