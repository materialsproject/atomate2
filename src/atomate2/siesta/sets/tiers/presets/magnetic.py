"""Magnetic calculation tier presets.

Presets for spin-polarized and strongly correlated magnetic systems.
"""

from __future__ import annotations

from typing import Any

MAGNETIC_PRESETS: dict[str, dict[str, Any]] = {
    "magnetic_2d": {
        "description": "2D magnetic materials (spin + constraints)",
        "tier": "intermediate",
        "enabled_modules": [],  # spin already in intermediate
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "spin": "polarized",
            "a2s_kpts": [8, 8, 1],
            "Mesh.Cutoff": "400 Ry",
            "SCF.Mixer.Weight": 0.005,
            "SCF.Mixer.Method": "Pulay",
        },
    },
    "magnetic_correlated": {
        "description": "Strongly correlated magnetic systems (DFT+U)",
        "tier": "advanced",
        "enabled_modules": ["dftu"],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "spin": "polarized",
            "a2s_kpts": [6, 6, 6],
            "Mesh.Cutoff": "400 Ry",
            "SCF.Mixer.Weight": 0.002,
            "SCF.Mixer.Method": "Pulay",
            "SCF.Mixer.History": 10,
        },
    },
}
