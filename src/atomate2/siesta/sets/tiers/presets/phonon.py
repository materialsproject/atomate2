"""Phonon calculation tier presets.

Presets optimized for phonon and vibrational property calculations.
"""

from __future__ import annotations

from typing import Any

PHONON_PRESETS: dict[str, dict[str, Any]] = {
    "phonon_dirty": {
        "description": "Standard phonon calculations",
        "tier": "advanced",
        "enabled_modules": ["phonons"],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [1, 1, 1],
            "Mesh.Cutoff": "150 Ry",
            "SCF.DM.Tolerance": 1e-5,
        },
    },
    "phonon_standard": {
        "description": "Standard phonon calculations",
        "tier": "advanced",
        "enabled_modules": ["phonons"],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [6, 6, 6],
            "Mesh.Cutoff": "300 Ry",
            "SCF.DM.Tolerance": 1e-6,
        },
    },
    "phonon_high_accuracy": {
        "description": "High-accuracy phonon calculations (tight forces)",
        "tier": "advanced",
        "enabled_modules": ["phonons", "dos_bands"],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "TZP",
            "a2s_kpts": [8, 8, 8],
            "Mesh.Cutoff": "500 Ry",
            "SCF.DM.Tolerance": 1e-7,
            "MD.MaxForceTol": "0.001 eV/Ang",
        },
    },
}
