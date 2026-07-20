"""Optical and spectroscopic calculation tier presets.

Presets for optical properties, band structure, and DOS calculations.
"""

from __future__ import annotations

from typing import Any

OPTICAL_PRESETS: dict[str, dict[str, Any]] = {
    "optical_response": {
        "description": "Optical absorption and dielectric properties",
        "tier": "advanced",
        "enabled_modules": ["optical", "dos_bands"],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [8, 8, 8],
            "Mesh.Cutoff": "400 Ry",
            "Optical.CalculationType": "Polarization",
            "Optical.NumberOfEnergies": 500,
        },
    },
    "band_structure": {
        "description": "Electronic band structure and DOS",
        "tier": "advanced",
        "enabled_modules": ["dos_bands"],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [8, 8, 8],
            "Mesh.Cutoff": "300 Ry",
        },
    },
}
