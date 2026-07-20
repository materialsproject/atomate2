"""Surface and interface calculation tier presets.

Presets optimized for surface energy calculations and slab models.
"""

from __future__ import annotations

from typing import Any

SURFACE_PRESETS: dict[str, dict[str, Any]] = {
    "surface_dirty": {
        "description": "Quick surface screening (low accuracy, fast)",
        "tier": "dirty",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "SZP",
            "a2s_kpts": [2, 2, 1],
            "Mesh.Cutoff": "150 Ry",
            "OccupationFunction": "MP",
            "OccupationMPOrder": 1,
            "ElectronicTemperature": "300 K",
            "SCF.MustConverge": False,
            "SCF.Mixer.Weight": 0.02,
        },
    },
    "surface_basic": {
        "description": "Standard surface calculations (reliable)",
        "tier": "basic",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZ",
            "a2s_kpts": [3, 3, 1],
            "Mesh.Cutoff": "250 Ry",
            "OccupationFunction": "MP",
            "OccupationMPOrder": 1,
            "ElectronicTemperature": "300 K",
            "SCF.Mixer.Weight": 0.01,
            "SCF.Mixer.Method": "Pulay",
            "SCF.Mixer.History": 6,
        },
    },
    "surface_metal": {
        "description": "Metallic surface calculations (occupation smearing)",
        "tier": "intermediate",
        "enabled_modules": [],  # SCF and electronic_structure already in intermediate
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [6, 6, 1],  # Dense in-plane, Γ in z
            "Mesh.Cutoff": "300 Ry",
            "OccupationFunction": "MP",
            "OccupationMPOrder": 1,
            "ElectronicTemperature": "300 K",
            "SCF.Mixer.Weight": 0.005,
            "SCF.Mixer.Method": "Pulay",
            "SCF.Mixer.History": 6,
        },
    },
    "surface_semiconductor": {
        "description": "Semiconductor surface calculations",
        "tier": "advanced",
        "enabled_modules": ["charge_dipole"],  # Dipole corrections
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [3, 3, 1],
            "Mesh.Cutoff": "300 Ry",
            "OccupationFunction": "FD",
            "ElectronicTemperature": "100 K",
        },
    },
}
