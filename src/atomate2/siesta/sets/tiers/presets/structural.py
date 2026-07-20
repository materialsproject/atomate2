"""Structural calculation tier presets.

Presets for structural relaxation, bulk calculations, and molecular systems.
"""

from __future__ import annotations

from typing import Any

STRUCTURAL_PRESETS: dict[str, dict[str, Any]] = {
    "relax_dirty": {
        "description": "Basic structural relaxation (minimal parameters)",
        "tier": "basic",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "SZ",
            "a2s_kpts": [1, 1, 1],
            "Mesh.Cutoff": "50 Ry",
        },
    },
    "relax_standard": {
        "description": "Standard structural relaxation (default settings)",
        "tier": "intermediate",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [4, 4, 4],
            "Mesh.Cutoff": "200 Ry",
        },
    },
    "relax_high_accuracy": {
        "description": "High-accuracy structural relaxation",
        "tier": "intermediate",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "TZP",
            "a2s_kpts": [8, 8, 8],
            "Mesh.Cutoff": "400 Ry",
            "SCF.Mixer.Weight": 0.05,
            "SCF.DM.Tolerance": 1e-6,
        },
    },
    "relax_bulk_metal": {
        "description": "Bulk metallic systems (occupation smearing)",
        "tier": "intermediate",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [6, 6, 6],
            "Mesh.Cutoff": "300 Ry",
            "OccupationFunction": "MP",
            "OccupationMPOrder": 1,
            "ElectronicTemperature": "300 K",
            "SCF.Mixer.Weight": 0.02,
            "SCF.Mixer.Method": "Pulay",
            "SCF.Mixer.History": 6,
        },
    },
    "relax_bulk_semiconductor": {
        "description": "Bulk semiconductor/insulator systems",
        "tier": "intermediate",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [6, 6, 6],
            "Mesh.Cutoff": "300 Ry",
            "OccupationFunction": "FD",
            "ElectronicTemperature": "100 K",
            "SCF.Mixer.Weight": 0.05,
        },
    },
}
