"""Electrocatalysis tier presets.

Presets optimized for electrocatalysis calculations including ORR, OER, and HER.
These presets are designed for surface adsorption calculations with appropriate
k-point sampling and convergence settings for catalytic activity studies.
"""

from __future__ import annotations

from typing import Any

ELECTROCATALYSIS_PRESETS: dict[str, dict[str, Any]] = {
    "electrocatalysis_dirty": {
        "description": "Fast electrocatalysis screening (low accuracy, quick)",
        "tier": "dirty",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "SZ",
            "a2s_kpts": [2, 2, 1],
            "Mesh.Cutoff": "150 Ry",
            "OccupationFunction": "MP",
            "OccupationMPOrder": 1,
            "ElectronicTemperature": "300 K",
            "SCF.MustConverge": False,
            "SCF.Mixer.Weight": 0.02,
        },
    },
    "electrocatalysis_basic": {
        "description": "Standard electrocatalysis calculations (reliable)",
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
    "electrocatalysis_intermediate": {
        "description": "Publication-quality electrocatalysis (DZP + vdW)",
        "tier": "intermediate",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [4, 4, 1],
            "Mesh.Cutoff": "300 Ry",
            "OccupationFunction": "MP",
            "OccupationMPOrder": 1,
            "ElectronicTemperature": "300 K",
            "SCF.Mixer.Weight": 0.005,
            "SCF.Mixer.Method": "Pulay",
            "SCF.Mixer.History": 8,
            "XC.functional": "VDW",
            "XC.authors": "DRSLL",
        },
    },
    "electrocatalysis_gas_phase": {
        "description": "Gas-phase molecule references (high accuracy)",
        "tier": "advanced",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "TZP",
            "a2s_kpts": [1, 1, 1],  # Gamma-only for molecules
            "Mesh.Cutoff": "400 Ry",
            "OccupationFunction": "FD",
            "ElectronicTemperature": "25 meV",
            "SCF.Mixer.Weight": 0.1,
        },
    },
}
