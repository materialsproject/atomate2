"""Molecular and adsorbate calculation tier presets.

Presets for isolated molecules and adsorbate screening calculations.
"""

from __future__ import annotations

from typing import Any

MOLECULAR_PRESETS: dict[str, dict[str, Any]] = {
    "molecule_gas_phase": {
        "description": "Isolated molecules in gas phase (large box, Γ-point)",
        "tier": "intermediate",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [1, 1, 1],  # Γ-point only for isolated molecules
            "Mesh.Cutoff": "300 Ry",
            "OccupationFunction": "FD",
            "ElectronicTemperature": "25 K",  # Low temperature for molecules
            "SCF.Mixer.Weight": 0.1,
        },
    },
    "adsorbate_screening": {
        "description": "Fast adsorbate screening (basic parameters for grid scans)",
        "tier": "basic",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [4, 4, 1],  # Reduced k-points for screening
            "Mesh.Cutoff": "200 Ry",
            "OccupationFunction": "MP",
            "ElectronicTemperature": "300 K",
            "SCF.Mixer.Weight": 0.01,
        },
    },
}
