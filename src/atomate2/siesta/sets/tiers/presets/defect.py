"""Defect calculation tier presets.

Presets for point defect calculations (vacancies, substitutions, interstitials).
"""

from __future__ import annotations

from typing import Any

DEFECT_PRESETS: dict[str, dict[str, Any]] = {
    "defect_dirty": {
        "description": "Quick defect calculation (minimal parameters for testing)",
        "tier": "basic",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "SZ",
            "a2s_kpts": [1, 1, 1],
            "Mesh.Cutoff": "100 Ry",
            "Spin": "polarized",
            "SCF.Mixer.Weight": 0.1,
            "SCF.DM.Tolerance": 1e-4,
        },
    },
    "defect_standard": {
        "description": "Standard defect calculation (balanced accuracy/cost)",
        "tier": "intermediate",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [4, 4, 4],
            "Mesh.Cutoff": "250 Ry",
            "Spin": "polarized",
            "SCF.Mixer.Weight": 0.05,
            "SCF.Mixer.Method": "Pulay",
            "SCF.Mixer.History": 6,
            "SCF.DM.Tolerance": 1e-5,
            "MD.MaxForceTol": "0.02 eV/Ang",
        },
    },
    "defect_accurate": {
        "description": "High-accuracy defect calculation (publication quality)",
        "tier": "advanced",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "TZP",
            "a2s_kpts": [6, 6, 6],
            "Mesh.Cutoff": "400 Ry",
            "Spin": "polarized",
            "SCF.Mixer.Weight": 0.02,
            "SCF.Mixer.Method": "Pulay",
            "SCF.Mixer.History": 8,
            "SCF.DM.Tolerance": 1e-6,
            "MD.MaxForceTol": "0.01 eV/Ang",
        },
    },
    "defect_oxide": {
        "description": "Defects in oxide materials (wider band gaps)",
        "tier": "intermediate",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [4, 4, 4],
            "Mesh.Cutoff": "300 Ry",
            "Spin": "polarized",
            "OccupationFunction": "FD",
            "ElectronicTemperature": "100 K",
            "SCF.Mixer.Weight": 0.05,
            "SCF.Mixer.Method": "Pulay",
            "SCF.Mixer.History": 6,
            "SCF.DM.Tolerance": 1e-5,
        },
    },
    "defect_metal": {
        "description": "Defects in metallic systems (Fermi smearing)",
        "tier": "intermediate",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [6, 6, 6],
            "Mesh.Cutoff": "300 Ry",
            "Spin": "polarized",
            "OccupationFunction": "MP",
            "OccupationMPOrder": 1,
            "ElectronicTemperature": "300 K",
            "SCF.Mixer.Weight": 0.02,
            "SCF.Mixer.Method": "Pulay",
            "SCF.Mixer.History": 8,
        },
    },
}
