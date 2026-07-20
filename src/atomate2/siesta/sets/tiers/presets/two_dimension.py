"""2D material calculation tier presets.

Presets optimized for 2D materials with vacuum in the z-direction.
These presets use dense in-plane k-point sampling and Γ-point in z-direction.
"""

from __future__ import annotations

from typing import Any

TWO_DIMENSION_PRESETS: dict[str, dict[str, Any]] = {
    "2d_metal": {
        "description": "2D metallic materials (graphene, MXenes) with z-vacuum",
        "tier": "intermediate",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [12, 12, 1],  # Dense in-plane, Γ in z for vacuum
            "Mesh.Cutoff": "300 Ry",
            "OccupationFunction": "MP",
            "OccupationMPOrder": 1,
            "ElectronicTemperature": "300 K",
            "SCF.Mixer.Weight": 0.01,
            "SCF.Mixer.Method": "Pulay",
            "SCF.Mixer.History": 6,
        },
    },
    "2d_metal_rough_auto": {
        "description": "2D metallic materials (graphene, MXenes) with z-vacuum",
        "tier": "intermediate",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [3, 3, 1],  # Dense in-plane, Γ in z for vacuum
            "Mesh.Cutoff": "300 Ry",
            "OccupationFunction": "MP",
            "OccupationMPOrder": 1,
            "ElectronicTemperature": "300 K",
            "SCF.Mixer.Weight": 0.01,
            "SCF.Mixer.Method": "Pulay",
            "SCF.Mixer.History": 6,
        },
    },
    "2d_semiconductor": {
        "description": "2D semiconductors (TMDs, h-BN) with z-vacuum",
        "tier": "intermediate",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [12, 12, 1],  # Dense in-plane, Γ in z for vacuum
            "Mesh.Cutoff": "300 Ry",
            "OccupationFunction": "FD",
            "ElectronicTemperature": "100 K",
            "SCF.Mixer.Weight": 0.05,
            "SCF.Mixer.Method": "Pulay",
        },
    },
    "2d_insulator": {
        "description": "2D insulators (h-BN, silicene oxide) with z-vacuum",
        "tier": "intermediate",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [10, 10, 1],  # Moderate in-plane, Γ in z for vacuum
            "Mesh.Cutoff": "300 Ry",
            "OccupationFunction": "FD",
            "ElectronicTemperature": "50 K",
            "SCF.Mixer.Weight": 0.1,
        },
    },
    "2d_magnetic": {
        "description": "2D magnetic materials (CrI3, VSe2) with z-vacuum",
        "tier": "advanced",
        "enabled_modules": ["spin"],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [12, 12, 1],  # Dense in-plane, Γ in z for vacuum
            "Mesh.Cutoff": "350 Ry",
            "Spin": "polarized",
            "OccupationFunction": "FD",
            "ElectronicTemperature": "100 K",
            "SCF.Mixer.Weight": 0.02,
            "SCF.Mixer.Method": "Pulay",
            "SCF.Mixer.History": 8,
        },
    },
    "2d_vdw": {
        "description": "2D materials with van der Waals corrections (bilayers)",
        "tier": "advanced",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [12, 12, 1],  # Dense in-plane, Γ in z for vacuum
            "Mesh.Cutoff": "300 Ry",
            # Grimme D3 van der Waals correction with automatic XC-dependent parameters
            "DFTD3": True,
            # Auto-select s6, s8, a1, a2 based on XC functional
            "DFTD3.UseXCDefaults": True,
            # Use Becke-Johnson damping (better than zero-damping)
            "DFTD3.BJdamping": True,
            # Optional: Adjust cutoffs if needed (defaults usually fine)
            # "DFTD3.2BodyCutOff": "60.0 Bohr",
            # "DFTD3.3BodyCutOff": "40.0 Bohr",
            # "DFTD3.CoordinationCutoff": "10.0 Bohr",
            "OccupationFunction": "FD",
            "ElectronicTemperature": "100 K",
            "SCF.Mixer.Weight": 0.05,
        },
    },
    "2d_vdw_dirty": {
        "description": "Fast dirty 2D calculations with vdW corrections (testing)",
        "tier": "dirty",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "SZ",
            "a2s_kpts": [1, 1, 1],  # Γ-point only (ultra-fast)
            "Mesh.Cutoff": "150 Ry",
            # Grimme D3 van der Waals correction with automatic XC-dependent parameters
            "DFTD3": True,
            # Auto-select s6, s8, a1, a2 based on XC functional
            "DFTD3.UseXCDefaults": True,
            # Use Becke-Johnson damping (better than zero-damping)
            "DFTD3.BJdamping": True,
            # Optional: Adjust cutoffs if needed (defaults usually fine)
            # "DFTD3.2BodyCutOff": "60.0 Bohr",
            # "DFTD3.3BodyCutOff": "40.0 Bohr",
            # "DFTD3.CoordinationCutoff": "10.0 Bohr",
            "OccupationFunction": "FD",
            "ElectronicTemperature": "300 K",
            "SCF.Mixer.Weight": 0.1,  # Faster convergence
            "SCF.DM.Tolerance": 1.0e-3,  # Looser tolerance
            "SCF.MustConverge": False,  # Don't fail if SCF doesn't converge
        },
    },
    "2d_optical": {
        "description": "2D materials for optical properties (excitons)",
        "tier": "advanced",
        "enabled_modules": ["optical"],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "TZP",
            "a2s_kpts": [16, 16, 1],  # Very dense in-plane for optical properties
            "Mesh.Cutoff": "350 Ry",
            "OccupationFunction": "FD",
            "ElectronicTemperature": "50 K",
            "SCF.Mixer.Weight": 0.05,
            "OpticalCalculation": True,
            "Optical.Broaden": "0.1 eV",
            "Optical.NumberOfBands": 100,
        },
    },
    "2d_screening": {
        "description": "Fast 2D material screening with z-vacuum",
        "tier": "basic",
        "enabled_modules": [],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZ",
            "a2s_kpts": [6, 6, 1],  # Reduced in-plane, Γ in z for vacuum
            "Mesh.Cutoff": "200 Ry",
            "OccupationFunction": "MP",
            "ElectronicTemperature": "300 K",
            "SCF.Mixer.Weight": 0.02,
        },
    },
}
