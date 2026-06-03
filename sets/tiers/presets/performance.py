"""Performance tuning tier presets.

Presets for large systems, HPC parallel calculations, and convergence testing.
"""

from __future__ import annotations

from typing import Any

PERFORMANCE_PRESETS: dict[str, dict[str, Any]] = {
    "large_system": {
        "description": "Large systems (>100 atoms) - performance optimizations",
        "tier": "expert",
        "enabled_modules": ["parallel", "solvers", "efficiency"],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [2, 2, 2],
            "Mesh.Cutoff": "200 Ry",
            "SolutionMethod": "OrderN",  # Linear-scaling
            "ON.MaximumIterations": 1000,
        },
    },
    "parallel_hpc": {
        "description": "HPC parallel calculations (MPI optimization)",
        "tier": "expert",
        "enabled_modules": ["parallel", "solvers"],
        "disabled_modules": [],
        "recommended_params": {
            "PAO.BasisSize": "DZP",
            "Diag.ParallelOverK": "true",
        },
    },
    "convergence_test": {
        "description": "Convergence testing (all modules enabled)",
        "tier": "expert",
        "enabled_modules": [],  # Use all modules from expert tier
        "disabled_modules": [],
        "recommended_params": {},
    },
}
