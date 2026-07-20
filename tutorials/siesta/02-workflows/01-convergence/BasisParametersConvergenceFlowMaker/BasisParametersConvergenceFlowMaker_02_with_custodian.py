#!/usr/bin/env python
"""Basis parameter convergence with automatic error handling (custodian)."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.basis import BasisParametersConvergenceFlowMaker
from atomate2.siesta.powerups import update_user_siesta_settings

structure = Structure.from_file("../../../00-structures/MgO_mp-1265_primitive.cif")

flow = BasisParametersConvergenceFlowMaker(
    energy_shifts=[0.01, 0.02, 0.03],
    split_norms=[0.10, 0.15, 0.20],
    use_custodian=True,
    custodian_max_errors=10,
).make(structure)

# workflow = flow.make(structure)

# Apply powerup to ALL jobs in workflow
workflow = update_user_siesta_settings(
    flow, {"a2s_kpts": [2, 2, 2], "Mesh.Cutoff": "150 Ry"}
)


results = run_locally(workflow, create_folders=True)

print("✓ MgO basis test complete: 3×3 grid")
