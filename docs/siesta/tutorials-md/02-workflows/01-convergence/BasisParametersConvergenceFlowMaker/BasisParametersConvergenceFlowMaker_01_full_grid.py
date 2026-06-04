#!/usr/bin/env python
"""Full 5×5 basis parameter grid."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.basis import BasisParametersConvergenceFlowMaker
from atomate2.siesta.powerups import update_user_siesta_settings

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")
# structure = Structure.from_file("../../../00-structures/MgO_mp-1265_primitive.cif")

flow = BasisParametersConvergenceFlowMaker(
    energy_shifts=[
        0.005,
        0.01,
        0.015,
        0.02,
        0.025,
    ],  # Without Custodian works for Si_mp-149_primitive
    split_norms=[
        0.10,
        0.125,
        0.15,
        0.175,
        0.20,
    ],  # Without Custodian works for Si_mp-149_primitive
    # energy_shifts=[0.01, 0.02, 0.03],                 # for MgO_mp-1265_primitive will Crash
    # split_norms=[0.10, 0.15, 0.20],                   # for MgO_mp-1265_primitive will Crash
)
workflow = flow.make(structure)

# Apply powerup to ALL jobs in workflow
workflow = update_user_siesta_settings(
    workflow, {"a2s_kpts": [2, 2, 2], "Mesh.Cutoff": "150 Ry"}
)

results = run_locally(workflow, create_folders=True, root_dir="01", ensure_success=True)

print("✓ 5×5 grid complete: 25 parameter combinations")
