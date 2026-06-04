#!/usr/bin/env python
"""Combined sequential convergence (cutoff then k-points)."""

from jobflow import run_locally, Flow
from pymatgen.core import Structure
from atomate2.siesta.flows.convergence import (
    MeshCutoffConvergenceFlowMaker,
    KpointsConvergenceFlowMaker,
)

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Step 1: Converge cutoff
cutoff_flow = MeshCutoffConvergenceFlowMaker(
    dry_run=False, mesh_cutoffs=[200, 250, 300, 350]
)
cutoff_workflow = cutoff_flow.make(structure)

# Step 2: Converge k-points
kpts_flow = KpointsConvergenceFlowMaker(
    dry_run=False, kpoints_list=[[4, 4, 4], [6, 6, 6], [8, 8, 8]]
)
kpts_workflow = kpts_flow.make(structure)

# Combine sequentially
combined = Flow([cutoff_workflow, kpts_workflow])
results = run_locally(combined, create_folders=True)

print("✓ Combined convergence complete")
print("  Step 1: Mesh cutoff (200-350 Ry)")
print("  Step 2: K-points (4x4x4 to 8x8x8)")
