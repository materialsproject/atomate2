#!/usr/bin/env python
"""Mesh cutoff convergence test."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.convergence import MeshCutoffConvergenceFlowMaker

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

flow = MeshCutoffConvergenceFlowMaker(
    dry_run=False, mesh_cutoffs=[100, 150, 200, 250, 300, 350, 400]
)
workflow = flow.make(structure)
results = run_locally(
    workflow, create_folders=True, root_dir="MeshCutoffConvergenceFlowMaker"
)

print("✓ Mesh cutoff convergence complete")
print("  Tested: 100-400 Ry in 50 Ry steps")
