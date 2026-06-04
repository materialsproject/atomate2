#!/usr/bin/env python
"""Customize entire workflows with powerups."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.convergence import KpointsConvergenceFlowMaker
from atomate2.siesta.powerups import update_user_siesta_settings, add_metadata

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Create convergence workflow
flow = KpointsConvergenceFlowMaker(
    kpoints_list=[[2, 2, 2], [4, 4, 4], [6, 6, 6]],
    dry_run=True,
    dry_run_output_dir="powerup_flow",
)
workflow = flow.make(structure)

# Apply powerup to ALL jobs in workflow
workflow = update_user_siesta_settings(
    workflow, {"PAO.BasisSize": "DZP", "Mesh.Cutoff": "350 Ry"}
)

# Add metadata
workflow = add_metadata(workflow, {"study": "kpoints_test", "basis": "DZP"})

run_locally(workflow, create_folders=True)
print("✓ Complete: powerup_flow/")
print("  All 3 k-point tests use DZP basis + 350 Ry cutoff")
