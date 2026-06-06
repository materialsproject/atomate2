#!/usr/bin/env python
"""Preview entire workflows with automatic dry-run propagation."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.convergence import (
    KpointsConvergenceFlowMaker,
    MeshCutoffConvergenceFlowMaker,
)
from atomate2.siesta.flows.eos import SiestaEosFlowMaker

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Example 1: K-points convergence
kpts_flow = KpointsConvergenceFlowMaker(
    kpoints_list=[[2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8]],
    dry_run=True,
    dry_run_output_dir="kpoints_preview",
)
workflow = kpts_flow.make(structure)
run_locally(workflow, create_folders=True)
print("✓ K-points preview: kpoints_preview/")

# Example 2: Mesh cutoff convergence
mesh_flow = MeshCutoffConvergenceFlowMaker(
    mesh_cutoffs=[150, 200, 250, 300, 350],
    dry_run=True,
    dry_run_output_dir="mesh_preview",
)
workflow = mesh_flow.make(structure)
run_locally(workflow, create_folders=True)
print("✓ Mesh cutoff preview: mesh_preview/")

# Example 3: EOS workflow
eos_flow = SiestaEosFlowMaker(
    linear_strain=(-0.05, 0.05),
    number_of_frames=7,
    dry_run=True,
    dry_run_output_dir="eos_preview",
)
workflow = eos_flow.make(structure)
run_locally(workflow, create_folders=True)
print("✓ EOS preview (7 volumes): eos_preview/")
