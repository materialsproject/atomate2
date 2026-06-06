#!/usr/bin/env python
"""K-points convergence test."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.convergence import KpointsConvergenceFlowMaker

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

flow = KpointsConvergenceFlowMaker(
    dry_run=False,
    kpoints_list=[[2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8], [10, 10, 10]],
)
workflow = flow.make(structure)
results = run_locally(
    workflow, create_folders=True, root_dir="KpointsConvergenceFlowMaker"
)

print("✓ K-points convergence complete")
print("  Tested: 2x2x2 to 10x10x10")
