#!/usr/bin/env python
"""Test 4 basis sets with EOS."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.basis import EOSBasisConvergenceFlowMaker

# structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")
structure = Structure.from_file("../../../00-structures/MoS2.cif")

flow = EOSBasisConvergenceFlowMaker(
    dry_run=False,
    basis_sets=["SZ", "DZ", "DZP", "DZDP"],
    linear_strain=(-0.05, 0.05),
    number_of_frames=7,
)
workflow = flow.make(structure)
results = run_locally(workflow, create_folders=True, root_dir="01_standard")

print("✓ EOS basis convergence: 4 basis × 7 volumes = 28 calculations")
