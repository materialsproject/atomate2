#!/usr/bin/env python
"""Complete basis convergence: comprehensive study (full)."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.basis import CompleteBasisConvergenceFlowMaker

# Load structure
structure = Structure.from_file("../../../00-structures/MgO_mp-1265_primitive.cif")

# Create complete basis convergence workflow
# Tests: 3 basis sizes × 3 energy_shifts × 3 split_norms = 27 calculations
maker = CompleteBasisConvergenceFlowMaker(
    basis_sizes=["SZ", "SZP", "DZ", "DZP", "TZ", "TZP"],  # Test 3 basis sizes
    # basis_sizes=["DZ", "DZP"],  # Test 3 basis sizes
    energy_shifts=[0.005, 0.01, 0.015, 0.025],  # Test 3 energy shifts (Ry)
    split_norms=[0.15, 0.20, 0.25, 0.35],  # Test 3 split norms
    kpts=[4, 4, 4],  # K-points grid
)

# Create and run workflow
flow = maker.make(structure)
results = run_locally(flow, create_folders=True)

print("✓ Complete basis convergence (full): 3×3×3 = 27 calculations")
print("  Basis sizes: DZ, DZP, TZP")
print("  Energy shifts: 0.005, 0.01, 0.015 Ry")
print("  Split norms: 0.15, 0.20, 0.25")
print("\nThis workflow finds optimal basis size AND parameters simultaneously.")
