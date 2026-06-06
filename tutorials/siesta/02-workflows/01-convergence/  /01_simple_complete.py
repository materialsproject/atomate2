#!/usr/bin/env python
"""Complete basis convergence: tests both sizes and parameters (simple)."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.basis import CompleteBasisConvergenceMaker

# Load structure
structure = Structure.from_file("../../../00-structures/MgO_mp-1265_primitive.cif")

# Create complete basis convergence workflow
# Tests: 2 basis sizes × 2 energy_shifts × 2 split_norms = 8 calculations
maker = CompleteBasisConvergenceMaker(
    basis_sizes=["DZ", "DZP"],  # Test 2 basis sizes
    energy_shifts=[0.01, 0.02],  # Test 2 energy shifts (Ry)
    split_norms=[0.15, 0.20],  # Test 2 split norms
    kpts=[3, 3, 3],  # K-points grid
)

# Create and run workflow
flow = maker.make(structure)
results = run_locally(flow, create_folders=True)

print("✓ Complete basis convergence (simple): 2×2×2 = 8 calculations")
print("  Basis sizes: DZ, DZP")
print("  Energy shifts: 0.01, 0.02 Ry")
print("  Split norms: 0.15, 0.20")
