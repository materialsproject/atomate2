#!/usr/bin/env python
"""Complete basis convergence with automatic error handling."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.basis import CompleteBasisConvergenceMaker

# Load structure
structure = Structure.from_file("../../../00-structures/MgO_mp-1265_primitive.cif")

# Create complete basis convergence workflow with custodian
# Tests: 3 basis sizes × 2 energy_shifts × 2 split_norms = 12 calculations
maker = CompleteBasisConvergenceMaker(
    basis_sizes=["DZ", "DZP", "TZP"],  # Test 3 basis sizes
    energy_shifts=[0.01, 0.02],  # Test 2 energy shifts (Ry)
    split_norms=[0.15, 0.20],  # Test 2 split norms
    kpts=[3, 3, 3],  # K-points grid
    use_custodian=True,  # Enable automatic error handling
    custodian_max_errors=10,  # Allow up to 10 recovery attempts
)

# Create and run workflow
flow = maker.make(structure)
results = run_locally(flow, create_folders=True)

print("✓ Complete basis convergence with custodian: 3×2×2 = 12 calculations")
print("  Automatic SCF convergence recovery enabled")
print("  high success rate even with difficult parameter combinations")
