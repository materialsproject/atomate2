#!/usr/bin/env python
"""Convergence Testing Recipes - K-points and basis set convergence."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook
from jobflow import run_locally

# Create silicon structure
silicon = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# ==============================================================================
# Example 1: K-Point Convergence
# ==============================================================================
print("Example 1: K-Point Convergence Testing")
kpoints_flow = RecipeBook.kpoints_convergence(
    silicon, kpts_range=[2, 4, 6, 8, 10], tolerance=0.001
)
# Uncomment to run:
results = run_locally(kpoints_flow, create_folders=True)

# ==============================================================================
# Example 2: Basis Set Convergence (PAO parameters)
# ==============================================================================
print("Example 2: Basis Set Convergence Testing")
basis_flow = RecipeBook.basis_convergence(
    silicon,
    energy_shifts=[0.01, 0.02, 0.05, 0.10, 0.15],  # Ry
    basis_size="DZP",
)
# Uncomment to run: results = run_locally(basis_flow, create_folders=True)

# ==============================================================================
# Example 3: Complete Convergence (K-points + Basis + Mesh Cutoff)
# ==============================================================================
print("Example 3: Complete Convergence Testing")
combined_flow = RecipeBook.complete_convergence(silicon)

# Dry-run mode
print("\nRunning k-point convergence dry-run...")
dry_run = RecipeBook.kpoints_convergence(silicon, dry_run=True)
# results = run_locally(dry_run, create_folders=True)
print("✅ Check folders for SIESTA input files")
