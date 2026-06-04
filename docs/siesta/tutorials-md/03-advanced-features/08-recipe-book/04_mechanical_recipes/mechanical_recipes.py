#!/usr/bin/env python
"""Mechanical Properties Recipes - Elastic constants, bulk modulus, EOS."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook
from jobflow import run_locally

# Create silicon structure
silicon = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# ==============================================================================
# Example 1: Elastic Constants
# ==============================================================================
print("Example 1: Elastic Constants")
elastic_flow = RecipeBook.elastic_constants_workflow(silicon)
# Uncomment to run:
# results = run_locally(elastic_flow, create_folders=True)

# ==============================================================================
# Example 2: Equation of State (EOS) and Bulk Modulus
# ==============================================================================
print("Example 2: Equation of State and Bulk Modulus")
# NOTE: EOS calculation gives you BOTH the E(V) curve AND bulk modulus
# The old bulk_modulus_workflow() was removed as it was a duplicate of eos_workflow()
eos_flow = RecipeBook.eos_workflow(silicon, number_of_frames=7)
# Uncomment to run:
results = run_locally(eos_flow, create_folders=True)
# Output includes: bulk_modulus, equilibrium_volume, E0, EOS_fit

# ==============================================================================
# Example 3: Complete Mechanical Properties
# ==============================================================================
print("Example 3: Complete Mechanical Properties")
mechanical_flow = RecipeBook.mechanical_properties(silicon)

# Dry-run mode
print("\nRunning dry-run mode...")
dry_run = RecipeBook.mechanical_properties(silicon, dry_run=True)
# results = run_locally(dry_run, create_folders=True)
print("✅ Check folders for SIESTA input files")
