#!/usr/bin/env python
"""Electronic Properties Recipes - Band structure and DOS calculations."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook
from jobflow import run_locally

# Create silicon structure
silicon = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# ==============================================================================
# Example 1: Band Structure
# ==============================================================================
print("Example 1: Band Structure Workflow")
bands_flow = RecipeBook.band_structure_workflow(silicon)
# Uncomment to run:
# results = run_locally(bands_flow, create_folders=True)

# ==============================================================================
# Example 2: Density of States (DOS)
# ==============================================================================
print("Example 2: DOS Workflow")
dos_flow = RecipeBook.dos_workflow(silicon)
# Uncomment to run:
# results = run_locally(dos_flow, create_folders=True)

# ==============================================================================
# Example 3: Complete Electronic Properties (Bands + DOS)
# ==============================================================================
print("Example 3: Complete Electronic Properties")
electronic_flow = RecipeBook.electronic_properties(silicon)
# Uncomment to run:
# results = run_locally(electronic_flow, create_folders=True)

# Dry-run mode (preview inputs)
print("\nRunning dry-run mode...")
dry_run = RecipeBook.electronic_properties(silicon, dry_run=True)
results = run_locally(dry_run, create_folders=True)
print("✅ Check folders for SIESTA input files")
