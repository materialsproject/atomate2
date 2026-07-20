#!/usr/bin/env python
"""Example 6: Dry-Run Mode - Preview inputs without running calculations."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook
from jobflow import run_locally

# Create example structure (Silicon)
silicon = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Dry-run mode: Generate SIESTA input files without running
dry_run_flow = RecipeBook.complete_material_study(silicon, dry_run=True)
results = run_locally(dry_run_flow, create_folders=True)

print("✅ Dry-run complete! Check folders for SIESTA input files (.fdf)")
