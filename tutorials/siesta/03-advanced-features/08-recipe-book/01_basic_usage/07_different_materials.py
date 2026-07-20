#!/usr/bin/env python
"""Example 7: Different Material Types - Automatic detection and parameter adjustment."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook
from jobflow import run_locally

# Semiconductor (Silicon)
silicon = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Metal (Aluminum)
al_lattice = Lattice.cubic(4.05)
aluminum = Structure(al_lattice, ["Al"], [[0, 0, 0]])

# Recipe Book automatically detects material type and adjusts parameters
print("Silicon analysis:")
RecipeBook.print_analysis(silicon)

print("\nAluminum analysis:")
RecipeBook.print_analysis(aluminum)

# Create workflow (Recipe Book uses metal-specific parameters for Al)
al_flow = RecipeBook.electronic_properties(aluminum, dry_run=False)
results = run_locally(al_flow, create_folders=True)
