#!/usr/bin/env python
"""Example 2: Complete Material Study - The ultimate one-liner."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook
from jobflow import run_locally

# Create example structure (Silicon)
silicon = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Complete material study: electronic + mechanical + thermal properties
flow = RecipeBook.complete_material_study(
    silicon,
    # properties=["electronic", "mechanical", "thermal"]
    properties=[
        "electronic",
        "mechanical",
    ],
)

# Run locally (uncomment to execute)
# results = run_locally(flow, create_folders=True)

# Or dry-run mode (preview inputs)
dry_run_flow = RecipeBook.complete_material_study(silicon, dry_run=False)
results = run_locally(dry_run_flow, create_folders=True)
