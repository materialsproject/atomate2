#!/usr/bin/env python
"""Example 4: Selective Properties - Calculate only specific properties."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook
from jobflow import run_locally

# Create example structure (Silicon)
silicon = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Only electronic and mechanical (skip thermal to save time)
selective_flow = RecipeBook.complete_material_study(
    silicon,
    properties=["electronic", "mechanical"],  # Options: electronic, mechanical, thermal
)

# Run locally (uncomment to execute)
results = run_locally(selective_flow, create_folders=True)

# Or dry-run mode
# dry_run_flow = RecipeBook.complete_material_study(
#    silicon,
#    properties=["electronic", "mechanical"],
#    dry_run=True
# )
# results = run_locally(dry_run_flow, create_folders=True)
