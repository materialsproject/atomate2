#!/usr/bin/env python
"""Example 3: Quick Characterization - Fast preliminary study (~1-2 hours)."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook
from jobflow import run_locally

# Create example structure (Silicon)
silicon = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Fast preliminary study
quick_flow = RecipeBook.quick_characterization(silicon)

# Run locally (uncomment to execute)
# results = run_locally(quick_flow, create_folders=True)

# Or dry-run mode
dry_run_flow = RecipeBook.quick_characterization(silicon, dry_run=True)
results = run_locally(dry_run_flow, create_folders=True)
