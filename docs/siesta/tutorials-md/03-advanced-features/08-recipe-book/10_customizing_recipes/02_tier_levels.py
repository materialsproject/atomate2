#!/usr/bin/env python
"""Example 2: Using Tier Levels for Computational Rigor.

Tier levels provide predefined parameter sets for different
computational requirements (fast screening vs. publication quality).
"""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook
from jobflow import run_locally

# Create silicon structure
silicon = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

print("=" * 80)
print("Example 2: Using Tier Levels")
print("=" * 80)
print()

print("Available tier levels:")
print("  • basic_dirty  - Ultra-fast (testing, ~1 min)")
print("  • basic        - Fast (screening, ~5-10 min)")
print("  • intermediate - Balanced (default, ~15-30 min)")
print("  • advanced     - High accuracy (publication, ~1-2 hours)")
print("  • expert       - Maximum precision (benchmarks, ~4-8 hours)")
print()

# Example: Advanced tier for publication-quality results
flow = RecipeBook.band_structure_workflow(
    silicon,
    tier="advanced",  # High-accuracy parameters
    dry_run=True,
)

print("✅ Created band structure workflow with 'advanced' tier")
print()
print("Advanced tier automatically sets:")
print("  • Dense k-point meshes")
print("  • Large basis sets (typically DZP or TZP)")
print("  • Tight SCF convergence criteria")
print("  • High mesh cutoff values")
print()

# Run in dry-run mode
results = run_locally(flow, create_folders=True)

print("✅ Input files generated!")
print("   Compare siesta.fdf with default (intermediate) tier")
print()
