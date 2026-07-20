#!/usr/bin/env python
"""Example 3: Using Material-Specific Presets.

Presets provide optimized parameter sets for specific material types
(surfaces, 2D materials, magnetic systems, etc.). 26 presets available!
"""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook
from jobflow import run_locally

# Create bulk Pt structure (FCC)
pt = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.924), ["Pt"], [[0, 0, 0]])

print("=" * 80)
print("Example 3: Using Material-Specific Presets")
print("=" * 80)
print()

print("Some available presets:")
print("  • relax_standard, relax_high_accuracy")
print("  • surface_metal, surface_semiconductor")
print("  • phonon_standard, phonon_high_accuracy")
print("  • 2d_metal, 2d_semiconductor, 2d_vdw")
print("  • magnetic_afm, magnetic_correlated")
print("  • ... and 16 more")
print()
print("View all: atomate2siesta-presets list")
print()

# Surface energy workflow with metal surface preset
flow = RecipeBook.surface_energy_workflow(
    pt,
    miller_indices=[(1, 1, 1)],  # Pt(111) surface
    preset="surface_metal",  # Optimized for metal surfaces
    dry_run=True,
)

print("✅ Created surface energy workflow with 'surface_metal' preset")
print()
print("The preset automatically applies:")
print("  • Dense k-point mesh in surface plane")
print("  • Electronic smearing (MP or FD)")
print("  • Dipole correction for surfaces")
print("  • Appropriate basis set for metals")
print("  • Optimized SCF mixing for metals")
print()

# Run in dry-run mode
results = run_locally(flow, create_folders=True)

print("✅ Input files generated!")
print()
print("To see preset details:")
print("  atomate2siesta-presets show surface_metal")
print()
