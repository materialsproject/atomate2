#!/usr/bin/env python
"""Example 4: Combining Preset + User Parameters.

This is the RECOMMENDED approach for most use cases:
Start with a preset, then override specific parameters as needed.

Priority order: tier defaults → preset → user_params (highest)
"""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook
from jobflow import run_locally

# Create bulk Pt structure
pt = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.924), ["Pt"], [[0, 0, 0]])

print("=" * 80)
print("Example 4: Combining Preset + User Parameters (RECOMMENDED!)")
print("=" * 80)
print()

# Surface energy with preset + custom parameters
flow = RecipeBook.surface_energy_workflow(
    pt,
    miller_indices=[(1, 1, 1)],
    auto_params=False,  # Disable auto-detection for full control
    preset="surface_metal",  # Start with optimized preset
    user_params={  # Override specific parameters
        "a2s_kpts": [8, 8, 1],  # Custom k-mesh (denser than preset)
        "ElectronicTemperature": "500 K",  # Higher smearing
        "Mesh.Cutoff": "400 Ry",  # Increase cutoff
    },
    dry_run=True,
)

print("✅ Created workflow combining preset + custom parameters")
print()
print("Parameter sources:")
print("  1. surface_metal preset provides:")
print("     - Electronic smearing (MP)")
print("     - Dipole correction")
print("     - Metal-optimized SCF mixing")
print()
print("  2. user_params overrides:")
print("     - K-mesh: [8,8,1] (denser than preset)")
print("     - Electronic temperature: 500 K (higher)")
print("     - Mesh cutoff: 400 Ry (increased)")
print()
print("Result: Preset base config + your fine-tuning!")
print()

# Run in dry-run mode
results = run_locally(flow, create_folders=True)

print("✅ Input files generated!")
print("   Check siesta.fdf to see combined parameters")
print()
print("Best practice:")
print("  1. Choose appropriate preset for your material type")
print("  2. Override only what you need with user_params")
print("  3. Keeps simplicity while maintaining control!")
print()
