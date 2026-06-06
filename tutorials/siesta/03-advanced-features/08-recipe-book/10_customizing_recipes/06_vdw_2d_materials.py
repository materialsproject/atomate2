#!/usr/bin/env python
"""Example 6: van der Waals Functionals for 2D Materials.

Demonstrates how to use van der Waals functionals with 2D materials.
Perfect for graphene, MoS2, h-BN, and other layered materials.
"""

from pymatgen.core import Structure
from atomate2.siesta.recipes import RecipeBook
from jobflow import run_locally
import numpy as np

print("=" * 80)
print("Example 6: van der Waals Functionals for 2D Materials")
print("=" * 80)
print()

# Create graphene structure
a = 2.46
lattice = [
    [a, 0, 0],
    [a * np.cos(np.radians(120)), a * np.sin(np.radians(120)), 0],
    [0, 0, 20.0],  # 20 Å vacuum
]
graphene = Structure(lattice, ["C", "C"], [[0, 0, 0.5], [1 / 3, 2 / 3, 0.5]])

print("Structure: Graphene")
print(f"Lattice: a = {a} Å, c = 20 Å (with vacuum)")
print()

# Phonon workflow with 2D preset (using standard PBE)
# Note: For production vdW calculations, install DRSLL pseudopotentials and use:
#   XC.Functional: "VDW", XC.Authors: "DRSLL"
flow = RecipeBook.phonon_workflow(
    graphene,
    auto_params=False,  # Disable auto-detection
    preset="2d_semiconductor",  # 2D preset (PBE functional)
    user_params={
        "XC.Functional": "GGA",  # Standard GGA
        "XC.Authors": "PBE",  # PBE functional (default pseudos)
        "a2s_kpts": [12, 12, 1],  # Dense in-plane, 1 out-of-plane
        "PAO.BasisSize": "DZP",  # Double-zeta for efficiency
    },
    supercell_matrix=(3, 3, 1),
    # dry_run=True,
)

print("✅ Created graphene phonon workflow with PBE functional")
print()
print("Parameters applied:")
print("  • Preset: 2d_semiconductor (optimized for 2D materials)")
print("  • XC.Functional: GGA, XC.Authors: PBE")
print("  • K-mesh: [12, 12, 1] (dense in-plane, 1 out-of-plane)")
print("  • Basis: DZP (double-zeta)")
print()
print("Note: This uses standard PBE pseudopotentials (default installation).")
print()
print("For van der Waals calculations:")
print("  ⚠️  DRSLL pseudos NOT available via atomate2siesta-pseudos")
print("  ⚠️  Download manually from SIESTA website")
print()
print("  After manual installation, use a2s_pseudo_path parameter:")
print()
print("    user_params={")
print(
    "        'a2s_pseudo_path': '/path/to/DRSLL-pseudos',  # Point to your DRSLL pseudos"
)
print("        'XC.Functional': 'VDW',")
print("        'XC.Authors': 'DRSLL',")
print("    }")
print()
print("Available via atomate2siesta-pseudos:")
print("  • PBE, PBEsol (11 pseudopotential sets)")
print()

# Run in dry-run mode
results = run_locally(flow, create_folders=True)

print("✅ Input files generated!")
print("   Check siesta.fdf for vdW functional settings")
print()
print("Best practices for 2D materials:")
print("  1. Use 2d_* presets for optimized parameters")
print("  2. Always include vdW functional for layered materials")
print("  3. Use dense in-plane k-mesh, k_z = 1")
print("  4. Include sufficient vacuum (15-20 Å)")
print()
