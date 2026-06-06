#!/usr/bin/env python
"""Example 7: Recipe-Specific Workflow Parameters.

Each recipe accepts workflow-specific parameters in addition to
FDF parameters (user_params). This example shows the most common ones.
"""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook
from jobflow import run_locally

# Create silicon structure
silicon = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Create Pt structure for surface
pt = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.924), ["Pt"], [[0, 0, 0]])

print("=" * 80)
print("Example 7: Recipe-Specific Workflow Parameters")
print("=" * 80)
print()

# ==============================================================================
# Part 1: Phonon Workflow - Supercell Parameters
# ==============================================================================
print("Part 1: Phonon Workflow - Supercell Parameters")
print("-" * 80)
print()

flow_phonon = RecipeBook.phonon_workflow(
    silicon,
    supercell_matrix=(2, 2, 2),  # Explicit 2x2x2 supercell
    # OR use min_length instead:
    # min_length=15.0,              # Auto supercell ≥15 Å
    auto_params=False,  # Disable auto-detection
    user_params={
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [4, 4, 4],
    },
    dry_run=True,
)

print("✅ Phonon workflow created")
print("   Workflow parameter: supercell_matrix=(2,2,2)")
print("   FDF parameters: PAO.BasisSize, a2s_kpts")
print()

results1 = run_locally(flow_phonon, create_folders=True)
print("✅ Input files generated")
print()

# ==============================================================================
# Part 2: Surface Energy Workflow - Surface Parameters
# ==============================================================================
print("\nPart 2: Surface Energy Workflow - Surface Parameters")
print("-" * 80)
print()

flow_surface = RecipeBook.surface_energy_workflow(
    pt,
    miller_indices=[(1, 0, 0), (1, 1, 0), (1, 1, 1)],  # Which surfaces
    slab_layers=5,  # Slab thickness (layers)
    vacuum=15.0,  # Vacuum spacing (Å)
    auto_params=False,  # Disable auto-detection
    user_params={
        "a2s_kpts": [6, 6, 1],  # Surface k-mesh
        "Mesh.Cutoff": "350 Ry",
    },
    dry_run=True,
)

print("✅ Surface energy workflow created")
print("   Workflow parameters:")
print("     - miller_indices: [(1,0,0), (1,1,0), (1,1,1)]")
print("     - slab_layers: 5")
print("     - vacuum: 15.0 Å")
print("   FDF parameters: a2s_kpts, Mesh.Cutoff")
print()

results2 = run_locally(flow_surface, create_folders=True)
print("✅ Input files generated")
print()

# ==============================================================================
# Part 3: EOS Workflow - Number of Volume Points
# ==============================================================================
print("\nPart 3: EOS Workflow - Number of Volume Points")
print("-" * 80)
print()

flow_eos = RecipeBook.eos_workflow(
    silicon,
    number_of_frames=9,  # 9 volume points
    auto_params=False,  # Disable auto-detection
    user_params={
        "PAO.BasisSize": "TZP",
        "a2s_kpts": [8, 8, 8],
        "Mesh.Cutoff": "400 Ry",
    },
    dry_run=True,
)

print("✅ EOS workflow created")
print("   Workflow parameter: number_of_frames=9")
print("   FDF parameters: PAO.BasisSize, a2s_kpts, Mesh.Cutoff")
print()

results3 = run_locally(flow_eos, create_folders=True)
print("✅ Input files generated")
print()

# ==============================================================================
# Summary
# ==============================================================================
print("=" * 80)
print("SUMMARY: Recipe-Specific Parameters")
print("=" * 80)
print()
print("Each recipe has its own workflow parameters:")
print()
print("Phonon:")
print("  supercell_matrix=(2,2,2) or min_length=15.0")
print()
print("Surface Energy:")
print("  miller_indices=[(1,0,0), ...]")
print("  slab_layers=5")
print("  vacuum=15.0")
print()
print("EOS:")
print("  number_of_frames=9")
print()
print("Adsorption Scanning:")
print("  grid_density=(7,7)")
print("  height_above_surface=2.0")
print()
print("These are SEPARATE from FDF parameters (user_params)!")
print("Workflow params control what calculations to run.")
print("FDF params (user_params) control HOW to run them.")
print()
