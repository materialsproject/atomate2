#!/usr/bin/env python
"""Thermal Properties Recipes - Phonons, QHA, thermal expansion."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook
from jobflow import run_locally

# Create silicon structure
# NOTE: from_spacegroup creates CONVENTIONAL cell (8 atoms for diamond cubic)
# For phonons, use PRIMITIVE cell (2 atoms) to avoid huge supercells!
silicon_conv = Structure.from_spacegroup(
    "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]
)
silicon = silicon_conv.get_primitive_structure()
print(
    f"Using primitive cell: {len(silicon)} atoms (conventional has {len(silicon_conv)} atoms)"
)

# ==============================================================================
# Example 1: Phonon Calculation
# ==============================================================================
print("\nExample 1: Phonon Workflow")
# Auto-calculated supercell for 2 atoms → (3,3,3) = 54 atoms
# phonon_flow = RecipeBook.phonon_workflow(silicon)
# OR manually specify smaller supercell with explicit k-points:
phonon_flow = RecipeBook.phonon_workflow(
    silicon,
    supercell_matrix=(2, 2, 2),
    auto_params=False,  # Disable automatic parameter detection
    user_params={
        "PAO.BasisSize": "DZP",
        "Mesh.Cutoff": "300 Ry",
        "a2s_kpts": [2, 2, 2],
    },
    phonon_user_params={"a2s_kpts": [2, 2, 2]},  # Specific k-points for phonon forces
)  # 16 atoms
# Note: phonon_user_params disables automatic k-point scaling
# Without it, k-points would be auto-scaled: [6,6,6] → [3,3,3] for 2x2x2 supercell
# Uncomment to run:
# results = run_locally(phonon_flow, create_folders=True)

# ==============================================================================
# Example 2: Quasi-Harmonic Approximation (QHA)
# ==============================================================================
print("Example 2: QHA Workflow (Thermal Expansion)")
qha_flow = RecipeBook.qha_workflow(
    silicon,
    supercell_matrix=(2, 2, 2),
    auto_params=False,  # Disable automatic parameter detection
    user_params={
        "PAO.BasisSize": "DZP",
        "Mesh.Cutoff": "300 Ry",
        "a2s_kpts": [2, 2, 2],
    },  # Fixed typo!
    phonon_user_params={"a2s_kpts": [2, 2, 2]},  # Specific k-points for phonon forces
)
# Uncomment to run:
# results = run_locally(qha_flow, create_folders=True)

# ==============================================================================
# Example 3: Grüneisen Parameters
# ==============================================================================
print("Example 3: Grüneisen Parameters")
gruneisen_flow = RecipeBook.gruneisen_workflow(
    silicon,
    supercell_matrix=(2, 2, 2),
    auto_params=False,  # Disable automatic parameter detection
    user_params={
        "PAO.BasisSize": "DZP",
        "Mesh.Cutoff": "300 Ry",
        "a2s_kpts": [2, 2, 2],
    },  # Fixed typo!
    phonon_user_params={"a2s_kpts": [1, 1, 1]},  # Specific k-points for phonon forces
)
# Uncomment to run:
# results = run_locally(gruneisen_flow, create_folders=True)

# ==============================================================================
# Example 4: Complete Thermal Properties
# ==============================================================================
print("Example 4: Complete Thermal Properties")
thermal_flow = RecipeBook.thermal_properties(
    silicon,
    supercell_matrix=(2, 2, 2),  # Now consistent with other workflows!
    auto_params=False,  # Disable automatic parameter detection
    user_params={
        "PAO.BasisSize": "DZP",
        "Mesh.Cutoff": "300 Ry",
        "a2s_kpts": [2, 2, 2],
    },
    phonon_user_params={"a2s_kpts": [1, 1, 1]},  # Specific k-points for phonon forces
    ignore_imaginary_modes=True,  # Use all volumes even with imaginary frequencies
)

# Uncomment to run:
results = run_locally(thermal_flow, create_folders=True)

# Dry-run mode
print("\nRunning dry-run mode...")
dry_run = RecipeBook.thermal_properties(silicon, dry_run=True)
# results = run_locally(dry_run, create_folders=True)
print("✅ Check folders for SIESTA input files")
