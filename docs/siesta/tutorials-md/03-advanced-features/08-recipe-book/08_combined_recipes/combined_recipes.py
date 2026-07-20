#!/usr/bin/env python
"""Combined Recipes - High-throughput and multi-property workflows."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook

# Create multiple structures for screening
# IMPORTANT: Use PRIMITIVE cell for phonon calculations (2 atoms vs 8 atoms)
silicon_conv = Structure.from_spacegroup(
    "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]
)
silicon = silicon_conv.get_primitive_structure()
print(
    f"Using primitive Si: {len(silicon)} atoms (conventional: {len(silicon_conv)} atoms)"
)

ge_lattice = Lattice.cubic(5.66)
germanium_conv = Structure(ge_lattice, ["Ge", "Ge"], [[0, 0, 0], [0.25, 0.25, 0.25]])
germanium = germanium_conv  # Already primitive (2 atoms)

# ==============================================================================
# Example 1: Complete Material Study (The Ultimate Workflow)
# ==============================================================================
print("\nExample 1: Complete Material Study")
complete_flow = RecipeBook.complete_material_study(
    silicon,
    properties=["electronic", "mechanical", "thermal"],
    auto_params=False,  # Disable auto-detection for explicit control
    user_params={
        "PAO.BasisSize": "DZP",
        "Mesh.Cutoff": "300 Ry",
        "a2s_kpts": [2, 2, 2],  # For relax/static
    },
    phonon_user_params={
        "a2s_kpts": [2, 2, 2]  # Explicit k-points for phonon forces (no auto-scaling)
    },
    supercell_matrix=(2, 2, 2),  # 16-atom supercell for phonons
    ignore_imaginary_modes=True,  # Handle imaginary frequencies gracefully
)
# Uncomment to run:
# results = run_locally(complete_flow, create_folders=True)

# ==============================================================================
# Example 2: High-Throughput Screening
# ==============================================================================
print("\nExample 2: High-Throughput Screening")
structures = [silicon, germanium]
screening_flows = [RecipeBook.quick_characterization(struct) for struct in structures]
# Uncomment to run:
# for flow in screening_flows:
#     results = run_locally(flow, create_folders=True)

# ==============================================================================
# Example 3: Property Comparison Workflow with Convergence
# ==============================================================================
print("\nExample 3: Property Comparison with Convergence Testing")
comparison_flow = RecipeBook.complete_material_study(
    silicon,
    properties=["electronic", "mechanical"],
    test_convergence=True,  # Automatically extract optimal k-points and cutoff
)
# Uncomment to run:
# results = run_locally(comparison_flow, create_folders=True)

# Dry-run mode
print("\nRunning complete study dry-run...")
dry_run = RecipeBook.complete_material_study(silicon, dry_run=True)
# results = run_locally(dry_run, create_folders=True)
print("✅ Check folders for SIESTA input files")
