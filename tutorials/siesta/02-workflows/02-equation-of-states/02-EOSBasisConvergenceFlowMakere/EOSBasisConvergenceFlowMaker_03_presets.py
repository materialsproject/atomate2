#!/usr/bin/env python
"""
EOS Basis Convergence with Tier Presets.

This tutorial demonstrates using tier presets with EOSBasisConvergenceFlowMaker.
Since EOSBasisConvergenceFlowMaker creates its own internal makers, we use
presets via powerups to apply production-quality parameter sets.

Key concepts:
- Tier presets: Pre-configured parameter sets for different use cases
- get_tier_preset(): Retrieve preset configuration
- Apply preset parameters via powerups
- Best practice for production calculations

Available presets:
- relax_standard: Standard relaxation (mesh_cutoff=200 Ry, kpts=[4,4,4])
- relax_high_accuracy: High-accuracy relaxation
- 2d_semiconductor: Optimized for 2D semiconductors like MoS2
- surface_metal: Optimized for metallic surfaces
- magnetic_*: Optimized for magnetic systems
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.flows.basis import EOSBasisConvergenceFlowMaker
from atomate2.siesta.powerups import update_user_siesta_settings
from atomate2.siesta.sets.tiers import get_tier_preset

# Load structure (MoS2 - 2D semiconductor)
structure = Structure.from_file("../../../00-structures/MoS2.cif")

# =============================================================================
# Method: Apply tier preset via powerups
# =============================================================================
print("EOS Basis Convergence with Tier Presets")
print("=" * 50)

# Step 1: Get the tier preset configuration
# Available presets: relax_standard, relax_high_accuracy, 2d_semiconductor, etc.
preset_name = "2d_semiconductor"
preset = get_tier_preset(preset_name)

print(f"\nUsing preset: {preset_name}")
print(f"  Description: {preset['description']}")
print(f"  Tier: {preset['tier']}")
print(f"  Recommended params: {preset['recommended_params']}")

# Step 2: Create the basic workflow
maker = EOSBasisConvergenceFlowMaker(
    basis_sets=["SZ", "DZ", "DZP", "DZDP"],
    linear_strain=(-0.05, 0.05),
    number_of_frames=7,
    dry_run=True,  # Use dry_run for testing
    dry_run_output_dir="03_presets",
)

workflow = maker.make(structure)

# Step 3: Apply preset parameters via powerups
# Merge preset recommended_params with custom overrides
preset_params = preset["recommended_params"].copy()

# Add custom overrides on top of preset
custom_overrides = {
    "Mesh.Cutoff": "350 Ry",  # Override preset mesh cutoff for higher accuracy
}
preset_params.update(custom_overrides)

workflow = update_user_siesta_settings(workflow, preset_params)

# Step 4: Run the workflow
results = run_locally(workflow, create_folders=True, root_dir="03_presets")

print("\n✓ EOS basis convergence with tier preset complete!")
print(f"  - Preset: {preset_name}")
print(f"  - Tier: {preset['tier']}")
print("  - Basis sets: SZ, DZ, DZP, DZDP")
print("  - Mesh.Cutoff: 350 Ry (overridden from preset)")
print("  - Other parameters: from preset configuration")
print("  - Total calculations: 4 basis × 7 volumes = 28 jobs")


# =============================================================================
# Alternative: Multiple presets comparison
# =============================================================================
print("\n" + "=" * 50)
print("Available presets for reference:")
print("=" * 50)

# List some useful presets for different materials
presets_to_show = [
    "relax_standard",
    "relax_high_accuracy",
    "2d_semiconductor",
    "2d_metal",
    "relax_bulk_semiconductor",
]

for preset_name in presets_to_show:
    try:
        p = get_tier_preset(preset_name)
        print(f"\n{preset_name}:")
        print(f"  Description: {p['description']}")
        print(f"  Tier: {p['tier']}")
        print(f"  Params: {list(p['recommended_params'].keys())}")
    except ValueError:
        print(f"\n{preset_name}: Not available")
