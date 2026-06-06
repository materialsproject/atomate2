#!/usr/bin/env python
"""
Full Basis Parameter Optimization with Tier Presets.

This tutorial demonstrates using tier presets with EOSFullBasisConvergenceFlowMaker.
Since the flow maker creates its own internal makers, we use presets via powerups
to apply production-quality parameter sets.

Key concepts:
- Tier presets: Pre-configured parameter sets for different use cases
- get_tier_preset(): Retrieve preset configuration
- Apply preset parameters via powerups
- Best practice for production calculations

Available presets:
- relax_standard: Standard relaxation (mesh_cutoff=200 Ry, kpts=[4,4,4])
- relax_high_accuracy: High-accuracy relaxation
- bulk_semiconductor: Optimized for bulk semiconductors
- bulk_metal: Optimized for bulk metals
- magnetic_*: Optimized for magnetic systems

When to use presets with EOSFullBasisConvergenceFlowMaker:
- Production calculations requiring validated parameter sets
- Material-specific settings (semiconductors, metals, magnetic systems)
- Reproducible, documented workflows
- Combining preset parameters with basis parameter optimization
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.flows.eos import EOSFullBasisConvergenceFlowMaker
from atomate2.siesta.powerups import update_user_siesta_settings
from atomate2.siesta.sets.tiers import get_tier_preset

# Load structure (Si - bulk semiconductor)
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# =============================================================================
# Method: Apply tier preset via powerups
# =============================================================================
print("Full Basis Convergence with Tier Presets")
print("=" * 50)

# Step 1: Get the tier preset configuration
preset_name = "bulk_semiconductor"
preset = get_tier_preset(preset_name)

print(f"\nUsing preset: {preset_name}")
print(f"  Description: {preset['description']}")
print(f"  Tier: {preset['tier']}")
print(f"  Recommended params: {preset['recommended_params']}")

# Step 2: Create the basic workflow with custodian
maker = EOSFullBasisConvergenceFlowMaker(
    dry_run=True,  # Use dry_run for testing
    basis_sizes=["DZ", "DZP"],
    energy_shifts=[0.01, 0.015],
    split_norms=[0.15, 0.20],
    linear_strain=(-0.04, 0.04),
    number_of_frames=5,
    a2s_kpts=[4, 4, 4],
    # Custodian error handling
    use_custodian=True,
    custodian_max_errors=10,
)

workflow = maker.make(structure)

# Step 3: Apply preset parameters via powerups
# Merge preset recommended_params with custom overrides
preset_params = preset["recommended_params"].copy()

# Add custom overrides on top of preset
custom_overrides = {
    "Mesh.Cutoff": "400 Ry",  # Override preset mesh cutoff for higher accuracy
    "SCF.DM.Tolerance": 1.0e-5,  # Tighter SCF convergence
}
preset_params.update(custom_overrides)

workflow = update_user_siesta_settings(workflow, preset_params)

# Step 4: Run the workflow
results = run_locally(workflow, create_folders=True, root_dir="04_presets")

print("\n✓ Full basis convergence with tier preset complete!")
print(f"  - Preset: {preset_name}")
print(f"  - Tier: {preset['tier']}")
print("  - Basis sizes: DZ, DZP")
print("  - Energy shifts: 0.01, 0.015 Ry")
print("  - Split norms: 0.15, 0.20")
print("  - Total combinations: 2 × 2 × 2 = 8 EOS calculations")
print("  - Mesh.Cutoff: 400 Ry (overridden from preset)")
print("  - Other parameters: from preset configuration")
print("  - Custodian: enabled for automatic error recovery")


# =============================================================================
# Show available presets for reference
# =============================================================================
print("\n" + "=" * 50)
print("Available presets for bulk materials:")
print("=" * 50)

presets_to_show = [
    "relax_standard",
    "relax_high_accuracy",
    "bulk_semiconductor",
    "bulk_metal",
]

for pname in presets_to_show:
    try:
        p = get_tier_preset(pname)
        print(f"\n{pname}:")
        print(f"  Description: {p['description']}")
        print(f"  Tier: {p['tier']}")
        print(f"  Params: {list(p['recommended_params'].keys())}")
    except ValueError:
        print(f"\n{pname}: Not available")
