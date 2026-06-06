#!/usr/bin/env python
"""
EOS with Mesh.Cutoff via Tier Presets.

This tutorial demonstrates using tier presets with override_params
to customize Mesh.Cutoff while using production-quality parameter sets.

Key concepts:
- Tier presets: Pre-configured parameter sets (basic, intermediate, advanced, expert)
- override_params: Modify specific preset parameters
- Best practice for production calculations
- Ensures consistent, validated parameter combinations

Available presets:
- relax_standard: Standard relaxation (mesh_cutoff=200, kpts=[4,4,4])
- accurate_static: High-accuracy static (mesh_cutoff=300, kpts=[6,6,6])
- 2d_*: Specialized for 2D materials
- surface_*: Specialized for surface calculations
- magnetic_*: Specialized for magnetic systems
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.flows.eos import SiestaEosFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset

# Load structure
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Method : Preset with mesh cutoff override
# ===========================================
print("Method : Standard preset + custom Mesh.Cutoff")

# Create relax makers and apply tier preset
# SiestaEosFlowMaker is a Flow-level maker, so we apply presets to the child RelaxMakers
initial_relax_maker = RelaxMaker.variable_cell_relaxation()
initial_relax_maker = apply_tier_preset(
    initial_relax_maker,
    "relax_standard",  # Use standard relaxation preset
    override_params={
        "Mesh.Cutoff": "400 Ry",  # Override default mesh cutoff (preset uses 200 Ry)
    },
)

eos_relax_maker = RelaxMaker.fixed_cell_relaxation()
eos_relax_maker = apply_tier_preset(
    eos_relax_maker,
    "relax_standard",  # Use standard relaxation preset
    override_params={
        "Mesh.Cutoff": "450 Ry",  # Override default mesh cutoff (preset uses 200 Ry)
    },
)

# Create EOS maker with tier-preset makers
maker = SiestaEosFlowMaker(
    initial_relax_maker=initial_relax_maker,  # Use tier-preset initial maker
    eos_relax_maker=eos_relax_maker,  # Use tier-preset EOS maker
    # dry_run=True,
    linear_strain=(-0.05, 0.05),
    number_of_frames=7,
)

workflow = maker.make(structure)
results = run_locally(
    workflow, create_folders=True, root_dir="SiestaEosFlowMaker_04_mesh_cutoff_presets"
)

print("✓ EOS with relax_standard preset + custom mesh cutoff complete!")
print("  - Preset: relax_standard")
print("  - Mesh.Cutoff: 450 Ry (overridden from preset default)")
print("  - Other parameters: from preset (k-points, basis, etc.)")
