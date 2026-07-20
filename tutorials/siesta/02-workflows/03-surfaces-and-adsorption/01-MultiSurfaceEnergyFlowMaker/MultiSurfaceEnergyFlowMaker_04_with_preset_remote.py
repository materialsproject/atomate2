#!/usr/bin/env python
"""Surface energy calculation with different presets for bulk and slab.

This example shows how to apply different material-specific presets to bulk
and slab calculations. Bulk and surface calculations often need different
parameter sets for optimal accuracy and efficiency.

Presets provide optimized parameter sets for specific calculation types:
- bulk_metal, bulk_semiconductor: For bulk calculations
- surface_metal, surface_semiconductor: For surface calculations
- phonon_high_accuracy: For high-accuracy phonon calculations
- etc.
"""

from pymatgen.core import Structure
from atomate2.siesta.flows.surface import MultiSurfaceEnergyFlowMaker
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow_remote import submit_flow

# Load structure (using Si, but demonstrating the pattern for any material)
bulk = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Create separate makers for bulk and slab with different presets
bulk_maker = StaticMaker(use_custodian=True, custodian_max_errors=10)
slab_maker = StaticMaker(use_custodian=True, custodian_max_errors=10)

# Apply different presets to bulk and slab makers
bulk_maker = apply_tier_preset(bulk_maker, "bulk_metal")  # Bulk-optimized settings
slab_maker = apply_tier_preset(
    slab_maker, "surface_metal"
)  # Surface-optimized settings

# Create flow with custom makers
flow = MultiSurfaceEnergyFlowMaker(
    miller_indices=[
        (1, 1, 1),  # {111} family - triangular faces
    ],
    bulk_static_maker=bulk_maker,  # Bulk with 'bulk_metal' preset
    slab_static_maker=slab_maker,  # Slab with 'surface_metal' preset
    slab_layers=4,
    vacuum_size=15.0,
    symmetrize=False,
)

workflow = flow.make(bulk)

results = submit_flow(
    workflow,
    # project="atomate2siesta",
    # worker="agustina_worker",
    project="alberto",
    worker="cesga_worker",
    resources={
        # "partition": "RES", # for Agustina
        # "account": "icn2100", # for Agustina
        # "mem": "500GB",
        "mem_per_cpu": "4G",  # For cesga
        "nodes": 1,
        "ntasks_per_node": 24,
        "cpus_per_task": 1,
        # "ntasks": 24,
        "time": "24:00:00",
    },
)

print("✓ Surface energy calculation complete with different presets")
print("\nBulk calculation ('bulk_metal' preset):")
print("  - Tier: 'intermediate'")
print("  - OccupationFunction: 'MP' (Methfessel-Paxton)")
print("  - ElectronicTemperature: '300 K'")
print("  - Optimized for bulk metallic systems")
print("\nSlab calculation ('surface_metal' preset):")
print("  - Tier: 'intermediate'")
print("  - OccupationFunction: 'MP' (Methfessel-Paxton)")
print("  - ElectronicTemperature: '300 K'")
print("  - SCF.Mixer.Weight: 0.005 (for surfaces)")
print("  - Enhanced convergence for surface calculations")
print("\nAvailable presets:")
print("  - Bulk: bulk_metal, bulk_semiconductor")
print("  - Surface: surface_metal, surface_semiconductor")
print("  - Phonons: phonon_high_accuracy")
print("  - Relaxation: relax_standard, careful_relax")
print("  - and more... (see tiers.py for full list)")
