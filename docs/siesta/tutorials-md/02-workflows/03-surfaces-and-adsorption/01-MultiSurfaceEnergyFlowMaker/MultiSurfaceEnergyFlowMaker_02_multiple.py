#!/usr/bin/env python
"""Multiple surface orientation comparison.

This example compares surface energies across different Miller indices
to identify the most stable surface orientation.
"""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.surface import MultiSurfaceEnergyFlowMaker
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset

bulk = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Create makers with custodian enabled
bulk_maker = StaticMaker(use_custodian=True, custodian_max_errors=10)
slab_maker = StaticMaker(use_custodian=True, custodian_max_errors=10)
slab_maker = apply_tier_preset(slab_maker, "surface_semiconductor")

# Compare multiple surface orientations
flow = MultiSurfaceEnergyFlowMaker(
    # dry_run=True,
    miller_indices=[
        (1, 0, 0),  # {100} family - cubic faces
        (1, 1, 0),  # {110} family - rectangular faces
        (1, 1, 1),  # {111} family - triangular faces
    ],
    bulk_static_maker=bulk_maker,  # Use custodian-enabled maker
    slab_static_maker=slab_maker,  # Use custodian-enabled maker
    slab_layers=4,  # Number of atomic layers in slab
    vacuum_size=15.0,  # Vacuum spacing in Angstroms
    symmetrize=False,  # Use asymmetric slabs to explore all terminations
    plot_results=True,  # Generate comparison plots
    write_summary=True,  # Write text summary
)
workflow = flow.make(bulk)
results = run_locally(workflow, create_folders=True)

print("✓ Multiple surface orientation comparison complete")
print("Results will include:")
print("  - Surface energies for each orientation")
print("  - Comparison plots")
print("  - Stability ranking")
