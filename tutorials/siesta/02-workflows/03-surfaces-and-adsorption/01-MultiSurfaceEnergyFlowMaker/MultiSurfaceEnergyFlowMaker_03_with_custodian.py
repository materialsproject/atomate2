#!/usr/bin/env python
"""Surface energy calculation with automatic error handling.

This example shows how to enable custodian error handling at the flow level,
which automatically propagates to all child makers (bulk and slab calculations).
"""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.surface import MultiSurfaceEnergyFlowMaker

bulk = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Enable custodian and set tier at flow level - automatically propagates to all calculations!
flow = MultiSurfaceEnergyFlowMaker(
    miller_indices=[
        (1, 1, 1),  # {111} family - triangular faces
    ],
    slab_layers=4,  # Number of atomic layers in slab
    vacuum_size=15.0,  # Vacuum spacing in Angstroms
    symmetrize=False,  # Use asymmetric slabs
    use_custodian=True,  # Enable automatic error handling
    custodian_max_errors=10,  # Allow up to 10 error corrections
    tier="dirty",  # Use basic tier for faster calculations
)

workflow = flow.make(bulk)
results = run_locally(workflow, create_folders=True)

print("✓ Surface energy calculation complete with custodian and tier='basic'")
print("Features automatically propagated to all calculations:")
print("  - Custodian: Automatic SCF convergence recovery")
print("  - Custodian: Geometry optimization error handling")
print("  - Custodian: Error logs saved to custodian.json")
print("  - Custodian: Up to 10 automatic error corrections")
print("  - Tier: 'basic' tier parameters for faster calculations")
