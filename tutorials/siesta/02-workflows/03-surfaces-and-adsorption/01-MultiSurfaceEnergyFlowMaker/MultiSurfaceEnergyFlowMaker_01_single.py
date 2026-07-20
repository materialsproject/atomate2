#!/usr/bin/env python
"""Single surface energy calculation."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.surface import MultiSurfaceEnergyFlowMaker

bulk = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

flow = MultiSurfaceEnergyFlowMaker(
    # dry_run=True,
    miller_indices=[(1, 0, 0)],
    slab_layers=4,
    vacuum_size=15.0,
)
workflow = flow.make(bulk)
results = run_locally(workflow, create_folders=True)

print("✓ Surface energy (100) complete")
