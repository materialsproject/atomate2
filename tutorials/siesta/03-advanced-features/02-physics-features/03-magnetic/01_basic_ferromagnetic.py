#!/usr/bin/env python
"""Basic ferromagnetic calculation with automatic DM.InitSpin generation.

This example shows the simplest way to perform magnetic calculations:
1. Set magnetic moments on the structure
2. Create a maker with Spin="polarized"
3. DM.InitSpin block is automatically generated!
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally

# Create simple Fe structure (BCC)
lattice = Lattice.cubic(2.87)  # Fe lattice parameter
structure = Structure(lattice, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])


# if we Set magnetic moments (Fe typically has ~2-3 μB) it will generate the blocks
structure.add_site_property("magmom", [2.5, 2.5])

print(f"Structure: {structure.composition}")
print(f"Magnetic moments: {structure.site_properties['magmom']}")


# Create maker - DM.InitSpin automatically generated from magmom!
maker = StaticMaker.scf(
    dry_run=True,
    user_params={
        "Spin": "polarized",
        "a2s_magnetic_ordering": "ferromagnetic",  # Explicit FM (default is AFM)
        "a2s_kpts": [4, 4, 4],
        "Mesh.Cutoff": "300 Ry",
        "PAO.BasisSize": "DZP",
    },
)

# Run calculation
job = maker.make(structure)
response = run_locally(job, create_folders=True)

print("\n✓ Calculation complete!")
print("✓ DM.InitSpin was automatically generated:")
print("  1  +2.5")
print("  2  +2.5")
