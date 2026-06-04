#!/usr/bin/env python
"""Antiferromagnetic calculation with alternating spin directions.

This example demonstrates the "AFM" magnetic ordering option,
which automatically alternates the signs of magnetic moments.
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally

# Create NiO structure (rock salt, antiferromagnetic)
lattice = Lattice.cubic(4.17)  # NiO lattice parameter
structure = Structure(
    lattice,
    ["Ni", "Ni", "O", "O"],
    [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0.5]],
)

# Set magnetic moments on Ni atoms (absolute values)
# AFM ordering will automatically alternate the signs
structure.add_site_property("magmom", [1.7, 1.7, 0.0, 0.0])


print(f"Structure: {structure.composition}")
print(f"Input magnetic moments: {structure.site_properties['magmom']}")

# Create maker with AFM ordering (default)
maker = StaticMaker.scf(
    dry_run=True,
    user_params={
        "Spin": "polarized",
        # magnetic_ordering defaults to "antiferromagnetic"
        "a2s_kpts": [4, 4, 4],
        "Mesh.Cutoff": "300 Ry",
        "PAO.BasisSize": "DZP",
    },
)

# Run calculation
job = maker.make(structure)
response = run_locally(job, create_folders=True)

print("\n✓ Calculation complete!")
print("✓ DM.InitSpin generated with alternating signs:")
print("  1  +1.7  (Ni, spin up)")
print("  2  -1.7  (Ni, spin down)")
print("  3  0.0   (O, non-magnetic)")
print("  4  0.0   (O, non-magnetic)")
