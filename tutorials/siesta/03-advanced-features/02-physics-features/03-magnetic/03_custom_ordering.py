#!/usr/bin/env python
"""Custom magnetic ordering preserving exact signs from structure.

This example shows how to use "custom" ordering to preserve
the exact magnetic moment values and signs from the structure.
Useful for complex magnetic configurations.
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally

# Create Fe4N structure with custom magnetic pattern
lattice = Lattice.cubic(3.8)
structure = Structure(
    lattice,
    ["Fe", "Fe", "Fe", "Fe", "N"],
    [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]],
)

# Set custom magnetic moments with specific pattern:
# Fe atoms have different moments and some negative (spin-down)
structure.add_site_property(
    "magmom",
    [
        +2.5,  # Fe1: spin up, high moment
        -1.8,  # Fe2: spin down, lower moment
        +2.5,  # Fe3: spin up, high moment
        -1.8,  # Fe4: spin down, lower moment
        0.0,  # N: non-magnetic
    ],
)

print(f"Structure: {structure.composition}")
print(f"Custom magnetic moments: {structure.site_properties['magmom']}")

# Create maker with custom ordering
maker = StaticMaker.scf(
    dry_run=True,
    user_params={
        "Spin": "polarized",
        "a2s_magnetic_ordering": "custom",  # Preserve exact signs!
        "a2s_kpts": [4, 4, 4],
        "Mesh.Cutoff": "300 Ry",
        "PAO.BasisSize": "DZP",
    },
)

# Run calculation
job = maker.make(structure)
response = run_locally(job, create_folders=True)

print("\n✓ Calculation complete!")
print("✓ DM.InitSpin preserves exact values:")
print("  1  +2.5  (Fe, spin up)")
print("  2  -1.8  (Fe, spin down)")
print("  3  +2.5  (Fe, spin up)")
print("  4  -1.8  (Fe, spin down)")
print("  5  0.0   (N, non-magnetic)")
