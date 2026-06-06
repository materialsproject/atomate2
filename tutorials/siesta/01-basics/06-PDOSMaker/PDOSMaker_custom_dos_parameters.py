#!/usr/bin/env python
"""
Tutorial: Custom DOS Parameters

This tutorial demonstrates how to customize DOS calculation parameters including:
- Energy range
- Energy resolution (smearing width)
- DOS and PDOS k-point grids

Use Case:
---------
When you need:
- Higher resolution DOS (smaller energy steps)
- Wider energy range to see deeper valence/conduction bands
- Denser k-point sampling for smoother DOS curves
- Custom energy grid around specific features

Example:
--------
Silicon with custom DOS parameters for high-quality DOS.
"""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import PDOSMaker
from jobflow import run_locally

# Load silicon structure
structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")

print("\n" + "=" * 70)
print("Tutorial: Custom DOS Parameters")
print("=" * 70)
print(f"\nStructure: {structure.composition}")
print(f"Number of atoms: {len(structure)}")

# Create PDOS maker with custom DOS parameters
pdos_maker = PDOSMaker.pdos_calculation(
    dry_run=True,
    user_params={
        # SCF parameters
        "a2s_kpts": [10, 10, 10],  # Dense SCF k-grid
        "Mesh.Cutoff": "400 Ry",  # High cutoff for accuracy
        # Custom DOS energy range and resolution
        # Note: These are handled by the DensityOfStatesAndBandStructure dataclass
        # Default: energy_range=(-20.0, 10.0), smearing_width=0.2
        # Custom PDOS k-grid (denser than default 10x10x10)
        "%block DOS.kgrid.MonkhorstPack": [
            "15 0 0 0.0",
            "0 15 0 0.0",
            "0 0 15 0.0",
        ],
    },
)

job = pdos_maker.make(structure)
response = run_locally(job, create_folders=True)
