#!/usr/bin/env python
"""
Tutorial 13, Example 1: Basic NetCDF Output

Demonstrates enabling NetCDF format for grid-based data output.

Learning objectives:
- Enable NetCDF output with CDF.Save
- Save charge density in NetCDF format
- Understand NetCDF advantages over binary
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally

print("=" * 70)
print("Tutorial 13, Example 1: Basic NetCDF Output")
print("=" * 70)
print()

# Create Si structure
structure = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Configure NetCDF output
user_params = {
    "CDF.Save": True,  # Enable NetCDF output
    "SaveRho": True,  # Save charge density
    "PAO.BasisSize": "DZP",
    "a2s_kpts": [4, 4, 4],
    "Mesh.Cutoff": "200 Ry",
}

print("NetCDF configuration:")
print(f"  CDF.Save: {user_params['CDF.Save']}")
print("  Compression: 0 (none, default)")
print("  Precision: single (default)")
print()

# maker = StaticMaker.scf(user_params=user_params, tier="expert", dry_run=True)
maker = StaticMaker.scf(user_params=user_params, dry_run=True)
job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("NetCDF output will be saved to: systemLabel.nc")
print()
print("Advantages:")
print("  ✅ Portable across platforms")
print("  ✅ Self-describing (metadata included)")
print("  ✅ Standard tools (Python xarray, ncdump)")
print("  ✅ Optional compression")
