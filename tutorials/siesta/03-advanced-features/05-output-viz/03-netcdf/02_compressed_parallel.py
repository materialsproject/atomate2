#!/usr/bin/env python
"""
Tutorial 13, Example 2: Compressed NetCDF with Parallel I/O

Demonstrates NetCDF compression and parallel I/O for large-scale calculations.

Learning objectives:
- Configure compression level for file size optimization
- Enable parallel I/O for MPI calculations
- Set double precision for high-accuracy data
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally

print("=" * 70)
print("Tutorial 13, Example 2: Compressed NetCDF with Parallel I/O")
print("=" * 70)
print()

structure = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Configure compressed NetCDF with parallel I/O
user_params = {
    "CDF.Save": True,
    "CDF.Compress": 6,  # Level 6: good balance (4-5× compression)
    "CDF.MPI": True,  # Parallel I/O for MPI runs
    "CDF.Grid.Precision": "double",  # Double precision
    "SaveRho": True,
    "SaveDeltaRho": True,
    "PAO.BasisSize": "DZP",
    "a2s_kpts": [4, 4, 4],
}

print("NetCDF configuration:")
print(f"  Compression: Level {user_params['CDF.Compress']} (4-5× smaller files)")
print(f"  Parallel I/O: {user_params['CDF.MPI']} (faster with MPI)")
print(f"  Precision: {user_params['CDF.Grid.Precision']} (15 sig figs)")
print()

maker = StaticMaker.scf(user_params=user_params, tier="expert", dry_run=True)
job = maker.make(structure)
results = run_locally(job, create_folders=True)
