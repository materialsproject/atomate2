#!/usr/bin/env python
"""
Tutorial 10, Example 1: Basic Charge Density Output

Demonstrates saving total and deformation charge densities in binary format.

This example shows how to configure SIESTA to output charge density grids
for visualization and analysis.

Learning objectives:
- Save total electron charge density
- Save deformation charge density (SCF - atomic)
- Use binary format (default, fastest)
- Access grid files after calculation
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally

print("=" * 70)
print("Tutorial 10, Example 1: Basic Charge Density Output")
print("=" * 70)
print()

# Create a simple silicon structure
print("Creating Si structure...")
structure = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])
print(f"  Formula: {structure.composition.reduced_formula}")
print(f"  Space group: {structure.get_space_group_info()}")
print()

# Configure grid output using SIESTA FDF parameter names
print("Configuring grid output...")
user_params = {
    # Charge densities
    "SaveRho": True,  # Total electron charge density
    "SaveDeltaRho": True,  # Deformation charge density (SCF - atomic)
    # File format (binary is default, fastest)
    "SaveGridFunc.Format": "binary",
    # Basic calculation parameters
    "PAO.BasisSize": "DZP",
    "a2s_kpts": [4, 4, 4],
    "Mesh.Cutoff": "200 Ry",
}
print("  Grid output parameters:")
print(f"    SaveRho: {user_params['SaveRho']}")
print(f"    SaveDeltaRho: {user_params['SaveDeltaRho']}")
print(f"    SaveGridFunc.Format: {user_params['SaveGridFunc.Format']}")
print()

# Create static calculation job
print("Creating static calculation job...")
maker = StaticMaker.scf(user_params=user_params, dry_run=True)
job = maker.make(structure)
print(f"  Job name: {job.name}")
print()

# Run calculation
print("Running calculation...")
print("  This will generate grid files in the calculation directory")
print("  Grid files: systemLabel.RHO (total), systemLabel.DRHO (deformation)")
print()

# Uncomment to run:
results = run_locally(job, create_folders=True)
#
# if results:
#     print("Calculation complete!")
#     print()
#     print("Output files:")
#     print("  - systemLabel.RHO: Total electron charge density (binary)")
#     print("  - systemLabel.DRHO: Deformation charge density (binary)")
#     print()
#     print("Visualize with:")
#     print("  - XCrySDen: xcrysden --siesta_rho systemLabel.RHO")
#     print("  - Convert to cube: grid2cube systemLabel.RHO systemLabel.RHO.cube")

print("=" * 70)
print("Key Concepts:")
print("=" * 70)
print(
    """
1. SaveRho: Total self-consistent electron charge density
   - Shows electron distribution in molecule/crystal
   - Essential for bonding analysis

2. SaveDeltaRho: Deformation charge density (ρ_SCF - ρ_atomic)
   - Shows charge redistribution upon bonding
   - Positive: charge accumulation (bonding regions)
   - Negative: charge depletion

3. Binary format:
   - Fastest I/O
   - Smallest file size (~1 MB for 50x50x50 grid)
   - SIESTA native format
   - Readable by XCrySDen

Next: Try example 02 to save potentials in NetCDF format
"""
)
