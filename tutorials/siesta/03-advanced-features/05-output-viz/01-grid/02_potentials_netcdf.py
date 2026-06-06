##!/usr/bin/env python
"""
Tutorial 10, Example 2: Potentials in NetCDF Format

Demonstrates saving various potentials in NetCDF format for visualization.

This example shows how to output electrostatic and total potentials
in portable NetCDF format with metadata.

Learning objectives:
- Save electrostatic potential (Hartree + external)
- Save total Kohn-Sham potential
- Use NetCDF format (portable, includes metadata)
- Understand different potential components
"""

from jobflow import run_locally
from pymatgen.core import Lattice, Structure

from atomate2.siesta.jobs.core import StaticMaker

print("=" * 70)
print("Tutorial 10, Example 2: Potentials in NetCDF Format")
print("=" * 70)
print()

# Create a water molecule
print("Creating H2O molecule...")
lattice = Lattice.cubic(15.0)  # Large box for molecule
structure = Structure(
    lattice,
    ["O", "H", "H"],
    [
        [0.5, 0.5, 0.5],  # O at center
        [0.5, 0.55, 0.5],  # H
        [0.55, 0.5, 0.5],  # H
    ],
    coords_are_cartesian=False,
)
print(f"  Formula: {structure.composition.reduced_formula}")
print(f"  Atoms: {len(structure)} ({structure.composition})")
print()

# Configure potential output using SIESTA FDF parameter names
print("Configuring potential output...")
user_params = {
    # Potentials
    "SaveElectrostaticPotential": True,  # V_H + V_ext
    "SaveTotalPotential": True,  # V_H + V_XC + V_ext
    "SaveNeutralAtomPotential": True,  # Superposition of atomic potentials
    # File format (NetCDF with metadata)
    "SaveGridFunc.Format": "netcdf",
    # Calculation parameters
    "PAO.BasisSize": "DZP",
    "a2s_kpts": [1, 1, 1],  # Gamma point for molecule
    "Mesh.Cutoff": "250 Ry",
}
print("  Potential output parameters:")
print(f"    SaveElectrostaticPotential: {user_params['SaveElectrostaticPotential']}")
print(f"    SaveTotalPotential: {user_params['SaveTotalPotential']}")
print(f"    SaveNeutralAtomPotential: {user_params['SaveNeutralAtomPotential']}")
print(f"    Format: {user_params['SaveGridFunc.Format']}")
print()

# Create static calculation job with dry_run mode
print("Creating static calculation job...")
maker = StaticMaker.scf(user_params=user_params, dry_run=True)
job = maker.make(structure)
print(f"  Job name: {job.name}")
print()

# Run calculation
print("Running calculation (dry-run mode)...")
print("  This will generate NetCDF grid files with metadata")
print("  Grid files:")
print("    - systemLabel.VH.nc (electrostatic potential)")
print("    - systemLabel.VT.nc (total potential)")
print("    - systemLabel.VNA.nc (neutral atom potential)")
print()

# Run calculation
results = run_locally(job, create_folders=True)
#
# if results:
#     print("Calculation complete!")
#     print()
#     print("Output files (NetCDF format):")
#     print("  - systemLabel.VH.nc: Electrostatic potential (V_Hartree + V_ext)")
#     print("  - systemLabel.VT.nc: Total Kohn-Sham potential (V_H + V_XC + V_ext)")
#     print("  - systemLabel.VNA.nc: Neutral atom superposition potential")
#     print()
#     print("NetCDF advantages:")
#     print("  ✓ Portable across platforms")
#     print("  ✓ Includes grid metadata (dimensions, units)")
#     print("  ✓ Compressed (~30% smaller than formatted)")
#     print("  ✓ Can be read by many analysis tools")
#     print()
#     print("Analyze with Python:")
#     print("  import netCDF4")
#     print("  data = netCDF4.Dataset('systemLabel.VH.nc')")
#     print("  potential = data.variables['potential'][:]")

print("=" * 70)
print("Understanding Potentials:")
print("=" * 70)
print(
    """
1. Electrostatic Potential (V_H + V_ext):
   - Hartree potential: electron-electron interaction
   - External potential: nuclei-electron interaction
   - Classical electrostatics
   - Used for: Work functions, band alignments

2. Total Kohn-Sham Potential (V_H + V_XC + V_ext):
   - Includes exchange-correlation potential
   - The effective potential "felt" by electrons
   - Determines electronic structure
   - Used for: Band structure interpretation

3. Neutral Atom Potential (V_NA):
   - Superposition of isolated atom potentials
   - Reference for charge transfer analysis
   - Shows bonding perturbation

4. NetCDF Format:
   - Modern HDF5-based format
   - Self-describing (metadata included)
   - Compressed by default
   - Tools: Python netCDF4, NCO, CDO

Typical file sizes (50x50x50 grid):
  - Binary: ~1 MB
  - NetCDF: ~1.5 MB (compressed)
  - Formatted: ~5 MB

Next: Try example 03 for Bader charge analysis
"""
)
