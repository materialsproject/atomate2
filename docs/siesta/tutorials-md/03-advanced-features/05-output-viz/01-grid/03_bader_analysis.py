#!/usr/bin/env python
"""
Tutorial 10, Example 3: Bader Charge Analysis

Demonstrates enabling Bader charge analysis for charge transfer calculations.

This example shows how to configure SIESTA for Bader charge partitioning,
useful for analyzing charge transfer in molecules and surfaces.

Learning objectives:
- Enable Bader charge analysis
- Understand required grid outputs
- Analyze atomic charges and charge transfer
- Use cube format for compatibility
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally

print("=" * 70)
print("Tutorial 10, Example 3: Bader Charge Analysis")
print("=" * 70)
print()

# Create a CO molecule (polar bond)
print("Creating CO molecule...")
lattice = Lattice.cubic(12.0)  # Large box for molecule
structure = Structure(
    lattice,
    ["C", "O"],
    [
        [0.5, 0.5, 0.5],  # C at center
        [0.55, 0.5, 0.5],  # O (C-O bond ~1.2 Å)
    ],
    coords_are_cartesian=False,
)
print(f"  Formula: {structure.composition.reduced_formula}")
print("  Bond: Polar C-O (expect charge transfer C → O)")
print()

# Configure Bader analysis
print("Configuring Bader charge analysis...")
user_params = {
    # Bader analysis (requires SaveRho!)
    "SaveBaderCharge": True,  # Enable Bader analysis
    "SaveRho": True,  # REQUIRED for Bader
    "SaveDeltaRho": True,  # Useful to visualize charge transfer
    # Use cube format for compatibility with Bader code
    "SaveGridFunc.Format": "cube",
    # Calculation parameters
    "PAO.BasisSize": "TZP",  # Better basis for accurate charges
    "a2s_kpts": [1, 1, 1],  # Gamma point for molecule
    "Mesh.Cutoff": "300 Ry",  # Fine grid for accurate integration
}
print("  Bader analysis parameters:")
print(f"    SaveBaderCharge: {user_params['SaveBaderCharge']}")
print(f"    SaveRho: {user_params['SaveRho']} (REQUIRED)")
print(f"    Format: {user_params['SaveGridFunc.Format']}")
print()

print("⚠️  Important: Bader analysis requires:")
print("   1. SaveBaderCharge: True")
print("   2. SaveRho: True (charge density)")
print("   3. Fine mesh cutoff (≥ 250 Ry recommended)")
print()

# Create static calculation job
print("Creating static calculation job...")
maker = StaticMaker.scf(user_params=user_params)
job = maker.make(structure)
print(f"  Job name: {job.name}")
print()

# Run calculation
print("Running calculation...")
print("  This will:")
print("    1. Calculate self-consistent charge density")
print("    2. Partition charge using Bader analysis")
print("    3. Compute atomic charges and volumes")
print()

# Uncomment to run:
results = run_locally(job, create_folders=True)
#
# if results:
#     print("Calculation complete!")
#     print()
#     print("Output files:")
#     print("  - systemLabel.RHO.cube: Total charge density (cube format)")
#     print("  - systemLabel.DRHO.cube: Deformation charge density")
#     print("  - ACF.dat: Bader charges and volumes")
#     print()
#     print("Reading Bader results:")
#     print("  # ACF.dat contains:")
#     print("  # ATOM    X      Y      Z    CHARGE   MIN DIST   ATOMIC VOL")
#     print("  #    1  6.000  6.000  6.000  5.234    1.234      12.345")
#     print("  #    2  7.200  6.000  6.000  8.766    1.234      23.456")
#     print()
#     print("Interpretation:")
#     print("  - Atom 1 (C): Charge = 5.234 (lost ~0.77 electrons)")
#     print("  - Atom 2 (O): Charge = 8.766 (gained ~0.77 electrons)")
#     print("  - Charge transfer: ~0.77 e⁻ from C to O (polar C-O bond)")

print("=" * 70)
print("Understanding Bader Charge Analysis:")
print("=" * 70)
print(
    """
1. What is Bader Analysis?
   - Partitions space into atomic basins
   - Based on zero-flux surfaces in charge density gradient
   - Integrates charge within each basin → atomic charge

2. Advantages:
   ✓ Well-defined, unique partitioning
   ✓ Minimal basis set dependence
   ✓ Works for molecules, surfaces, crystals
   ✓ Physically meaningful volumes

3. Interpreting Results:
   - Neutral atom: Charge ≈ Z (atomic number)
   - Positive charge: Electron depletion (cation)
   - Negative charge: Electron accumulation (anion)
   - Charge transfer: |Q₁ - Z₁|

4. Best Practices:
   - Use fine mesh (≥ 250 Ry)
   - Use large basis (DZP minimum, TZP better)
   - Check convergence with respect to mesh
   - Compare to Mulliken/Voronoi charges

5. Applications:
   - Ionic vs covalent bonding
   - Charge transfer in complexes
   - Oxidation states
   - Surface charge analysis

6. External Bader Code:
   If SIESTA's built-in Bader doesn't work, use:
   http://theory.cm.utexas.edu/henkelman/code/bader/

   bader systemLabel.RHO.cube

   This generates ACF.dat and BCF.dat files.

Common charge transfer examples:
  - C-O: ~0.5-1.0 e⁻ (polar covalent)
  - NaCl: ~0.9 e⁻ (ionic)
  - C-C: ~0.0-0.1 e⁻ (covalent)
  - Metal-O: ~1-2 e⁻ (ionic/polar)

Next: Explore tutorial 11 for denchar visualization
"""
)
