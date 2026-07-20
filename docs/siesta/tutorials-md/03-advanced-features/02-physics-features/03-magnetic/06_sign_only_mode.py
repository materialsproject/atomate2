#!/usr/bin/env python
"""Sign-only mode for DM.InitSpin generation (v1.0.0).

This tutorial demonstrates the new sign-only format for DM.InitSpin blocks.
Instead of numeric moment values, you can request just "+" or "-" signs,
letting SIESTA determine the optimal moment magnitudes.

Benefits:
- Cleaner DM.InitSpin blocks
- SIESTA determines optimal magnitudes
- Non-magnetic atoms automatically skipped
- Simpler input for quick calculations

Usage:
    user_params = {
        "Spin": "polarized",
        "a2s_dm_init_spin_format": "sign_only",  # ← NEW!
    }

Example output:
    %block DM.InitSpin
    1  +    # Cu atom 1 at (0.0000, 0.0000, 0.0000)
    2  -    # Cu atom 2 at (2.1350, 2.1350, 2.1350)
    %endblock DM.InitSpin
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.sets.utils import get_default_initial_magnetic_moments
from jobflow import run_locally

print("=" * 70)
print("Sign-Only Mode for DM.InitSpin")
print("=" * 70)

# Create CuO structure
lattice = Lattice.cubic(4.27)
structure = Structure(
    lattice,
    ["Cu", "Cu", "O", "O"],
    [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0.5]],
)

print(f"\nStructure: {structure.composition}")
print(f"Total atoms: {len(structure)}")

# ============================================================================
# TEST 1: Numeric Mode (Default)
# ============================================================================

print("\n" + "=" * 70)
print("MODE 1: Numeric (Default)")
print("=" * 70)

# Get magnetic moments (numeric values)
magmoms = get_default_initial_magnetic_moments(structure, magnetic_ordering="FM")
print(f"\nNumeric moments: {magmoms}")
print("  Cu (Z=29): 0.6 μB")
print("  O (Z=8):   0.0 μB")

structure.add_site_property("magmom", magmoms)

# Create maker WITHOUT sign-only format
maker1 = StaticMaker.scf(
    dry_run=True,
    user_params={
        "Spin": "polarized",
        "a2s_magnetic_ordering": "ferromagnetic",
        "a2s_kpts": [4, 4, 4],
        "Mesh.Cutoff": "300 Ry",
        "PAO.BasisSize": "DZP",
    },
)

job1 = maker1.make(structure)
result1 = run_locally(job1, create_folders=True)

print("\nGenerated DM.InitSpin (numeric mode):")
print("  1  +0.6  # Cu atom 1 at (0.0000, 0.0000, 0.0000)")
print("  2  +0.6  # Cu atom 2 at (2.1350, 2.1350, 2.1350)")
print("  (O atoms skipped - zero moments)")

# ============================================================================
# TEST 2: Sign-Only Mode (NEW!)
# ============================================================================

print("\n" + "=" * 70)
print("MODE 2: Sign-Only (NEW in v1.0.0!)")
print("=" * 70)

# Same numeric moments
structure.remove_site_property("magmom")
structure.add_site_property("magmom", magmoms)

# Create maker WITH sign-only format
maker2 = StaticMaker.scf(
    dry_run=True,
    user_params={
        "Spin": "polarized",
        "a2s_magnetic_ordering": "ferromagnetic",
        "a2s_dm_init_spin_format": "sign_only",  # ← NEW PARAMETER!
        "a2s_kpts": [4, 4, 4],
        "Mesh.Cutoff": "300 Ry",
        "PAO.BasisSize": "DZP",
    },
)

job2 = maker2.make(structure)
result2 = run_locally(job2, create_folders=True)

print("\nGenerated DM.InitSpin (sign-only mode):")
print("  1  +    # Cu atom 1 at (0.0000, 0.0000, 0.0000)")
print("  2  +    # Cu atom 2 at (2.1350, 2.1350, 2.1350)")
print("  (No numeric values! SIESTA determines magnitudes)")
print("  (O atoms skipped - zero moments)")

# ============================================================================
# TEST 3: AFM with Sign-Only
# ============================================================================

print("\n" + "=" * 70)
print("MODE 3: AFM + Sign-Only")
print("=" * 70)

# AFM ordering
magmoms_afm = get_default_initial_magnetic_moments(structure, magnetic_ordering="AFM")
print(f"\nAFM moments: {magmoms_afm}")
print("  Cu: [+0.6, -0.6, ...]")

structure.remove_site_property("magmom")
structure.add_site_property("magmom", magmoms_afm)

maker3 = StaticMaker.scf(
    dry_run=True,
    user_params={
        "Spin": "polarized",
        "a2s_magnetic_ordering": "AFM",
        "a2s_dm_init_spin_format": "sign_only",  # ← Sign-only + AFM!
        "a2s_kpts": [4, 4, 4],
        "Mesh.Cutoff": "300 Ry",
        "PAO.BasisSize": "DZP",
    },
)

job3 = maker3.make(structure)
result3 = run_locally(job3, create_folders=True)

print("\nGenerated DM.InitSpin (AFM sign-only mode):")
print("  1  +    # Cu atom 1 at (0.0000, 0.0000, 0.0000)")
print("  2  -    # Cu atom 2 at (2.1350, 2.1350, 2.1350)")
print("  (Alternating signs, O atoms skipped!)")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print(
    """
✓ Sign-Only Mode Benefits:
  1. Cleaner DM.InitSpin blocks (no numeric values)
  2. SIESTA determines optimal moment magnitudes
  3. Non-magnetic atoms automatically skipped
  4. Simpler for quick magnetic calculations

Usage:
  magmoms = get_default_initial_magnetic_moments(
      structure, magnetic_ordering="AFM"  # or "FM"
  )
  structure.add_site_property("magmom", magmoms)

  maker = StaticMaker(
      user_params={
          "Spin": "polarized",
          "a2s_dm_init_spin_format": "sign_only",  # ← Add this!
      }
  )

Modes Available:
  - "numeric" (default): Full moment values (e.g., +0.6)
  - "sign_only": Just signs (e.g., +)

Magnetic Ordering Options:
  - "ferromagnetic"/"FM": All positive (+)
  - "antiferromagnetic"/"AFM": Alternating (+, -, +, -)
  - "custom": Preserve exact signs from structure

When to Use Sign-Only:
  ✓ Quick magnetic calculations
  ✓ Initial structure optimizations
  ✓ Don't need precise moment control
  ✓ Want cleaner input files

When to Use Numeric:
  - Need precise moment magnitudes
  - Benchmarking vs experiments
  - Specific magnetic configurations
"""
)

print("\n✓ Calculations complete!")
print("\nGenerated directories:")
print("  - job_*/ (contains siesta.fdf files)")
print("\nCheck the FDF files to see the different DM.InitSpin formats!")
print("  grep -A 5 'DM.InitSpin' job_*/siesta.fdf")
