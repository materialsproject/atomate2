"""Tutorial 2: Cell Standardization Workflow.

This tutorial demonstrates how to use the standardize command to:
1. Convert structures to primitive cells (for efficient DFT)
2. Convert to conventional cells (for visualization/databases)
3. Apply international standard settings
4. Understand the impact on calculations

The standardize command is essential for:
- Reducing computational cost (primitive cells)
- Database comparison (conventional cells)
- Literature matching (international settings)
- Understanding crystal symmetry

Key Features:
- Automatic symmetry detection
- Space group analysis
- Cell reduction/expansion
- Multi-format output
"""

from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

print("=" * 80)
print("Tutorial 2: Cell Standardization Workflow")
print("=" * 80)

# ============================================================================
# Example 1: Conventional → Primitive (DFT Efficiency)
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: Conventional → Primitive (For DFT Efficiency)")
print("=" * 80)

# Create conventional FCC Si cell (8 atoms)
si_conventional = Structure.from_spacegroup(
    "Fd-3m",  # FCC space group
    [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],  # Conventional cubic cell
    ["Si"],
    [[0, 0, 0]],
)
si_conventional.to(filename="si_conventional.cif", fmt="cif")
print("\nCreated conventional Si cell:")
print("  Space group: Fd-3m (227)")
print(f"  Atoms: {si_conventional.num_sites}")
print(f"  Volume: {si_conventional.volume:.2f} Ų")
print(f"  Lattice: a=b=c={si_conventional.lattice.a:.3f} Å, α=β=γ=90°")

print("\nConvert to primitive cell:")
print("  atomate2siesta-structure standardize si_conventional.cif --primitive")

print("\nExpected results:")
print("  - Atoms: 8 → 2 (4x reduction!)")
print("  - Volume: ~160 → ~40 Ų (4x reduction)")
print("  - Lattice: rhombohedral (α=β=γ=60°)")
print("  - Computational cost: ~4x faster DFT")

print("\nWhy use primitive cells for DFT?")
print("  ✓ Fewer atoms = less computation time")
print("  ✓ Smaller k-point mesh needed")
print("  ✓ Less memory required")
print("  ✓ Same physical properties")

# ============================================================================
# Example 2: Primitive → Conventional (Visualization)
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Primitive → Conventional (For Visualization)")
print("=" * 80)

# Create primitive Si cell (2 atoms, rhombohedral representation of FCC)
# Start with correct diamond structure and get its primitive cell
si_conventional = Structure.from_spacegroup(
    "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]
)
si_primitive = SpacegroupAnalyzer(si_conventional).get_primitive_standard_structure()
si_primitive.to(filename="si_primitive.cif", fmt="cif")
print("\nCreated primitive Si cell:")
print(f"  Atoms: {si_primitive.num_sites}")
print(f"  Volume: {si_primitive.volume:.2f} Ų")
print(
    f"  Lattice angles: α={si_primitive.lattice.alpha:.1f}°, "
    f"β={si_primitive.lattice.beta:.1f}°, γ={si_primitive.lattice.gamma:.1f}°"
)

print("\nConvert to conventional cell:")
print("  atomate2siesta-structure standardize si_primitive.cif --conventional")

print("\nExpected results:")
print("  - Atoms: 2 → 8 (4x expansion)")
print("  - Lattice: rhombohedral → cubic (α=β=γ=90°)")
print("  - Easier to visualize and understand")
print("  - Standard for crystallographic databases")

print("\nWhen to use conventional cells?")
print("  ✓ Creating figures for publications")
print("  ✓ Comparing with experimental data")
print("  ✓ Teaching crystal structures")
print("  ✓ Building supercells")

# ============================================================================
# Example 3: International Standard Setting
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: International Standard Setting")
print("=" * 80)

# Create rutile TiO2 structure
tio2_structure = Structure.from_spacegroup(
    "P42/mnm",  # Rutile space group
    [[4.59, 0, 0], [0, 4.59, 0], [0, 0, 2.96]],
    ["Ti", "O"],
    [[0, 0, 0], [0.3, 0.3, 0]],
)
tio2_structure.to(filename="tio2_input.cif", fmt="cif")
print("\nCreated TiO2 rutile structure:")
print("  Space group: P42/mnm (136)")
print(f"  Atoms: {tio2_structure.num_sites}")

print("\nApply international standard setting:")
print("  atomate2siesta-structure standardize tio2_input.cif --international")

print("\nExpected results:")
print("  - Refined structure in standard orientation")
print("  - Follows International Tables conventions")
print("  - Axes aligned according to standards")
print("  - Useful for literature comparison")

# ============================================================================
# Example 4: Symmetry Analysis During Standardization
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Symmetry Analysis During Standardization")
print("=" * 80)

# Create graphite structure
a = 2.46
c = 6.71
# Hexagonal lattice: a=b, γ=120°
lattice = [
    [a, 0, 0],
    [a * np.cos(np.radians(120)), a * np.sin(np.radians(120)), 0],
    [0, 0, c],
]
graphite = Structure.from_spacegroup(
    "P63/mmc",  # Hexagonal space group
    lattice,
    ["C"],
    [[0, 0, 0.25]],
)
graphite.to(filename="graphite.cif", fmt="cif")
print("\nCreated graphite structure:")
print("  Space group: P63/mmc (194)")
print("  Crystal system: hexagonal")

print("\nStandardize and view symmetry analysis:")
print("  atomate2siesta-structure standardize graphite.cif --primitive")

print("\nSymmetry information displayed:")
print("  ✓ Space group symbol and number")
print("  ✓ Crystal system (hexagonal)")
print("  ✓ Point group")
print("  ✓ Before/after comparison")

# ============================================================================
# Example 5: Custom Symmetry Precision
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: Custom Symmetry Precision")
print("=" * 80)

# Create Si structure for this example
si_structure = Structure.from_spacegroup(
    "Fd-3m", [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]], ["Si"], [[0, 0, 0]]
)

# Create slightly distorted structure
si_distorted = si_structure.copy()
si_distorted.perturb(distance=0.02)  # Small distortion
si_distorted.to(filename="si_distorted.cif", fmt="cif")
print("\nCreated slightly distorted Si structure (0.02 Å)")

print("\nDefault precision (may not detect symmetry):")
print("  atomate2siesta-structure standardize si_distorted.cif --primitive")

print("\nRelaxed precision (will detect symmetry):")
print(
    "  atomate2siesta-structure standardize si_distorted.cif --primitive --symprec 0.1"
)

print("\nSymmetry precision guidelines:")
print("  - 0.01 Å (default): High precision, strict matching")
print("  - 0.1 Å: Relaxed for experimental structures")
print("  - 0.3 Å: Very relaxed for MD snapshots")

# ============================================================================
# Example 6: Workflow Integration
# ============================================================================
print("\n" + "=" * 80)
print("Example 6: Complete Workflow Integration")
print("=" * 80)

print(
    """
Typical Standardization Workflows:

1. DFT Calculation Preparation:
   ┌─────────────┐
   │   Input     │  (Any cell)
   │  Structure  │
   └──────┬──────┘
          │
          ↓  atomate2siesta-structure standardize --primitive
   ┌─────────────┐
   │  Primitive  │  (Minimal atoms)
   │    Cell     │
   └──────┬──────┘
          │
          ↓  Run DFT calculation
   ┌─────────────┐
   │  Optimized  │
   │  Structure  │
   └──────┬──────┘
          │
          ↓  atomate2siesta-structure standardize --conventional
   ┌─────────────┐
   │Conventional │  (For visualization)
   │    Cell     │
   └─────────────┘

2. Database Submission:
   Input → --conventional → Upload to database

3. Literature Comparison:
   Downloaded structure → --international → Compare with paper

4. Phonon Calculations:
   Primitive cell → Generate supercell → Run phonons
"""
)

# ============================================================================
# Example 7: Before/After Comparison
# ============================================================================
print("\n" + "=" * 80)
print("Example 7: Before/After Comparison")
print("=" * 80)

print("\nView detailed before/after statistics:")
print(
    "  atomate2siesta-structure standardize si_conventional.cif --primitive --show-before-after"
)

print("\nComparison table shows:")
print("  - Site count changes")
print("  - Volume scaling")
print("  - Lattice parameter changes")
print("  - Angle transformations")

# ============================================================================
# Example 8: Multi-Format Output
# ============================================================================
print("\n" + "=" * 80)
print("Example 8: Multi-Format Output")
print("=" * 80)

print("\nStandardize and save in different formats:")
print("  # POSCAR for VASP")
print("  atomate2siesta-structure standardize si_input.cif --primitive --format poscar")
print()
print("  # XSF for visualization")
print("  atomate2siesta-structure standardize si_input.cif --primitive --format xsf")
print()
print("  # FDF for SIESTA")
print("  atomate2siesta-structure standardize si_input.cif --primitive --format fdf")

# ============================================================================
# Summary and Best Practices
# ============================================================================
print("\n" + "=" * 80)
print("Summary and Best Practices")
print("=" * 80)

print(
    """
Decision Tree for Cell Standardization:

┌─────────────────────┐
│   What's your       │
│   goal?             │
└──────┬──────────────┘
       │
       ├─── DFT calculation? ────→ Use --primitive
       │                          (Faster, fewer atoms)
       │
       ├─── Visualization? ──────→ Use --conventional
       │                          (Easier to understand)
       │
       ├─── Database upload? ────→ Use --conventional
       │                          (Standard format)
       │
       └─── Literature match? ───→ Use --international
                                   (Standard setting)

Best Practices:

1. Always Standardize Before DFT:
   ✓ Use primitive cells for efficiency
   ✓ Check atom count reduction
   ✓ Verify symmetry is preserved

2. Symmetry Precision:
   - Start with default (0.01 Å)
   - Increase for experimental structures
   - Decrease for theoretical structures

3. Verify Results:
   - Use --show-before-after
   - Compare atom counts
   - Check lattice angles
   - Verify space group

4. Common Transformations:
   - FCC: 8 → 2 atoms (primitive)
   - BCC: 4 → 2 atoms (primitive)
   - Hexagonal: usually 2 → 1 atom

5. Computational Savings:
   Atoms  | Speedup (approx)
   ───────┼─────────────────
   8 → 2  | 4x faster
   4 → 2  | 2x faster
   6 → 3  | 2x faster

6. Integration with Other Commands:
   standardize → supercell → DFT
   DFT output → standardize → compare

Command Reference:
  atomate2siesta-structure standardize --help
"""
)

# ============================================================================
# Cleanup
# ============================================================================
print("\nCleaning up generated files...")
for f in [
    "si_conventional.cif",
    "si_primitive.cif",
    "tio2_input.cif",
    "graphite.cif",
    "si_distorted.cif",
]:
    if Path(f).exists():
        Path(f).unlink()

print("✓ Tutorial complete!")
print("\nNext tutorial: 03_surface_preparation.py")
