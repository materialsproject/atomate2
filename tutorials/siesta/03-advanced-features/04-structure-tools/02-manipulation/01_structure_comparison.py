"""Tutorial 1: Structure Comparison Workflow.

This tutorial demonstrates how to use the compare command to:
1. Compare structures before/after optimization
2. Verify structure preservation during format conversion
3. Check symmetry standardization results
4. Validate supercell generation

The compare command is essential for:
- Quality assurance in computational workflows
- Debugging structure manipulation operations
- Verifying file format conversions
- Comparing calculation results

Key Features:
- Quantitative RMSD calculation
- Lattice parameter comparison
- Composition analysis
- Site-by-site matching
"""

from pathlib import Path

from pymatgen.core import Structure

# Create test structures
print("=" * 80)
print("Tutorial 1: Structure Comparison Workflow")
print("=" * 80)

# ============================================================================
# Example 1: Compare Before/After Optimization
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: Compare Before/After Optimization")
print("=" * 80)

# Create a slightly distorted Si structure
si_structure = Structure.from_spacegroup(
    "Fd-3m", [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]], ["Si"], [[0, 0, 0]]
)

# Save original
si_structure.to(filename="si_original.cif", fmt="cif")
print("\nCreated original Si structure (5.43 Å lattice)")

# Create "optimized" version with slightly different lattice
si_optimized = si_structure.copy()
si_optimized.scale_lattice(si_optimized.volume * 0.98)  # 2% compression
si_optimized.to(filename="si_optimized.cif", fmt="cif")
print("Created optimized Si structure (2% volume reduction)")

print("\nCompare using CLI:")
print("  atomate2siesta-structure compare si_original.cif si_optimized.cif")
print("\nExpected results:")
print("  - Lattice mismatch: ~2% volume difference")
print("  - Same number of sites")
print("  - Same composition")
print("  - Small RMSD due to lattice compression")

# ============================================================================
# Example 2: Verify Format Conversion
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Verify Format Conversion")
print("=" * 80)

# Save in different formats
si_structure.to(filename="si_test.cif", fmt="cif")
print("\nSaved structure as CIF")

print("\nConvert to POSCAR:")
print("  atomate2siesta-structure convert si_test.cif")
print("  (This creates structure.cif as output)")

print("\nVerify conversion preserved structure:")
print("  atomate2siesta-structure compare si_test.cif structure.cif")
print("\nExpected results:")
print("  - Structures should be identical")
print("  - RMSD should be ~0 Å")
print("  - All lattice parameters match")

# ============================================================================
# Example 3: Check Standardization Results
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Check Standardization Results")
print("=" * 80)

# Create a non-standard cell (2x2x2 supercell)
si_supercell = si_structure.copy()
si_supercell.make_supercell([2, 2, 2])
si_supercell.to(filename="si_supercell.cif", fmt="cif")
print(f"\nCreated 2x2x2 supercell ({si_supercell.num_sites} atoms)")

print("\nStandardize to primitive cell:")
print("  atomate2siesta-structure standardize si_supercell.cif --primitive")

print("\nCompare with original primitive:")
print("  atomate2siesta-structure compare si_original.cif primitive_si_supercell.cif")
print("\nExpected results:")
print("  - Should match original primitive cell")
print("  - May have different site ordering")
print("  - RMSD should be very small (< 0.01 Å)")

# ============================================================================
# Example 4: Verify Supercell Generation
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Verify Supercell Generation")
print("=" * 80)

print("\nGenerate 3x3x3 supercell:")
print("  atomate2siesta-structure supercell si_original.cif --matrix 3 3 3")

print("\nCompare unit cell counts:")
print("  atomate2siesta-structure compare si_original.cif supercell_si_original.cif")
print("\nExpected results:")
print("  - Supercell should have 27x more atoms (2 → 54)")
print("  - Volume should be 27x larger")
print("  - Lattice parameters ~3x larger")
print("  - Composition ratio should be identical")

# ============================================================================
# Example 5: Advanced Comparison with Tolerance
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: Advanced Comparison with Tolerance")
print("=" * 80)

# Create structure with small perturbation
si_perturbed = si_structure.copy()
si_perturbed.perturb(distance=0.05)  # 0.05 Å perturbation
si_perturbed.to(filename="si_perturbed.cif", fmt="cif")
print("\nCreated perturbed structure (0.05 Å displacements)")

print("\nCompare with tight tolerance (default 0.01 Å):")
print("  atomate2siesta-structure compare si_original.cif si_perturbed.cif")
print("\nExpected: Structures differ (perturbation > tolerance)")

print("\nCompare with relaxed tolerance (0.1 Å):")
print(
    "  atomate2siesta-structure compare si_original.cif si_perturbed.cif --tolerance 0.1"
)
print("\nExpected: Structures match (perturbation < tolerance)")

print("\nUse verbose mode for detailed site analysis:")
print("  atomate2siesta-structure compare si_original.cif si_perturbed.cif --verbose")

# ============================================================================
# Example 6: Compare Different Compositions
# ============================================================================
print("\n" + "=" * 80)
print("Example 6: Compare Different Compositions")
print("=" * 80)

# Create SiGe alloy
sige_structure = si_structure.copy()
sige_structure.replace(0, "Ge")  # Replace first Si with Ge
sige_structure.to(filename="sige_alloy.cif", fmt="cif")
print("\nCreated SiGe alloy (50% substitution)")

print("\nCompare compositions:")
print("  atomate2siesta-structure compare si_original.cif sige_alloy.cif")
print("\nExpected results:")
print("  - Composition mismatch detected")
print("  - Same lattice (similar atomic radii)")
print("  - Different RMSD due to composition")
print("  - Site matching: 50% matched (only Si sites match)")

# ============================================================================
# Summary and Best Practices
# ============================================================================
print("\n" + "=" * 80)
print("Summary and Best Practices")
print("=" * 80)

print(
    """
Best Practices for Structure Comparison:

1. Format Conversion Verification:
   - Always compare after converting between formats
   - Use tight tolerance (default 0.01 Å) for exact match
   - Check that RMSD ≈ 0

2. Optimization Validation:
   - Compare before/after optimization to track changes
   - Expected: lattice changes, small RMSD
   - Monitor volume and lattice parameter changes

3. Standardization Checks:
   - Compare primitive ↔ conventional cells
   - Account for different site orderings
   - Focus on lattice symmetry, not site order

4. Supercell Verification:
   - Check atom count multiplier
   - Verify volume scaling
   - Ensure composition ratio unchanged

5. Tolerance Selection:
   - 0.01 Å (default): Exact matching
   - 0.1 Å: Relaxed structures, thermal effects
   - 0.5 Å: Major structural differences

6. Verbose Mode:
   - Use --verbose for detailed mismatch analysis
   - Shows up to 10 unmatched sites
   - Helpful for debugging

7. Selective Comparison:
   - Use --no-compare-sites to skip site matching
   - Use --no-calculate-rmsd for composition-only check
   - Useful for large structures

Command Reference:
  atomate2siesta-structure compare --help
"""
)

# ============================================================================
# Cleanup
# ============================================================================
print("\nCleaning up generated files...")
for f in [
    "si_original.cif",
    "si_optimized.cif",
    "si_test.cif",
    "si_supercell.cif",
    "si_perturbed.cif",
    "sige_alloy.cif",
]:
    if Path(f).exists():
        Path(f).unlink()

print("✓ Tutorial complete!")
print("\nNext tutorial: 02_cell_standardization.py")
