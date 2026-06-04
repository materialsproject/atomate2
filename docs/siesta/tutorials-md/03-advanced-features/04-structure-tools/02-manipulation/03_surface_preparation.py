"""Tutorial 3: Surface Preparation Pipeline.

This tutorial demonstrates the complete workflow for preparing surface slabs:
1. Create surface slabs with different Miller indices
2. Add appropriate vacuum spacing
3. Standardize the slab structure
4. Optimize cell for DFT calculations
5. Compare different surface orientations

The surface preparation pipeline is essential for:
- Surface energy calculations
- Adsorption studies
- Interface modeling
- Catalysis research

Key Workflow:
  Bulk Structure → Slab → Vacuum → Standardize → Optimize → DFT

Features Demonstrated:
- Multiple Miller indices (100, 110, 111)
- Vacuum thickness optimization
- Symmetric vs asymmetric slabs
- Cell standardization for surfaces
- Orthogonalization benefits
"""

from pathlib import Path

from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator

print("=" * 80)
print("Tutorial 3: Surface Preparation Pipeline")
print("=" * 80)

# ============================================================================
# Example 1: Basic Surface Slab Generation
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: Basic Surface Slab Generation")
print("=" * 80)

# Create bulk Al structure (FCC)
al_bulk = Structure.from_spacegroup(
    "Fm-3m",  # FCC space group
    [[4.05, 0, 0], [0, 4.05, 0], [0, 0, 4.05]],
    ["Al"],
    [[0, 0, 0]],
)
al_bulk.to(filename="al_bulk.cif", fmt="cif")
print("\nCreated bulk Al structure:")
print("  Space group: Fm-3m (225)")
print(f"  Atoms: {al_bulk.num_sites}")
print(f"  Lattice: {al_bulk.lattice.a:.3f} Å")

# Generate (111) surface slab
slabgen = SlabGenerator(
    al_bulk,
    miller_index=[1, 1, 1],
    min_slab_size=10.0,  # Minimum slab thickness (Å)
    min_vacuum_size=15.0,  # Minimum vacuum thickness (Å)
    center_slab=True,
)

al_111_slab = slabgen.get_slab()
al_111_slab.to(filename="al_111_slab.cif", fmt="cif")
print("\nGenerated Al(111) surface slab:")
print(f"  Atoms: {al_111_slab.num_sites}")
print(f"  Lattice c (slab + vacuum): {al_111_slab.lattice.c:.3f} Å")
print("  Miller index: (1,1,1)")

print("\nStandardize the slab:")
print("  atomate2siesta-structure standardize al_111_slab.cif --primitive")
print("\nExpected results:")
print("  - Reduced to primitive surface cell")
print("  - Symmetry-equivalent atoms removed")
print("  - Optimal for DFT calculations")

# ============================================================================
# Example 2: Multiple Surface Orientations
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Multiple Surface Orientations")
print("=" * 80)

# Create Cu bulk structure
cu_bulk = Structure.from_spacegroup(
    "Fm-3m",
    [[3.61, 0, 0], [0, 3.61, 0], [0, 0, 3.61]],
    ["Cu"],
    [[0, 0, 0]],
)
cu_bulk.to(filename="cu_bulk.cif", fmt="cif")
print(f"\nCreated bulk Cu structure (FCC, a={cu_bulk.lattice.a:.3f} Å)")

# Generate multiple surface orientations
miller_indices = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
surface_info = []

for miller in miller_indices:
    slabgen = SlabGenerator(
        cu_bulk,
        miller_index=miller,
        min_slab_size=10.0,
        min_vacuum_size=15.0,
        center_slab=True,
    )

    slab = slabgen.get_slab()
    miller_str = "".join(map(str, miller))
    filename = f"cu_{miller_str}_slab.cif"
    slab.to(filename=filename, fmt="cif")

    surface_info.append(
        {
            "miller": miller,
            "atoms": slab.num_sites,
            "area": slab.surface_area,
            "thickness": slab.lattice.c,
            "filename": filename,
        }
    )

    print(f"\n{miller} surface:")
    print(f"  Atoms: {slab.num_sites}")
    print(f"  Surface area: {slab.surface_area:.2f} Ų")
    print(f"  Total thickness: {slab.lattice.c:.3f} Å")

print("\n" + "-" * 80)
print("Compare all surfaces:")
print("-" * 80)

for info in surface_info:
    miller_str = "".join(map(str, info["miller"]))
    print(f"\nCu({miller_str}):")
    print(f"  atomate2siesta-structure compare cu_bulk.cif cu_{miller_str}_slab.cif")
    print(
        f"  Expected: Surface has {info['atoms']} atoms, bulk has {cu_bulk.num_sites}"
    )

# ============================================================================
# Example 3: Vacuum Thickness Optimization
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Vacuum Thickness Optimization")
print("=" * 80)

# Test different vacuum thicknesses
vacuum_sizes = [10.0, 15.0, 20.0, 25.0]
print("\nGenerate slabs with varying vacuum:")

for vacuum in vacuum_sizes:
    slabgen = SlabGenerator(
        al_bulk,
        miller_index=[1, 1, 1],
        min_slab_size=10.0,
        min_vacuum_size=vacuum,
        center_slab=True,
    )

    slab = slabgen.get_slab()
    filename = f"al_111_vacuum_{int(vacuum)}.cif"
    slab.to(filename=filename, fmt="cif")

    print(f"\n  Vacuum = {vacuum} Å:")
    print(f"    Total c: {slab.lattice.c:.3f} Å")
    print(f"    Atoms: {slab.num_sites}")
    print(f"    Filename: {filename}")

print("\n" + "-" * 80)
print("Vacuum Thickness Guidelines:")
print("-" * 80)
print(
    """
  10 Å: Minimum for small molecules
  15 Å: Standard for most calculations
  20 Å: Large adsorbates or dipole-sensitive
  25 Å: Very accurate surface energy

  Rule of thumb: Vacuum ≥ 1.5 × slab thickness

  Convergence test:
    for vacuum in 10 15 20 25; do
        atomate2siesta-structure slab al_bulk.cif --miller-indices 1,1,1 \\
            --min-vacuum-size $vacuum
    done
"""
)

# ============================================================================
# Example 4: Symmetric vs Asymmetric Slabs
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Symmetric vs Asymmetric Slabs")
print("=" * 80)

# Symmetric slab (default behavior of SlabGenerator)
slabgen_sym = SlabGenerator(
    al_bulk,
    miller_index=[1, 1, 1],
    min_slab_size=10.0,
    min_vacuum_size=15.0,
    center_slab=True,
    primitive=False,
)

slab_sym = slabgen_sym.get_slab()
slab_sym.to(filename="al_111_symmetric.cif", fmt="cif")

print("\nSymmetric slab (default):")
print(f"  Atoms: {slab_sym.num_sites}")
print("  Both surfaces identical: Yes (by default)")
print("  Use case: Surface energy calculations")

# For asymmetric slab, we can manually remove top layer atoms
# or use different parameters. Here we'll just note the concept.
print("\nAsymmetric slab:")
print("  Can be created by manually removing/adding atoms")
print("  Or by using different terminations")
print("  Use case: Adsorption on one surface only")

print("\nNote on slab symmetry:")
print("  - SlabGenerator creates symmetric slabs by default")
print("  - For asymmetric slabs, manually modify structure")
print("  - Asymmetric more efficient for single-sided adsorption")

# ============================================================================
# Example 5: Complete Surface Preparation Pipeline
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: Complete Surface Preparation Pipeline")
print("=" * 80)

# Create MgO bulk structure
mgo_bulk = Structure.from_spacegroup(
    "Fm-3m",
    [[4.21, 0, 0], [0, 4.21, 0], [0, 0, 4.21]],
    ["Mg", "O"],
    [[0, 0, 0], [0.5, 0.5, 0.5]],
)
mgo_bulk.to(filename="mgo_bulk.cif", fmt="cif")
print("\nCreated MgO bulk structure:")
print(f"  Atoms: {mgo_bulk.num_sites}")
print(f"  Formula: {mgo_bulk.formula}")

print("\n" + "-" * 80)
print("Complete Pipeline:")
print("-" * 80)

print(
    """
Step 1: Generate surface slab
  atomate2siesta-structure slab mgo_bulk.cif \\
      --miller-indices 1,0,0 \\
      --min-slab-size 12.0 \\
      --min-vacuum-size 15.0 \\
      --symmetric

Step 2: Standardize to primitive cell
  atomate2siesta-structure standardize slab_mgo_bulk.cif --primitive

Step 3: Check lattice angles
  atomate2siesta-structure info primitive_slab_mgo_bulk.cif

Step 4: Optimize cell if not orthogonal
  atomate2siesta-structure optimize-cell primitive_slab_mgo_bulk.cif --orthogonalize

Step 5: Compare with original
  atomate2siesta-structure compare slab_mgo_bulk.cif orthogonal_primitive_slab_mgo_bulk.cif

Step 6: Ready for DFT!
  # Use optimized structure in workflow
"""
)

# ============================================================================
# Example 6: Surface Termination Analysis
# ============================================================================
print("\n" + "=" * 80)
print("Example 6: Surface Termination Analysis")
print("=" * 80)

print(
    """
For polar surfaces (like MgO 111), multiple terminations exist:

Generate all terminations:
  atomate2siesta-structure slab mgo_bulk.cif \\
      --miller-indices 1,1,1 \\
      --min-slab-size 12.0 \\
      --min-vacuum-size 15.0 \\
      --all-terminations

This creates multiple files:
  - slab_mgo_bulk_term1.cif  (Mg-terminated)
  - slab_mgo_bulk_term2.cif  (O-terminated)
  - slab_mgo_bulk_term3.cif  (Mixed termination)

Analyze each termination:
  for f in slab_mgo_bulk_term*.cif; do
      echo "Analyzing: $f"
      atomate2siesta-structure info $f
      atomate2siesta-structure standardize $f --primitive
  done

Select most stable termination based on:
  ✓ Stoichiometry (Mg:O ratio)
  ✓ Dipole moment (zero preferred)
  ✓ Surface energy (DFT calculation needed)
"""
)

# ============================================================================
# Example 7: Layer-by-Layer Analysis
# ============================================================================
print("\n" + "=" * 80)
print("Example 7: Layer-by-Layer Analysis")
print("=" * 80)

# Generate slab with specific number of layers
print(
    """
Control slab thickness by number of layers:

# Thin slab (3 layers)
atomate2siesta-structure slab cu_bulk.cif \\
    --miller-indices 1,1,1 \\
    --min-slab-size 6.0 \\
    --min-vacuum-size 15.0

# Medium slab (5 layers)
atomate2siesta-structure slab cu_bulk.cif \\
    --miller-indices 1,1,1 \\
    --min-slab-size 10.0 \\
    --min-vacuum-size 15.0

# Thick slab (7 layers)
atomate2siesta-structure slab cu_bulk.cif \\
    --miller-indices 1,1,1 \\
    --min-slab-size 14.0 \\
    --min-vacuum-size 15.0

Layer Convergence Test:
  1. Generate slabs with 3, 5, 7, 9 layers
  2. Standardize each to primitive cell
  3. Run DFT relaxation on each
  4. Compare surface energies
  5. Select minimum converged thickness

Rule of thumb:
  - 3 layers: Quick tests, trends only
  - 5 layers: Standard for most surfaces
  - 7 layers: Accurate surface energies
  - 9+ layers: Very accurate, expensive
"""
)

# ============================================================================
# Example 8: Integration with DFT Workflow
# ============================================================================
print("\n" + "=" * 80)
print("Example 8: Integration with DFT Workflow")
print("=" * 80)

# Generate final optimized surface
slabgen_final = SlabGenerator(
    cu_bulk,
    miller_index=[1, 1, 1],
    min_slab_size=10.0,
    min_vacuum_size=15.0,
    center_slab=True,
)

cu_111_final = slabgen_final.get_slab()
cu_111_final.to(filename="cu_111_final.cif", fmt="cif")

print(
    """
Complete Workflow for DFT Surface Calculation:

1. Prepare surface:
   atomate2siesta-structure slab cu_bulk.cif \\
       --miller-indices 1,1,1 \\
       --min-slab-size 10.0 \\
       --min-vacuum-size 15.0 \\
       --symmetric

2. Standardize:
   atomate2siesta-structure standardize slab_cu_bulk.cif --primitive

3. Optimize cell:
   atomate2siesta-structure optimize-cell primitive_slab_cu_bulk.cif --orthogonalize

4. Verify structure:
   atomate2siesta-structure info orthogonal_primitive_slab_cu_bulk.cif

5. Generate DFT workflow:
   atomate2siesta-maker relax orthogonal_primitive_slab_cu_bulk.cif \\
       --preset surface_standard \\
       --execution-mode local

6. Run calculation:
   python relax_orthogonal_primitive_slab_cu_bulk.py

Benefits of this pipeline:
  ✓ Minimal number of atoms (primitive cell)
  ✓ Orthogonal cell (better k-point sampling)
  ✓ Standardized structure (reproducible)
  ✓ Optimal vacuum (converged surface energy)
  ✓ Symmetric slab (no net dipole)
"""
)

# ============================================================================
# Example 9: Advanced Surface Features
# ============================================================================
print("\n" + "=" * 80)
print("Example 9: Advanced Surface Features")
print("=" * 80)

print(
    """
Advanced slab manipulation:

1. Selective relaxation (fix middle layers):
   # This requires modifying the structure
   # Tag atoms in bottom layers as "selective_dynamics"

   from pymatgen.core import Structure

   slab = Structure.from_file("cu_111_final.cif")

   # Calculate z-coordinates
   z_coords = [site.frac_coords[2] for site in slab]
   z_min, z_max = min(z_coords), max(z_coords)
   z_range = z_max - z_min

   # Fix middle 40% of slab
   selective_dynamics = []
   for site in slab:
       z = site.frac_coords[2]
       z_rel = (z - z_min) / z_range

       # Allow relaxation if in top/bottom 30%
       if z_rel < 0.3 or z_rel > 0.7:
           selective_dynamics.append([True, True, True])
       else:
           selective_dynamics.append([False, False, False])

   slab.add_site_property("selective_dynamics", selective_dynamics)
   slab.to("cu_111_selective.cif", fmt="cif")

2. Add adsorbate on surface:
   atomate2siesta-structure molecule --formula CO
   atomate2siesta-structure attach cu_111_final.cif molecule_CO.cif \\
       --position top --distance 2.0

3. Create stepped surface:
   # Generate high-index surface (e.g., 211)
   atomate2siesta-structure slab cu_bulk.cif \\
       --miller-indices 2,1,1 \\
       --min-slab-size 15.0 \\
       --min-vacuum-size 20.0

4. Surface reconstruction:
   # Manually modify top layer atom positions
   # Then run geometry optimization with SIESTA
"""
)

# ============================================================================
# Summary and Best Practices
# ============================================================================
print("\n" + "=" * 80)
print("Summary and Best Practices")
print("=" * 80)

print(
    """
Surface Preparation Checklist:

1. Bulk Structure Quality:
   ✓ Use experimental lattice parameters
   ✓ Ensure correct space group
   ✓ Verify stoichiometry
   ✓ Standardize to primitive cell first

2. Slab Generation:
   ✓ Choose appropriate Miller index
   ✓ Use enough layers (≥5 for accuracy)
   ✓ Add sufficient vacuum (≥15 Å)
   ✓ Center slab in cell
   ✓ Consider symmetric vs asymmetric

3. Cell Optimization:
   ✓ Standardize to primitive surface cell
   ✓ Orthogonalize if needed (better k-points)
   ✓ Verify no artificial strain introduced
   ✓ Check lattice angles (close to 90°?)

4. DFT Preparation:
   ✓ Use tier preset (e.g., surface_standard)
   ✓ Increase k-points in slab plane
   ✓ Reduce k-points perpendicular to slab
   ✓ Consider dipole corrections if asymmetric
   ✓ Fix bottom layers if needed

5. Validation:
   ✓ Compare with original bulk
   ✓ Verify surface area is reasonable
   ✓ Check atom count is expected
   ✓ Run test calculation before production

Common Surface Orientations:

Material  | Stable Surfaces | Notes
----------|----------------|------------------------
FCC       | (111), (100)   | Close-packed vs open
BCC       | (110), (100)   | (110) most stable
Diamond   | (100), (111)   | Reconstructions common
Rocksalt  | (100), (111)   | (100) non-polar
Rutile    | (110), (100)   | (110) most stable

Surface Energy Convergence:

Parameter        | Typical Values
-----------------|----------------------------------
Slab thickness   | 5-7 layers (10-14 Å)
Vacuum size      | 15-20 Å
k-point density  | 2× bulk density in slab plane
Energy cutoff    | Same as bulk
SCF tolerance    | 1×10⁻⁵ eV (tighter than bulk)

Troubleshooting:

Issue: Non-zero dipole moment
Fix: Use symmetric slab or add dipole correction

Issue: Too many atoms
Fix: Standardize to primitive cell first

Issue: Non-orthogonal cell
Fix: Use --orthogonalize option

Issue: Surface atoms too close
Fix: Increase min_slab_size

Issue: Vacuum too small
Fix: Increase min_vacuum_size (≥15 Å)

Command Quick Reference:
  atomate2siesta-structure slab --help
  atomate2siesta-structure standardize --help
  atomate2siesta-structure optimize-cell --help
  atomate2siesta-structure compare --help
"""
)

# ============================================================================
# Cleanup
# ============================================================================
print("\nCleaning up generated files...")
cleanup_files = [
    "al_bulk.cif",
    "al_111_slab.cif",
    "cu_bulk.cif",
    "cu_100_slab.cif",
    "cu_110_slab.cif",
    "cu_111_slab.cif",
    "al_111_vacuum_10.cif",
    "al_111_vacuum_15.cif",
    "al_111_vacuum_20.cif",
    "al_111_vacuum_25.cif",
    "al_111_symmetric.cif",
    "mgo_bulk.cif",
    "cu_111_final.cif",
]

for f in cleanup_files:
    if Path(f).exists():
        Path(f).unlink()

print("✓ Tutorial complete!")
print("\nNext tutorial: 04_complete_workflow.py")
