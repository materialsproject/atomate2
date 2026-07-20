#!/usr/bin/env python3
"""
Surface Energy Convergence Testing Tutorial
============================================

This tutorial demonstrates how to use SurfaceEnergyConvergenceFlowMaker
to systematically test convergence of surface energy calculations.

Surface energy calculations require careful convergence testing with respect to:
1. **Slab thickness** (number of atomic layers) - ensures bulk-like interior
2. **Vacuum thickness** - eliminates periodic image interactions

This workflow automates the convergence testing process and generates:
- Convergence plots (surface energy vs. parameter)
- Summary file with recommendations
- Automatic detection of converged parameters

Three convergence modes are available:
- "layers": Test slab thickness with fixed vacuum (most common)
- "vacuum": Test vacuum thickness with fixed layers
- "both": Full 2D grid of calculations (comprehensive)

IMPORTANT: Symmetric slabs (default)
-----------------------------------------
By default, the workflow generates symmetric slabs where both surfaces have
the same termination. This is required for physically meaningful surface
energy calculations. For rocksalt structures like MgO (100), use ODD layer
counts (5, 7, 9, 11...) to ensure symmetric terminations:
- 5 layers: Mg-O-Mg-O-Mg (Mg-terminated both sides)
- 7 layers: Mg-O-Mg-O-Mg-O-Mg (Mg-terminated both sides)

Author: atomate2siesta team
"""

# %% Imports
from pymatgen.core import Lattice, Structure
from jobflow import run_locally

from atomate2.siesta.flows.surface.convergence import (
    SurfaceEnergyConvergenceFlowMaker,
)
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset

# %% Example 1: Layer Convergence (Most Common Use Case)
# ======================================================
# Test how many atomic layers are needed for converged surface energy.
# This is the most important convergence test for surface calculations.

print("=" * 60)
print("Example 1: Slab Thickness (Layer) Convergence")
print("=" * 60)

# Create MgO bulk structure
lattice = Lattice.cubic(4.212)  # MgO lattice constant
mgo_bulk = Structure(
    lattice,
    ["Mg", "O"],
    [[0, 0, 0], [0.5, 0.5, 0.5]],
)

# Create makers with appropriate settings
# Bulk: use isotropic k-points
bulk_maker = StaticMaker()
bulk_maker = apply_tier_preset(bulk_maker, "surface_dirty")

# Slab: use reduced k-points in vacuum direction
slab_maker = StaticMaker()
slab_maker = apply_tier_preset(
    slab_maker,
    "surface_dirty",
    override_params={"a2s_kpts": [2, 2, 1]},  # Reduced along z
)

# Create convergence workflow
# NOTE: For rocksalt (100) like MgO, use ODD layer counts for symmetric slabs
# You can choose termination: "O" for O-terminated, "Mg" for Mg-terminated
layer_conv_maker = SurfaceEnergyConvergenceFlowMaker(
    name="MgO_100_layer_convergence",
    miller_index=(1, 0, 0),  # (100) surface
    bulk_static_maker=bulk_maker,
    slab_static_maker=slab_maker,
    # Test these slab thicknesses (ODD numbers for symmetric rocksalt slabs)
    slab_layers=[5, 7, 9, 11, 13],
    # Mode: only vary layers
    convergence_mode="layers",
    fixed_vacuum=15.0,  # Fixed vacuum during layer testing
    # Convergence criterion
    convergence_threshold=0.05,  # J/m² (use 0.01 for publication quality)
    # symmetrize=True is the default - generates symmetric slabs
    # termination="Mg",  # Uncomment for Mg-terminated slabs (default: first available)
)

# Create the flow
layer_flow = layer_conv_maker.make(mgo_bulk)

print(f"Created flow with {len(layer_flow.jobs)} jobs:")
for job in layer_flow.jobs:
    print(f"  - {job.name}")

# To run:
# results = run_locally(layer_flow, create_folders=True)


# %% Example 1b: Mg-Terminated Slabs
# ==================================
# For rocksalt structures, you can choose the termination.
# MgO (100) can be either O-terminated or Mg-terminated.

print("\n" + "=" * 60)
print("Example 1b: Mg-Terminated Slabs")
print("=" * 60)

mg_term_maker = SurfaceEnergyConvergenceFlowMaker(
    name="MgO_100_Mg_terminated",
    miller_index=(1, 0, 0),
    bulk_static_maker=bulk_maker,
    slab_static_maker=slab_maker,
    slab_layers=[5, 7, 9],
    convergence_mode="layers",
    fixed_vacuum=15.0,
    termination="Mg",  # <-- Explicitly request Mg-terminated slabs
)

mg_term_flow = mg_term_maker.make(mgo_bulk)

print(f"Created Mg-terminated flow with {len(mg_term_flow.jobs)} jobs:")
for job in mg_term_flow.jobs:
    print(f"  - {job.name}")


# %% Example 1c: With Chemical Potential (O-rich conditions)
# ==========================================================
# For non-stoichiometric slabs (Mg4O5 or Mg5O4), surface energy depends
# on chemical potential. The workflow calculates at bulk equilibrium by
# default, but you can provide explicit chemical potentials for O-rich
# or Mg-rich conditions.

print("\n" + "=" * 60)
print("Example 1c: O-rich Chemical Potential")
print("=" * 60)

# O-rich conditions: use μ_O from O₂ molecule (≈ -4.9 eV per O atom)
# This represents oxidizing conditions (e.g., air atmosphere)
o_rich_maker = SurfaceEnergyConvergenceFlowMaker(
    name="MgO_100_O_rich",
    miller_index=(1, 0, 0),
    bulk_static_maker=bulk_maker,
    slab_static_maker=slab_maker,
    slab_layers=[5, 7, 9],
    convergence_mode="layers",
    termination="O",  # O-terminated (has excess O atoms)
    # Chemical potential for excess O in O-rich limit
    # μ_O ≈ (1/2) × E(O₂) ≈ -4.9 eV (typical DFT value)
    excess_species_chemical_potential=-4.9,  # eV per excess O atom
)

o_rich_flow = o_rich_maker.make(mgo_bulk)

print(f"Created O-rich workflow with {len(o_rich_flow.jobs)} jobs")
print("Surface energy will be calculated using μ_O = -4.9 eV (O-rich limit)")
print("")
print("For other conditions:")
print("  - O-rich limit: μ_O ≈ -4.9 eV (from O₂ molecule)")
print("  - Mg-rich limit: μ_Mg ≈ -1.5 eV (from Mg bulk metal)")
print("  - Bulk equilibrium: μ calculated from bulk MgO energy (default)")


# %% Example 2: Vacuum Convergence
# ================================
# Test how much vacuum is needed to avoid periodic image interactions.
# Usually less critical than layer convergence, but important for
# systems with large dipole moments.

print("\n" + "=" * 60)
print("Example 2: Vacuum Thickness Convergence")
print("=" * 60)

vacuum_conv_maker = SurfaceEnergyConvergenceFlowMaker(
    name="MgO_100_vacuum_convergence",
    miller_index=(1, 0, 0),
    bulk_static_maker=bulk_maker,
    slab_static_maker=slab_maker,
    # Test these vacuum thicknesses
    vacuum_sizes=[10.0, 12.5, 15.0, 17.5, 20.0, 25.0],
    # Mode: only vary vacuum
    convergence_mode="vacuum",
    fixed_layers=9,  # Use 9 layers (odd for symmetric rocksalt slab)
    convergence_threshold=0.02,  # J/m²
)

vacuum_flow = vacuum_conv_maker.make(mgo_bulk)

print(f"Created flow with {len(vacuum_flow.jobs)} jobs:")
for job in vacuum_flow.jobs:
    print(f"  - {job.name}")


# %% Example 3: Full 2D Convergence Grid
# =====================================
# For publication-quality results, test both parameters simultaneously.
# This creates a grid of calculations and shows how the parameters interact.

print("\n" + "=" * 60)
print("Example 3: Full 2D Convergence Grid (layers × vacuum)")
print("=" * 60)

full_conv_maker = SurfaceEnergyConvergenceFlowMaker(
    name="MgO_111_full_convergence",
    miller_index=(1, 1, 1),  # (111) surface
    bulk_static_maker=bulk_maker,
    slab_static_maker=slab_maker,
    # Test grid: 4 layers × 4 vacuums = 16 slab calculations + 1 bulk
    # Note: (111) surface symmetry differs from (100), check terminations
    slab_layers=[5, 7, 9, 11],
    vacuum_sizes=[10.0, 15.0, 20.0, 25.0],
    convergence_mode="both",
    convergence_threshold=0.01,  # Publication quality
)

full_flow = full_conv_maker.make(mgo_bulk)

print(f"Created flow with {len(full_flow.jobs)} jobs:")
print("  - 1 bulk calculation")
print(f"  - {len(full_flow.jobs) - 2} slab calculations (4 layers × 4 vacuums)")
print("  - 1 analysis job")


# %% Example 4: With Custodian Error Handling
# ===========================================
# Enable automatic error recovery for production calculations.

print("\n" + "=" * 60)
print("Example 4: With Custodian Error Handling")
print("=" * 60)

# Create makers with custodian enabled
bulk_maker_cust = StaticMaker(use_custodian=True, custodian_max_errors=5)
bulk_maker_cust = apply_tier_preset(bulk_maker_cust, "surface_basic")

slab_maker_cust = StaticMaker(use_custodian=True, custodian_max_errors=5)
slab_maker_cust = apply_tier_preset(
    slab_maker_cust,
    "surface_basic",
    override_params={"a2s_kpts": [3, 3, 1]},
)

robust_conv_maker = SurfaceEnergyConvergenceFlowMaker(
    name="MgO_110_robust_convergence",
    miller_index=(1, 1, 0),
    bulk_static_maker=bulk_maker_cust,
    slab_static_maker=slab_maker_cust,
    # (110) surface: check terminations for your specific structure
    slab_layers=[5, 7, 9, 11, 13, 15],
    convergence_mode="layers",
    fixed_vacuum=15.0,
    convergence_threshold=0.01,
    # Custodian settings propagate from makers
)

robust_flow = robust_conv_maker.make(mgo_bulk)
print(f"Created robust workflow with {len(robust_flow.jobs)} jobs")
print("Custodian will automatically handle SCF convergence issues")


# %% Example 5: Dry Run Mode (Testing)
# ====================================
# Use dry_run=True to test the workflow without running SIESTA.
# This generates all input files but skips actual calculations.

print("\n" + "=" * 60)
print("Example 5: Dry Run Mode (for testing)")
print("=" * 60)

# Create makers with dry_run, custodian, AND presets applied
dry_bulk_maker = StaticMaker(dry_run=False, use_custodian=True, custodian_max_errors=5)
# dry_bulk_maker = apply_tier_preset(dry_bulk_maker, "surface_dirty")
dry_bulk_maker = apply_tier_preset(dry_bulk_maker, "surface_basic")

dry_slab_maker = StaticMaker(dry_run=False, use_custodian=True, custodian_max_errors=5)
# dry_slab_maker = apply_tier_preset(dry_slab_maker, "surface_dirty")
dry_slab_maker = apply_tier_preset(dry_slab_maker, "surface_basic")

dry_conv_maker = SurfaceEnergyConvergenceFlowMaker(
    name="MgO_100_dry_run_test",
    miller_index=(1, 0, 0),
    bulk_static_maker=dry_bulk_maker,
    slab_static_maker=dry_slab_maker,
    slab_layers=[1, 2, 3, 4, 5, 6, 7],  # Odd layers for symmetric rocksalt slabs
    convergence_mode="layers",
    # dry_run=True,
    termination="Mg",
    # Diffuse basis parameters
    apply_diffuse_basis=True,  # Enable diffuse basis for surface atoms
    surface_basis="DZP",  # Larger basis for surface atoms
    bulk_basis="DZ",  # Standard basis for interior atoms
)

dry_flow = dry_conv_maker.make(mgo_bulk)

print("Dry run mode: Will generate input files without running SIESTA")
print("Useful for verifying workflow setup before production runs")

# Actually run the dry-run workflow
print("\nRunning dry-run workflow...")
dry_results = run_locally(dry_flow, create_folders=True)
print("Dry run complete! Check job_* folders for generated input files.")


# %% Expected Outputs
# ===================
"""
After running a convergence workflow, you will find:

1. **surface_convergence_summary.txt**
   - Complete analysis with all calculation results
   - Convergence detection (converged or not)
   - Recommended parameters for production calculations
   - Surface energy statistics

2. **Convergence Plots** (depends on mode):
   - "layers" mode: surface_convergence_layers.png
     - Left: Surface energy vs. number of layers
     - Right: ΔSurface energy vs. layers (with threshold line)

   - "vacuum" mode: surface_convergence_vacuum.png
     - Left: Surface energy vs. vacuum thickness
     - Right: ΔSurface energy vs. vacuum

   - "both" mode:
     - surface_convergence_grid.png (heatmap)
     - surface_convergence_lines.png (line plots)

3. **Analysis Output Dictionary**:
   ```python
   {
       "miller_index": (1, 0, 0),
       "converged": True,
       "converged_at": {"n_layers": 8, "vacuum": 15.0, ...},
       "recommended_layers": 8,
       "recommended_vacuum": 15.0,
       "final_surface_energy_Jm2": 1.234,
       "results": [...],  # All calculation results
       "summary_file": "surface_convergence_summary.txt",
       "plot_files": {"layers": "surface_convergence_layers.png"},
   }
   ```

Convergence Criteria Guidelines:
- Publication quality: < 0.01 J/m² change
- Standard calculations: < 0.05 J/m² change
- Quick screening: < 0.1 J/m² change

Typical Convergence Behavior:
- Simple metals (Al, Cu): 4-6 layers usually sufficient
- Oxides (MgO, TiO₂): 6-10 layers often needed
- Surfaces with strong relaxation: May need 10-15 layers
- Vacuum: 12-15 Å is typically sufficient for most systems

Symmetric Slab Guidelines:
- By default (symmetrize=True), slabs have identical terminations on both surfaces
- For rocksalt (100) like MgO/NaCl: Use ODD layer counts (5, 7, 9, 11...)
- For FCC (111) metals: Even or odd may work depending on termination desired
- For FCC (100) metals: Usually any count works (single element)
- Always verify terminations visually for new materials!
"""

print("\n" + "=" * 60)
print("Tutorial Complete!")
print("=" * 60)
print(
    """
Key points:
1. Use "layers" mode for slab thickness convergence (most common)
2. Use "vacuum" mode for vacuum thickness convergence
3. Use "both" mode for comprehensive publication-quality testing
4. Set convergence_threshold based on required accuracy:
   - 0.01 J/m² for publication
   - 0.05 J/m² for standard calculations
5. Always use reduced k-points in the vacuum direction for slabs
6. Use ODD layer counts for rocksalt (100) surfaces (MgO, NaCl, etc.)
   to ensure symmetric terminations (e.g., 5, 7, 9, 11 layers)
7. Symmetric slabs (default) are required for physically meaningful
   surface energies - both surfaces must have the same termination
8. Choose termination with termination="O" or termination="Mg"
9. For non-stoichiometric slabs (like O-terminated MgO with Mg4O5):
   - Default: bulk equilibrium chemical potential (reference value)
   - O-rich: excess_species_chemical_potential=-4.9 (from O2)
   - Mg-rich: excess_species_chemical_potential=-1.5 (from Mg metal)
"""
)
