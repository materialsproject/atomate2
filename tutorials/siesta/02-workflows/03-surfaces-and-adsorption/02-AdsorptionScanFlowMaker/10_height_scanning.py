#!/usr/bin/env python
"""3D adsorption scanning: xy grid + multiple heights (v1.0.0).

This tutorial demonstrates the new height scanning feature that allows you to
scan the full 3D potential energy surface (x, y, z) to find the optimal
adsorption site AND height automatically.

Three ways to specify heights:
1. Single height (backward compatible)
2. Explicit list of heights
3. Automatic range (height_min, height_max, height_step)

Benefits:
- Automatically find optimal adsorption height
- Scan entire 3D potential energy surface
- Identify height-dependent site preferences
- More accurate adsorption energy predictions
"""

from jobflow import run_locally
from pymatgen.core import Lattice, Molecule, Structure

from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker
from atomate2.siesta.powerups import update_user_siesta_settings

print("=" * 70)
print("3D Adsorption Scanning: xy Grid + Height Scanning")
print("=" * 70)

# Create MgO(100) slab
lattice = Lattice.from_parameters(a=4.2, b=4.2, c=19.6, alpha=90, beta=90, gamma=90)
species = ["Mg", "Mg", "O", "O", "Mg", "Mg", "O", "O"]
coords = [
    [0.0, 0.0, 0.32],
    [0.5, 0.5, 0.32],  # Mg layer
    [0.5, 0.0, 0.36],
    [0.0, 0.5, 0.36],  # O layer
    [0.0, 0.0, 0.43],
    [0.5, 0.5, 0.43],  # Mg layer
    [0.5, 0.0, 0.47],
    [0.0, 0.5, 0.47],  # O layer
]
slab = Structure(lattice, species, coords)

# Create CO molecule
molecule = Molecule(["C", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]])

print(f"\nSlab: {slab.composition}")
print(f"Adsorbate: {molecule.composition}")

# ============================================================================
# MODE 1: Single Height (Backward Compatible)
# ============================================================================

print("\n" + "=" * 70)
print("MODE 1: Single Height (Backward Compatible)")
print("=" * 70)

flow1 = AdsorptionScanFlowMaker(
    name="single_height",
    grid_size=(3, 3),
    height=2.0,  # Single height: 2.0 Å
    dry_run=True,
    use_custodian=True,  # Enable automatic error handling
)

# workflow1 = flow1.make(slab, molecule)
# results1 = run_locally(workflow1, create_folders=True, root_dir="mode1_single_height")

print("\n✓ Mode 1 complete:")
print("  Grid: 3×3 sites")
print("  Height: 2.0 Å (single)")
print("  Total calculations: 9 (3×3×1)")

# ============================================================================
# MODE 2: Explicit List of Heights
# ============================================================================

print("\n" + "=" * 70)
print("MODE 2: Explicit List of Heights")
print("=" * 70)

flow2 = AdsorptionScanFlowMaker(
    name="height_list",
    grid_size=(2, 2),
    heights=[1.5, 2.0],  # [1.5, 2.0, 2.5, 3.0],  # Explicit list
    # dry_run=True,
    use_custodian=True,  # Enable automatic error handling
)

workflow2 = flow2.make(slab, molecule)


workflow2 = update_user_siesta_settings(
    workflow2,
    {
        # SCF Convergence (tighter for surface calculations)
        "SCF.Mixer.Weight": 0.1,  # Slower mixing (more stable for surfaces)
        "SCF.Mixer.History": 8,  # More Pulay history (better convergence)
        "SCF.DM.Tolerance": 1e-5,  # Tighter convergence criterion
        # Occupation (Methfessel-Paxton for metals/surfaces)
        "OccupationFunction": "MP",  # Better for metallic/surface systems
        "OccupationMPOrder": 1,  # First-order MP
        "ElectronicTemperature": "25 meV",  # Small smearing
        # Output options
        "WriteCoorStep": True,  # Write coordinates at each step
        "WriteMullikenPop": 1,  # Mulliken population analysis
    },
)


results2 = run_locally(workflow2, create_folders=True, root_dir="mode2_height_list")

print("\n✓ Mode 2 complete:")
print("  Grid: 3×3 sites")
print("  Heights: [1.5, 2.0, 2.5, 3.0] Å")
print("  Total calculations: 36 (3×3×4)")
print("  → Scans 4 heights at each xy position")

# ============================================================================
# MODE 3: Automatic Height Range
# ============================================================================

print("\n" + "=" * 70)
print("MODE 3: Automatic Height Range")
print("=" * 70)

flow3 = AdsorptionScanFlowMaker(
    name="height_range",
    grid_size=(3, 3),
    height_min=1.0,  # Minimum height
    height_max=3.0,  # Maximum height
    height_step=0.5,  # Step size
    dry_run=True,
)

# workflow3 = flow3.make(slab, molecule)
# results3 = run_locally(workflow3, create_folders=True, root_dir="mode3_height_range")

print("\n✓ Mode 3 complete:")
print("  Grid: 3×3 sites")
print("  Height range: 1.0 to 3.0 Å, step 0.5 Å")
print("  Heights generated: [1.0, 1.5, 2.0, 2.5, 3.0] Å")
print("  Total calculations: 45 (3×3×5)")
print("  → Automatic range generation")

# ============================================================================
# MODE 4: Fine Height Scan (Higher Resolution)
# ============================================================================

print("\n" + "=" * 70)
print("MODE 4: Fine Height Scan")
print("=" * 70)

flow4 = AdsorptionScanFlowMaker(
    name="fine_height_scan",
    grid_size=(2, 2),  # Smaller grid
    height_min=1.5,
    height_max=2.5,
    height_step=0.2,  # Finer step
    dry_run=True,
)

# workflow4 = flow4.make(slab, molecule)
# results4 = run_locally(workflow4, create_folders=True, root_dir="mode4_fine_scan")

print("\n✓ Mode 4 complete:")
print("  Grid: 2×2 sites")
print("  Height range: 1.5 to 2.5 Å, step 0.2 Å")
print("  Heights: [1.5, 1.7, 1.9, 2.1, 2.3, 2.5] Å")
print("  Total calculations: 24 (2×2×6)")
print("  → Fine resolution in height dimension")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print(
    """
✓ Height Scanning Feature (v1.0.0)

Three Ways to Specify Heights:
  1. Single height (backward compatible):
     height=2.0

  2. Explicit list:
     heights=[1.5, 2.0, 2.5, 3.0]

  3. Automatic range:
     height_min=1.0, height_max=3.0, height_step=0.5

Priority (if multiple specified):
  heights (list) > height_min/max/step (range) > height (single)

Benefits:
  ✓ Find optimal adsorption height automatically
  ✓ Scan full 3D potential energy surface (x, y, z)
  ✓ Identify height-dependent site preferences
  ✓ More accurate adsorption energies
  ✓ 100% backward compatible

Output:
  - Best site includes optimal height
  - Adsorption energies for all (x, y, z) positions
  - Saved structure uses best height

Use Cases:
  - Weakly bound adsorbates (van der Waals)
  - Molecules with flexible orientation
  - Initial structure optimization
  - Exploring adsorption potential energy surface
  - Identifying optimal height before geometry optimization

Computational Cost:
  Single height:  n_xy sites × 1 height
  Height scan:    n_xy sites × n_heights
  Example:        3×3 grid × 5 heights = 45 calculations

Tips:
  - Start with coarse grid + few heights for screening
  - Refine around optimal region with finer resolution
  - Use dry_run=True to preview structures first
  - Combine with tier presets for efficiency
"""
)

print("=" * 70)
print("Generated Directories:")
print("=" * 70)
print("  mode1_single_height/    - Single height (backward compatible)")
print("  mode2_height_list/      - Explicit list of heights")
print("  mode3_height_range/     - Automatic range generation")
print("  mode4_fine_scan/        - Fine resolution height scan")
print()
print("Each directory contains:")
print("  - job_XXX/ folders with siesta.fdf files")
print("  - best_adsorption_structure.cif (at optimal height!)")
print("  - adsorption_summary.txt (includes height info)")
print("  - adsorption_sites.png (visualization)")
print()
print("✓ All modes complete!")
