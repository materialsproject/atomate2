"""
Tutorial: Surface Vacancy Generation with SurfaceVacancyGenerator

Category: Defects
Difficulty: Advanced
Time: ~10 min (dry-run), ~hours (full calculations)

This tutorial demonstrates surface-aware vacancy generation for slab structures.
Critical for catalysis, electrocatalysis, and surface chemistry calculations.

Key Features:
- Automatic surface layer identification by z-coordinate
- Top/bottom/both surface selection
- Multiple surface layer support
- In-plane symmetry reduction
- Integration with DefectFlowMaker

Why SurfaceVacancyGenerator vs SiestaVacancyGenerator?
- SiestaVacancyGenerator: Uses 3D bulk symmetry, treats all atoms equally
- SurfaceVacancyGenerator: Surface-aware, distinguishes surface vs bulk atoms
"""

from pymatgen.core import Lattice, Structure
from jobflow import run_locally

from atomate2.siesta.flows.defects import (
    SurfaceVacancyGenerator,
    DefectFlowMaker,
)

# ============================================================================
# EXAMPLE 1: Basic surface vacancy - MoS₂ monolayer
# ============================================================================

print("=" * 70)
print("EXAMPLE 1: Surface S vacancy on MoS₂ slab (topmost layer only)")
print("=" * 70)

# Create MoS₂ slab (2H structure, 3 layers thick)
# This is a simplified structure for demonstration
# In practice, use pymatgen's SlabGenerator or atomate2siesta's SurfaceMaker
lattice = Lattice.hexagonal(3.16, 20.0)  # c-axis includes vacuum

mos2_slab = Structure(
    lattice=lattice,
    species=["Mo", "S", "S", "Mo", "S", "S", "Mo", "S", "S"],
    coords=[
        # Bottom layer
        [0.333, 0.667, 0.35],  # Mo
        [0.667, 0.333, 0.32],  # S
        [0.667, 0.333, 0.38],  # S
        # Middle layer
        [0.000, 0.000, 0.45],  # Mo
        [0.333, 0.667, 0.42],  # S
        [0.333, 0.667, 0.48],  # S
        # Top layer (surface)
        [0.333, 0.667, 0.55],  # Mo
        [0.667, 0.333, 0.52],  # S (lower)
        [0.667, 0.333, 0.58],  # S (upper) <-- This is the top surface!
    ],
)

# Save structure for inspection
mos2_slab.to(filename="mos2_slab_3layer.cif")
print(f"\nMoS₂ slab: {mos2_slab.composition}")
print(f"  Total atoms: {len(mos2_slab)}")
print(f"  Lattice c: {mos2_slab.lattice.c:.2f} Å (includes vacuum)")

# Create surface vacancy generator
generator = SurfaceVacancyGenerator(
    slab_structure=mos2_slab,
    surface_layers=1,  # Only topmost layer
    surface_side="top",  # Top surface only (not bottom)
    layer_tolerance=0.5,  # Å tolerance for grouping atoms
    use_ghost_atoms=True,  # SIESTA-specific ghost atoms
    use_in_plane_symmetry=True,  # Use 2D symmetry (not 3D)
)

# Generate S vacancies on top surface
print("\n--- Generating S vacancies on top surface ---")
surface_defects = generator.generate_defects(species="S")

print(f"\nGenerated {len(surface_defects)} S vacancy defect(s) on top surface")

for i, defect in enumerate(surface_defects):
    print(f"\nDefect {i+1}:")
    print(f"  Species: V_{defect['species']}")
    print(f"  Layer: {defect['layer_index']}")
    print(f"  Layer z-position: {defect['layer_z_position']:.3f} Å")
    print(f"  Is top surface: {defect['is_top_surface']}")
    print(f"  Is bottom surface: {defect['is_bottom_surface']}")
    print(f"  Fractional coords: {defect['frac_coords']}")
    print(f"  Uses ghost atom: {defect['use_ghost']}")

    # Save defect structure
    defect["structure"].to(filename=f"mos2_V_S_layer{defect['layer_index']}.cif")
    print(f"  Saved: mos2_V_S_layer{defect['layer_index']}.cif")

# ============================================================================
# EXAMPLE 2: Subsurface vacancies (top 2 layers)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 2: Surface + subsurface vacancies (top 2 layers)")
print("=" * 70)

# Generate vacancies in top TWO layers (surface + subsurface)
generator_subsurface = SurfaceVacancyGenerator(
    slab_structure=mos2_slab,
    surface_layers=2,  # Top two layers
    surface_side="top",
    use_ghost_atoms=True,
)

subsurface_defects = generator_subsurface.generate_defects(species="S")

print(f"\nGenerated {len(subsurface_defects)} S vacancy defects (surface + subsurface)")

# Group by layer
defects_by_layer = {}
for defect in subsurface_defects:
    layer = defect["layer_index"]
    if layer not in defects_by_layer:
        defects_by_layer[layer] = []
    defects_by_layer[layer].append(defect)

for layer, layer_defects in sorted(defects_by_layer.items()):
    print(f"\nLayer {layer} (z = {layer_defects[0]['layer_z_position']:.2f} Å):")
    print(f"  {len(layer_defects)} S vacancy site(s)")

# ============================================================================
# EXAMPLE 3: Both surfaces (symmetric slab)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 3: Both top and bottom surfaces")
print("=" * 70)

generator_both = SurfaceVacancyGenerator(
    slab_structure=mos2_slab,
    surface_layers=1,
    surface_side="both",  # Both top AND bottom
    use_ghost_atoms=True,
)

both_surface_defects = generator_both.generate_defects(species="S")

print(f"\nGenerated {len(both_surface_defects)} S vacancy defects (both surfaces)")

top_defects = [d for d in both_surface_defects if d["is_top_surface"]]
bottom_defects = [d for d in both_surface_defects if d["is_bottom_surface"]]

print(f"  Top surface: {len(top_defects)} defect(s)")
print(f"  Bottom surface: {len(bottom_defects)} defect(s)")

# ============================================================================
# EXAMPLE 4: Multiple charge states with supercell
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 4: Multiple charge states with 3×3×1 supercell")
print("=" * 70)

# For charged defects, use larger supercell (in-plane expansion only!)
generator_charged = SurfaceVacancyGenerator(
    slab_structure=mos2_slab,
    surface_layers=1,
    surface_side="top",
    use_ghost_atoms=True,
)

charged_defects = generator_charged.generate_defects(
    species="S",
    supercell_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 1]],  # 3×3×1 (in-plane only!)
    charge_states=[0, +1, +2],  # Neutral, +1, +2
)

print(f"\nGenerated {len(charged_defects)} defects (1 site × 3 charge states)")

for defect in charged_defects:
    q = defect["charge_state"]
    print(f"  V_S^{q:+d} in {defect['structure'].composition}")

# ============================================================================
# EXAMPLE 5: Integration with DefectFlowMaker
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 5: Create full defect workflow with DefectFlowMaker")
print("=" * 70)

# Generate surface defect
generator = SurfaceVacancyGenerator(
    slab_structure=mos2_slab,
    surface_layers=1,
    surface_side="top",
    use_ghost_atoms=True,
)

defects = generator.generate_defects(
    species="S",
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 1]],  # 2×2×1
    charge_states=[0, +2],  # Neutral and +2
)

print(f"\nCreating {len(defects)} DefectFlowMaker workflows...")

# Create workflow for each defect
for i, defect_info in enumerate(defects):
    q = defect_info["charge_state"]
    layer = defect_info["layer_index"]

    print(f"\nDefect {i+1}: V_S^{q:+d} in layer {layer}")

    # Create DefectFlowMaker
    flow_maker = DefectFlowMaker(
        epsilon_static=15.0,  # MoS₂ dielectric constant (in-plane)
        epsilon_parallel=15.0,  # In-plane ε (for 2D slab correction)
        epsilon_perpendicular=7.0,  # Out-of-plane ε (for 2D slab correction)
        correction_scheme="slab2d",  # Use 2D slab correction (anisotropic)
        defect_type="vacancy",
        charge_state=q,
        use_ghost_atoms=True,
        dry_run=True,  # Set to False for real calculations
        chemical_potentials={"S": -2.5},  # S chemical potential (example)
    )

    # Create workflow
    flow = flow_maker.make(
        defect_structure=defect_info["structure"],
        host_structure=defect_info["host_structure"],
        defect_site=defect_info["frac_coords"],
        defect_species="S",
    )

    print(f"  Created workflow: {flow.name}")
    print(f"  Jobs in workflow: {len(flow.jobs)}")

    # Run workflow (dry-run mode)
    if i == 0:  # Only run first one for demo
        print("  Running dry-run workflow...")
        results = run_locally(flow, create_folders=True)
        print("  Dry-run completed! Check job folders for FDF files.")

# ============================================================================
# EXAMPLE 6: Disable in-plane symmetry (get ALL surface sites)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 6: Disable in-plane symmetry (all surface sites)")
print("=" * 70)

# Useful for asymmetric surfaces (steps, kinks, reconstructions)
generator_no_sym = SurfaceVacancyGenerator(
    slab_structure=mos2_slab,
    surface_layers=1,
    surface_side="top",
    use_ghost_atoms=True,
    use_in_plane_symmetry=False,  # Disable symmetry reduction
)

no_sym_defects = generator_no_sym.generate_defects(species="S")

print(f"\nGenerated {len(no_sym_defects)} S vacancy defects (no symmetry reduction)")
print("  This returns ALL surface S atoms, not just unique sites")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("TUTORIAL SUMMARY")
print("=" * 70)
print(
    """
✓ Example 1: Basic surface vacancy (topmost layer only)
✓ Example 2: Subsurface vacancies (top 2 layers)
✓ Example 3: Both top and bottom surfaces
✓ Example 4: Multiple charge states with supercell
✓ Example 5: Integration with DefectFlowMaker
✓ Example 6: Disable in-plane symmetry

Key Takeaways:
1. SurfaceVacancyGenerator automatically identifies surface layers by z-coordinate
2. Use surface_layers=1 for topmost layer only (most common)
3. Use surface_side="top"/"bottom"/"both" to select which surface
4. For slabs, expand supercell in-plane only: [[n,0,0],[0,m,0],[0,0,1]]
5. Use correction_scheme="slab2d" for 2D slab corrections (anisotropic ε)
6. Ghost atoms are highly recommended for SIESTA surface calculations

Next Steps:
- Try with real surface structures (TiO₂, graphene, metal surfaces)
- Calculate formation energies with different chemical potentials
- Compare surface vs subsurface defect energies
- Study defect-adsorbate interactions

For bulk defects (3D periodicity), use SiestaVacancyGenerator instead.
"""
)

print("\nGenerated structure files:")
print("  - mos2_slab_3layer.cif (pristine slab)")
print("  - mos2_V_S_layer*.cif (vacancy structures)")
print("\nDry-run FDF files created in job_* directories")
