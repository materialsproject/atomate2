"""
Tutorial: Surface Interstitial Generation with SurfaceInterstitialGenerator

Category: Defects
Difficulty: Advanced
Time: ~10 min (dry-run), ~hours (full calculations)

This tutorial demonstrates surface-aware interstitial generation for slab structures.
Critical for adsorption, intercalation, and surface doping studies.

Key Features:
- Automatic interstitial placement at surface sites
- Ontop/hollow site types (bridge sites coming soon)
- Configurable offset (adsorption vs intercalation)
- Minimum distance filtering
- Integration with DefectFlowMaker

Why SurfaceInterstitialGenerator vs SiestaInterstitialGenerator?
- SiestaInterstitialGenerator: Uses 3D bulk Voronoi sites
- SurfaceInterstitialGenerator: Surface-aware, places atoms near surface layers
"""

from pymatgen.core import Lattice, Structure
from jobflow import run_locally

from atomate2.siesta.flows.defects import (
    SurfaceInterstitialGenerator,
    DefectFlowMaker,
)

# ============================================================================
# EXAMPLE 1: H adsorption on MoS₂ surface (ontop site) - HER catalysis
# ============================================================================

print("=" * 70)
print("EXAMPLE 1: H adsorption on MoS₂ surface (ontop S sites)")
print("=" * 70)

# Load bulk MoS₂ structure and create slab by adding vacuum
print("\n--- Creating MoS₂ slab from bulk structure ---")
bulk_mos2 = Structure.from_file("../../../00-structures/Mos2.cif")
print(f"Bulk MoS₂: {bulk_mos2.composition}")
print(f"  Lattice: a={bulk_mos2.lattice.a:.3f} Å, c={bulk_mos2.lattice.c:.3f} Å")

# The bulk structure has 2 MoS₂ layers (6 atoms: 2 Mo + 4 S)
# Add vacuum by scaling the c-axis: c_new = c_bulk + vacuum
vacuum_thickness = 15.0  # Å
new_c = bulk_mos2.lattice.c + vacuum_thickness

# Create new lattice with vacuum
new_lattice = Lattice.from_parameters(
    a=bulk_mos2.lattice.a,
    b=bulk_mos2.lattice.b,
    c=new_c,
    alpha=bulk_mos2.lattice.alpha,
    beta=bulk_mos2.lattice.beta,
    gamma=bulk_mos2.lattice.gamma,
)

# Rescale fractional coordinates to center the slab
# Original: z in [0, 1] → New: z in [0, c_bulk/c_new]
scale_factor = bulk_mos2.lattice.c / new_c
new_coords = []
for site in bulk_mos2:
    new_z = (
        site.frac_coords[2] * scale_factor + (1 - scale_factor) / 2
    )  # Center in cell
    new_coords.append([site.frac_coords[0], site.frac_coords[1], new_z])

# Create slab structure
mos2_slab = Structure(
    new_lattice,
    [site.specie for site in bulk_mos2],
    new_coords,
    site_properties=bulk_mos2.site_properties,
)

print(f"  Created slab: {len(mos2_slab)} atoms")
print(f"  Slab thickness: {mos2_slab.lattice.c:.3f} Å (bulk + vacuum)")

# Verify structure quality
mo_indices = [i for i, site in enumerate(mos2_slab) if site.specie.symbol == "Mo"]
s_indices = [i for i, site in enumerate(mos2_slab) if site.specie.symbol == "S"]
print(f"  Mo atoms: {len(mo_indices)}, S atoms: {len(s_indices)}")
print(f"  MoS₂ layers: {len(mo_indices)}")

# Check Mo-S bond length (should be ~1.57 Å)
# Verify structure by checking z-coordinates directly
if mo_indices and s_indices:
    print("\n  Verifying structure geometry:")
    # Get z-coordinates in Cartesian (Å)
    for i, mo_idx in enumerate(mo_indices):
        mo_z = mos2_slab[mo_idx].coords[2]
        print(f"    Mo{i}: z = {mo_z:.3f} Å")

    for i, s_idx in enumerate(s_indices):
        s_z = mos2_slab[s_idx].coords[2]
        print(f"    S{i}: z = {s_z:.3f} Å")

    # Calculate minimum Mo-S distance (should be ~1.57 Å)
    min_dist = float("inf")
    for mo_idx in mo_indices:
        for s_idx in s_indices:
            dist = mos2_slab.get_distance(mo_idx, s_idx)
            if dist < min_dist:
                min_dist = dist
                best_mo = mo_idx
                best_s = s_idx

    print(f"\n  Mo-S bond length: {min_dist:.3f} Å ✓")
    print("    (2H-MoS₂ experimental value: ~2.41 Å)")
    print("    Note: z-component ~1.57 Å + xy-offset in hexagonal lattice")

print(f"\nMoS₂ slab: {mos2_slab.composition}")
print(f"  Total atoms: {len(mos2_slab)}")

# Create H adsorption generator (ontop sites)
generator = SurfaceInterstitialGenerator(
    slab_structure=mos2_slab,
    surface_layers=1,  # Top layer only
    surface_side="top",  # Top surface
    interstitial_offset=1.5,  # 1.5 Å above surface (typical for H-S bond)
    interstitial_site_type="ontop",  # Directly above S atoms
    min_dist_from_atoms=1.0,  # Minimum 1.0 Å from existing atoms
)

print("\n--- Generating H interstitials on surface S atoms (ontop) ---")
h_adsorption = generator.generate_defects(
    species="S",  # Reference surface S atoms
    interstitial_species="H",  # Add H atoms
)

print(f"\nGenerated {len(h_adsorption)} H adsorption site(s)")

for i, defect in enumerate(h_adsorption):
    print(f"\nDefect {i+1}: H adsorption")
    print(f"  Reference species: {defect['reference_species']}")
    print(f"  Interstitial species: {defect['species']}")
    print(f"  Site type: {defect['site_type']}")
    print(f"  Layer: {defect['layer_index']}")
    print(f"  Reference layer z: {defect['layer_z_position']:.3f} Å")
    print(f"  Interstitial z: {defect['interstitial_z']:.3f} Å")
    print(f"  Offset: {defect['interstitial_z'] - defect['layer_z_position']:.3f} Å")
    print(f"  Fractional coords: {defect['frac_coords']}")

# ============================================================================
# EXAMPLE 2: H adsorption with multiple charge states (HER study)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 2: H adsorption with charge states (HER mechanism)")
print("=" * 70)

generator_charged = SurfaceInterstitialGenerator(
    slab_structure=mos2_slab,
    surface_layers=1,
    surface_side="top",
    interstitial_offset=1.5,
    interstitial_site_type="ontop",
)

# Generate H with neutral and charged states
h_charged = generator_charged.generate_defects(
    species="S",
    interstitial_species="H",
    charge_states=[0, -1],  # Neutral H and H⁻ (hydride)
)

print(f"\nGenerated {len(h_charged)} H adsorption configurations")

charges_count = {}
for defect in h_charged:
    q = defect["charge_state"]
    charges_count[q] = charges_count.get(q, 0) + 1

for q, count in sorted(charges_count.items()):
    print(f"  H^{q:+d}: {count} configuration(s)")

# ============================================================================
# EXAMPLE 3: Hollow site interstitials (Li intercalation)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 3: Li intercalation at hollow sites")
print("=" * 70)

# For intercalation, place Li at hollow sites (cell center in xy, gap center in z)
# NEGATIVE offset triggers automatic interlayer gap detection
# The generator finds the van der Waals gap below the surface and places Li there
# Structure: Layer 1 (9.28-12.41 Å) | Gap (12.41-15.97 Å) | Layer 2 (15.97-19.10 Å)
generator_hollow = SurfaceInterstitialGenerator(
    slab_structure=mos2_slab,
    surface_layers=1,
    surface_side="top",
    interstitial_offset=-0.2,  # Small negative offset from gap center
    # Gap center auto-detected at ~14.19 Å (between MoS₂ layers)
    # offset=-0.2 → Li at ~13.99 Å (slightly below center, still in gap)
    # For exact gap center, use offset=0.0
    interstitial_site_type="hollow",  # At cell center (xy), gap center (z)
    min_dist_from_atoms=0.5,  # Smaller for intercalation (atoms are further away)
)

li_intercalation = generator_hollow.generate_defects(
    species="S",  # Reference S atoms (for layer identification)
    interstitial_species="Li",
    charge_states=[0, +1],
)

print(f"\nGenerated {len(li_intercalation)} Li intercalation site(s)")

for i, defect in enumerate(li_intercalation):
    if i < 2:  # Show first 2 for brevity
        print(f"\nLi intercalation {i+1}:")
        print(f"  Site type: {defect['site_type']}")
        print(f"  Reference layer z: {defect['layer_z_position']:.3f} Å")
        print(f"  Li z-position: {defect['interstitial_z']:.3f} Å")
        print(
            f"  Offset: {defect['interstitial_z'] - defect['layer_z_position']:.3f} Å"
        )
        print(f"  Charge: {defect['charge_state']:+d}")

# Run Li intercalation through DefectFlowMaker
print("\nRunning Li intercalation workflows...")
for i, defect_info in enumerate(li_intercalation[:1]):  # First one for demo
    q = defect_info["charge_state"]
    print(f"\n  Li intercalation workflow (charge {q:+d})...")

    flow_maker = DefectFlowMaker(
        epsilon_static=15.0,
        epsilon_parallel=15.0,
        epsilon_perpendicular=7.0,
        correction_scheme="slab2d",
        defect_type="interstitial",
        charge_state=q,
        use_ghost_atoms=False,
        dry_run=True,
        chemical_potentials={"Li": -1.9},  # Li metal reference
    )

    flow = flow_maker.make(
        defect_structure=defect_info["structure"],
        host_structure=defect_info["host_structure"],
        defect_site=defect_info["frac_coords"],
        defect_species="Li",
    )

    results = run_locally(flow, create_folders=True)
    print("  ✓ Li intercalation workflow completed (check job_* folders)")

# ============================================================================
# EXAMPLE 4: O adsorption on metal surface (multiple sites)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 4: O adsorption on metal surface (catalysis)")
print("=" * 70)

# Create simple Pt(111) slab (for demonstration)
pt_lattice = Lattice.from_parameters(
    a=2.77, b=2.77, c=15.0, alpha=90, beta=90, gamma=60
)

pt_slab = Structure(
    pt_lattice,
    ["Pt", "Pt", "Pt"],  # 3-layer Pt slab
    [
        [0.000, 0.000, 0.40],  # Bottom layer
        [0.333, 0.667, 0.47],  # Middle layer
        [0.667, 0.333, 0.54],  # Top layer (surface)
    ],
)

# Generate O adsorption on all surface Pt sites (no symmetry reduction)
generator_pt = SurfaceInterstitialGenerator(
    slab_structure=pt_slab,
    surface_layers=1,
    surface_side="top",
    interstitial_offset=2.0,  # 2.0 Å above Pt (typical for O-Pt)
    interstitial_site_type="ontop",
    use_in_plane_symmetry=False,  # Get ALL surface sites
)

o_adsorption = generator_pt.generate_defects(
    species="Pt",  # Reference Pt atoms
    interstitial_species="O",
    charge_states=[-2, 0],  # O²⁻ and O⁰
)

print(f"\nGenerated {len(o_adsorption)} O adsorption configurations on Pt(111)")
# Get indices of Pt atoms, then count surface layer (last atom in this simple slab)
pt_indices = pt_slab.indices_from_symbol("Pt")
print(f"  Surface Pt atoms: {len(pt_indices[-1:])}")
print("  Charge states: [-2, 0]")

# ============================================================================
# EXAMPLE 5: Adsorption with supercell (realistic surface coverage)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 5: H adsorption with 3×3×1 supercell (low coverage)")
print("=" * 70)

generator_supercell = SurfaceInterstitialGenerator(
    slab_structure=mos2_slab,
    surface_layers=1,
    surface_side="top",
    interstitial_offset=1.5,
    interstitial_site_type="ontop",
)

# Generate with supercell for low coverage studies
h_supercell = generator_supercell.generate_defects(
    species="S",
    interstitial_species="H",
    supercell_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 1]],  # 3×3×1
    charge_states=[0],
)

print(f"\nGenerated {len(h_supercell)} H adsorption in 3×3×1 supercell")

for defect in h_supercell[:1]:  # Show first one
    print(f"\n  Host structure: {defect['host_structure'].composition}")
    print(f"  Total atoms in host: {len(defect['host_structure'])}")
    print(f"  Defect structure: {defect['structure'].composition}")
    print(f"  H coverage: 1 H per {len(defect['host_structure'])} atoms")

# ============================================================================
# EXAMPLE 6: Integration with DefectFlowMaker (full workflow)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 6: Create DefectFlowMaker workflow for H adsorption")
print("=" * 70)

# Generate H adsorption
generator = SurfaceInterstitialGenerator(
    slab_structure=mos2_slab,
    surface_layers=1,
    surface_side="top",
    interstitial_offset=1.5,
    interstitial_site_type="ontop",
)

defects = generator.generate_defects(
    species="S",
    interstitial_species="H",
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 1]],  # 2×2×1
    charge_states=[0, -1],
)

print(f"\nCreating {len(defects)} DefectFlowMaker workflows...")

# Create workflow for first defect (demo)
for i, defect_info in enumerate(defects[:1]):  # Only first one for demo
    q = defect_info["charge_state"]

    print(f"\nDefect {i+1}: H^{q:+d} adsorption")

    # Create DefectFlowMaker
    flow_maker = DefectFlowMaker(
        epsilon_static=15.0,  # MoS₂ in-plane dielectric constant
        epsilon_parallel=15.0,  # In-plane ε
        epsilon_perpendicular=7.0,  # Out-of-plane ε
        correction_scheme="slab2d",  # 2D slab correction (anisotropic)
        defect_type="interstitial",
        charge_state=q,
        use_ghost_atoms=False,  # Interstitials don't use ghost atoms
        dry_run=True,  # Dry-run mode
        chemical_potentials={"H": -3.4},  # H₂ molecule reference
    )

    # Create workflow
    flow = flow_maker.make(
        defect_structure=defect_info["structure"],
        host_structure=defect_info["host_structure"],
        defect_site=defect_info["frac_coords"],
        defect_species="H",
    )

    print(f"  Created workflow: {flow.name}")
    print(f"  Jobs in workflow: {len(flow.jobs)}")

    # Run dry-run
    print("  Running dry-run workflow...")
    results = run_locally(flow, create_folders=True)
    print("  Dry-run completed! Check job folders for FDF files.")

# ============================================================================
# EXAMPLE 7: Both surfaces (symmetric slab study)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 7: H adsorption on both top and bottom surfaces")
print("=" * 70)

generator_both = SurfaceInterstitialGenerator(
    slab_structure=mos2_slab,
    surface_layers=1,
    surface_side="both",  # Both surfaces!
    interstitial_offset=1.5,
    interstitial_site_type="ontop",
)

h_both_surfaces = generator_both.generate_defects(
    species="S",
    interstitial_species="H",
)

print(f"\nGenerated {len(h_both_surfaces)} H adsorption sites (both surfaces)")

top_count = sum(1 for d in h_both_surfaces if d["is_top_surface"])
bottom_count = sum(1 for d in h_both_surfaces if d["is_bottom_surface"])

print(f"  Top surface: {top_count} site(s)")
print(f"  Bottom surface: {bottom_count} site(s)")

# ============================================================================
# EXAMPLE 8: Multiple interstitial species (screening study)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 8: Screen different adsorbates (H, O, N)")
print("=" * 70)

# Screen multiple adsorbates on MoS₂ surface
adsorbates = ["H", "O", "N"]
all_adsorbates = []

for adsorbate in adsorbates:
    generator = SurfaceInterstitialGenerator(
        slab_structure=mos2_slab,
        surface_layers=1,
        surface_side="top",
        interstitial_offset=1.5,
        interstitial_site_type="ontop",
    )

    defects = generator.generate_defects(
        species="S",
        interstitial_species=adsorbate,
    )

    all_adsorbates.extend(defects)
    print(f"  {adsorbate} adsorption: {len(defects)} site(s)")

print(f"\nTotal: {len(all_adsorbates)} adsorbate configurations")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("TUTORIAL SUMMARY")
print("=" * 70)
print(
    """
✓ Example 1: H adsorption on MoS₂ (ontop S sites)
✓ Example 2: H with charge states (HER mechanism)
✓ Example 3: Li intercalation (hollow sites, negative offset)
✓ Example 4: O adsorption on Pt(111) (metal surface)
✓ Example 5: H with supercell (low coverage)
✓ Example 6: Integration with DefectFlowMaker
✓ Example 7: Both top and bottom surfaces
✓ Example 8: Multiple adsorbates screening (H, O, N)

Key Takeaways:
1. SurfaceInterstitialGenerator automatically places atoms at surface sites
2. Ontop: directly above surface atoms (most common for adsorption)
3. Hollow: at cell center (intercalation, bridge sites)
4. Positive offset: adsorption above surface (1.0-2.0 Å typical)
5. Negative offset: intercalation below surface (between layers)
6. Use supercells for realistic coverage (3×3×1, 4×4×1)
7. Multiple charge states for reaction mechanism studies
8. Use correction_scheme="slab2d" for 2D slab corrections

Important Parameters:
- min_dist_from_atoms: Default 1.0 Å for adsorbates, use 0.8 Å for intercalation
- Hollow sites generate ONE position per surface layer (not one per atom)
- Ontop sites generate one position per surface atom

Common Use Cases:
- H adsorption for HER (hydrogen evolution reaction)
- O adsorption for ORR/OER (oxygen reduction/evolution)
- Li intercalation for batteries
- N/C doping near surface
- Multi-adsorbate screening for catalysis

Next Steps:
- Try with real slab structures (from SurfaceMaker)
- Calculate adsorption energies with DefectFlowMaker
- Study coverage effects (1/9, 1/16 ML with larger supercells)
- Combine with surface vacancies (defect-adsorbate interactions)

For bulk interstitials (3D periodicity), use SiestaInterstitialGenerator instead.
"""
)

print("\nWorkflow output:")
print("  - job_*/dry_run_output/Defect_Calculation_*/")
print("    - structure.cif (defect structure)")
print("    - siesta.fdf (SIESTA input file)")
print("    - *.psml (pseudopotential symlinks)")
print("\nAll defect structures are saved in their respective job directories.")
