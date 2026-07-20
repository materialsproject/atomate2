"""
Tutorial: Surface Substitution Generation with SurfaceSubstitutionGenerator

Category: Defects
Difficulty: Advanced
Time: ~10 min (dry-run), ~hours (full calculations)

This tutorial demonstrates surface-aware substitution generation for slab structures.
Critical for surface doping, catalysis, and alloy surface studies.

Key Features:
- Automatic surface atom replacement (dopants, alloying)
- Single or multiple dopants
- Antisite defect generation
- Surface-only substitution (no bulk)
- Integration with DefectFlowMaker

Why SurfaceSubstitutionGenerator vs SiestaSubstitutionGenerator?
- SiestaSubstitutionGenerator: Uses 3D bulk symmetry, all atomic sites
- SurfaceSubstitutionGenerator: Surface-aware, replaces only surface atoms
"""

from pymatgen.core import Lattice, Structure
from jobflow import run_locally

from atomate2.siesta.flows.defects import (
    SurfaceSubstitutionGenerator,
    DefectFlowMaker,
)

# ============================================================================
# EXAMPLE 1: Single Mo→W substitution on MoS₂ surface (single-atom catalyst)
# ============================================================================

print("=" * 70)
print("EXAMPLE 1: Mo→W substitution on MoS₂ surface (catalysis)")
print("=" * 70)

# Load bulk MoS₂ structure and create slab by adding vacuum
print("\n--- Creating MoS₂ slab from bulk structure ---")
bulk_mos2 = Structure.from_file("../../../00-structures/Mos2.cif")
print(f"Bulk MoS₂: {bulk_mos2.composition}")

# Add vacuum by scaling the c-axis
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
scale_factor = bulk_mos2.lattice.c / new_c
new_coords = []
for site in bulk_mos2:
    new_z = site.frac_coords[2] * scale_factor + (1 - scale_factor) / 2
    new_coords.append([site.frac_coords[0], site.frac_coords[1], new_z])

# Create slab structure
mos2_slab = Structure(
    new_lattice,
    [site.specie for site in bulk_mos2],
    new_coords,
    site_properties=bulk_mos2.site_properties,
)

print(f"MoS₂ slab: {mos2_slab.composition}")
print(f"  Total atoms: {len(mos2_slab)}")
print(f"  Slab thickness: {mos2_slab.lattice.c:.3f} Å (bulk + vacuum)")

# Create Mo→W substitution generator
# Note: MoS₂ has S-terminated surface, so we need surface_layers=2 to include Mo
generator = SurfaceSubstitutionGenerator(
    slab_structure=mos2_slab,
    surface_layers=2,  # Top 2 layers (S + Mo)
    surface_side="top",  # Top surface
    use_in_plane_symmetry=True,  # Reduce by symmetry
)

print("\n--- Replacing surface Mo with W (single-atom catalyst) ---")
mo_to_w = generator.generate_defects(
    species="Mo",  # Replace Mo atoms
    dopants="W",  # With W atoms
)

print(f"\nGenerated {len(mo_to_w)} W-doped MoS₂ configuration(s)")

for i, defect in enumerate(mo_to_w):
    print(f"\nDefect {i+1}: W_{defect['removed_species']}")
    print(f"  Dopant: {defect['species']} (added)")
    print(f"  Host: {defect['removed_species']} (removed)")
    print(f"  Layer: {defect['layer_index']}")
    print(f"  Layer z: {defect['layer_z_position']:.3f} Å")
    print(f"  Is top surface: {defect['is_top_surface']}")
    print(f"  Charge state: {defect['charge_state']:+d}")

# ============================================================================
# EXAMPLE 2: Multiple dopant screening (catalysis screening)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 2: Screen multiple transition metal dopants (W, Nb, Ta, Re)")
print("=" * 70)

# Screen different transition metals as Mo replacements
generator_screening = SurfaceSubstitutionGenerator(
    slab_structure=mos2_slab,
    surface_layers=2,  # Include S + Mo layers
    surface_side="top",
)

# Generate Mo→TM substitutions for multiple dopants
dopants = ["W", "Nb", "Ta", "Re"]
tm_doped = generator_screening.generate_defects(
    species="Mo",
    dopants=dopants,  # Multiple dopants!
    charge_states=[0],  # Neutral substitutions
)

print(f"\nGenerated {len(tm_doped)} TM-doped MoS₂ configurations")
print(f"  Dopants tested: {dopants}")
print(f"  Configurations per dopant: {len(tm_doped) // len(dopants)}")

# Group by dopant
for dopant in dopants:
    dopant_defects = [d for d in tm_doped if d["species"] == dopant]
    print(f"  {dopant}_Mo: {len(dopant_defects)} configuration(s)")

# ============================================================================
# EXAMPLE 3: Surface alloying (Pt-Pd bimetallic catalyst)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 3: Pt-Pd surface alloying (bimetallic catalyst)")
print("=" * 70)

# Create Pt(111) slab
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

# Replace surface Pt with Pd (surface alloying)
generator_alloy = SurfaceSubstitutionGenerator(
    slab_structure=pt_slab,
    surface_layers=1,
    surface_side="top",
    use_in_plane_symmetry=False,  # Get all surface sites (for coverage studies)
)

pt_pd_alloy = generator_alloy.generate_defects(
    species="Pt",
    dopants="Pd",
)

print(f"\nGenerated {len(pt_pd_alloy)} Pt-Pd alloy surface configuration(s)")
print("  Surface Pt atoms replaced with Pd")
print("  Ideal for bimetallic catalyst studies")

# ============================================================================
# EXAMPLE 4: Antisite defects on surface
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 4: Surface antisite defects (Mo_S and S_Mo)")
print("=" * 70)

generator_antisite = SurfaceSubstitutionGenerator(
    slab_structure=mos2_slab,
    surface_layers=1,
    surface_side="top",
)

# Mo on S site
mo_on_s = generator_antisite.generate_defects(
    species="S",  # Replace S
    dopants="Mo",  # With Mo (antisite)
)

# S on Mo site
s_on_mo = generator_antisite.generate_defects(
    species="Mo",  # Replace Mo
    dopants="S",  # With S (antisite)
)

print("\nGenerated antisite defects:")
print(f"  Mo_S (Mo on S site): {len(mo_on_s)} configuration(s)")
print(f"  S_Mo (S on Mo site): {len(s_on_mo)} configuration(s)")

# Show details for first antisite
if mo_on_s:
    print("\nMo_S antisite example:")
    print(f"  Host species removed: {mo_on_s[0]['removed_species']}")
    print(f"  Dopant added: {mo_on_s[0]['species']}")
    print(f"  Layer: {mo_on_s[0]['layer_index']}")

# ============================================================================
# EXAMPLE 5: Charge states for doping studies
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 5: N doping on surface O sites (charge states)")
print("=" * 70)

# Create TiO₂ anatase (101) slab (simplified)
tio2_lattice = Lattice.from_parameters(
    a=3.78, b=3.78, c=18.0, alpha=90, beta=90, gamma=90
)

tio2_slab = Structure(
    tio2_lattice,
    ["Ti", "Ti", "O", "O", "O", "O"],  # Simplified 2-layer slab
    [
        [0.0, 0.0, 0.45],  # Ti (bottom)
        [0.5, 0.5, 0.55],  # Ti (top) - surface
        [0.25, 0.25, 0.42],  # O
        [0.75, 0.75, 0.48],  # O
        [0.25, 0.75, 0.52],  # O - surface
        [0.75, 0.25, 0.58],  # O - surface
    ],
)

# N doping on surface O sites (for visible light absorption)
generator_n_doping = SurfaceSubstitutionGenerator(
    slab_structure=tio2_slab,
    surface_layers=1,
    surface_side="top",
)

n_doped_tio2 = generator_n_doping.generate_defects(
    species="O",  # Replace O
    dopants="N",  # With N
    charge_states=[-1, 0, +1],  # N oxidation states
)

print(f"\nGenerated {len(n_doped_tio2)} N-doped TiO₂ configurations")

charges_count = {}
for defect in n_doped_tio2:
    q = defect["charge_state"]
    charges_count[q] = charges_count.get(q, 0) + 1

print("  Charge states:")
for q, count in sorted(charges_count.items()):
    print(f"    N_O^{q:+d}: {count} configuration(s)")

# ============================================================================
# EXAMPLE 6: Subsurface substitution (near-surface doping)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 6: Subsurface doping (top 2 layers)")
print("=" * 70)

# Replace atoms in top TWO layers (surface + subsurface)
generator_subsurface = SurfaceSubstitutionGenerator(
    slab_structure=mos2_slab,
    surface_layers=2,  # Top 2 layers!
    surface_side="top",
)

subsurface_doped = generator_subsurface.generate_defects(
    species="Mo",
    dopants="W",
)

print(f"\nGenerated {len(subsurface_doped)} W-doped configurations (2 layers)")

# Group by layer
layer_count = {}
for defect in subsurface_doped:
    layer = defect["layer_index"]
    layer_count[layer] = layer_count.get(layer, 0) + 1

print("  Substitutions by layer:")
for layer, count in sorted(layer_count.items()):
    print(f"    Layer {layer}: {count} substitution(s)")

# ============================================================================
# EXAMPLE 7: Substitution with supercell (low doping concentration)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 7: W doping with 3×3×1 supercell (low concentration)")
print("=" * 70)

generator_supercell = SurfaceSubstitutionGenerator(
    slab_structure=mos2_slab,
    surface_layers=1,
    surface_side="top",
)

# Generate with supercell for low doping concentration
w_doped_supercell = generator_supercell.generate_defects(
    species="Mo",
    dopants="W",
    supercell_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 1]],  # 3×3×1
)

print(f"\nGenerated {len(w_doped_supercell)} W-doped structure(s) in 3×3×1 supercell")

for defect in w_doped_supercell[:1]:  # Show first one
    host_comp = defect["host_structure"].composition
    defect_comp = defect["structure"].composition
    print(f"\n  Host: {host_comp}")
    print(f"  Doped: {defect_comp}")
    print(f"  Doping concentration: 1 W per {len(defect['host_structure'])} atoms")

# ============================================================================
# EXAMPLE 8: Integration with DefectFlowMaker (full workflow)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 8: Create DefectFlowMaker workflow for W doping")
print("=" * 70)

# Generate W-doped MoS₂
generator = SurfaceSubstitutionGenerator(
    slab_structure=mos2_slab,
    surface_layers=2,  # Include S + Mo layers
    surface_side="top",
)

defects = generator.generate_defects(
    species="Mo",
    dopants="W",
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 1]],  # 2×2×1
    charge_states=[0],
)

print(f"\nCreating {len(defects)} DefectFlowMaker workflow(s)...")

# Create workflow for first defect (demo)
for i, defect_info in enumerate(defects[:1]):  # Only first one for demo
    print(f"\nDefect {i+1}: W_{defect_info['removed_species']} substitution")

    # Create DefectFlowMaker
    flow_maker = DefectFlowMaker(
        epsilon_static=15.0,  # MoS₂ in-plane dielectric constant
        epsilon_parallel=15.0,  # In-plane ε
        epsilon_perpendicular=7.0,  # Out-of-plane ε
        correction_scheme="slab2d",  # 2D slab correction
        defect_type="substitution",
        charge_state=defect_info["charge_state"],
        use_ghost_atoms=False,  # Substitutions don't use ghost atoms
        dry_run=True,  # Dry-run mode
        chemical_potentials={
            "Mo": -7.0,  # Mo bulk reference
            "W": -8.5,  # W bulk reference
        },
    )

    # Create workflow
    flow = flow_maker.make(
        defect_structure=defect_info["structure"],
        host_structure=defect_info["host_structure"],
        defect_site=defect_info["frac_coords"],
        defect_species="W",
        removed_species="Mo",
    )

    print(f"  Created workflow: {flow.name}")
    print(f"  Jobs in workflow: {len(flow.jobs)}")

    # Run dry-run
    print("  Running dry-run workflow...")
    results = run_locally(flow, create_folders=True)
    print("  Dry-run completed! Check job folders for FDF files.")

# ============================================================================
# EXAMPLE 9: Both surfaces (symmetric slab)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 9: W doping on both top and bottom surfaces")
print("=" * 70)

generator_both = SurfaceSubstitutionGenerator(
    slab_structure=mos2_slab,
    surface_layers=1,
    surface_side="both",  # Both surfaces!
)

w_both_surfaces = generator_both.generate_defects(
    species="Mo",
    dopants="W",
)

print(f"\nGenerated {len(w_both_surfaces)} W-doped configuration(s) (both surfaces)")

top_count = sum(1 for d in w_both_surfaces if d["is_top_surface"])
bottom_count = sum(1 for d in w_both_surfaces if d["is_bottom_surface"])

print(f"  Top surface: {top_count} substitution(s)")
print(f"  Bottom surface: {bottom_count} substitution(s)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("TUTORIAL SUMMARY")
print("=" * 70)
print(
    """
✓ Example 1: Single Mo→W substitution (single-atom catalyst)
✓ Example 2: Multiple dopant screening (W, Nb, Ta, Re)
✓ Example 3: Pt-Pd surface alloying (bimetallic catalyst)
✓ Example 4: Antisite defects (Mo_S, S_Mo)
✓ Example 5: N doping with charge states (TiO₂)
✓ Example 6: Subsurface doping (top 2 layers)
✓ Example 7: Low doping concentration (3×3×1 supercell)
✓ Example 8: Integration with DefectFlowMaker
✓ Example 9: Both top and bottom surfaces

Key Takeaways:
1. SurfaceSubstitutionGenerator replaces ONLY surface atoms (not bulk)
2. Single dopant: dopants="W" → one defect type
3. Multiple dopants: dopants=["W", "Nb", "Ta"] → screening study
4. Antisite defects: A on B site, B on A site (same structure)
5. Charge states: important for doping (valence difference)
6. Subsurface: surface_layers=2 for near-surface doping
7. Low concentration: use larger supercells (3×3×1, 4×4×1)
8. Symmetry: use_in_plane_symmetry=False for all sites

Nomenclature:
- W_Mo: W substituting on Mo site (W dopant, Mo removed)
- Mo_S: Mo substituting on S site (Mo antisite)
- N_O: N substituting on O site (N doping)

Common Use Cases:
- Single-atom catalysts (Mo→W, Mo→Pt on MoS₂)
- Surface doping (N-doped TiO₂ for visible light)
- Bimetallic catalysts (Pt-Pd, Pt-Au alloy surfaces)
- Antisite defects (defect engineering)
- Dopant screening (high-throughput catalysis)

Charge State Guidelines:
- Same valence: typically q=0 (W⁶⁺ on Mo⁶⁺)
- Acceptor: negative charge (Li⁺ on Mg²⁺ → q=-1)
- Donor: positive charge (Nb⁵⁺ on Ti⁴⁺ → q=+1)

Next Steps:
- Try with real slab structures (from SurfaceMaker)
- Calculate substitution/doping energies
- Compare different dopants (formation energy trends)
- Study charge compensation mechanisms
- Combine with surface vacancies (complex defects)

For bulk substitutions (3D periodicity), use SiestaSubstitutionGenerator instead.
"""
)

print("\nWorkflow output:")
print("  - job_*/dry_run_output/Defect_Calculation_*/")
print("    - structure.cif (substitution structure)")
print("    - siesta.fdf (SIESTA input file)")
print("    - *.psml (pseudopotential symlinks)")
print("\nAll substitution structures are saved in their respective job directories.")
