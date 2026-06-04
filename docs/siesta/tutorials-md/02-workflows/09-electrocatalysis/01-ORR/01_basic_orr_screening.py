#!/usr/bin/env python
"""Basic ORR Catalyst Screening

Demonstrates ORRFlowMaker for screening catalyst surfaces for ORR activity.
Results include free energy diagrams, overpotential analysis, and plots.
See orr_analysis_summary.txt for detailed interpretation.

IMPORTANT: This tutorial uses RelaxMaker for adsorption calculations to obtain
proper adsorption energies. The adsorbate geometry is optimized at each site,
which is essential for realistic thermodynamics.

Expected results for Pt(111):
- Overpotential: η ≈ 0.4-0.6 V (literature: ~0.45 V)
- Rate-limiting step: typically OH* → H₂O or O* → OH*

Spin handling:
- O₂: Triplet state (S=1), spin-polarized calculation, 2.0 μB
- H₂O, H₂: Singlet (S=0), non-spin-polarized
- Automatic spin detection via electrocatalysis_gas_phase_dirty preset

Diffuse orbitals for surfaces:
- Surface atoms automatically detected (outermost atomic layer at vacuum)
- DZ basis for surface atoms (more diffuse, better for vacuum interface)
- SZ basis for bulk atoms (standard)
- Improves surface energies, work functions, and adsorption energies

Slab energy optimization:
- ORRFlowMaker automatically reuses slab energy across multiple adsorbates
- The clean slab is calculated once (for O₂ adsorption), then reused for OOH, O, OH
- This saves 3 redundant slab calculations, significantly reducing compute time
- For custom workflows, use AdsorptionScanFlowMaker's precalc_slab_energy parameter
"""

from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice

from atomate2.siesta.flows.electrocatalysis import ORRFlowMaker
from atomate2.siesta.flows.molecular import GasPhaseMoleculeMaker
from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from atomate2.siesta.sets.utils import apply_diffuse_basis_to_surface
from jobflow import run_locally

# Load or create catalyst surface
a = 3.92
lattice = Lattice.from_parameters(
    a=a * 3**0.5, b=a * 3, c=20.0, alpha=90, beta=90, gamma=90
)

pt_coords = []
for layer in range(4):
    z = layer * 2.27
    pt_coords.extend(
        [
            [0.0, 0.0, z],
            [a * 3**0.5 / 2, a / 2, z],
            [a * 3**0.5 / 2, 3 * a / 2, z],
        ]
    )

pt111_slab = Structure(
    lattice=lattice,
    species=["Pt"] * 12,
    coords=pt_coords,
    coords_are_cartesian=True,
)

# Or load from file:
# pt111_slab = Structure.from_file("Pt111_relaxed.cif")

# ============================================================================
# OPTIONAL: Apply diffuse basis to surface atoms for better accuracy
# ============================================================================
# Surface atoms need larger (more diffuse) basis sets because electrons
# extend further into the vacuum than in bulk. This improves:
# - Surface energies, work functions, adsorption energies
#
# Layer-based detection: automatically identifies outermost atomic layers
# - Surface atoms: DZ (double-zeta, more diffuse orbitals)
# - Bulk atoms: SZ (single-zeta, standard)

species_labels, pao_basissizes, surface_info = apply_diffuse_basis_to_surface(
    pt111_slab,
    surface_basis="DZP",  # Larger basis for surface atoms
    bulk_basis="DZ",  # Standard basis for bulk atoms
    surface_layers=1,  # 1 outermost layer per surface (top + bottom)
)
pt111_slab.add_site_property("species_label", species_labels)

print(f"Surface atoms ({len(surface_info['surface'])}): {surface_info['surface']}")
print(f"Bulk atoms ({len(surface_info['bulk'])}): {surface_info['bulk']}")
print(f"PAO.BasisSizes: {pao_basissizes}")

# Create makers with tier presets
# Options: electrocatalysis_dirty (1-2h), electrocatalysis_basic (4-6h),
#          electrocatalysis_intermediate (8-12h)

# Clean surface calculation (with diffuse basis for surface atoms)
slab_maker = StaticMaker()
slab_maker = apply_tier_preset(
    slab_maker,
    "electrocatalysis_dirty",
    override_params={"%block PAO.BasisSizes": pao_basissizes},
)

# Gas-phase reference calculations (O₂, H₂O, H₂)
# Uses spin-polarized DFT for O₂ (triplet), non-polarized for H₂O/H₂
gas_relax_maker = RelaxMaker()
gas_relax_maker = apply_tier_preset(gas_relax_maker, "electrocatalysis_gas_phase_dirty")
gas_phase_maker = GasPhaseMoleculeMaker(relax_maker=gas_relax_maker)

# Adsorption calculations (slab+adsorbate and gas-phase molecules)
# IMPORTANT: Use RelaxMaker for proper adsorption energies!
# StaticMaker gives non-physical results because adsorbate geometry is not optimized.
# RelaxMaker allows the adsorbate to find its optimal position on the surface.

# Option 1: RelaxMaker (RECOMMENDED - proper adsorption energies)
adsorption_slab_maker = RelaxMaker.fixed_cell_relaxation()
adsorption_slab_maker = apply_tier_preset(
    adsorption_slab_maker,
    "electrocatalysis_dirty",
    override_params={"%block PAO.BasisSizes": pao_basissizes},
)

# Option 2: StaticMaker (faster but qualitative only - adsorbate at fixed height)
# adsorption_slab_maker = StaticMaker()
# adsorption_slab_maker = apply_tier_preset(
#     adsorption_slab_maker, "electrocatalysis_dirty"
# )

# Gas-phase adsorbate reference
adsorption_adsorbate_maker = RelaxMaker.fixed_cell_relaxation()
adsorption_adsorbate_maker = apply_tier_preset(
    adsorption_adsorbate_maker, "electrocatalysis_gas_phase"
)

adsorption_maker = AdsorptionScanFlowMaker(
    slab_static_maker=adsorption_slab_maker,
    adsorbate_static_maker=adsorption_adsorbate_maker,
)

# Create workflow
orr_maker = ORRFlowMaker(
    name="Pt111_ORR",
    gas_phase_maker=gas_phase_maker,
    surface_static_maker=slab_maker,
    adsorption_maker=adsorption_maker,
    grid_size=(2, 2),
    height=1.0,
    temperature=298.15,
    ph=0.0,
    potential=0.0,
    use_custodian=True,  # Enable automatic error recovery
    custodian_max_errors=10,  # Allow up to 10 retry attempts
    # dry_run=True,  # Uncomment to test workflow without running SIESTA
)

orr_flow = orr_maker.make(pt111_slab)

# Run workflow
results = run_locally(orr_flow, create_folders=True, ensure_success=True)

# Or submit to remote:
# from jobflow_remote import submit_flow
# submit_flow(orr_flow)

# Results will include:
# - orr_free_energy_diagram.png
# - orr_overpotential_summary.png
# - orr_analysis_summary.txt (detailed interpretation with benchmarks, scaling relations, references)

# ============================================================================
# ADVANCED: Manual slab energy reuse for custom workflows
# ============================================================================
# If you're building custom multi-adsorbate workflows (not using ORRFlowMaker),
# you can manually reuse slab energies using the precalc_slab_energy parameter:
#
# from pymatgen.core import Molecule
# from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker
# from jobflow import Flow
#
# # First adsorbate - calculates slab energy
# o2_molecule = Molecule(["O", "O"], [[0, 0, 0], [0, 0, 1.21]])
# first_maker = AdsorptionScanFlowMaker(
#     slab_static_maker=adsorption_slab_maker,
#     adsorbate_static_maker=adsorption_adsorbate_maker,
#     grid_size=(3, 3),
#     height=2.0,
# )
# o2_flow = first_maker.make(pt111_slab, o2_molecule)
#
# # Second adsorbate - reuses slab energy from first scan
# oh_molecule = Molecule(["O", "H"], [[0, 0, 0], [0.96, 0, 0]])
# reuse_maker = AdsorptionScanFlowMaker(
#     slab_static_maker=adsorption_slab_maker,
#     adsorbate_static_maker=adsorption_adsorbate_maker,
#     grid_size=(3, 3),
#     height=2.0,
#     precalc_slab_energy=o2_flow.output.slab_energy,  # Reuse from first scan
# )
# oh_flow = reuse_maker.make(pt111_slab, oh_molecule)
#
# # Combine into single workflow
# combined_flow = Flow([o2_flow, oh_flow], name="multi_adsorbate_scan")
# results = run_locally(combined_flow, create_folders=True)
