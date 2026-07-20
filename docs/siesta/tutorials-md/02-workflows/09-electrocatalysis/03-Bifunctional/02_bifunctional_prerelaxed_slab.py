#!/usr/bin/env python
"""Bifunctional ORR/OER Workflow with Pre-Relaxed Slab

This tutorial demonstrates BifunctionalFlowMaker for combined ORR+OER screening
using a pre-relaxed slab. Key advantages:

1. Pre-relaxed slab: No redundant slab relaxation
2. Fixed slab atoms: Only adsorbates move during relaxation
3. Single workflow: Both ORR and OER calculated together
4. Shared intermediates: O*, OH*, OOH*, O₂* calculated once
5. Bifunctional gap: η_ORR + η_OER output for battery applications

This workflow is ideal for:
- Rechargeable metal-air batteries (need both ORR and OER)
- Regenerative fuel cells
- High-throughput catalyst screening

Expected output:
- bifunctional_orr_diagram.png (ORR free energy diagram)
- bifunctional_oer_diagram.png (OER free energy diagram)
- bifunctional_summary.txt (combined analysis with benchmarks)
"""

from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice

from atomate2.siesta.flows.electrocatalysis import BifunctionalFlowMaker
from atomate2.siesta.flows.molecular import GasPhaseMoleculeMaker
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from atomate2.siesta.sets.utils import apply_diffuse_basis_to_surface
from jobflow import run_locally

# ============================================================================
# Step 1: Create or load pre-relaxed slab
# ============================================================================
# In practice, you would load a pre-relaxed slab from a previous calculation:
# relaxed_slab = Structure.from_file("Pt111_relaxed.cif")

# For this tutorial, we create a 4-layer Pt(111) slab
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

# ============================================================================
# Step 2: Apply diffuse basis to surface atoms (optional but recommended)
# ============================================================================
species_labels, pao_basissizes, surface_info = apply_diffuse_basis_to_surface(
    pt111_slab,
    surface_basis="DZ",
    bulk_basis="SZ",
    surface_layers=1,
)
pt111_slab.add_site_property("species_label", species_labels)

print(f"Surface atoms: {surface_info['surface']}")
print(f"Bulk atoms: {surface_info['bulk']}")

# ============================================================================
# Step 3: Create makers with tier presets
# ============================================================================
# Gas-phase maker for O₂, H₂O, H₂ references
gas_relax_maker = RelaxMaker()
gas_relax_maker = apply_tier_preset(gas_relax_maker, "electrocatalysis_gas_phase_dirty")
gas_phase_maker = GasPhaseMoleculeMaker(relax_maker=gas_relax_maker)

# Static maker for clean surface (if energy not provided)
surface_static_maker = StaticMaker()
surface_static_maker = apply_tier_preset(
    surface_static_maker,
    "electrocatalysis_dirty",
    override_params={"%block PAO.BasisSizes": pao_basissizes},
)

# Relax maker for adsorbate optimization
# NOTE: Slab atoms will be automatically fixed by BifunctionalFlowMaker
# using %block Geometry.Constraints
adsorption_relax_maker = RelaxMaker.fixed_cell_relaxation()
adsorption_relax_maker = apply_tier_preset(
    adsorption_relax_maker,
    "electrocatalysis_dirty",
    override_params={"%block PAO.BasisSizes": pao_basissizes},
)

# ============================================================================
# Step 4: Create BifunctionalFlowMaker
# ============================================================================
# Option A: Calculate clean surface energy automatically
maker = BifunctionalFlowMaker(
    name="Pt111_bifunctional",
    gas_phase_maker=gas_phase_maker,
    surface_static_maker=surface_static_maker,
    adsorption_relax_maker=adsorption_relax_maker,
    grid_size=(2, 2),
    height=2.0,
    temperature=298.15,
    ph=0.0,
    potential_orr=0.0,
    potential_oer=1.23,
    plot_results=True,
    write_summary=True,
)

# Option B: Pass pre-calculated clean surface energy (faster)
# maker = BifunctionalFlowMaker(
#     clean_surface_energy=-1234.567,  # eV from previous calculation
#     gas_phase_maker=gas_phase_maker,
#     adsorption_relax_maker=adsorption_relax_maker,
# )

# ============================================================================
# Step 5: Create and run workflow
# ============================================================================
flow = maker.make(pt111_slab)

print(f"Created workflow with {len(flow.jobs)} jobs")
print("Jobs: gas-phase (O₂, H₂O, H₂), clean surface, O₂*, OOH*, O*, OH*, analysis")

# Run locally
results = run_locally(flow, create_folders=True)

# Or submit to remote cluster:
# from jobflow_remote import submit_flow
# submit_flow(flow)

# ============================================================================
# Results interpretation
# ============================================================================
"""
Output files:
- bifunctional_orr_diagram.png: ORR free energy diagram
- bifunctional_oer_diagram.png: OER free energy diagram
- bifunctional_summary.txt: Combined analysis

Key metrics in summary:
- η_ORR: ORR overpotential (target < 0.4 V)
- η_OER: OER overpotential (target < 0.4 V)
- Bifunctional gap: η_ORR + η_OER (target < 0.8 V)

Benchmarks:
- Pt/C: ~0.8 V gap (ORR active, poor OER)
- IrO₂: ~0.4 V gap (OER active, poor ORR)
- Best bifunctional: ~0.6-0.7 V

For rechargeable metal-air batteries:
- Lower bifunctional gap = higher round-trip efficiency
- Target: < 0.8 V for practical applications
"""
