"""Slab2D with automatic potential alignment from VT files."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.flows.defects import DefectFlowMaker, create_vacancy_with_ghost
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

unit_cell = Structure(
    Lattice.hexagonal(2.5, 20.0),
    ["B", "N"],
    [[0, 0, 0.5], [1 / 3, 2 / 3, 0.5]],
)

slab = unit_cell.make_supercell([2, 2, 1])

# Use a nitrogen vacancy: N has an automatic chemical-potential reference
# (elemental boron does not - it would need explicit chemical_potentials)
n_indices = [i for i, site in enumerate(slab) if site.specie.symbol == "N"]
defect_slab = create_vacancy_with_ghost(slab, n_indices[0])

print(f"V_N^+1 in hBN 2×2×1 supercell ({len(slab)} atoms)")

# Slab2D: Anisotropic dielectric screening for 2D materials
# In-plane (ε∥): stronger screening due to π-electrons
# Out-of-plane (ε⊥): weaker screening due to weak vdW interactions
epsilon_parallel = 6.8  # In-plane
epsilon_perpendicular = 3.0  # Out-of-plane

# Slab2D: AUTO-ENABLES SaveElectrostaticPotential, reads .VT, aligns potential
# Also AUTO-GENERATES potential alignment plot
maker = DefectFlowMaker(
    epsilon_parallel=epsilon_parallel,
    epsilon_perpendicular=epsilon_perpendicular,
    correction_scheme="slab2d",  # Auto: SaveElectrostaticPotential=True
    defect_type="vacancy",
    charge_state=+1,
    auto_calculate_chemical_potentials=True,
    dry_run=False,
    skip_relax=True,
)

maker.defect_relax_maker = apply_tier_preset(maker.defect_relax_maker, "defect_dirty")
maker.host_static_maker = apply_tier_preset(maker.host_static_maker, "defect_dirty")

flow = maker.make(
    defect_slab,
    slab,
    slab[n_indices[0]].frac_coords.tolist(),
    "N",
)

print("Running workflow...")
results = run_locally(flow, create_folders=True)
print("✓ Potential alignment automatically calculated from .VT files")
print(
    "✓ Plots: potential_alignment.png, dielectric_profile.png, density_difference.png, radial_distribution.png"
)
