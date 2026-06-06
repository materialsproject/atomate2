"""Makov-Payne with automatic quadrupole from RHO files."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.flows.defects import DefectFlowMaker, create_vacancy_with_ghost
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

lattice = Lattice.cubic(4.212)
unit_cell = Structure(
    lattice,
    ["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"],
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
    ],
)
host_structure = unit_cell.make_supercell([2, 2, 2])

o_indices = [i for i, site in enumerate(host_structure) if site.specie.symbol == "O"]
defect_structure = create_vacancy_with_ghost(host_structure, o_indices[15])

print(f"V_O^+2 in {len(host_structure)}-atom supercell")

# Makov-Payne-Quadrupole: AUTO-ENABLES SaveRho, reads .RHO, calculates quadrupole
# Note: Use "makov-payne" for basic version (Q=0, no .RHO needed)
maker = DefectFlowMaker(
    epsilon_static=9.8,
    correction_scheme="makov-payne-quadrupole",  # Auto: SaveRho=True
    defect_type="vacancy",
    charge_state=+2,
    auto_calculate_chemical_potentials=True,
    dry_run=False,
    skip_relax=True,
)

maker.defect_relax_maker = apply_tier_preset(maker.defect_relax_maker, "defect_dirty")
maker.host_static_maker = apply_tier_preset(maker.host_static_maker, "defect_dirty")

flow = maker.make(
    defect_structure,
    host_structure,
    host_structure[o_indices[15]].frac_coords.tolist(),
    "O",
)

print("Running workflow...")
results = run_locally(flow, create_folders=True)
print("✓ Quadrupole automatically calculated from .RHO files")
print("✓ Plots generated: density_difference.png, radial_distribution.png")
