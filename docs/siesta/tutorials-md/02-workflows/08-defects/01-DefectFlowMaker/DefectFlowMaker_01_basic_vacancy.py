"""Minimal: Neutral oxygen vacancy in MgO with ghost atoms."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.flows.defects import DefectFlowMaker, create_vacancy_with_ghost
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

# Create MgO 2×2×2 supercell
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

# Create vacancy with ghost atom (CRITICAL for SIESTA)
o_indices = [i for i, site in enumerate(host_structure) if site.specie.symbol == "O"]
defect_structure = create_vacancy_with_ghost(host_structure, o_indices[15])

print(f"Host: {len(host_structure)} atoms, Defect: {len(defect_structure)} atoms")

# Create workflow with defect_dirty preset
maker = DefectFlowMaker(
    epsilon_static=9.8,
    defect_type="vacancy",
    charge_state=0,
    dry_run=False,
    skip_relax=True,
    auto_calculate_chemical_potentials=True,
)

# Apply defect preset for quick calculations
maker.defect_relax_maker = apply_tier_preset(maker.defect_relax_maker, "defect_dirty")
maker.host_static_maker = apply_tier_preset(maker.host_static_maker, "defect_dirty")

flow = maker.make(
    defect_structure,
    host_structure,
    host_structure[o_indices[15]].frac_coords.tolist(),
    "O",
)

# Run
results = run_locally(flow, create_folders=True)
defect_doc = results[flow.jobs[-1].uuid][1].output

print(f"E_f = {defect_doc.corrected_formation_energy:.4f} eV (dry-run)")
print(
    f"Correction: {defect_doc.correction_scheme} ({defect_doc.correction_energy:.4f} eV)"
)
