"""Minimal: Charged vacancy (V_O^+2) with finite-size corrections."""

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

# Create charged vacancy (q=+2)
o_indices = [i for i, site in enumerate(host_structure) if site.specie.symbol == "O"]
defect_structure = create_vacancy_with_ghost(host_structure, o_indices[15])

print(f"Charged defect: V_O^+2 in {len(host_structure)}-atom supercell")

# CRITICAL: Chemical potentials for correct formation energies
# For oxygen vacancy: E_f = E_defect - E_host + μ_O + q*E_F + E_corr
# μ_O should be calculated from O2 molecule energy: μ_O = E(O2)/2
# Example: If E(O2) = -9.0 eV, then μ_O = -4.5 eV
chemical_potentials = {
    "O": -4.5,  # eV (example value - calculate from O2 in large box)
    "Mg": -1.5,  # eV (example value - calculate from bulk Mg)
}

# Create workflow with correction scheme
# Available corrections: "lany-zunger" (default), "makov-payne", "freysoldt", "kumagai"
# When using freysoldt/kumagai: .VT files are AUTO-ENABLED for potential alignment!
maker = DefectFlowMaker(
    epsilon_static=9.8,
    # correction_scheme="kumagai",  # State-of-the-art for relaxed systems
    # correction_scheme="lany-zunger",  # Simple, fast (default)
    # correction_scheme="makov-payne",  # With quadrupole term
    correction_scheme="freysoldt",  # With potential alignment
    defect_type="vacancy",
    # charge_state=[0, +1, +2],  # Multiple charges → returns list[Flow]
    charge_state=2,  # Single charge → returns Flow
    auto_calculate_chemical_potentials=True,
    dry_run=False,
    skip_relax=True,
)

maker.defect_relax_maker = apply_tier_preset(maker.defect_relax_maker, "defect_dirty")
maker.host_static_maker = apply_tier_preset(maker.host_static_maker, "defect_dirty")

flows = maker.make(
    defect_structure,
    host_structure,
    host_structure[o_indices[15]].frac_coords.tolist(),
    "O",
)

results = run_locally(flows, create_folders=True)
