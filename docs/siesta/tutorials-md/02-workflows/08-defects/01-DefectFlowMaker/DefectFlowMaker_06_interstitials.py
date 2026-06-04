"""Generate interstitial defects automatically using from_pristine_structure()."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.flows.defects import DefectFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

# Create MgO unit cell
lattice = Lattice.cubic(4.212)
mgo = Structure(
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

# Create makers with tier presets
defect_relax_maker = apply_tier_preset(
    RelaxMaker.fixed_cell_relaxation(), "defect_dirty"
)
host_static_maker = apply_tier_preset(StaticMaker(), "defect_dirty")

flows = DefectFlowMaker.from_pristine_structure(
    mgo,
    defect_type="interstitial",
    species="Li",
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    charge_states=[+1],
    epsilon_static=9.8,
    dry_run=False,
    skip_relax=True,
    defect_relax_maker=defect_relax_maker,
    host_static_maker=host_static_maker,
    auto_calculate_chemical_potentials=True,
)

run_locally(flows, create_folders=True, ensure_success=True)
