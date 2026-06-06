"""Formation energy diagrams and charge transition levels."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.flows.defects import FormationEnergyDiagramFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

# Create MgO structure
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

maker = FormationEnergyDiagramFlowMaker(
    defect_type="vacancy",
    species="O",
    charge_states=[0, +1, +2, -1, -2],
    epsilon_static=9.8,
    # Bandgap auto-extracted from calculations (auto_bandgap=True by default)
    # Alternatively, set bandgap=7.8 to use a fixed value
    dry_run=False,
    skip_relax=True,
    defect_relax_maker=defect_relax_maker,
    host_static_maker=host_static_maker,
    auto_calculate_chemical_potentials=True,
)

flow = maker.make(mgo)
run_locally(flow, create_folders=True, ensure_success=True)
