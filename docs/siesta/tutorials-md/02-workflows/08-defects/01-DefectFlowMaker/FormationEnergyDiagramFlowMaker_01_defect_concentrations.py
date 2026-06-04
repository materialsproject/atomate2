"""Defect concentration calculations with Fermi level solver."""

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

# Create makers
defect_relax_maker = apply_tier_preset(
    RelaxMaker.fixed_cell_relaxation(), "defect_dirty"
)
host_static_maker = apply_tier_preset(StaticMaker(), "defect_dirty")

# Run formation energy diagram workflow with concentration analysis
# Bandgap will be auto-extracted from calculations (auto_bandgap=True by default)
diagram_maker = FormationEnergyDiagramFlowMaker(
    defect_type="vacancy",
    species="O",
    charge_states=[0, +1, +2],  # Multiple charge states for concentration analysis
    epsilon_static=9.8,
    # Bandgap auto-extracted from calculations
    # Alternatively, set bandgap=7.8 to use a fixed value
    vbm_energy=0.0,
    dry_run=False,
    skip_relax=True,
    defect_relax_maker=defect_relax_maker,
    host_static_maker=host_static_maker,
    auto_calculate_chemical_potentials=True,
    include_concentration_analysis=True,  # Enable built-in concentration analysis
    temperature=300.0,  # Room temperature
)

flow = diagram_maker.make(mgo)

run_locally(flow, create_folders=True, ensure_success=True)
