"""
Tutorial 13: SRH Recombination Analysis

Complete workflow: Defect generation → Formation energies → Concentrations → SRH analysis
"""

from pymatgen.core import Lattice, Structure
from jobflow import run_locally

from atomate2.siesta.flows.defects.analysis import (
    FormationEnergyDiagramFlowMaker,
    CaptureParameters,
)
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset

# MgO structure
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

# Create makers with tier preset
defect_maker = RelaxMaker.fixed_cell_relaxation()
defect_maker = apply_tier_preset(defect_maker, "defect_dirty")

host_maker = StaticMaker()
host_maker = apply_tier_preset(host_maker, "defect_dirty")

# Create flow with SRH analysis enabled
maker = FormationEnergyDiagramFlowMaker(
    defect_type="vacancy",
    species="O",
    charge_states=[0, +1, +2],
    epsilon_static=9.8,
    # Bandgap auto-extracted from calculations (auto_bandgap=True by default)
    # Alternatively, set bandgap=5.5 to use a fixed value
    dry_run=False,
    defect_relax_maker=defect_maker,
    host_static_maker=host_maker,
    auto_calculate_chemical_potentials=True,  # Auto chemical potentials
    include_srh_analysis=True,
    temperature=300.0,
    # Effective masses for SRH analysis
    # For accurate results, provide material-specific values
    # Auto-extraction from band structure not yet implemented (TODO)
    effective_mass_electron=0.35,
    effective_mass_hole=0.7,
    capture_parameters={
        "V_O": CaptureParameters(
            sigma_n=5e-15, sigma_p=1e-16, method="custom", temperature=300.0
        )
    },
)

flow = maker.make(mgo)

# Run
run_locally(flow, create_folders=True)

print("\n✓ Complete workflow done!")
