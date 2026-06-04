"""Brouwer diagrams for defect analysis."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.flows.defects import FormationEnergyDiagramFlowMaker
from atomate2.siesta.flows.defects.analysis import (
    calculate_brouwer_vs_fermi_level_job,
    calculate_brouwer_vs_temperature_job,
    plot_brouwer_diagram_job,
)
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally, Flow

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

# Run formation energy diagram workflow
# Bandgap will be auto-extracted from calculations
diagram_maker = FormationEnergyDiagramFlowMaker(
    defect_type="vacancy",
    species="O",
    charge_states=[0, +1, +2],  # Multiple charge states for Brouwer analysis
    epsilon_static=9.8,
    # Bandgap auto-extracted from calculations (auto_bandgap=True by default)
    vbm_energy=0.0,
    dry_run=False,
    skip_relax=True,
    defect_relax_maker=defect_relax_maker,
    host_static_maker=host_static_maker,
    auto_calculate_chemical_potentials=True,
)

diagram_flow = diagram_maker.make(mgo)

# Get extracted bandgap from flow output
# This will be the bandgap computed from the host calculation
extracted_bandgap = diagram_flow.output.get("extracted_bandgap", 7.8)

# 1. Brouwer diagram vs. Fermi level
# Shows how charge state populations change across the band gap
brouwer_fermi_job = calculate_brouwer_vs_fermi_level_job(
    defect_documents=diagram_flow.output["defect_outputs"],
    bandgap=extracted_bandgap,
    vbm_energy=0.0,
    temperature=300.0,
    n_sites=5e22,
    effective_dos={"N_C": 1e19, "N_V": 1e19},
    n_points=100,  # Sample 100 Fermi level points
)

# Plot Brouwer diagram vs. Fermi level
plot_fermi_job = plot_brouwer_diagram_job(
    data=brouwer_fermi_job.output,
    filename="brouwer_vs_fermi_level.png",
    show_carriers=True,
    vbm_energy=0.0,
    bandgap=extracted_bandgap,
)

# 2. Brouwer diagram vs. temperature
# Shows thermal activation of defects
brouwer_temp_job = calculate_brouwer_vs_temperature_job(
    defect_documents=diagram_flow.output["defect_outputs"],
    bandgap=extracted_bandgap,
    vbm_energy=0.0,
    n_sites=5e22,
    effective_dos={"N_C": 1e19, "N_V": 1e19},
    temp_min=200.0,  # 200 K
    temp_max=1000.0,  # 1000 K
    n_points=100,  # Sample 100 temperature points
    solve_fermi_level=True,  # Self-consistent Fermi level at each T
)

# Plot Brouwer diagram vs. temperature
plot_temp_job = plot_brouwer_diagram_job(
    data=brouwer_temp_job.output,
    filename="brouwer_vs_temperature.png",
    show_carriers=True,
)

# Create combined flow
flow = Flow(
    [
        diagram_flow,
        brouwer_fermi_job,
        plot_fermi_job,
        brouwer_temp_job,
        plot_temp_job,
    ],
    output={
        "brouwer_fermi": brouwer_fermi_job.output,
        "brouwer_temp": brouwer_temp_job.output,
        "plot_fermi": plot_fermi_job.output,
        "plot_temp": plot_temp_job.output,
    },
    name="brouwer_diagram_analysis",
)

run_locally(flow, create_folders=True, ensure_success=True)
