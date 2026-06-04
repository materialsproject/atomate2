"""Interactive formation energy diagrams and JSON export."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.flows.defects import FormationEnergyDiagramFlowMaker
from atomate2.siesta.flows.defects.analysis import (
    plot_formation_energy_diagram_plotly_job,
    export_formation_energy_json_job,
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
    charge_states=[0, +1, +2],
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
extracted_bandgap = diagram_flow.output.get("extracted_bandgap", 7.8)

# Create interactive Plotly diagram
plotly_job = plot_formation_energy_diagram_plotly_job(
    defect_documents=diagram_flow.output["defect_outputs"],
    bandgap=extracted_bandgap,
    vbm_energy=0.0,
    filename="formation_energy_interactive.html",
    show_ctls=True,
    show_band_edges=True,
)

# Export formation energy data to JSON
json_job = export_formation_energy_json_job(
    defect_documents=diagram_flow.output["defect_outputs"],
    bandgap=extracted_bandgap,
    vbm_energy=0.0,
    filename="formation_energy_data.json",
    include_metadata=True,
)

# Create combined flow
flow = Flow(
    [diagram_flow, plotly_job, json_job],
    output={
        "plotly_html": plotly_job.output,
        "json_data": json_job.output,
    },
    name="interactive_formation_energy_analysis",
)

run_locally(flow, create_folders=True, ensure_success=True)

print("\n" + "=" * 80)
print("INTERACTIVE ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print("  1. formation_energy_interactive.html - Open in browser for interactive plot")
print("     - Hover over lines to see detailed defect information")
print("     - Zoom, pan, and explore the diagram interactively")
print("     - Save as PNG directly from the plot")
print("\n  2. formation_energy_data.json - Machine-readable data export")
print("     - Complete formation energy data")
print("     - Charge transition levels")
print("     - Calculation metadata")
print("     - Ready for external analysis or custom plotting")
print("\nInteractive features:")
print("  • Hover tooltips with E_F, E_f, charge state")
print("  • Zoom and pan for detailed examination")
print("  • Toggle defects on/off in legend")
print("  • Professional styling for presentations")
print("  • Export as PNG/SVG from browser")
print("=" * 80)
