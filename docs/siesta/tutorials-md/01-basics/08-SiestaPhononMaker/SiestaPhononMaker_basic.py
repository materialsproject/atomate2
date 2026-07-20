#!/usr/bin/env python
"""Basic phonon calculation with automatic plotting and tier preset.

The SiestaPhononFlowMaker automatically generates:
- phonon_bands.png
- phonon_dos.png
- thermal_properties.png
- phonon_summary.txt

This example uses the phonon_standard tier preset for optimized parameters.

Runtime: ~2 min (dry-run), ~20-30 minutes (full calculation)
"""

from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.siesta.jobs.core import SiestaPhononMaker, StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset

# Load structure
structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")

# Create static maker and apply tier preset for force calculations
static_maker = StaticMaker()
static_maker = apply_tier_preset(static_maker, "phonon_dirty")

# Create phonon workflow with automatic plotting
maker = SiestaPhononMaker(
    # Makers
    static_maker=static_maker,  # Use tier-preset static maker
    relax_maker=None,  # No relaxation
    # Supercell size (two options):
    min_length=10.0,  # Auto-generate supercell (e.g., 2×2×2 for Si)
    # OR use explicit matrix:
    # supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],  # 2×2×2 supercell
    prefer_90_degrees=True,  # Prefer ~90° angles when auto-generating
    # Displacement
    displacement=0.01,  # Atomic displacement (Å)
    use_symmetry=True,  # Reduce calculations via symmetry
    # Q-point sampling
    mesh=(30, 30, 30),  # Q-point mesh for DOS
    # Thermal properties
    create_thermal_properties=True,
    t_min=0,
    t_max=1000,
    t_step=10,
    # Plotting (automatic)
    generate_plots=True,
    plot_band_structure=True,
    plot_dos=True,
    plot_thermal=True,
    write_summary=True,
    # Execution mode
    # dry_run=True,  # Set to False to run actual calculations
)

# Run workflow (includes phonon calc + automatic plotting)
flow = maker.make(structure)
results = run_locally(flow, create_folders=True, ensure_success=True)

print(
    "\n✓ Complete! Check: phonon_bands.png, phonon_dos.png, "
    "thermal_properties.png, phonon_summary.txt"
)
