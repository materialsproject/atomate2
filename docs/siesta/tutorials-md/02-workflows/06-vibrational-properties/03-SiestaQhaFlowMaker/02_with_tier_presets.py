#!/usr/bin/env python
"""QHA calculation with tier presets and optimized parameters.

This example demonstrates using tier presets with the QHA workflow for faster
calculations. It uses smaller k-points and the phonon_dirty preset.

Note: This example runs actual calculations (dry_run=False) because QHA
requires real force and energy data. Dry-run mode will successfully generate
input files for all volume points but will fail at the analysis step since
it uses dummy forces and energies.

Important: The example uses ignore_imaginary_modes=True because low-accuracy
calculations (phonon_dirty preset + small k-points) may produce imaginary modes
at some volumes. Without this flag, QHA fitting would fail if fewer than 4
valid volumes remain after filtering out imaginary modes.
"""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.phonon import SiestaQhaFlowMaker
from atomate2.siesta.jobs.core import StaticMaker, LuaMaker
from atomate2.siesta.jobs.phonon import PhonopyMaker
from atomate2.siesta.sets.tiers import apply_tier_preset

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Setup relax maker with tier preset for structure optimization
# relax_maker = RelaxMaker.variable_cell_relaxation(use_custodian=True)
relax_maker = LuaMaker.variable_cell_relaxation(use_custodian=True)
relax_maker = apply_tier_preset(relax_maker, "phonon_dirty")

# Setup static maker with smaller k-points for phonon force calculations
static_maker = StaticMaker.scf(user_params={"a2s_kpts": [2, 2, 2]}, use_custodian=True)
static_maker = apply_tier_preset(static_maker, "phonon_dirty")

# Create phonon maker with custom static maker
phonon_maker = PhonopyMaker(
    static_maker=static_maker,
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],  # 2x2x2 supercell
    mesh=(50, 50, 50),  # q-point mesh for phonon DOS
)

# phonon_maker.apply_tier_preset(phonon_maker,"phonon_dirty")

# Run QHA calculation with custom settings
flow = SiestaQhaFlowMaker(
    structure_optimizer=relax_maker,
    phonon_maker=phonon_maker,
    number_of_frames=5,  # Number of volumes to sample
    ignore_imaginary_modes=True,  # Use all volumes even if some have imaginary modes
    dry_run=False,
)
workflow = flow.make(structure)
results = run_locally(workflow, create_folders=True, root_dir="02_with_tier_presets")

print("✓ QHA calculation complete")
print("\nKey improvements in this example:")
print("  ✓ Tier preset applied to both relax and static makers")
print("  ✓ Smaller k-points [2, 2, 2] for faster phonon calculations")
print("  ✓ StaticMaker.scf() for clean parameter passing")
print("  ✓ Custom PhonopyMaker with explicit supercell")
