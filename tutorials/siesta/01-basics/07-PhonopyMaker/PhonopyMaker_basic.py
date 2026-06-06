#!/usr/bin/env python
"""Basic phonon calculation."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.phonon import PhonopyMaker
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset

structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")

# Create static maker with tier preset
static_maker = StaticMaker()
static_maker = apply_tier_preset(static_maker, "phonon_dirty")

# Create phonon workflow with custom static maker
flow = PhonopyMaker(
    static_maker=static_maker,
    dry_run=False,
    min_length=10.0,  # Auto-generate supercell
)
workflow = flow.make(structure)
results = run_locally(workflow, create_folders=True, root_dir="01_basic")

print("✓ Phonon calculation complete")
