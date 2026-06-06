#!/usr/bin/env python
"""Calculate Grüneisen parameters.

Note: This example runs actual calculations (dry_run=False) because Grüneisen
parameter calculation requires real force data. Dry-run mode will successfully
generate input files for all three volume points but will fail at the analysis
step since it uses dummy forces.
"""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.jobs.phonon import PhonopyMaker
from atomate2.siesta.flows.phonon import SiestaGruneisenFlowMaker
from atomate2.siesta.sets.tiers import apply_tier_preset


structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Setup relax maker with tier preset
relax_maker = RelaxMaker()
relax_maker = apply_tier_preset(relax_maker, "phonon_dirty")

# Setup static maker with smaller k-points for phonon calculations
static_maker = StaticMaker.scf(user_params={"a2s_kpts": [2, 2, 2]})
static_maker = apply_tier_preset(static_maker, "phonon_dirty")

# Create phonon maker with custom static maker
phonon_maker = PhonopyMaker(static_maker=static_maker)

# Run actual calculation (not dry-run) to get meaningful Grüneisen parameters
flow = SiestaGruneisenFlowMaker(
    structure_optimizer=relax_maker, phonon_maker=phonon_maker, dry_run=False
)
workflow = flow.make(structure)
results = run_locally(workflow, create_folders=True, root_dir="01_calculate")

print("✓ Grüneisen parameters complete")
