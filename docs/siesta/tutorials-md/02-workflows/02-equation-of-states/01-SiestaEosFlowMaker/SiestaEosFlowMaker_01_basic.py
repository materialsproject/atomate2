#!/usr/bin/env python
"""Basic 7-point EOS calculation."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.eos import SiestaEosFlowMaker

# structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")
structure = Structure.from_file("../../../00-structures/MoS2.cif")

flow = SiestaEosFlowMaker(
    dry_run=False, linear_strain=(-0.05, 0.05), number_of_frames=5
)
workflow = flow.make(structure)
results = run_locally(workflow, create_folders=True)

print("✓ EOS complete: 7 volumes (-5% to +5%)")
