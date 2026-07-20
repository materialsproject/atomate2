#!/usr/bin/env python
"""Basic FDF block input usage."""

from collections import OrderedDict
from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.sets.base import SiestaInputGenerator

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

fdf_blocks = OrderedDict(
    {"PAO.Basis": ["Si 2", " n=3 0 2 P 1", "   4.5 0.0", " n=3 1 1 P 1", "   5.0 0.0"]}
)

input_gen = SiestaInputGenerator(fdf_arguments=fdf_blocks)
maker = StaticMaker(input_set_generator=input_gen, dry_run=False)
job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✓ FDF block calculation complete")
