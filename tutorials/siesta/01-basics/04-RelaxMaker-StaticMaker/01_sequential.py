#!/usr/bin/env python
"""Sequential workflow example."""

from jobflow import run_locally, Flow
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker

structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")

relax_job = RelaxMaker.fixed_cell_relaxation(dry_run=True).make(structure)
static_job = StaticMaker(dry_run=True).make(structure)

workflow = Flow([relax_job, static_job])
results = run_locally(workflow, create_folders=True)

print("✓ Sequential workflow complete")
