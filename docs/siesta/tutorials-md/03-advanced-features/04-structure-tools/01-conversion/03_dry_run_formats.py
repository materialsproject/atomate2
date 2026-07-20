#!/usr/bin/env python
"""Generate structures in different formats using dry-run mode."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Generate XSF format
maker_xsf = RelaxMaker.fixed_cell_relaxation(
    dry_run=True, dry_run_format="xsf", dry_run_output_dir="output_xsf"
)
job = maker_xsf.make(structure)
run_locally(job, create_folders=True)

# Generate JSON format
maker_json = RelaxMaker.fixed_cell_relaxation(
    dry_run=True, dry_run_format="json", dry_run_output_dir="output_json"
)
job = maker_json.make(structure)
run_locally(job, create_folders=True)

# Generate POSCAR format
maker_poscar = RelaxMaker.fixed_cell_relaxation(
    dry_run=True, dry_run_format="POSCAR", dry_run_output_dir="output_poscar"
)
job = maker_poscar.make(structure)
run_locally(job, create_folders=True)

print("✓ Generated structures in XSF, JSON, and POSCAR formats")
print("  output_xsf/structure.xsf")
print("  output_json/structure.json")
print("  output_poscar/POSCAR")
