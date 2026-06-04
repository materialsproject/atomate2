#!/usr/bin/env python
"""Preview individual jobs with dry-run mode."""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Example 1: Relax job preview
relax_maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=True, dry_run_output_dir="preview_relax"
)
job = relax_maker.make(structure)
run_locally(job, create_folders=True)
print("✓ Relax preview: preview_relax/")

# Example 2: Static job with custom parameters
static_maker = StaticMaker.scf(
    dry_run=True,
    dry_run_output_dir="preview_static",
    user_params={"PAO.BasisSize": "DZP", "a2s_kpts": [8, 8, 8]},
)
job = static_maker.make(structure)
run_locally(job, create_folders=True)
print("✓ Static preview: preview_static/")
print("  Check siesta.fdf for DZP basis and 8x8x8 k-points")
