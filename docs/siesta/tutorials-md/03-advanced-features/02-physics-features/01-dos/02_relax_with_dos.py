#!/usr/bin/env python
"""
DOS Calculation with RelaxMaker - Calculate DOS during geometry optimization

Shows how to enable DOS calculation during structural relaxation.
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import run_locally

# Silicon
structure = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Direct FDF format
user_params = {
    "xc.functional": "GGA",
    "xc.authors": "PBE",
    "Mesh.Cutoff": "200 Ry",
    "a2s_kpts": [4, 4, 4],
    "PAO.BasisSize": "DZP",
    # ProjectedDensityOfStates block (requires %block prefix)
    "%block ProjectedDensityOfStates": ["-10.0 10.0 0.1 200 eV"],
}

# Create RelaxMaker with dry_run enabled
maker = RelaxMaker.fixed_cell_relaxation(
    user_params=user_params, dry_run=False, dry_run_output_dir="relax_dos_preview"
)

job = maker.make(structure)

# Run with dry_run
results = run_locally(job, create_folders=True)

print(f"\nDry-run completed: {results}")
print("Check relax_dos_preview/ for generated SIESTA input files")
print("Output files (when run): siesta.DOS, siesta.PDOS, siesta.PDOS.xml")
