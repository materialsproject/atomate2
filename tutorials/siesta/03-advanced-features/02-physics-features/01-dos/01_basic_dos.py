#!/usr/bin/env python
"""
Basic DOS Calculation - Generates siesta.DOS automatically

The %block ProjectedDensityOfStates generates both total DOS and PDOS.
SIESTA outputs: siesta.DOS, siesta.PDOS, siesta.PDOS.xml
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally

# Silicon
structure = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Direct FDF format - proper SIESTA syntax
user_params = {
    "xc.functional": "GGA",
    "xc.authors": "PBE",
    "Mesh.Cutoff": "200 Ry",
    "a2s_kpts": [4, 4, 4],
    "PAO.BasisSize": "DZP",
    # ProjectedDensityOfStates block (requires %block prefix)
    "%block ProjectedDensityOfStates": ["-10.0 10.0 0.1 200 eV"],
}

# Create maker with dry_run enabled
maker = StaticMaker.scf(
    user_params=user_params, dry_run=True, dry_run_output_dir="dos_preview"
)

job = maker.make(structure)

# Run with dry_run (generates input files only, no calculation)
results = run_locally(job, create_folders=True, root_dir="01_basic_dos")

print(f"\nDry-run completed: {results}")
print("Check dos_preview/ for generated SIESTA input files")
print("Output files (when run): siesta.DOS, siesta.PDOS, siesta.PDOS.xml")
