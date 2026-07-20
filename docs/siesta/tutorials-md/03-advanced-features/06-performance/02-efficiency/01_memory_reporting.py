#!/usr/bin/env python
"""Memory allocation monitoring - reports allocations larger than threshold."""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import run_locally

# Create Si structure
structure = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Configure memory reporting
user_params = {
    "AllocReportLevel": 2,  # Moderate verbosity
    "AllocReportThreshold": 5.0,  # Report allocations >5 MB
    "PAO.BasisSize": "DZP",
    "a2s_kpts": [4, 4, 4],
    "Mesh.Cutoff": "200 Ry",
    "MD.TypeOfRun": "CG",
    "MD.NumCGsteps": 0,  # Single point
}

# Create job with memory monitoring (tier="expert" enables EfficiencyOptions)
maker = RelaxMaker.fixed_cell_relaxation(
    user_params=user_params, tier="expert", dry_run=False
)
job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✓ Memory reporting configured. Check siesta.out for allocation reports.")
