#!/usr/bin/env python
"""Walltime limits for cluster jobs - automatic restart on time limit."""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import run_locally

# Create Si structure
structure = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Calculate walltime settings for 1-hour cluster queue
cluster_time_limit = 3600  # seconds
max_walltime = cluster_time_limit * 0.90  # 90% of limit
max_walltime_slack = 100.0  # Stop 100s early

# Configure walltime limits with restart
user_params = {
    "MaxWalltime": max_walltime,  # 3240 s
    "MaxWalltimeSlack": max_walltime_slack,  # 100 s
    "UseSaveData": True,  # Enable restart
    "PAO.BasisSize": "DZP",
    "a2s_kpts": [4, 4, 4],
    "Mesh.Cutoff": "200 Ry",
    "MD.TypeOfRun": "CG",
    "MD.NumCGsteps": 100,
}

# Create job with walltime limits (tier="expert" enables EfficiencyOptions)
maker = RelaxMaker.fixed_cell_relaxation(
    user_params=user_params, tier="expert", dry_run=True
)
job = maker.make(structure)
results = run_locally(job, create_folders=True)

print(
    "✓ Walltime limits configured. SIESTA will save restart files if time limit reached."
)
