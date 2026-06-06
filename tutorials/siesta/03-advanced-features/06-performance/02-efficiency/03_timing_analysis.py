#!/usr/bin/env python
"""Comprehensive timing analysis - hierarchical tree, per-process, and SCF step timing."""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import StaticMaker
from jobflow import run_locally

# Create Si structure
structure = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Configure comprehensive timing
user_params = {
    "UserTreeTimer": True,  # Hierarchical timing tree
    "UserParallelTimer": True,  # Per-process timing
    "TimingSplitSCFSteps": True,  # Individual SCF step timing
    "TimerReportThreshold": 0.01,  # Report times >10ms
    "PAO.BasisSize": "DZP",
    "a2s_kpts": [6, 6, 6],
    "Mesh.Cutoff": "300 Ry",
}

# Create job with timing analysis (tier="expert" enables EfficiencyOptions)
maker = StaticMaker.scf(user_params=user_params, tier="expert", dry_run=True)
job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✓ Timing analysis configured. Check siesta.out for:")
print("  - Hierarchical timing tree (find bottlenecks)")
print("  - Per-process timing (load balance)")
print("  - SCF step timing (convergence analysis)")
