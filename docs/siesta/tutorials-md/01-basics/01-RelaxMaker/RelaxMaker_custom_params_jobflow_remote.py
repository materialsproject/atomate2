#!/usr/bin/env python
"""Relaxation with custom parameters."""

from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow_remote import submit_flow

structure = Structure.from_file("../../00-structures/Si_mp-149_primitive.cif")

maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=False,
    user_params={
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [6, 6, 6],
        "xc.functional": "GGA",
        "xc.authors": "PBE",  # Good default for solids
        "a2s_pseudo_relativistic": "SR",  # Scalar relativistic
    },
)
job = maker.make(structure)

# Set custom job name
job.name = "2-Test atomate2siesta - Custom Relaxation"

results = submit_flow(
    job,
    project="mn5",
    worker="mn5_worker",
    # project="alberto",
    # worker="cesga_worker",
    resources={
        # "partition": "RES", # for Agustina
        # "account": "icn2100", # for Agustina
        "qos": "gp_debug",  # for mn5
        "account": "icn85",  # for mn5
        # "mem": "500GB",
        # "mem_per_cpu": "4G",  # For cesga
        "nodes": 1,
        "ntasks_per_node": 4,  # 24
        "cpus_per_task": 1,
        # "ntasks": 24,
        "time": "1:00:00",
    },
)


print("✓ Relax with custom parameters complete")
