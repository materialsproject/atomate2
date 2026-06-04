#!/usr/bin/env python
"""Basic QHA calculation.

Note: This example runs actual calculations (dry_run=False) because QHA
requires real force and energy data. Dry-run mode will successfully generate
input files for all volume points but will fail at the analysis step since
it uses dummy forces and energies.
"""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.phonon import SiestaQhaFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.jobs.phonon import PhonopyMaker
from collections import OrderedDict

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# Configure custom calculation parameters
user_params = OrderedDict(
    {
        "a2s_kpts": [2, 2, 2],  # 2x2x2 k-point mesh
        "PAO.BasisSize": "SZ",  # Single-zeta basis
        "Mesh.Cutoff": "50 Ry",  # 50 Ry mesh cutoff
    }
)

# Create structure optimizer with custom parameters
structure_optimizer = RelaxMaker.variable_cell_relaxation(user_params=user_params)

# Run QHA calculation with custom settings
flow = SiestaQhaFlowMaker(
    structure_optimizer=structure_optimizer,
    ignore_imaginary_modes=True,  # Use all volumes even if some have imaginary modes
    number_of_frames=5,  # Number of volumes
    phonon_maker=PhonopyMaker(
        supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        mesh=(50, 50, 50),  # q-point mesh for phonon DOS
    ),
)
workflow = flow.make(structure)
results = run_locally(workflow, create_folders=True, root_dir="01_basic")

print("✓ QHA calculation complete")
