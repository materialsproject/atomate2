#!/usr/bin/env python
"""Adsorption scanning with auto_allocate_resources() for jobflow-remote.

Replaces the manual get_optimal_cores() approach from 05_dynamic_resources_remote.py
with the built-in auto_allocate_resources() powerup. One line instead of a loop!

Comparison:
  05_dynamic_resources_remote.py: Manual loop over jobs, custom helper function
  THIS FILE:                      One-line powerup call (auto_allocate_resources)
"""

from pymatgen.core import Lattice, Molecule, Structure
from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.sets.core import StaticSetGenerator
from atomate2.siesta.powerups import auto_allocate_resources

# from jobflow_remote import submit_flow  # Uncomment when submitting

# ============================================================================
# CONFIGURATION
# ============================================================================

SLAB_PARAMS = {
    "a2s_kpts": [2, 2, 1],
    "Mesh.Cutoff": "200 Ry",
    "PAO.BasisSize": "DZP",
    "OccupationFunction": "MP",
    "ElectronicTemperature": "300 K",
    "SCF.Mixer.Weight": 0.01,
    "SCF.Mixer.Method": "Pulay",
    "SCF.DM.Tolerance": 1e-4,
}

MOLECULE_PARAMS = {
    "a2s_kpts": [1, 1, 1],
    "Mesh.Cutoff": "300 Ry",
    "PAO.BasisSize": "DZP",
    "OccupationFunction": "FD",
    "ElectronicTemperature": "25 K",
    "SCF.Mixer.Weight": 0.1,
    "SCF.DM.Tolerance": 1e-5,
}

# ============================================================================
# STRUCTURES
# ============================================================================

# MgO(100) slab (8 atoms)
lattice = Lattice.from_parameters(a=4.2, b=4.2, c=19.6, alpha=90, beta=90, gamma=90)
species = ["Mg", "Mg", "O", "O", "Mg", "Mg", "O", "O"]
coords = [
    [0.0, 0.0, 0.32],
    [0.5, 0.5, 0.32],
    [0.5, 0.0, 0.36],
    [0.0, 0.5, 0.36],
    [0.0, 0.0, 0.43],
    [0.5, 0.5, 0.43],
    [0.5, 0.0, 0.47],
    [0.0, 0.5, 0.47],
]
slab = Structure(lattice, species, coords)

# CO molecule (2 atoms)
molecule = Molecule(["C", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]])

print(f"Slab: {len(slab)} atoms, Molecule: {len(molecule)} atoms")

# ============================================================================
# WORKFLOW SETUP
# ============================================================================

slab_maker = StaticMaker(
    input_set_generator=StaticSetGenerator(user_params=SLAB_PARAMS),
    use_custodian=True,
)

adsorbate_maker = StaticMaker(
    input_set_generator=StaticSetGenerator(user_params=MOLECULE_PARAMS),
    use_custodian=True,
)

flow_maker = AdsorptionScanFlowMaker(
    grid_size=(2, 2),
    height=2.0,
    slab_static_maker=slab_maker,
    adsorbate_static_maker=adsorbate_maker,
)

workflow = flow_maker.make(slab, molecule)

# ============================================================================
# AUTO-ALLOCATE RESOURCES (one line!)
# ============================================================================

workflow = auto_allocate_resources(
    workflow,
    base_resources={
        "partition": "RES",
        "account": "icn2100",
        "mem_per_cpu": "4G",
    },
    verbose=True,
)

# ============================================================================
# SUBMIT
# ============================================================================

# Uncomment to submit (simple):
# submit_flow(workflow, project="myproject", worker="hpc_worker")

# Or submit with cluster-specific resources + exec_config:
# submit_flow(
#     workflow,
#     project="mn5",
#     worker="mn5_worker",
#     resources={
#         "qos": "gp_debug",
#         "account": "icn85",
#         "nodes": 1,
#         "ntasks_per_node": 24,  # global default; auto_allocate overrides per job
#         "cpus_per_task": 1,
#         "time": "2:00:00",
#     },
#     exec_config={
#         "modules": [
#             "intel", "impi mkl", "hdf5/1.14.1-2",
#             "pnetcdf/1.12.3 netcdf", "openblas", "lapack",
#             "scalapack/2.1.0", "elpa", "siesta/5.0.0",
#         ],
#     },
# )

print("\nTo submit: uncomment submit_flow() and set your project/worker names")
print("\nCompare with 05_dynamic_resources_remote.py:")
print("  Before: 30 lines of manual loop + helper function")
print("  Now:    1 line — auto_allocate_resources(workflow, base_resources={...})")
