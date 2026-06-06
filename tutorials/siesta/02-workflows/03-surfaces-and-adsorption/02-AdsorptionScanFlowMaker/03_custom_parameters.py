#!/usr/bin/env python
"""Adsorption scanning with custom parameters defined as configuration.

This example shows how to define all calculation parameters explicitly
in a configuration dictionary. This gives you complete control over:
- k-point sampling
- Mesh cutoff
- Basis set size
- SCF convergence parameters
- Occupation functions
- And more...

Simply modify the CONFIG dictionary to customize your calculations!
"""

from jobflow import run_locally
from pymatgen.core import Lattice, Molecule, Structure
from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.sets.core import StaticSetGenerator

# ============================================================================
# CONFIGURATION - Define all parameters here
# ============================================================================
CONFIG = {
    # Slab+adsorbate calculation parameters
    "slab_params": {
        "a2s_kpts": [2, 2, 1],  # k-point mesh (dense in-plane, Γ in z)
        "Mesh.Cutoff": "50 Ry",  # Real-space grid cutoff 200 Ry
        "PAO.BasisSize": "SZ",  # Double-zeta polarized basis
        "OccupationFunction": "MP",  # Methfessel-Paxton smearing
        "ElectronicTemperature": "300 K",
        "SCF.Mixer.Weight": 0.01,  # Mixing weight for SCF
        "SCF.Mixer.Method": "Pulay",
        "SCF.DM.Tolerance": 1e-4,
    },
    # Isolated molecule parameters
    "molecule_params": {
        "a2s_kpts": [1, 1, 1],  # Γ-point only for molecules
        "Mesh.Cutoff": "50 Ry",  # Higher cutoff for accuracy 300 Ry
        "PAO.BasisSize": "SZ",
        "OccupationFunction": "FD",  # Fermi-Dirac (no smearing)
        "ElectronicTemperature": "25 K",  # Low temperature
        "SCF.Mixer.Weight": 0.1,
        "SCF.DM.Tolerance": 1e-5,
    },
    # Grid scanning parameters
    "grid_size": (3, 3),
    "height": 2.0,
    # Error handling
    "use_custodian": True,
    "custodian_max_errors": 10,
}

# Alternative configurations:
#
# # High-accuracy configuration
# CONFIG_HIGH_ACCURACY = {
#     "slab_params": {
#         "a2s_kpts": [6, 6, 1],
#         "Mesh.Cutoff": "400 Ry",
#         "PAO.BasisSize": "TZP",
#         "OccupationFunction": "MP",
#         "ElectronicTemperature": "300 K",
#         "SCF.Mixer.Weight": 0.005,
#         "SCF.DM.Tolerance": 1e-6,
#     },
#     "molecule_params": {
#         "a2s_kpts": [1, 1, 1],
#         "Mesh.Cutoff": "500 Ry",
#         "PAO.BasisSize": "TZP",
#         "OccupationFunction": "FD",
#         "ElectronicTemperature": "25 K",
#         "SCF.DM.Tolerance": 1e-7,
#     },
#     "grid_size": (5, 5),
#     "height": 2.0,
#     "use_custodian": True,
#     "custodian_max_errors": 15,
# }
#
# # Fast screening configuration
# CONFIG_FAST = {
#     "slab_params": {
#         "a2s_kpts": [2, 2, 1],
#         "Mesh.Cutoff": "150 Ry",
#         "PAO.BasisSize": "SZP",
#         "OccupationFunction": "MP",
#         "ElectronicTemperature": "500 K",
#         "SCF.Mixer.Weight": 0.02,
#     },
#     "molecule_params": {
#         "a2s_kpts": [1, 1, 1],
#         "Mesh.Cutoff": "200 Ry",
#         "PAO.BasisSize": "DZP",
#         "OccupationFunction": "FD",
#         "ElectronicTemperature": "50 K",
#     },
#     "grid_size": (2, 2),
#     "height": 2.0,
#     "use_custodian": False,
#     "custodian_max_errors": 5,
# }

# ============================================================================
# WORKFLOW SETUP
# ============================================================================

# Create MgO(100) slab
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

# Create CO molecule
molecule = Molecule(["C", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]])

# Create input generators with custom parameters
slab_generator = StaticSetGenerator(user_params=CONFIG["slab_params"])
molecule_generator = StaticSetGenerator(user_params=CONFIG["molecule_params"])

# Create makers
slab_maker = StaticMaker(
    input_set_generator=slab_generator,
    use_custodian=CONFIG["use_custodian"],
    custodian_max_errors=CONFIG["custodian_max_errors"],
)

adsorbate_maker = StaticMaker(
    input_set_generator=molecule_generator,
    use_custodian=CONFIG["use_custodian"],
    custodian_max_errors=CONFIG["custodian_max_errors"],
)

# Print configuration
print("Configuration:")
print("\nSlab parameters:")
for key, value in CONFIG["slab_params"].items():
    print(f"  {key}: {value}")
print("\nMolecule parameters:")
for key, value in CONFIG["molecule_params"].items():
    print(f"  {key}: {value}")
print(f"\nGrid size: {CONFIG['grid_size']}")
print(f"Custodian: {'enabled' if CONFIG['use_custodian'] else 'disabled'}")

# Create workflow
flow = AdsorptionScanFlowMaker(
    grid_size=CONFIG["grid_size"],
    height=CONFIG["height"],
    slab_static_maker=slab_maker,
    adsorbate_static_maker=adsorbate_maker,
)

# Run workflow
workflow = flow.make(slab, molecule)
results = run_locally(workflow, create_folders=True, root_dir="03_custom_parameters")


print("\n✓ Adsorption scan complete!")
print("\nTo change calculation parameters:")
print("  1. Modify the CONFIG dictionary at the top of this file")
print("  2. Available parameters: kpts, Mesh.Cutoff, PAO.BasisSize,")
print("     OccupationFunction, ElectronicTemperature, SCF.Mixer.Weight, etc.")
print("  3. See commented examples for high-accuracy and fast-screening configs")

"""
results = submit_flow(
    workflow,
    # project="atomate2siesta",
    # worker="agustina_worker",
    project="alberto",
    worker="cesga_worker",
    resources={
        # "partition": "RES", # for Agustina
        # "account": "icn2100", # for Agustina
        # "mem": "500GB",
        "mem_per_cpu":"4G", # For cesga
        "nodes": 1,
        "ntasks_per_node": 24,
        "cpus_per_task": 1,
        # "ntasks": 24,
        "time": "24:00:00",
    },
)
"""
