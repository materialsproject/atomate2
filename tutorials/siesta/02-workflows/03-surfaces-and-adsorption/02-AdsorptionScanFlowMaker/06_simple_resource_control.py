#!/usr/bin/env python
"""Simple resource control for adsorbate jobs.

This example shows the SIMPLEST way to set different core counts
for molecule vs slab calculations. Just set resources when creating
the makers!

Key points:
- Molecule maker: 4 cores (small system)
- Slab maker: 24 cores (default, larger system)
- No loops or complexity - just set it once!
"""

from pymatgen.core import Lattice, Molecule, Structure
from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.sets.core import StaticSetGenerator
from jobflow_remote import submit_flow

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "slab_params": {
        "a2s_kpts": [2, 2, 1],
        "Mesh.Cutoff": "200 Ry",
        "PAO.BasisSize": "DZP",
        "OccupationFunction": "MP",
        "ElectronicTemperature": "300 K",
    },
    "molecule_params": {
        "a2s_kpts": [1, 1, 1],
        "Mesh.Cutoff": "300 Ry",
        "PAO.BasisSize": "DZP",
        "OccupationFunction": "FD",
        "ElectronicTemperature": "25 K",
    },
    "grid_size": (2, 2),
    "height": 2.0,
    "use_custodian": True,
}

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

# Create input generators
slab_generator = StaticSetGenerator(user_params=CONFIG["slab_params"])
molecule_generator = StaticSetGenerator(user_params=CONFIG["molecule_params"])

# ============================================================================
# CREATE MAKERS WITH DIFFERENT RESOURCES (SIMPLEST WAY!)
# ============================================================================

# Slab maker - uses default 24 cores from submit_flow()
slab_maker = StaticMaker(
    input_set_generator=slab_generator,
    use_custodian=CONFIG["use_custodian"],
)

# Molecule maker - explicitly set to 4 cores
adsorbate_maker = StaticMaker(
    input_set_generator=molecule_generator,
    use_custodian=CONFIG["use_custodian"],
)
# Set resources after creating the maker
adsorbate_maker.config_dict = {
    "manager_config": {
        "resources": {
            "mem_per_cpu": "4G",
            "nodes": 1,
            "ntasks_per_node": 4,  # ← 4 cores for molecule!
            "cpus_per_task": 1,
            "time": "24:00:00",
        }
    }
}

print("Resource Configuration:")
print("  Slab jobs: 24 cores (from submit_flow defaults)")
print("  Molecule job: 4 cores (from adsorbate_maker config)")

# Create workflow
flow = AdsorptionScanFlowMaker(
    grid_size=CONFIG["grid_size"],
    height=CONFIG["height"],
    slab_static_maker=slab_maker,
    adsorbate_static_maker=adsorbate_maker,
)

workflow = flow.make(slab, molecule)

# ============================================================================
# SUBMIT TO JOBFLOW-REMOTE
# ============================================================================
print("\nSubmitting workflow...")

results = submit_flow(
    workflow,
    project="alberto",
    worker="cesga_worker",
    resources={
        # Default resources (used by slab jobs)
        "mem_per_cpu": "4G",
        "nodes": 1,
        "ntasks_per_node": 24,  # Slab gets 24 cores
        "cpus_per_task": 1,
        "time": "24:00:00",
    },
)

print("\n✓ Workflow submitted!")
print("\nHow it works:")
print("  1. Create adsorbate_maker normally")
print("  2. Set adsorbate_maker.config_dict = {...} with 4 cores")
print("  3. Molecule job uses those 4 cores")
print("  4. All other jobs (slab, sites) use default 24 cores from submit_flow()")
print("\nThat's it! Just set config_dict after creating the maker!")
