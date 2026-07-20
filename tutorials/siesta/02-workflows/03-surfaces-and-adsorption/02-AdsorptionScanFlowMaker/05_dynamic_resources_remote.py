#!/usr/bin/env python
"""Adsorption scanning with dynamic resource allocation for jobflow-remote.

This example shows how to set appropriate computational resources for different
jobs based on system size. This prevents the "too many processors for system size"
error in SIESTA and improves job scheduling efficiency.

Key Concepts:
1. Small molecules (e.g., H2, CO) need only 1-4 cores
2. Medium systems (slab+adsorbate) need 8-12 cores
3. Large systems (>50 atoms) can use 24+ cores
4. Use job.update_config() to set per-job resources

This approach is more efficient than using a fixed 24 cores for all jobs!
"""

from pymatgen.core import Lattice, Molecule, Structure
from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.sets.core import StaticSetGenerator
from jobflow_remote import submit_flow

# ============================================================================
# HELPER FUNCTION - Determine core count based on system size
# ============================================================================


def get_optimal_cores(num_atoms: int) -> int:
    """Determine optimal core count based on number of atoms.

    Parameters
    ----------
    num_atoms : int
        Number of atoms in the system

    Returns
    -------
    int
        Recommended number of cores

    Rules:
    - Small molecules (<5 atoms): 1 core
    - Medium molecules (5-10 atoms): 4 cores
    - Small slabs (10-20 atoms): 8 cores
    - Medium slabs (20-50 atoms): 12 cores
    - Large systems (>50 atoms): 24 cores
    """
    if num_atoms < 5:
        return 1
    elif num_atoms < 10:
        return 4
    elif num_atoms < 20:
        return 8
    elif num_atoms < 50:
        return 12
    else:
        return 24


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Slab+adsorbate calculation parameters
    "slab_params": {
        "a2s_kpts": [2, 2, 1],  # k-point mesh (dense in-plane, Γ in z)
        "Mesh.Cutoff": "200 Ry",  # Real-space grid cutoff
        "PAO.BasisSize": "DZP",  # Double-zeta polarized basis
        "OccupationFunction": "MP",  # Methfessel-Paxton smearing
        "ElectronicTemperature": "300 K",
        "SCF.Mixer.Weight": 0.01,  # Mixing weight for SCF
        "SCF.Mixer.Method": "Pulay",
        "SCF.DM.Tolerance": 1e-4,
    },
    # Isolated molecule parameters
    "molecule_params": {
        "a2s_kpts": [1, 1, 1],  # Γ-point only for molecules
        "Mesh.Cutoff": "300 Ry",  # Higher cutoff for accuracy
        "PAO.BasisSize": "DZP",
        "OccupationFunction": "FD",  # Fermi-Dirac (no smearing)
        "ElectronicTemperature": "25 K",  # Low temperature
        "SCF.Mixer.Weight": 0.1,
        "SCF.DM.Tolerance": 1e-5,
    },
    # Grid scanning parameters
    "grid_size": (2, 2),
    "height": 2.0,
    # Error handling
    "use_custodian": True,
    "custodian_max_errors": 10,
}

# Base resources for all jobs (can be overridden per-job)
BASE_RESOURCES = {
    "mem_per_cpu": "4G",
    "nodes": 1,
    "cpus_per_task": 1,
    "time": "24:00:00",
}

# ============================================================================
# WORKFLOW SETUP
# ============================================================================

# Create MgO(100) slab (8 atoms)
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

# Create CO molecule (2 atoms)
molecule = Molecule(["C", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]])

# Print system sizes and recommended cores
print("System Sizes and Resource Allocation:")
print(f"  Slab: {len(slab)} atoms → {get_optimal_cores(len(slab))} cores")
print(f"  Molecule: {len(molecule)} atoms → {get_optimal_cores(len(molecule))} cores")
print(
    f"  Slab+molecule: ~{len(slab) + len(molecule)} atoms → {get_optimal_cores(len(slab) + len(molecule))} cores"
)

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

# Create workflow
flow = AdsorptionScanFlowMaker(
    grid_size=CONFIG["grid_size"],
    height=CONFIG["height"],
    slab_static_maker=slab_maker,
    adsorbate_static_maker=adsorbate_maker,
)

workflow = flow.make(slab, molecule)

# ============================================================================
# DYNAMIC RESOURCE ALLOCATION
# ============================================================================
print("\nApplying dynamic resource allocation...")

# Set resources for each job based on system size
for job in workflow:
    job_name = job.name

    # Determine system size and set appropriate cores
    if "molecule" in job_name.lower() or "adsorbate" in job_name.lower():
        # Small molecule jobs - use minimal cores
        num_cores = get_optimal_cores(len(molecule))
        job_type = "molecule"
    elif "slab" in job_name.lower():
        # Check if this is bare slab or slab+adsorbate
        if "site" in job_name.lower():
            # Slab + adsorbate at specific site
            num_cores = get_optimal_cores(len(slab) + len(molecule))
            job_type = "slab+adsorbate"
        else:
            # Bare slab
            num_cores = get_optimal_cores(len(slab))
            job_type = "bare slab"
    else:
        # Default to medium resources for unknown jobs
        num_cores = 8
        job_type = "unknown"

    # Update job configuration with appropriate resources
    job_resources = BASE_RESOURCES.copy()
    job_resources["ntasks_per_node"] = num_cores

    job.update_config(manager_config={"resources": job_resources})

    print(f"  {job_name[:50]:50s} [{job_type:15s}] → {num_cores:2d} cores")

# ============================================================================
# SUBMIT TO JOBFLOW-REMOTE
# ============================================================================
print("\nSubmitting workflow to jobflow-remote...")
print("Note: Each job will use appropriate core count based on system size!")

results = submit_flow(
    workflow,
    project="alberto",
    worker="cesga_worker",
    # Resources will be set per-job via job.update_config() above
    # No need to specify global resources here!
)

print("\n✓ Workflow submitted!")
print("\nKey Benefits of Dynamic Resource Allocation:")
print("  1. Prevents 'too many processors' errors for small molecules")
print("  2. Improves job scheduling efficiency on HPC clusters")
print("  3. Reduces queue wait times (small jobs start faster)")
print("  4. Better resource utilization (no wasted cores)")
print("\nResource Assignment Rules:")
print("  - Molecules (2 atoms): 1 core")
print("  - Bare slab (8 atoms): 8 cores")
print("  - Slab+adsorbate (10 atoms): 8 cores")
print("\nTo customize:")
print("  - Modify get_optimal_cores() function")
print("  - Adjust BASE_RESOURCES dictionary")
print("  - Change thresholds based on your system sizes")
