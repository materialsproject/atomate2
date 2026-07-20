#!/usr/bin/env python
"""THE EASIEST way to set resources for adsorbate jobs.

This example shows the ABSOLUTE SIMPLEST approach using a helper function.
Just call set_cores() and you're done!

Usage:
    adsorbate_maker = set_cores(StaticMaker(...), cores=4)

That's it! One line!
"""

from pymatgen.core import Lattice, Molecule, Structure
from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.sets.core import StaticSetGenerator
from jobflow_remote import submit_flow


# ============================================================================
# HELPER FUNCTION - Set core count in one line!
# ============================================================================
def set_cores(maker, cores=4):
    """Set the number of cores for a maker.

    Parameters
    ----------
    maker : Maker
        The maker to configure
    cores : int
        Number of cores to use (default: 4)

    Returns
    -------
    Maker
        The configured maker

    Example
    -------
    >>> maker = StaticMaker()
    >>> maker = set_cores(maker, cores=4)  # Use 4 cores!
    """
    maker.config_dict = {
        "manager_config": {
            "resources": {
                "mem_per_cpu": "4G",
                "nodes": 1,
                "ntasks_per_node": cores,
                "cpus_per_task": 1,
                "time": "24:00:00",
            }
        }
    }
    return maker


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
slab_generator = StaticSetGenerator(
    user_params={
        "a2s_kpts": [2, 2, 1],
        "Mesh.Cutoff": "200 Ry",
        "PAO.BasisSize": "DZP",
    }
)

molecule_generator = StaticSetGenerator(
    user_params={
        "a2s_kpts": [1, 1, 1],
        "Mesh.Cutoff": "300 Ry",
        "PAO.BasisSize": "DZP",
    }
)

# ============================================================================
# THE EASIEST WAY - Just use set_cores()!
# ============================================================================

# Slab maker - uses default 24 cores
slab_maker = StaticMaker(
    input_set_generator=slab_generator,
    use_custodian=True,
)

# Molecule maker - ONE LINE to set 4 cores!
adsorbate_maker = StaticMaker(
    input_set_generator=molecule_generator,
    use_custodian=True,
)
adsorbate_maker = set_cores(adsorbate_maker, cores=4)  # ← That's it!

print("✓ Adsorbate maker configured for 4 cores")

# Create workflow
flow = AdsorptionScanFlowMaker(
    grid_size=(2, 2),
    height=2.0,
    slab_static_maker=slab_maker,
    adsorbate_static_maker=adsorbate_maker,
)

workflow = flow.make(slab, molecule)

# ============================================================================
# SUBMIT
# ============================================================================
results = submit_flow(
    workflow,
    project="alberto",
    worker="cesga_worker",
    resources={
        "mem_per_cpu": "4G",
        "nodes": 1,
        "ntasks_per_node": 24,  # Default for slab jobs
        "cpus_per_task": 1,
        "time": "24:00:00",
    },
)

print("\n✓ Done!")
print("\nTo change cores:")
print("  adsorbate_maker = set_cores(adsorbate_maker, cores=1)   # 1 core")
print("  adsorbate_maker = set_cores(adsorbate_maker, cores=4)   # 4 cores")
print("  adsorbate_maker = set_cores(adsorbate_maker, cores=12)  # 12 cores")
