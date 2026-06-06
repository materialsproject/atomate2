"""Neutral oxygen vacancy in MgO — submit to HPC via jobflow-remote.

Same as DefectFlowMaker_01_basic_vacancy.py but uses submit_flow() instead of
run_locally(). Demonstrates two approaches for setting HPC resources:

  1. manager_config on individual Makers — per-maker resource control
  2. auto_allocate_resources() — auto-detect from atom count (post-generation)

Changes from local version:
  - run_locally() → submit_flow()
  - Added auto_allocate_resources() to set per-job core counts
"""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.flows.defects import DefectFlowMaker, create_vacancy_with_ghost
from atomate2.siesta.sets.tiers import apply_tier_preset
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.powerups import auto_allocate_resources

# from jobflow_remote import submit_flow  # Uncomment when submitting

# ============================================================================
# Structure setup (same as local version)
# ============================================================================

lattice = Lattice.cubic(4.212)
unit_cell = Structure(
    lattice,
    ["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"],
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
    ],
)
host_structure = unit_cell.make_supercell([2, 2, 2])

o_indices = [i for i, site in enumerate(host_structure) if site.specie.symbol == "O"]
defect_structure = create_vacancy_with_ghost(host_structure, o_indices[15])

print(f"Host: {len(host_structure)} atoms, Defect: {len(defect_structure)} atoms")

# ============================================================================
# Option A: Set resources on individual makers (known system sizes)
# ============================================================================

# Defect supercell (64 atoms) — needs many cores
defect_relax_maker = apply_tier_preset(
    RelaxMaker.fixed_cell_relaxation(
        use_custodian=True,
        manager_config={
            "resources": {
                "ntasks_per_node": 48,
                "time": "24:00:00",
                "partition": "RES",
                "account": "icn2100",
                "mem_per_cpu": "4G",
            }
        },
    ),
    "defect_dirty",
)

# Host static (64 atoms) — same resources as defect
host_static_maker = apply_tier_preset(
    StaticMaker(
        use_custodian=True,
        manager_config={
            "resources": {
                "ntasks_per_node": 48,
                "time": "12:00:00",
                "partition": "RES",
                "account": "icn2100",
                "mem_per_cpu": "4G",
            }
        },
    ),
    "defect_dirty",
)

maker = DefectFlowMaker(
    epsilon_static=9.8,
    defect_type="vacancy",
    charge_state=0,
    skip_relax=True,
    auto_calculate_chemical_potentials=True,
    defect_relax_maker=defect_relax_maker,
    host_static_maker=host_static_maker,
)

flow = maker.make(
    defect_structure,
    host_structure,
    host_structure[o_indices[15]].frac_coords.tolist(),
    "O",
)

# ============================================================================
# Option B: Auto-allocate resources based on atom count (simpler!)
# ============================================================================

# Each job gets cores/time based on its structure size:
#   - O2 molecule reference (2 atoms) → 2 cores, 02:00:00
#   - Host static (64 atoms)          → 64 cores, 12:00:00
#   - Defect supercell (64 atoms)     → 64 cores, 12:00:00
#   - Post-processing (0 atoms)       → 1 core,  00:30:00
flow = auto_allocate_resources(
    flow,
    base_resources={
        "partition": "RES",
        "account": "icn2100",
        "mem_per_cpu": "4G",
    },
    verbose=True,
)

# ============================================================================
# Submit to HPC
# ============================================================================

# Uncomment to submit (simple):
# submit_flow(flow, project="myproject", worker="hpc_worker")

# Or submit with cluster-specific resources + exec_config:
# submit_flow(
#     flow,
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
print("Monitor: jf job list")
