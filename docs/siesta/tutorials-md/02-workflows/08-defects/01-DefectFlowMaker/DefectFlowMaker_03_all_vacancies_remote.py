"""
All vacancy defects in MgO — submit to HPC via jobflow-remote.

Same as DefectFlowMaker_03_all_vacancies.py but uses submit_flow() instead
of run_locally(). Shows three resource allocation strategies:

  Strategy A: auto_allocate_resources() — auto-detect from atom count (recommended)
  Strategy B: update_jobflow_resources() — name-pattern matching
  Strategy C: manager_config on individual Makers — per-maker control

The defect workflow is heterogeneous: it contains small molecule references
(O2 = 2 atoms), host static (64 atoms), and defect supercells (63-64 atoms).
Without per-job resources, the O2 molecule would run on 48 cores — wasteful!
"""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.flows.defects import DefectFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset

# from jobflow_remote import submit_flow  # Uncomment when submitting

# ============================================================================
# Structure setup (same as local version)
# ============================================================================

lattice = Lattice.cubic(4.212)
mgo = Structure(
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

# ============================================================================
# Strategy A: auto_allocate_resources() — RECOMMENDED
# ============================================================================
print("=" * 70)
print("Strategy A: auto_allocate_resources() (Recommended)")
print("=" * 70)
print("Auto-detects atom count from each job's structure.\n")

from atomate2.siesta.powerups import auto_allocate_resources  # noqa: E402

flow_a = DefectFlowMaker.from_pristine_structure(
    mgo,
    defect_type="vacancy",
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    charge_states=[0],
    epsilon_static=9.8,
    use_symmetry=False,
    use_custodian=True,
    tier_preset="defect_dirty",
    skip_relax=True,
    auto_calculate_chemical_potentials=True,
)

# Auto-allocate cores/time per job
flow_a = auto_allocate_resources(
    flow_a,
    base_resources={"partition": "RES", "account": "icn2100", "mem_per_cpu": "4G"},
    verbose=True,
)

# Uncomment to submit:
# submit_flow(flow_a, project="myproject", worker="hpc_worker")

# ============================================================================
# Strategy B: update_jobflow_resources() — pattern matching
# ============================================================================
print("\n" + "=" * 70)
print("Strategy B: update_jobflow_resources() (Pattern matching)")
print("=" * 70)
print("Match job names to resource configs.\n")

from atomate2.siesta.powerups import update_jobflow_resources  # noqa: E402

flow_b = DefectFlowMaker.from_pristine_structure(
    mgo,
    defect_type="vacancy",
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    charge_states=[0],
    epsilon_static=9.8,
    use_symmetry=False,
    use_custodian=True,
    tier_preset="defect_dirty",
    skip_relax=True,
    auto_calculate_chemical_potentials=True,
)

flow_b = update_jobflow_resources(
    flow_b,
    resource_configs={
        # Molecule references (O2, Mg bulk) — minimal resources
        "ref_": {
            "ntasks_per_node": 2,
            "time": "02:00:00",
            "partition": "RES",
            "account": "icn2100",
            "mem_per_cpu": "4G",
        },
        # Chemical potential extraction — single core
        "mu_": {
            "ntasks_per_node": 1,
            "time": "00:30:00",
            "partition": "RES",
            "account": "icn2100",
            "mem_per_cpu": "4G",
        },
    },
    # Default for host static + defect relaxations
    default_resources={
        "ntasks_per_node": 48,
        "time": "24:00:00",
        "partition": "RES",
        "account": "icn2100",
        "mem_per_cpu": "4G",
    },
    verbose=True,
)

# Uncomment to submit:
# submit_flow(flow_b, project="myproject", worker="hpc_worker")

# ============================================================================
# Strategy C: manager_config on individual Makers
# ============================================================================
print("\n" + "=" * 70)
print("Strategy C: manager_config on individual Makers")
print("=" * 70)
print("Set resources per maker type before creating the flow.\n")

# Create makers with different resource configs
defect_relax_maker = apply_tier_preset(
    RelaxMaker.fixed_cell_relaxation(
        use_custodian=True,
        manager_config={
            "resources": {
                "ntasks_per_node": 48,
                "time": "48:00:00",
                "partition": "RES",
                "account": "icn2100",
                "mem_per_cpu": "4G",
            }
        },
    ),
    "defect_dirty",
)

host_static_maker = apply_tier_preset(
    StaticMaker(
        use_custodian=True,
        manager_config={
            "resources": {
                "ntasks_per_node": 24,
                "time": "12:00:00",
                "partition": "RES",
                "account": "icn2100",
                "mem_per_cpu": "4G",
            }
        },
    ),
    "defect_dirty",
)

print(
    f"defect_relax_maker: "
    f"{defect_relax_maker.manager_config['resources']['ntasks_per_node']} cores"
)
print(
    f"host_static_maker:  "
    f"{host_static_maker.manager_config['resources']['ntasks_per_node']} cores"
)

flow_c = DefectFlowMaker.from_pristine_structure(
    mgo,
    defect_type="vacancy",
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    charge_states=[0],
    epsilon_static=9.8,
    use_symmetry=False,
    skip_relax=True,
    auto_calculate_chemical_potentials=True,
    defect_relax_maker=defect_relax_maker,
    host_static_maker=host_static_maker,
)

# Uncomment to submit:
# submit_flow(flow_c, project="myproject", worker="hpc_worker")

# ============================================================================
# Strategy D: submit_flow() with global resources + exec_config
# ============================================================================
print("\n" + "=" * 70)
print("Strategy D: submit_flow() with resources= and exec_config=")
print("=" * 70)
print("Set GLOBAL defaults via submit_flow(). Per-job overrides still apply.\n")

flow_d = DefectFlowMaker.from_pristine_structure(
    mgo,
    defect_type="vacancy",
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    charge_states=[0],
    epsilon_static=9.8,
    use_symmetry=False,
    use_custodian=True,
    tier_preset="defect_dirty",
    skip_relax=True,
    auto_calculate_chemical_potentials=True,
)

# Auto-allocate per-job resources (overrides submit_flow defaults per job)
flow_d = auto_allocate_resources(
    flow_d,
    base_resources={"qos": "gp_debug", "account": "icn85"},
    verbose=True,
)

# Uncomment to submit with global resources + exec_config:
# submit_flow(
#     flow_d,
#     project="mn5",
#     worker="mn5_worker",
#     resources={
#         "qos": "gp_debug",
#         "account": "icn85",
#         "nodes": 1,
#         "ntasks_per_node": 24,
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

print(
    """
Note on resource precedence:
  1. submit_flow(resources={...}) sets GLOBAL defaults for all jobs
  2. auto_allocate_resources() sets PER-JOB overrides via job.update_config()
  3. Per-job overrides take priority over submit_flow() defaults

  So you can use BOTH:
    - submit_flow(resources={...}) for cluster-specific settings (qos, account, modules)
    - auto_allocate_resources() for job-specific ntasks_per_node and time
"""
)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("Summary: Resource Allocation for Defect Workflows")
print("=" * 70)
print(
    """
Job types in a defect workflow and their typical sizes:

  Job                    Atoms   Recommended cores   Time
  ---                    -----   -----------------   ----
  O2 molecule ref        2       2                   02:00:00
  Mg bulk ref            8       8                   04:00:00
  Host static            64      48-64               12:00:00
  Defect supercell       63-64   48-64               24:00:00
  mu extraction          0       1                   00:30:00
  Summary/analysis       0       1                   00:30:00

Without per-job resources: O2 runs on 48 cores (wasteful, may even crash)
With auto_allocate_resources(): each job gets appropriate resources

Recommendation:
  1. Use auto_allocate_resources() for automatic detection (Strategy A)
  2. Set base_resources with your cluster's partition/account
  3. Override specific jobs with update_jobflow_resources() if needed (Strategy B)
  4. Use manager_config on makers for full manual control (Strategy C)
"""
)
