#!/usr/bin/env python
"""
Auto Resource Allocation for Heterogeneous Workflows.

When using jobflow-remote, all jobs in a workflow get the same default HPC
resources from submit_flow(). This wastes resources on small jobs (molecule
references, post-processing) and may under-allocate for large supercells.

This tutorial shows 4 approaches to solve this, from simplest to most powerful:

  1. manager_config on FlowMaker  — propagates to all child makers (like dry_run)
  2. manager_config on individual Makers — per-maker control via factory methods
  3. update_jobflow_resources()    — name-pattern matching (existing powerup)
  4. auto_allocate_resources()     — NEW: auto-detect from atom count

All approaches use job.update_config() under the hood to set per-job SLURM
resources, overriding the global resources from submit_flow().

Note: Approach 1 (manager_config on FlowMaker) works with any FlowMaker that
extends BaseSiestaFlowMaker (EOS, NEB, Elastic, Surface, Convergence, etc.).
DefectFlowMaker extends plain Maker, so use Approaches 2-4 for defects.
"""

from pymatgen.core import Lattice, Structure

# ============================================================================
# Create example structure
# ============================================================================

# MgO unit cell (8 atoms)
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
# Approach 1: manager_config on FlowMaker (simplest — like dry_run)
# ============================================================================
print("=" * 70)
print("Approach 1: manager_config on FlowMaker")
print("=" * 70)
print("Sets the SAME resources for ALL child makers automatically.")
print("Works with: EOS, NEB, Elastic, Surface, Convergence, etc.")
print("(Any FlowMaker that extends BaseSiestaFlowMaker)\n")

from atomate2.siesta.flows.eos import SiestaEosFlowMaker  # noqa: E402
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker  # noqa: E402

# manager_config propagates to ALL child makers (like dry_run, use_custodian)
eos_maker = SiestaEosFlowMaker(
    number_of_frames=7,
    initial_relax_maker=RelaxMaker.fixed_cell_relaxation(),
    manager_config={
        "resources": {
            "ntasks_per_node": 24,
            "time": "24:00:00",
            "partition": "RES",
            "account": "icn2100",
            "mem_per_cpu": "4G",
        }
    },
)

# Verify propagation
print("EOS FlowMaker manager_config: set")
print(
    f"  initial_relax_maker.manager_config: "
    f"{eos_maker.initial_relax_maker.manager_config is not None}"
)

# To submit:
# from jobflow_remote import submit_flow
# flow = eos_maker.make(mgo)
# submit_flow(flow, project="myproject", worker="hpc_worker")

# ============================================================================
# Approach 2: manager_config on individual Makers (per-maker control)
# ============================================================================
print("\n" + "=" * 70)
print("Approach 2: manager_config on individual Makers")
print("=" * 70)
print("Set DIFFERENT resources for different maker types.")
print("Best for: when you know which makers need more/fewer resources.\n")

# Heavy supercell relaxation — many cores, long walltime
defect_relax_maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,
    manager_config={
        "resources": {
            "ntasks_per_node": 48,
            "time": "48:00:00",
            "partition": "RES",
            "mem_per_cpu": "4G",
        }
    },
)

# Light host static — fewer cores, shorter walltime
host_static_maker = StaticMaker(
    use_custodian=True,
    manager_config={
        "resources": {
            "ntasks_per_node": 24,
            "time": "12:00:00",
            "partition": "RES",
            "mem_per_cpu": "4G",
        }
    },
)

print(
    f"defect_relax_maker: "
    f"{defect_relax_maker.manager_config['resources']['ntasks_per_node']} cores, "
    f"{defect_relax_maker.manager_config['resources']['time']}"
)
print(
    f"host_static_maker:  "
    f"{host_static_maker.manager_config['resources']['ntasks_per_node']} cores, "
    f"{host_static_maker.manager_config['resources']['time']}"
)

# Pass these makers to any FlowMaker:
# from atomate2.siesta.flows.defects import DefectFlowMaker
# maker = DefectFlowMaker(
#     defect_relax_maker=defect_relax_maker,
#     host_static_maker=host_static_maker,
#     ...
# )

# ============================================================================
# Approach 3: update_jobflow_resources() — name-pattern matching
# ============================================================================
print("\n" + "=" * 70)
print("Approach 3: update_jobflow_resources() powerup")
print("=" * 70)
print("Match job names to resource configs using substring patterns.")
print("Best for: fine-grained control when job names are predictable.\n")

from atomate2.siesta.powerups import update_jobflow_resources  # noqa: E402
from atomate2.siesta.flows.defects import DefectFlowMaker  # noqa: E402
from atomate2.siesta.sets.tiers import apply_tier_preset  # noqa: E402

# Create a defect flow (dry-run to inspect without running)
flow_maker = DefectFlowMaker(
    epsilon_static=9.8,
    defect_relax_maker=apply_tier_preset(
        RelaxMaker.fixed_cell_relaxation(use_custodian=True), "defect_dirty"
    ),
    host_static_maker=apply_tier_preset(
        StaticMaker(use_custodian=True), "defect_dirty"
    ),
    auto_calculate_chemical_potentials=True,
    dry_run=True,
)

flow = DefectFlowMaker.from_pristine_structure(
    mgo,
    defect_type="vacancy",
    species="Mg",
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    charge_states=[0],
    epsilon_static=9.8,
    use_custodian=True,
    tier_preset="defect_dirty",
    dry_run=True,
    auto_calculate_chemical_potentials=True,
)

# Apply different resources based on job name patterns
flow = update_jobflow_resources(
    flow,
    resource_configs={
        # Molecule reference jobs (O2, N2, etc.) — tiny, 2 cores
        "ref_": {
            "ntasks_per_node": 2,
            "time": "02:00:00",
            "mem_per_cpu": "4G",
        },
        # Host static calculation — medium, 24 cores
        "host_static": {
            "ntasks_per_node": 24,
            "time": "12:00:00",
            "mem_per_cpu": "4G",
        },
        # Defect relaxation — heavy, 48 cores
        "defect_relax": {
            "ntasks_per_node": 48,
            "time": "48:00:00",
            "mem_per_cpu": "4G",
        },
    },
    # Default for jobs that don't match any pattern
    default_resources={
        "ntasks_per_node": 12,
        "time": "12:00:00",
        "mem_per_cpu": "4G",
    },
    verbose=True,
)

# ============================================================================
# Approach 4: auto_allocate_resources() — auto-detect from atom count (NEW!)
# ============================================================================
print("\n" + "=" * 70)
print("Approach 4: auto_allocate_resources() powerup (NEW!)")
print("=" * 70)
print("Automatically estimates resources from each job's atom count.")
print("Best for: heterogeneous workflows with varying system sizes.\n")

from atomate2.siesta.powerups import auto_allocate_resources  # noqa: E402

# Create a fresh defect flow
flow2 = DefectFlowMaker.from_pristine_structure(
    mgo,
    defect_type="vacancy",
    species="Mg",
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    charge_states=[0],
    epsilon_static=9.8,
    use_custodian=True,
    tier_preset="defect_dirty",
    dry_run=True,
    auto_calculate_chemical_potentials=True,
)

# Auto-allocate: each job gets resources based on its structure size
flow2 = auto_allocate_resources(
    flow2,
    base_resources={
        "partition": "RES",
        "account": "icn2100",
        "mem_per_cpu": "4G",
    },
    verbose=True,
)

# To submit:
# from jobflow_remote import submit_flow
# submit_flow(flow2, project="myproject", worker="hpc_worker")

# ============================================================================
# Resource Estimation Heuristics
# ============================================================================
print("\n" + "=" * 70)
print("Resource Estimation Heuristics (auto_allocate_resources)")
print("=" * 70)

from atomate2.siesta.powerups import _estimate_resources  # noqa: E402

print(f"\n{'Atoms':>8}  {'Cores':>6}  {'Time':>10}  {'Typical use case'}")
print("-" * 55)
for n, desc in [
    (0, "Post-processing / analysis"),
    (2, "O2 molecule reference"),
    (3, "CO molecule reference"),
    (8, "Small bulk unit cell"),
    (16, "Medium unit cell"),
    (32, "Small supercell"),
    (64, "2x2x2 supercell (defects)"),
    (128, "Large supercell"),
    (256, "Very large supercell"),
]:
    r = _estimate_resources(n)
    print(f"{n:>8}  {r['ntasks_per_node']:>6}  {r['time']:>10}  {desc}")

# ============================================================================
# Complete Example: Defect workflow with jobflow-remote submission
# ============================================================================
print("\n" + "=" * 70)
print("Complete Example: Submit defect workflow to HPC")
print("=" * 70)

print(
    """
# Full working example (uncomment submit_flow to run)

from pymatgen.core import Lattice, Structure
from atomate2.siesta.flows.defects import DefectFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from atomate2.siesta.powerups import auto_allocate_resources
from jobflow_remote import submit_flow

# 1. Create structure
lattice = Lattice.cubic(4.212)
mgo = Structure(lattice, ["Mg"]*4 + ["O"]*4, [...])

# 2. Generate all vacancy defect flows with tier preset
flow = DefectFlowMaker.from_pristine_structure(
    mgo,
    defect_type="vacancy",
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    charge_states=[0],
    epsilon_static=9.8,
    use_custodian=True,
    tier_preset="defect_dirty",
    auto_calculate_chemical_potentials=True,
)

# 3. Auto-allocate cores/time based on atom count
flow = auto_allocate_resources(
    flow,
    base_resources={"partition": "RES", "account": "icn2100", "mem_per_cpu": "4G"},
)

# 4. Submit to HPC (simple — auto_allocate_resources handles per-job resources)
# submit_flow(flow, project="myproject", worker="hpc_worker")

# 4b. Or submit with global resources + exec_config (cluster-specific settings)
# submit_flow(
#     flow,
#     project="mn5",
#     worker="mn5_worker",
#     resources={
#         "qos": "gp_debug",        # cluster-specific
#         "account": "icn85",        # cluster-specific
#         "nodes": 1,
#         "ntasks_per_node": 24,     # global default (auto_allocate overrides per job)
#         "cpus_per_task": 1,
#         "time": "2:00:00",         # global default (auto_allocate overrides per job)
#     },
#     exec_config={
#         "modules": [
#             "intel", "impi mkl", "hdf5/1.14.1-2",
#             "pnetcdf/1.12.3 netcdf", "openblas", "lapack",
#             "scalapack/2.1.0", "elpa", "siesta/5.0.0",
#         ],
#     },
# )
"""
)

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("Summary: Which approach to use?")
print("=" * 70)
print(
    """
  Approach 1 (manager_config on FlowMaker):
    Use when: All jobs need the same resources
    Works with: EOS, NEB, Elastic, Surface, Convergence FlowMakers
    Effort: 1 line (add manager_config={...} to FlowMaker)

  Approach 2 (manager_config on Makers):
    Use when: You know which makers need more/fewer resources
    Works with: Any maker (RelaxMaker, StaticMaker, etc.)
    Effort: Set manager_config on each maker type

  Approach 3 (update_jobflow_resources):
    Use when: You want pattern-based control (match job names)
    Works with: Any flow (post-generation powerup)
    Effort: Define resource_configs dict with patterns

  Approach 4 (auto_allocate_resources):
    Use when: Heterogeneous workflows with varying system sizes
    Works with: Any flow (post-generation powerup)
    Effort: 1 line powerup call after generating the flow

  Approach 5 (submit_flow resources= and exec_config=):
    Use when: Setting cluster-specific global defaults (qos, modules)
    Works with: Any flow submitted via submit_flow()
    Note: Per-job overrides (Approaches 1-4) take priority over these globals

  Resource precedence (highest to lowest):
    1. Per-job: auto_allocate_resources() / update_jobflow_resources()
    2. Per-maker: manager_config on individual Makers
    3. Global: submit_flow(resources={...})

  Recommended combo: auto_allocate_resources() + submit_flow(exec_config={...})
    - auto_allocate handles ntasks_per_node and time per job
    - submit_flow sets cluster-specific settings (qos, account, modules)
"""
)
