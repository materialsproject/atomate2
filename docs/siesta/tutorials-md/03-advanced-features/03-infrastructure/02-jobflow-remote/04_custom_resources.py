#!/usr/bin/env python
"""
Customize SLURM/PBS Resources in submit_flow().

This tutorial demonstrates how to override default submission script parameters
using the 'resources' parameter in submit_flow().
"""

from pathlib import Path
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker

# Check if jobflow-remote is configured
config_file = Path.home() / ".jfremote" / "atomate2siesta.yaml"
if not config_file.exists():
    print("=" * 80)
    print("Configuration Not Found")
    print("=" * 80)
    print("\n⚠️  Jobflow-remote is not configured yet!")
    print("\nTo set up:")
    print("  1. Run: atomate2siesta-jobflow-remote setup")
    print("  2. Configure worker in ~/.jfremote/atomate2siesta.yaml")
    print("  3. Run: jf admin reset")
    print("  4. Start runner: jf runner start")
    print("\n" + "=" * 80)
    exit(1)

print("=" * 80)
print("Tutorial: Customizing Submission Resources")
print("=" * 80)

# Load structure
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# ============================================================================
# Example 1: Basic Submission (Uses Worker Defaults)
# ============================================================================
print("\n### Example 1: Basic Submission")
print("Using default resources from worker configuration...")

basic_job = RelaxMaker.fixed_cell_relaxation().make(structure)

# Uncomment to submit:
# job_id = submit_flow(
#     basic_job,
#     project="atomate2siesta",
#     worker="local_shell"
# )
# print(f"✓ Job submitted: {job_id}")

print("Code:")
print("  submit_flow(job, project='atomate2siesta', worker='local_shell')")
print("Result:")
print("  → Uses default resources from worker config (nodes=1, time=24:00:00, etc.)")

# ============================================================================
# Example 2: Override Resources for Large Calculation
# ============================================================================
print("\n" + "=" * 80)
print("### Example 2: Large Calculation with Custom Resources")
print("Override: nodes=4, ntasks_per_node=48, time=72 hours")

large_job = RelaxMaker.fixed_cell_relaxation().make(structure)

# Uncomment to submit:
# job_id = submit_flow(
#     large_job,
#     project="atomate2siesta",
#     worker="slurm_worker",
#     resources={
#         "nodes": 4,
#         "ntasks_per_node": 48,
#         "time": "72:00:00",
#         "partition": "large",
#         "mem_per_cpu": "4GB"
#     }
# )
# print(f"✓ Job submitted: {job_id}")

print("Code:")
print(
    """  submit_flow(
    job,
    project="atomate2siesta",
    worker="slurm_worker",
    resources={
        "nodes": 4,
        "ntasks_per_node": 48,
        "time": "72:00:00",
        "partition": "large",
        "mem_per_cpu": "4GB"
    }
)"""
)
print("\nGenerated SLURM script includes:")
print("  #SBATCH --nodes=4")
print("  #SBATCH --ntasks-per-node=48")
print("  #SBATCH --time=72:00:00")
print("  #SBATCH --partition=large")
print("  #SBATCH --mem-per-cpu=4GB")

# ============================================================================
# Example 3: GPU Calculation
# ============================================================================
print("\n" + "=" * 80)
print("### Example 3: GPU Calculation")
print("Request: 4 GPUs, 24-hour walltime")

gpu_job = RelaxMaker.fixed_cell_relaxation().make(structure)

# Uncomment to submit:
# job_id = submit_flow(
#     gpu_job,
#     project="atomate2siesta",
#     worker="gpu_worker",
#     resources={
#         "gres": "gpu:4",
#         "time": "24:00:00",
#         "partition": "gpu"
#     }
# )
# print(f"✓ Job submitted: {job_id}")

print("Code:")
print(
    """  submit_flow(
    job,
    worker="gpu_worker",
    resources={"gres": "gpu:4", "time": "24:00:00", "partition": "gpu"}
)"""
)
print("\nGenerated SLURM script includes:")
print("  #SBATCH --gres=gpu:4")
print("  #SBATCH --partition=gpu")

# ============================================================================
# Example 4: High-Memory Job
# ============================================================================
print("\n" + "=" * 80)
print("### Example 4: High-Memory Job")
print("Request: 500GB total memory, single node")

highmem_job = RelaxMaker.fixed_cell_relaxation().make(structure)

# Uncomment to submit:
# job_id = submit_flow(
#     highmem_job,
#     project="atomate2siesta",
#     worker="slurm_worker",
#     resources={
#         "mem": "500GB",
#         "nodes": 1,
#         "ntasks_per_node": 24,
#         "time": "48:00:00"
#     }
# )
# print(f"✓ Job submitted: {job_id}")

print("Code:")
print(
    """  submit_flow(
    job,
    worker="slurm_worker",
    resources={"mem": "500GB", "nodes": 1, "time": "48:00:00"}
)"""
)
print("\nGenerated SLURM script includes:")
print("  #SBATCH --mem=500GB")

# ============================================================================
# Example 5: Quick Test with Short Walltime
# ============================================================================
print("\n" + "=" * 80)
print("### Example 5: Quick Test (1-hour limit)")
print("Testing with minimal resources")

test_job = RelaxMaker.fixed_cell_relaxation().make(structure)

# Uncomment to submit:
# job_id = submit_flow(
#     test_job,
#     project="atomate2siesta",
#     worker="slurm_worker",
#     resources={
#         "time": "01:00:00",
#         "nodes": 1,
#         "partition": "debug"
#     }
# )
# print(f"✓ Job submitted: {job_id}")

print("Code:")
print(
    """  submit_flow(
    job,
    worker="slurm_worker",
    resources={"time": "01:00:00", "nodes": 1, "partition": "debug"}
)"""
)
print("\nGenerated SLURM script includes:")
print("  #SBATCH --time=01:00:00")
print("  #SBATCH --partition=debug")

# ============================================================================
# Example 6: Production Run with QoS
# ============================================================================
print("\n" + "=" * 80)
print("### Example 6: Production Run with High Priority")
print("Using QoS for priority scheduling")

production_job = RelaxMaker.fixed_cell_relaxation().make(structure)

# Uncomment to submit:
# job_id = submit_flow(
#     production_job,
#     project="atomate2siesta",
#     worker="slurm_worker",
#     resources={
#         "qos": "high",
#         "time": "96:00:00",
#         "account": "research_project",
#         "nodes": 2,
#         "ntasks_per_node": 48
#     }
# )
# print(f"✓ Job submitted: {job_id}")

print("Code:")
print(
    """  submit_flow(
    job,
    worker="slurm_worker",
    resources={
        "qos": "high",
        "time": "96:00:00",
        "account": "research_project"
    }
)"""
)
print("\nGenerated SLURM script includes:")
print("  #SBATCH --qos=high")
print("  #SBATCH --account=research_project")
print("  #SBATCH --time=96:00:00")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: Available Resources Parameters")
print("=" * 80)

print(
    """
Common SLURM Parameters:
  - nodes              : Number of compute nodes
  - ntasks_per_node    : MPI tasks per node
  - time               : Walltime (HH:MM:SS or days-HH:MM:SS)
  - partition          : SLURM partition/queue
  - account            : Account/allocation to charge
  - mem                : Total memory per node (e.g., "500GB")
  - mem_per_cpu        : Memory per CPU (e.g., "4GB")
  - qos                : Quality of service (priority)
  - gres               : Generic resources (e.g., "gpu:4")
  - constraint         : Node constraints (e.g., "haswell")

PBS/Torque Parameters:
  - nodes              : Number of nodes
  - ppn                : Processors per node
  - walltime           : Walltime (HH:MM:SS)
  - queue              : PBS queue name
  - mem                : Memory request

Priority of Settings:
  1. resources in submit_flow()     ← HIGHEST (runtime override)
  2. Worker defaults in YAML config ← Fallback

To Enable Submission:
  1. Ensure runner is active: jf runner start
  2. Uncomment submit_flow() calls in this script
  3. Replace "slurm_worker" with your actual worker name
  4. Run: python 04_custom_resources.py
"""
)

print("=" * 80)
print("✓ Tutorial complete!")
print("=" * 80)
print("\nNext: 05_exec_config.py - Customize execution environment")
