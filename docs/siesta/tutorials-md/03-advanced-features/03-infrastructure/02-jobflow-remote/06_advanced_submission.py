#!/usr/bin/env python
"""
Advanced Job Submission: Combining resources + exec_config.

This tutorial demonstrates real-world scenarios combining:
- Custom SLURM/PBS resources
- Execution environment configuration
- Workflow-specific optimizations
- Batch submission strategies
"""

from pathlib import Path
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.flows.convergence import KpointsConvergenceFlowMaker
from atomate2.siesta.flows.eos import SiestaEosFlowMaker

# Check if jobflow-remote is configured
config_file = Path.home() / ".jfremote" / "atomate2siesta.yaml"
if not config_file.exists():
    print("=" * 80)
    print("Configuration Not Found")
    print("=" * 80)
    print("\n⚠️  Run: atomate2siesta-jobflow-remote setup")
    print("\n" + "=" * 80)
    exit(1)

print("=" * 80)
print("Tutorial: Advanced Job Submission Strategies")
print("=" * 80)

# Load structure
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# ============================================================================
# Scenario 1: Production Calculation - Large System
# ============================================================================
print("\n### Scenario 1: Production Calculation (Large System)")
print("System: 200+ atoms")
print("Requirements: 8 nodes, 48 cores/node, 72-hour walltime, high-memory")

production_job = RelaxMaker.fixed_cell_relaxation().make(structure)

# Uncomment to submit:
# job_id = submit_flow(
#     production_job,
#     project="atomate2siesta",
#     worker="slurm_worker",
#     resources={
#         "nodes": 8,
#         "ntasks_per_node": 48,
#         "time": "72:00:00",
#         "partition": "large",
#         "account": "research_grant",
#         "qos": "normal",
#         "mem_per_cpu": "4GB"
#     },
#     exec_config={
#         "modules": ["siesta/4.1", "intel-mpi/2021", "mkl/2021"],
#         "export": {
#             "SIESTA_PP_PATH": "/scratch/pseudos/ONCVPSP-PBE-SR-PDv0.4-Standard",
#             "OMP_NUM_THREADS": "1",
#             "MKL_NUM_THREADS": "1",
#             "I_MPI_PIN_DOMAIN": "omp"
#         },
#         "pre_run": """
# ulimit -s unlimited
# echo "=== Job Started at $(date) ==="
# echo "Nodes: $SLURM_JOB_NUM_NODES"
# echo "Tasks: $SLURM_NTASKS"
# echo "CPUs per task: $SLURM_CPUS_PER_TASK"
# """.strip(),
#         "post_run": """
# echo "=== Job Finished at $(date) ==="
# echo "Disk usage:"
# du -sh .
# echo "Compressing wavefunctions..."
# gzip -9 *.WFSX *.DM 2>/dev/null || true
# """.strip()
#     }
# )
# print(f"✓ Production job submitted: {job_id}")

print("\nConfiguration:")
print("  Resources: 8 nodes × 48 cores = 384 total cores")
print("  Memory: 4GB/core × 384 = ~1.5TB total")
print("  Walltime: 72 hours")
print("  Software: SIESTA 4.1 + Intel MPI + MKL")
print("  Optimization: Thread pinning, stack limit, output compression")

# ============================================================================
# Scenario 2: Quick Test - Debug Queue
# ============================================================================
print("\n" + "=" * 80)
print("### Scenario 2: Quick Test (Debug Queue)")
print("Purpose: Verify input files before production run")
print("Requirements: 1 node, 30-minute limit, debug partition")

test_job = StaticMaker().make(structure)

# Uncomment to submit:
# job_id = submit_flow(
#     test_job,
#     project="atomate2siesta",
#     worker="slurm_worker",
#     resources={
#         "nodes": 1,
#         "ntasks_per_node": 24,
#         "time": "00:30:00",
#         "partition": "debug",
#         "qos": "debug"
#     },
#     exec_config={
#         "modules": ["siesta/4.1"],
#         "pre_run": "echo 'Quick test run...'"
#     }
# )
# print(f"✓ Test job submitted: {job_id}")

print("\nConfiguration:")
print("  Resources: 1 node, 24 cores, 30 minutes")
print("  Purpose: Fast turnaround for debugging")
print("  Strategy: Minimal resources for rapid scheduling")

# ============================================================================
# Scenario 3: GPU-Accelerated Calculation
# ============================================================================
print("\n" + "=" * 80)
print("### Scenario 3: GPU-Accelerated Calculation")
print("Requirements: 4 GPUs, GPU partition, CUDA modules")

gpu_job = RelaxMaker.fixed_cell_relaxation().make(structure)

# Uncomment to submit:
# job_id = submit_flow(
#     gpu_job,
#     project="atomate2siesta",
#     worker="gpu_worker",
#     resources={
#         "nodes": 1,
#         "ntasks_per_node": 4,
#         "gres": "gpu:4",
#         "partition": "gpu",
#         "time": "24:00:00",
#         "mem": "128GB"
#     },
#     exec_config={
#         "modules": ["cuda/12.0", "siesta-gpu/4.1", "openmpi/4.1"],
#         "export": {
#             "CUDA_VISIBLE_DEVICES": "0,1,2,3",
#             "CUDA_MPS_PIPE_DIRECTORY": "/tmp/nvidia-mps",
#             "SIESTA_PP_PATH": "/scratch/pseudos"
#         },
#         "pre_run": """
# nvidia-smi
# echo "GPUs: $CUDA_VISIBLE_DEVICES"
# nvidia-cuda-mps-control -d  # Start MPS daemon
# """.strip(),
#         "post_run": """
# echo quit | nvidia-cuda-mps-control  # Stop MPS
# nvidia-smi
# """.strip()
#     }
# )
# print(f"✓ GPU job submitted: {job_id}")

print("\nConfiguration:")
print("  GPUs: 4× NVIDIA (using CUDA MPS for sharing)")
print("  Software: CUDA 12.0 + GPU-enabled SIESTA")
print("  Optimization: Multi-Process Service (MPS) for GPU efficiency")

# ============================================================================
# Scenario 4: Convergence Study - Batch Submission
# ============================================================================
print("\n" + "=" * 80)
print("### Scenario 4: Convergence Study Workflow")
print("Multiple calculations with consistent resources")

convergence_flow = KpointsConvergenceFlowMaker(
    kpoints_list=[[2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8], [10, 10, 10]]
).make(structure)

# Uncomment to submit:
# job_id = submit_flow(
#     convergence_flow,
#     project="atomate2siesta",
#     worker="slurm_worker",
#     resources={
#         "nodes": 1,
#         "ntasks_per_node": 24,
#         "time": "12:00:00",
#         "partition": "normal"
#     },
#     exec_config={
#         "modules": ["siesta/4.1", "openmpi/4.1"],
#         "export": {"SIESTA_PP_PATH": "/scratch/pseudos"}
#     }
# )
# print(f"✓ Convergence workflow submitted: {job_id}")

print("\nConfiguration:")
print("  Workflow: 5 k-point calculations + 1 analysis")
print("  Resources: Shared across all jobs (1 node each)")
print("  Strategy: Each job uses same resources for fair comparison")

# ============================================================================
# Scenario 5: High-Throughput Screening
# ============================================================================
print("\n" + "=" * 80)
print("### Scenario 5: High-Throughput Material Screening")
print("Submit 50 structures with optimized resources")

# Example: Batch submission for multiple structures
structures = [structure] * 5  # Simplified example with 5 structures

print(f"\nSubmitting {len(structures)} calculations...")

job_ids = []
for i, struct in enumerate(structures):
    job = RelaxMaker.fixed_cell_relaxation().make(struct)

    # Uncomment to submit:
    # job_id = submit_flow(
    #     job,
    #     project="atomate2siesta",
    #     worker="slurm_worker",
    #     resources={
    #         "nodes": 1,
    #         "ntasks_per_node": 12,  # Smaller jobs for faster scheduling
    #         "time": "06:00:00",      # 6-hour limit
    #         "partition": "normal"
    #     },
    #     exec_config={
    #         "modules": ["siesta/4.1"],
    #         "export": {"SIESTA_PP_PATH": "/scratch/pseudos"},
    #         "post_run": "gzip -9 *.WFSX *.DM 2>/dev/null || true"
    #     }
    # )
    # job_ids.append(job_id)
    # print(f"  Job {i+1}/{len(structures)}: {job_id}")

print("\nStrategy:")
print("  - Smaller resource requests (12 cores) for faster queue times")
print("  - Shorter walltime (6 hours) to fit more queue windows")
print("  - Automatic compression to save disk space")
print("  - Parallel submission for maximum throughput")

# ============================================================================
# Scenario 6: EOS Workflow with Custom Resources
# ============================================================================
print("\n" + "=" * 80)
print("### Scenario 6: Equation of State (EOS) Workflow")
print("7 volume calculations + fitting, different resources per stage")

eos_flow = SiestaEosFlowMaker(
    number_of_frames=7, initial_relax_maker=RelaxMaker.fixed_cell_relaxation()
).make(structure)

# Uncomment to submit:
# job_id = submit_flow(
#     eos_flow,
#     project="atomate2siesta",
#     worker="slurm_worker",
#     resources={
#         "nodes": 2,
#         "ntasks_per_node": 24,
#         "time": "24:00:00",
#         "partition": "normal",
#         "account": "research_project"
#     },
#     exec_config={
#         "modules": ["siesta/4.1", "openmpi/4.1", "python/3.11"],
#         "export": {
#             "SIESTA_PP_PATH": "/scratch/pseudos",
#             "OMP_NUM_THREADS": "1"
#         },
#         "pre_run": "echo 'Starting EOS workflow...'"
#     }
# )
# print(f"✓ EOS workflow submitted: {job_id}")

print("\nConfiguration:")
print("  Calculations: 7 volume points + 1 fitting job")
print("  Resources: 2 nodes per relaxation (moderate parallelization)")
print("  Software: SIESTA + Python (for EOS fitting)")
print("  Walltime: 24 hours total for full workflow")

# ============================================================================
# Scenario 7: Modifying Submitted Jobs
# ============================================================================
print("\n" + "=" * 80)
print("### Scenario 7: Post-Submission Modifications")
print("Change resources/exec_config after job is submitted")

print("\nCLI commands to modify submitted jobs:")
print(
    """
# Increase walltime for job that's taking longer
jf job set resources <job_id> --time 96:00:00

# Add more nodes for better parallelization
jf job set resources <job_id> --nodes 4 --ntasks-per-node 48

# Change to different partition
jf job set resources <job_id> --partition large

# Load additional module
jf job set exec-config <job_id> --modules siesta/5.0 mkl/2023

# Change worker (move to different cluster)
jf job set worker <job_id> --worker different_cluster

# Rerun job with modifications
jf job rerun <job_id>
"""
)

print("Note: Job must be in READY or FAILED state to modify resources")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: Best Practices for Job Submission")
print("=" * 80)

print(
    """
1. Resource Allocation Strategy:
   ✓ Small test: 1 node, 30 min (debug queue)
   ✓ Standard: 1-2 nodes, 24 hours
   ✓ Large production: 4-8 nodes, 48-72 hours
   ✓ High-throughput: 1 node, 6 hours (many small jobs)

2. Queue Selection:
   ✓ Debug: Quick tests (<30 min, 1 node)
   ✓ Normal: Standard calculations (<48 hours)
   ✓ Large: Big systems (48-72 hours, 4+ nodes)
   ✓ GPU: GPU-accelerated calculations

3. Execution Environment:
   ✓ Always set SIESTA_PP_PATH
   ✓ Disable threading (OMP_NUM_THREADS=1) for pure MPI
   ✓ Set stack limit (ulimit -s unlimited)
   ✓ Compress large outputs (*.WFSX, *.DM)

4. Priority System:
   Runtime (submit_flow) > Worker Config (YAML)
   - Use resources= for per-job overrides
   - Use exec_config= for per-job environment
   - Configure worker defaults in YAML for common settings

5. Common Patterns:

   Quick test:
     resources={"nodes": 1, "time": "00:30:00", "partition": "debug"}

   Standard production:
     resources={"nodes": 2, "time": "24:00:00"}
     exec_config={"modules": ["siesta/4.1"], "pre_run": "ulimit -s unlimited"}

   GPU calculation:
     resources={"gres": "gpu:4", "partition": "gpu"}
     exec_config={"modules": ["cuda/12.0", "siesta-gpu/4.1"]}

   High-memory:
     resources={"mem": "500GB", "nodes": 1}

6. Workflow Considerations:
   ✓ Convergence: Same resources for all jobs
   ✓ EOS: Moderate parallelization (2-4 nodes)
   ✓ High-throughput: Small resources, many jobs
   ✓ Production: Maximum resources available

To Enable Submission:
  1. Configure worker in ~/.jfremote/atomate2siesta.yaml
  2. Start runner: jf runner start
  3. Uncomment submit_flow() calls
  4. Adjust resources for your cluster
  5. Run: python 06_advanced_submission.py
"""
)

print("=" * 80)
print("✓ Tutorial complete!")
print("=" * 80)
print("\nFor more information:")
print("  - CLI management: atomate2siesta-jobflow-remote info")
print("  - Job monitoring: jf job list && jf job info <id>")
print("  - Runner commands: atomate2siesta-jobflow-remote runner")
