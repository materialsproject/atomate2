#!/usr/bin/env python
"""
Customize Execution Environment with exec_config.

This tutorial demonstrates how to configure the execution environment
using the 'exec_config' parameter in submit_flow(). This controls:
- Module loading
- Environment variables
- Pre/post-run commands
- Execution scripts
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
    print("\nRun: atomate2siesta-jobflow-remote setup")
    print("\n" + "=" * 80)
    exit(1)

print("=" * 80)
print("Tutorial: Customizing Execution Environment (exec_config)")
print("=" * 80)

# Load structure
structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

# ============================================================================
# Example 1: Load Specific Modules
# ============================================================================
print("\n### Example 1: Load Specific Software Modules")
print("Load SIESTA 4.1 and Intel MPI before execution")

job1 = RelaxMaker.fixed_cell_relaxation().make(structure)

# Uncomment to submit:
# job_id = submit_flow(
#     job1,
#     project="atomate2siesta",
#     worker="slurm_worker",
#     exec_config={
#         "modules": ["intel","impi mkl", "hdf5/1.14.1-2", "pnetcdf/1.12.3 netcdf","openblas", "lapack" ,"scalapack/2.1.0","elpa","siesta/5.0.0"],
#     }
# )
# print(f"✓ Job submitted: {job_id}")

print("Code:")
print(
    """  submit_flow(
    job,
    worker="slurm_worker",
    exec_config={
        "modules": ["siesta/4.1", "intel-mpi/2021"]
    }
)"""
)
print("\nGenerated submission script includes:")
print("  module load siesta/4.1")
print("  module load intel-mpi/2021")

# ============================================================================
# Example 2: Set Environment Variables
# ============================================================================
print("\n" + "=" * 80)
print("### Example 2: Set Custom Environment Variables")
print("Configure pseudopotential path and OpenMP threads")

job2 = RelaxMaker.fixed_cell_relaxation().make(structure)

# Uncomment to submit:
# job_id = submit_flow(
#     job2,
#     project="atomate2siesta",
#     worker="slurm_worker",
#     exec_config={
#         "export": {
#             "SIESTA_PP_PATH": "/scratch/pseudos",
#             "OMP_NUM_THREADS": "1",
#             "MKL_NUM_THREADS": "1"
#         }
#     }
# )
# print(f"✓ Job submitted: {job_id}")

print("Code:")
print(
    """  submit_flow(
    job,
    worker="slurm_worker",
    exec_config={
        "export": {
            "SIESTA_PP_PATH": "/scratch/pseudos",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1"
        }
    }
)"""
)
print("\nGenerated submission script includes:")
print("  export SIESTA_PP_PATH=/scratch/pseudos")
print("  export OMP_NUM_THREADS=1")
print("  export MKL_NUM_THREADS=1")

# ============================================================================
# Example 3: Pre-run Commands
# ============================================================================
print("\n" + "=" * 80)
print("### Example 3: Pre-run System Configuration")
print("Set stack size and create scratch directory")

job3 = RelaxMaker.fixed_cell_relaxation().make(structure)

# Uncomment to submit:
# job_id = submit_flow(
#     job3,
#     project="atomate2siesta",
#     worker="slurm_worker",
#     exec_config={
#         "pre_run": """
# ulimit -s unlimited
# mkdir -p $SCRATCH/siesta_tmp
# export TMPDIR=$SCRATCH/siesta_tmp
# """.strip()
#     }
# )
# print(f"✓ Job submitted: {job_id}")

print("Code:")
print(
    """  submit_flow(
    job,
    worker="slurm_worker",
    exec_config={
        "pre_run": '''
ulimit -s unlimited
mkdir -p $SCRATCH/siesta_tmp
export TMPDIR=$SCRATCH/siesta_tmp
        '''
    }
)"""
)
print("\nGenerated submission script includes (before job execution):")
print("  ulimit -s unlimited")
print("  mkdir -p $SCRATCH/siesta_tmp")
print("  export TMPDIR=$SCRATCH/siesta_tmp")

# ============================================================================
# Example 4: Post-run Commands
# ============================================================================
print("\n" + "=" * 80)
print("### Example 4: Post-run Cleanup and Archiving")
print("Compress output files after calculation")

job4 = RelaxMaker.fixed_cell_relaxation().make(structure)

# Uncomment to submit:
# job_id = submit_flow(
#     job4,
#     project="atomate2siesta",
#     worker="slurm_worker",
#     exec_config={
#         "post_run": """
# echo 'Compressing output files...'
# gzip -9 *.WFSX *.DM 2>/dev/null || true
# echo 'Job completed successfully'
# """.strip()
#     }
# )
# print(f"✓ Job submitted: {job_id}")

print("Code:")
print(
    """  submit_flow(
    job,
    worker="slurm_worker",
    exec_config={
        "post_run": '''
echo 'Compressing output files...'
gzip -9 *.WFSX *.DM 2>/dev/null || true
echo 'Job completed successfully'
        '''
    }
)"""
)
print("\nGenerated submission script includes (after job execution):")
print("  echo 'Compressing output files...'")
print("  gzip -9 *.WFSX *.DM 2>/dev/null || true")
print("  echo 'Job completed successfully'")

# ============================================================================
# Example 5: Complete exec_config (All Options)
# ============================================================================
print("\n" + "=" * 80)
print("### Example 5: Complete Configuration")
print("Combining modules, environment variables, and pre/post-run commands")

job5 = RelaxMaker.fixed_cell_relaxation().make(structure)

# Uncomment to submit:
# job_id = submit_flow(
#     job5,
#     project="atomate2siesta",
#     worker="slurm_worker",
#     exec_config={
#         "modules": ["siesta/4.1", "intel-mpi/2021", "python/3.11"],
#         "export": {
#             "SIESTA_PP_PATH": "/scratch/pseudos",
#             "OMP_NUM_THREADS": "1",
#             "SIESTA_CMD": "mpirun -np $SLURM_NTASKS siesta < siesta.fdf > siesta.out"
#         },
#         "pre_run": """
# ulimit -s unlimited
# echo "Starting job at $(date)"
# echo "Running on: $(hostname)"
# echo "Working directory: $(pwd)"
# """.strip(),
#         "post_run": """
# echo "Job finished at $(date)"
# gzip -9 *.WFSX *.DM 2>/dev/null || true
# du -sh .
# """.strip()
#     }
# )
# print(f"✓ Job submitted: {job_id}")

print("Code:")
print(
    """  submit_flow(
    job,
    worker="slurm_worker",
    exec_config={
        "modules": ["siesta/4.1", "intel-mpi/2021", "python/3.11"],
        "export": {
            "SIESTA_PP_PATH": "/scratch/pseudos",
            "OMP_NUM_THREADS": "1",
            "SIESTA_CMD": "mpirun -np $SLURM_NTASKS siesta"
        },
        "pre_run": "ulimit -s unlimited; echo 'Starting...'",
        "post_run": "gzip -9 *.WFSX *.DM; echo 'Done.'"
    }
)"""
)
print("\nGenerated submission script structure:")
print("  1. Load modules (siesta/4.1, intel-mpi/2021, python/3.11)")
print("  2. Export environment variables")
print("  3. Execute pre_run commands")
print("  4. Run SIESTA calculation")
print("  5. Execute post_run commands")

# ============================================================================
# Example 6: GPU-Specific Configuration
# ============================================================================
print("\n" + "=" * 80)
print("### Example 6: GPU Calculation Setup")
print("Configure CUDA environment for GPU calculations")

job6 = RelaxMaker.fixed_cell_relaxation().make(structure)

# Uncomment to submit:
# job_id = submit_flow(
#     job6,
#     project="atomate2siesta",
#     worker="gpu_worker",
#     resources={"gres": "gpu:2"},
#     exec_config={
#         "modules": ["cuda/12.0", "siesta-gpu/4.1"],
#         "export": {
#             "CUDA_VISIBLE_DEVICES": "0,1",
#             "CUDA_MPS_PIPE_DIRECTORY": "/tmp/nvidia-mps",
#             "CUDA_MPS_LOG_DIRECTORY": "/tmp/nvidia-log"
#         },
#         "pre_run": """
# nvidia-smi
# echo "GPUs available: $CUDA_VISIBLE_DEVICES"
# """.strip()
#     }
# )
# print(f"✓ Job submitted: {job_id}")

print("Code:")
print(
    """  submit_flow(
    job,
    worker="gpu_worker",
    resources={"gres": "gpu:2"},
    exec_config={
        "modules": ["cuda/12.0", "siesta-gpu/4.1"],
        "export": {"CUDA_VISIBLE_DEVICES": "0,1"},
        "pre_run": "nvidia-smi"
    }
)"""
)
print("\nGenerated submission script includes:")
print("  module load cuda/12.0")
print("  module load siesta-gpu/4.1")
print("  export CUDA_VISIBLE_DEVICES=0,1")
print("  nvidia-smi  # Check GPU availability")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: exec_config Options")
print("=" * 80)

print(
    """
exec_config Dictionary Keys:

  modules: List[str]
    - Software modules to load before execution
    - Example: ["siesta/4.1", "intel-mpi/2021"]
    - Generates: module load siesta/4.1; module load intel-mpi/2021

  export: Dict[str, str]
    - Environment variables to set
    - Example: {"OMP_NUM_THREADS": "1", "SIESTA_PP_PATH": "/path"}
    - Generates: export OMP_NUM_THREADS=1; export SIESTA_PP_PATH=/path

  pre_run: str
    - Shell commands executed BEFORE job
    - Example: "ulimit -s unlimited; mkdir -p $TMPDIR"
    - Common uses: System limits, directory setup, logging

  post_run: str
    - Shell commands executed AFTER job
    - Example: "gzip -9 *.WFSX; echo 'Done'"
    - Common uses: Cleanup, compression, archiving, notifications

Priority of Settings:
  1. exec_config in submit_flow()   ← HIGHEST (runtime override)
  2. Worker pre_run in YAML config  ← Merged with runtime config

Note: exec_config settings are MERGED with worker defaults, not replaced!

Common Use Cases:

  1. Load different SIESTA version:
     exec_config={"modules": ["siesta/5.0"]}

  2. Set pseudopotential path:
     exec_config={"export": {"SIESTA_PP_PATH": "/custom/path"}}

  3. Increase stack limit:
     exec_config={"pre_run": "ulimit -s unlimited"}

  4. Compress large files:
     exec_config={"post_run": "gzip -9 *.WFSX *.DM"}

  5. GPU setup:
     exec_config={
         "modules": ["cuda/12.0"],
         "export": {"CUDA_VISIBLE_DEVICES": "0,1"},
         "pre_run": "nvidia-smi"
     }

To Enable Submission:
  1. Ensure runner is active: jf runner start
  2. Uncomment submit_flow() calls in this script
  3. Adjust modules/paths for your cluster
  4. Run: python 05_exec_config.py
"""
)

print("=" * 80)
print("✓ Tutorial complete!")
print("=" * 80)
print("\nNext: 06_advanced_submission.py - Combine resources + exec_config")
