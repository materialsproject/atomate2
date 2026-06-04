# HPC Job Submission Tutorial

## Overview

This tutorial demonstrates **jobflow-remote**, a modern system for submitting calculations to HPC clusters with automatic queue management, MongoDB backend, and comprehensive CLI tools.

**Why jobflow-remote?**
- ✅ **Simpler** than FireWorks (YAML configuration)
- ✅ **Modern** CLI with comprehensive commands
- ✅ **Flexible** workers (local, SLURM, PBS, SGE)
- ✅ **Automatic** queue management
- ✅ **Built-in** MongoDB integration

---

## Quick Start (5 Commands)

```bash
# 1. Install jobflow-remote
atomate2siesta-jobflow-remote install

# 2. Generate configuration (~/.jfremote/atomate2siesta.yaml)
atomate2siesta-jobflow-remote setup

# 3. Initialize database
jf admin reset

# 4. Start runner daemon
jf runner start

# 5. Submit jobs (see Python examples below)
```

---

## Installation

### Using CLI Helper (Recommended)

```bash
# Stable version
atomate2siesta-jobflow-remote install

# Development version
atomate2siesta-jobflow-remote install --dev
```

### Direct Installation

```bash
pip install jobflow-remote

# Or development version
pip install git+https://github.com/Matgenix/jobflow-remote.git
```

---

## Configuration

### Generate Configuration File

```bash
# Default (local worker)
atomate2siesta-jobflow-remote setup

# Custom MongoDB
atomate2siesta-jobflow-remote setup --host server.com --port 27018

# Custom project/worker names
atomate2siesta-jobflow-remote setup --project-name my_project --worker-name hpc_worker
```

Creates `~/.jfremote/atomate2siesta.yaml` with three sections:

### 1. Workers (Execution Environments)

**Local Worker** (testing):
```yaml
workers:
  local_shell:
    type: local
    scheduler:
      type: shell
    pre_run: |
      export SIESTA_PP_PATH=$HOME/.siesta/pseudos
      export SIESTA_CMD="siesta < siesta.fdf > siesta.out"
```

**SLURM Worker** (HPC):
```yaml
workers:
  slurm_worker:
    type: remote
    host: cluster.university.edu
    user: username
    scheduler:
      type: slurm
      partition: normal
      account: project_name
      time: "24:00:00"
      nodes: 1
      ntasks_per_node: 24
      pre_run: |
        module load siesta/4.1
        export SIESTA_PP_PATH=/scratch/pseudos
        export SIESTA_CMD="mpirun -np 24 siesta < siesta.fdf > siesta.out"
```

**PBS Worker** (HPC):
```yaml
workers:
  pbs_worker:
    type: remote
    host: cluster.university.edu
    scheduler:
      type: pbs
      queue: normal
      walltime: "24:00:00"
      nodes: 1
      ppn: 24
      pre_run: |
        module load siesta
        export SIESTA_PP_PATH=/gpfs/pseudos
```

### 2. Queue Store (MongoDB for job queue)

```yaml
queue:
  store:
    type: MongoStore
    database: atomate2siesta
    collection_name: queue
    host: localhost
    port: 27017
```

### 3. Job Store (MongoDB for results)

```yaml
jobstore:
  docs_store:
    type: MongoStore
    database: atomate2siesta
    collection_name: tasks
    host: localhost
    port: 27017
```

---

## Submitting Jobs

### Basic Workflow

```python
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow_remote import submit_flow

# Load structure
structure = Structure.from_file("Si.cif")

# Create job
relax_maker = RelaxMaker.fixed_cell_relaxation()
job = relax_maker.make(structure)

# Submit to worker
job_id = submit_flow(
    job,
    project="atomate2siesta",
    worker="local_shell"  # or "slurm_worker" for HPC
)

print(f"Job ID: {job_id}")
```

### With Custom Parameters

```python
from atomate2.siesta.powerups import update_user_siesta_settings

# Create and customize job
job = relax_maker.make(structure)
job = update_user_siesta_settings(
    job,
    {
        "PAO.BasisSize": "DZP",
        "kpts": [8, 8, 8],
        "Mesh.Cutoff": "300 Ry",
    }
)

# Submit
job_id = submit_flow(job, project="atomate2siesta", worker="slurm_worker")
```

### Submitting Workflows

```python
from atomate2.siesta.flows.convergence import KpointsConvergenceMaker

# Create convergence workflow
flow = KpointsConvergenceMaker(
    kpoints_list=[[2,2,2], [4,4,4], [6,6,6], [8,8,8]]
)
workflow = flow.make(structure)

# Submit entire workflow
job_id = submit_flow(workflow, project="atomate2siesta", worker="slurm_worker")
```

---

## Customizing Submission Parameters

### Overview: resources and exec_config

When submitting jobs, you can customize two key aspects:

1. **`resources`**: SLURM/PBS submission script parameters (nodes, walltime, memory, etc.)
2. **`exec_config`**: Execution environment (modules, environment variables, pre/post-run commands)

Both parameters override worker defaults defined in your YAML configuration.

### Customizing Resources

The `resources` parameter controls SLURM/PBS submission script directives:

#### Basic Example

```python
job_id = submit_flow(
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
)
```

This generates a SLURM script with:
```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=48
#SBATCH --time=72:00:00
#SBATCH --partition=large
#SBATCH --mem-per-cpu=4GB
```

#### Available Resources Parameters

**SLURM Parameters**:
```python
resources = {
    "nodes": 4,                  # Number of compute nodes
    "ntasks_per_node": 48,       # MPI tasks per node
    "time": "72:00:00",          # Walltime (HH:MM:SS or D-HH:MM:SS)
    "partition": "normal",       # SLURM partition/queue
    "account": "project_name",   # Account/allocation
    "mem": "500GB",              # Total memory per node
    "mem_per_cpu": "4GB",        # Memory per CPU
    "qos": "high",               # Quality of service (priority)
    "gres": "gpu:4",             # Generic resources (GPUs, etc.)
    "constraint": "haswell",     # Node constraints
}
```

**PBS/Torque Parameters**:
```python
resources = {
    "nodes": 4,           # Number of nodes
    "ppn": 24,            # Processors per node
    "walltime": "72:00:00",  # Walltime
    "queue": "normal",    # PBS queue
    "mem": "500GB",       # Memory
}
```

#### Common Use Cases

**GPU Calculation**:
```python
submit_flow(
    job,
    worker="gpu_worker",
    resources={
        "gres": "gpu:4",
        "partition": "gpu",
        "time": "24:00:00"
    }
)
```

**High-Memory Job**:
```python
submit_flow(
    job,
    worker="slurm_worker",
    resources={
        "mem": "500GB",
        "nodes": 1,
        "time": "48:00:00"
    }
)
```

**Quick Test (Debug Queue)**:
```python
submit_flow(
    job,
    worker="slurm_worker",
    resources={
        "time": "00:30:00",
        "nodes": 1,
        "partition": "debug"
    }
)
```

**Production Run with Priority**:
```python
submit_flow(
    job,
    worker="slurm_worker",
    resources={
        "qos": "high",
        "time": "96:00:00",
        "account": "research_grant",
        "nodes": 8,
        "ntasks_per_node": 48
    }
)
```

### Customizing Execution Environment (exec_config)

The `exec_config` parameter controls the execution environment and commands:

#### Structure

```python
exec_config = {
    "modules": [...],           # Software modules to load
    "export": {...},            # Environment variables to set
    "pre_run": "...",          # Commands before job
    "post_run": "..."          # Commands after job
}
```

#### Loading Software Modules

```python
submit_flow(
    job,
    worker="slurm_worker",
    exec_config={
        "modules": ["siesta/4.1", "intel-mpi/2021", "python/3.11"]
    }
)
```

Generated script includes:
```bash
module load siesta/4.1
module load intel-mpi/2021
module load python/3.11
```

#### Setting Environment Variables

```python
submit_flow(
    job,
    worker="slurm_worker",
    exec_config={
        "export": {
            "SIESTA_PP_PATH": "/scratch/pseudos",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1"
        }
    }
)
```

Generated script includes:
```bash
export SIESTA_PP_PATH=/scratch/pseudos
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

#### Pre-run Commands

Commands executed **before** the main job:

```python
submit_flow(
    job,
    worker="slurm_worker",
    exec_config={
        "pre_run": """
ulimit -s unlimited
echo "Job starting at $(date)"
echo "Running on: $(hostname)"
mkdir -p $SCRATCH/siesta_tmp
export TMPDIR=$SCRATCH/siesta_tmp
        """.strip()
    }
)
```

Common uses:
- Set system limits (`ulimit`)
- Create temporary directories
- Log job information
- Check GPU availability (`nvidia-smi`)

#### Post-run Commands

Commands executed **after** the main job:

```python
submit_flow(
    job,
    worker="slurm_worker",
    exec_config={
        "post_run": """
echo "Job finished at $(date)"
echo "Disk usage:"
du -sh .
echo "Compressing large files..."
gzip -9 *.WFSX *.DM 2>/dev/null || true
        """.strip()
    }
)
```

Common uses:
- Compress output files
- Clean up temporary files
- Log completion status
- Archive results

#### Complete Example

Combining all exec_config options:

```python
submit_flow(
    job,
    worker="slurm_worker",
    resources={
        "nodes": 4,
        "ntasks_per_node": 48,
        "time": "48:00:00"
    },
    exec_config={
        "modules": ["siesta/4.1", "intel-mpi/2021", "mkl/2021"],
        "export": {
            "SIESTA_PP_PATH": "/scratch/pseudos",
            "OMP_NUM_THREADS": "1",
            "I_MPI_PIN_DOMAIN": "omp"
        },
        "pre_run": """
ulimit -s unlimited
echo "=== Job Started at $(date) ==="
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
        """.strip(),
        "post_run": """
echo "=== Job Finished at $(date) ==="
gzip -9 *.WFSX *.DM 2>/dev/null || true
du -sh .
        """.strip()
    }
)
```

### GPU-Specific Configuration

For GPU calculations, combine resources and exec_config:

```python
submit_flow(
    job,
    worker="gpu_worker",
    resources={
        "gres": "gpu:2",
        "partition": "gpu",
        "time": "24:00:00",
        "mem": "128GB"
    },
    exec_config={
        "modules": ["cuda/12.0", "siesta-gpu/4.1"],
        "export": {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "CUDA_MPS_PIPE_DIRECTORY": "/tmp/nvidia-mps"
        },
        "pre_run": """
nvidia-smi
echo "GPUs available: $CUDA_VISIBLE_DEVICES"
nvidia-cuda-mps-control -d  # Start MPS daemon
        """.strip(),
        "post_run": """
echo quit | nvidia-cuda-mps-control  # Stop MPS
        """.strip()
    }
)
```

### Priority and Merging Behavior

#### Priority Order
1. **Highest**: `resources` and `exec_config` in `submit_flow()` (runtime)
2. **Lowest**: Worker defaults in `~/.jfremote/projects/atomate2siesta.yaml`

#### Merging Behavior
- **resources**: Overrides matching keys, keeps non-matching defaults
- **exec_config**: Merges with worker config (both are executed)

Example:
```yaml
# Worker config (YAML)
workers:
  slurm_worker:
    scheduler:
      time: "24:00:00"
      nodes: 1
      pre_run: "module load siesta/4.1"
```

```python
# Runtime override
submit_flow(
    job,
    worker="slurm_worker",
    resources={"nodes": 4},  # Override nodes, keep time="24:00:00"
    exec_config={
        "pre_run": "ulimit -s unlimited"  # Runs AFTER "module load"
    }
)
```

Result:
- `nodes`: 4 (overridden)
- `time`: "24:00:00" (from worker default)
- Pre-run: Both "module load siesta/4.1" AND "ulimit -s unlimited"

### Modifying Jobs After Submission

Use CLI commands to modify resources/exec_config after job is submitted:

```bash
# Change resources
jf job set resources <job_id> --nodes 4 --time 96:00:00

# Change exec_config
jf job set exec-config <job_id> --modules siesta/5.0

# Change worker
jf job set worker <job_id> --worker different_worker

# Rerun with modifications
jf job rerun <job_id>
```

**Note**: Job must be in `READY` or `FAILED` state to modify.

### Tutorial Scripts

Explore complete examples in the tutorial directory:

- `04_custom_resources.py` - Comprehensive resources examples (6 scenarios)
- `05_exec_config.py` - Execution environment configuration (6 examples)
- `06_advanced_submission.py` - Real-world workflows combining both (7 scenarios)

### Best Practices

**Resource Allocation**:
- Start small: Test with 1 node, 30 min in debug queue
- Standard: 1-2 nodes, 24 hours for production
- Large systems: 4-8 nodes, 48-72 hours
- High-throughput: Many small jobs (1 node, 6 hours each)

**Execution Environment**:
- Always set `SIESTA_PP_PATH` in export
- Disable threading for pure MPI: `OMP_NUM_THREADS=1`
- Set stack limit: `ulimit -s unlimited` in pre_run
- Compress large outputs in post_run: `gzip -9 *.WFSX *.DM`

**Queue Strategy**:
- Debug queue: Quick tests (<30 min)
- Normal queue: Standard calculations (<48 hours)
- Large queue: Big jobs (48-72 hours, 4+ nodes)
- GPU queue: GPU-accelerated calculations

---

## Runner Management

### Start Runner

```bash
# Start in foreground (see output)
jf runner start

# Start in background (daemon mode)
jf runner start -d

# Start with specific project
jf runner start -p atomate2siesta
```

### Check Runner Status

```bash
jf runner status

# Output shows:
# - Runner state (running/stopped)
# - Active workers
# - Jobs in queue
# - Recent activity
```

### Stop Runner

```bash
jf runner stop

# Force stop
jf runner stop --force
```

---

## Monitoring Jobs

### List Jobs

```bash
# All jobs
jf job list

# Recent jobs
jf job list --limit 10

# Filter by state
jf job list --state RUNNING
jf job list --state COMPLETED
jf job list --state FAILED
```

### Job Details

```bash
# Detailed information
jf job info <job_id>

# Shows:
# - Job state (WAITING, RUNNING, COMPLETED, FAILED)
# - Worker assigned
# - Submission time
# - Runtime
# - Error messages (if failed)
```

### View Output

```bash
# View job output
jf job output <job_id>

# View SIESTA output file
jf job output <job_id> --file siesta.out
```

### Get Results

```bash
# Retrieve job results
jf job get <job_id>

# Returns:
# - Final structure
# - Energy
# - Forces/stresses
# - All calculation outputs
```

---

## Job Management Commands

### Rerun Failed Jobs

```bash
jf job rerun <job_id>
```

### Stop Running Job

```bash
jf job stop <job_id>
```

### Retry Job

```bash
jf job retry <job_id>
```

### Unlock Stuck Job

```bash
jf job unlock <job_id>
```

### Delete Job

```bash
jf job delete <job_id>

# Delete with confirmation
jf job delete <job_id> --force
```

---

## Best Practices

### 1. Test Locally First

```python
# Dry-run to validate
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
job = maker.make(structure)
run_locally(job)  # Check generated files

# Then submit to local worker
maker = RelaxMaker.fixed_cell_relaxation()
job = maker.make(structure)
submit_flow(job, project="test", worker="local_shell")

# Finally submit to HPC
submit_flow(job, project="production", worker="slurm_worker")
```

### 2. Monitor Runner Status

```bash
# Check runner regularly
jf runner status

# View logs
tail -f ~/.jfremote/logs/runner.log
```

### 3. Use Appropriate Worker

- **Local**: Testing, quick calculations
- **SLURM/PBS**: Production, expensive calculations
- **Configure memory/time** based on calculation size

### 4. Handle Failures

```bash
# Check failed jobs
jf job list --state FAILED

# View error
jf job info <failed_job_id>

# Rerun after fixing issue
jf job rerun <failed_job_id>
```

### 5. Database Backups

```bash
# Backup MongoDB (includes queue and results)
mongodump --db atomate2siesta --out /backup/path

# Restore
mongorestore --db atomate2siesta /backup/path/atomate2siesta
```

---

## Troubleshooting

### Runner Not Starting

**Problem**: `jf runner start` fails

**Solutions**:
```bash
# Check configuration
cat ~/.jfremote/atomate2siesta.yaml

# Check MongoDB
mongosh --eval "db.version()"

# Initialize database
jf admin reset

# Check logs
cat ~/.jfremote/logs/runner.log
```

### Jobs Stuck in WAITING

**Problem**: Jobs not running

**Solutions**:
```bash
# Check runner status
jf runner status

# Start runner if not running
jf runner start

# Check worker configuration
jf worker list

# Unlock stuck jobs
jf job unlock <job_id>
```

### Remote Worker Connection Failed

**Problem**: Cannot connect to HPC cluster

**Solutions**:
```yaml
# ~/.jfremote/atomate2siesta.yaml
workers:
  slurm_worker:
    host: cluster.edu  # Check hostname
    user: username     # Check username
    # Add SSH key path if needed
    ssh_config: ~/.ssh/config
```

Test SSH connection:
```bash
ssh username@cluster.edu
```

### SIESTA Not Found on Worker

**Problem**: "SIESTA command not found"

**Solutions**:
```yaml
# Update pre_run in worker config
workers:
  slurm_worker:
    pre_run: |
      module load siesta/4.1  # Load SIESTA module
      export SIESTA_CMD="mpirun siesta < siesta.fdf > siesta.out"
      export SIESTA_PP_PATH=/path/to/pseudos
```

---

## Advanced Usage

### Multiple Projects

```python
# Separate projects for different studies
submit_flow(job, project="surfaces", worker="slurm_worker")
submit_flow(job, project="defects", worker="slurm_worker")
submit_flow(job, project="testing", worker="local_shell")
```

### Batch Submission

```python
# Submit multiple jobs
job_ids = []
for structure in structures:
    job = relax_maker.make(structure)
    job_id = submit_flow(job, project="batch", worker="slurm_worker")
    job_ids.append(job_id)

print(f"Submitted {len(job_ids)} jobs")
```

**Note**: See the "Customizing Submission Parameters" section above for comprehensive `resources` and `exec_config` examples.

---

## Integration with Database Storage

Jobflow-remote automatically stores results in MongoDB (configured in jobstore section). Query results using maggma:

```python
from maggma.stores import MongoStore

# Connect to jobstore
store = MongoStore(
    database="atomate2siesta",
    collection_name="tasks",
    host="localhost",
    port=27017
)

store.connect()

# Query completed jobs
docs = list(store.query({"state": "COMPLETED"}))
print(f"Completed calculations: {len(docs)}")

# Query by formula
si_docs = list(store.query({"formula_pretty": "Si"}))

# Get energies
for doc in si_docs:
    energy = doc["output"]["energy"]
    print(f"Energy: {energy:.6f} eV")
```

---

## Summary

### Key Takeaways

✅ **jobflow-remote** manages HPC job submission automatically
✅ **MongoDB backend** stores queue and results
✅ **Runner daemon** processes jobs in background
✅ **Flexible workers** support local, SLURM, PBS, SGE
✅ **CLI tools** simplify setup and monitoring

### Essential Workflow

```bash
# Setup (once)
atomate2siesta-jobflow-remote setup
jf admin reset

# Run (always)
jf runner start

# Submit (Python)
submit_flow(job, project, worker)

# Monitor
jf job list
jf job info <id>
jf job output <id>
```

### Tutorial Scripts

Explore the complete examples in this directory:

1. **01_cli_setup.py** - CLI commands and configuration
2. **02_local_submit.py** - Submit to local worker (dry-run preview)
3. **03_monitor_jobs.py** - Job monitoring and status
4. **04_custom_resources.py** - Customize SLURM/PBS resources (6 scenarios)
5. **05_exec_config.py** - Configure execution environment (6 examples)
6. **06_advanced_submission.py** - Real-world workflows (7 scenarios)

### Next Steps

1. **Setup jobflow-remote**: `atomate2siesta-jobflow-remote setup`
2. **Try basic tutorials**: `python 01_cli_setup.py`, `python 02_local_submit.py`
3. **Learn customization**: `python 04_custom_resources.py`, `python 05_exec_config.py`
4. **Advanced patterns**: `python 06_advanced_submission.py`
5. **Configure HPC worker** for production in `~/.jfremote/projects/atomate2siesta.yaml`
6. **Database integration**: See `04-infrastructure/01-database-storage/`

---

**Make jobflow-remote your default for HPC calculations!** 🚀
