=============================
Infrastructure Learning Paths
=============================

Production deployment: database integration, HPC job submission, and automatic error recovery.

----

Tutorial 13: Database Storage
==============================

**Learning Objectives**:

* Store calculation results in MongoDB
* Use Maggma stores for data management
* Query and analyze results programmatically
* Set up high-throughput workflows

**Key Concepts**:

* **Maggma**: Materials Genome Project data management framework
* **MongoStore**: MongoDB interface for storing results
* **Task documents**: Structured result schemas
* High-throughput data management

**Basic Setup**:

.. code-block:: python

   from maggma.stores import MongoStore
   from jobflow import run_locally

   # Create MongoDB store
   store = MongoStore(
       database="my_materials_database",
       collection_name="siesta_calculations",
       host="localhost",
       port=27017,
   )

   # Run workflow with database storage
   results = run_locally(
       flow,
       create_folders=True,
       store=store,
   )

**Querying Results**:

.. code-block:: python

   # Connect to database
   store = MongoStore(
       database="my_materials_database",
       collection_name="siesta_calculations",
   )

   # Query all calculations
   for doc in store.query():
       print(f"{doc['formula']}: {doc['output']['energy']} eV")

   # Query specific materials
   silicon_docs = list(store.query({"formula": "Si2"}))

   # Query by energy range
   low_energy = list(store.query(
       {"output.energy": {"$lt": -10.0}}
   ))

**High-Throughput Example**:

.. code-block:: python

   from pymatgen.core import Structure
   from atomate2.siesta.jobs.core import RelaxMaker
   from jobflow import Flow

   # Load multiple structures
   structures = [
       Structure.from_file(f"structure_{i}.cif")
       for i in range(100)
   ]

   # Create jobs for all structures
   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "PAO.BasisSize": "DZP",
           "a2s_kpts": [4, 4, 4],
       }
   )

   jobs = [maker.make(s) for s in structures]
   flow = Flow(jobs)

   # Run and store in database
   results = run_locally(flow, create_folders=True, store=store)

**Document Structure**:

Results stored as structured documents:

.. code-block:: python

   {
       "_id": ObjectId("..."),
       "uuid": "...",
       "formula": "Si2",
       "structure": {...},
       "input": {
           "parameters": {...},
           "basis_size": "DZP",
       },
       "output": {
           "energy": -213.456,
           "forces": [...],
           "stress": [...],
       },
       "runtime": {
           "started": "2024-01-01T12:00:00",
           "completed": "2024-01-01T12:30:00",
           "wall_time": 1800,
       },
   }

**Advanced Features**:

* **Provenance tracking**: Full calculation history
* **Metadata**: Custom tags and annotations
* **Version control**: Track code/parameter versions
* **Data aggregation**: Complex queries and analysis

📁 **Location**: ``tutorials/04-infrastructure/01-database-storage/``

⏱️ **Time**: 45 minutes

⭐ **Difficulty**: Intermediate

----

Tutorial 14: HPC Job Submission with jobflow-remote ⭐ RECOMMENDED
===================================================================

**Learning Objectives**:

* Submit workflows to HPC clusters with jobflow-remote
* Configure job schedulers (SLURM, PBS, SGE, local)
* Manage job queue and monitor progress
* Use modern CLI tools for job management

**Why jobflow-remote?**

✅ **Simpler** than FireWorks - YAML configuration, no complex setup

✅ **Modern CLI** - Comprehensive commands for all operations

✅ **Flexible** - Local, SLURM, PBS, SGE workers with easy switching

✅ **Automatic** - Queue management and job dependencies handled automatically

✅ **Built-in** - MongoDB integration for queue and results

**Key Concepts**:

* **Workers**: Execution environments (local, remote HPC)
* **Queue Store**: MongoDB-backed job queue
* **Job Store**: MongoDB storage for results
* **Runner Daemon**: Background process that executes jobs
* **CLI Tools**: Complete command-line interface for management

**Quick Start (5 Commands)**:

.. code-block:: bash

   # 1. Install jobflow-remote
   atomate2siesta-jobflow-remote install

   # 2. Generate configuration
   atomate2siesta-jobflow-remote setup

   # 3. Initialize database
   jf admin reset

   # 4. Start runner daemon
   jf runner start

   # 5. Submit jobs (see Python examples below)

**Installation**:

Using the CLI helper (recommended):

.. code-block:: bash

   # Stable version
   atomate2siesta-jobflow-remote install

   # Development version
   atomate2siesta-jobflow-remote install --dev

Direct installation:

.. code-block:: bash

   pip install jobflow-remote

**Configuration**:

Generate configuration file at ``~/.jfremote/atomate2siesta.yaml``:

.. code-block:: bash

   # Default (local worker)
   atomate2siesta-jobflow-remote setup

   # Custom MongoDB
   atomate2siesta-jobflow-remote setup --host server.com --port 27018

   # Custom project/worker names
   atomate2siesta-jobflow-remote setup --project-name my_project --worker-name hpc_worker

**Configuration Structure**:

The configuration file has three main sections:

**1. Workers** (execution environments):

Local worker for testing:

.. code-block:: yaml

   workers:
     local_shell:
       type: local
       scheduler:
         type: shell
       pre_run: |
         export SIESTA_PP_PATH=$HOME/.siesta/pseudos
         export SIESTA_CMD="siesta < siesta.fdf > siesta.out"

SLURM worker for HPC:

.. code-block:: yaml

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
           export SIESTA_CMD="mpirun -np 24 siesta < siesta.fdf > siesta.out"
           export SIESTA_PP_PATH=/scratch/pseudos

PBS worker for HPC:

.. code-block:: yaml

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

**2. Queue Store** (MongoDB for job queue):

.. code-block:: yaml

   queue:
     store:
       type: MongoStore
       database: atomate2siesta
       collection_name: queue
       host: localhost
       port: 27017

**3. Job Store** (MongoDB for results):

.. code-block:: yaml

   jobstore:
     docs_store:
       type: MongoStore
       database: atomate2siesta
       collection_name: tasks
       host: localhost
       port: 27017

**Submitting Jobs**:

Basic workflow submission:

.. code-block:: python

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

With custom parameters:

.. code-block:: python

   from atomate2.siesta.powerups import update_user_siesta_settings

   # Customize job
   job = relax_maker.make(structure)
   job = update_user_siesta_settings(
       job,
       {
           "PAO.BasisSize": "DZP",
           "a2s_kpts": [8, 8, 8],
           "Mesh.Cutoff": "300 Ry",
       }
   )

   # Submit to HPC
   job_id = submit_flow(job, project="atomate2siesta", worker="slurm_worker")

Submitting workflows:

.. code-block:: python

   from atomate2.siesta.flows.convergence import KpointsConvergenceFlowMaker

   # Create convergence workflow
   flow = KpointsConvergenceFlowMaker(
       kpoints_list=[[2,2,2], [4,4,4], [6,6,6], [8,8,8]]
   )
   workflow = flow.make(structure)

   # Submit entire workflow
   job_id = submit_flow(workflow, project="atomate2siesta", worker="slurm_worker")

**Runner Management**:

Start the runner daemon:

.. code-block:: bash

   # Foreground (see output)
   jf runner start

   # Background (daemon mode)
   jf runner start -d

   # With specific project
   jf runner start -p atomate2siesta

Check runner status:

.. code-block:: bash

   jf runner status

   # Shows:
   # - Runner state (running/stopped)
   # - Active workers
   # - Jobs in queue
   # - Recent activity

Stop the runner:

.. code-block:: bash

   jf runner stop

   # Force stop
   jf runner stop --force

**Monitoring Jobs**:

List all jobs:

.. code-block:: bash

   # All jobs
   jf job list

   # Recent jobs
   jf job list --limit 10

   # Filter by state
   jf job list --state RUNNING
   jf job list --state COMPLETED
   jf job list --state FAILED

View job details:

.. code-block:: bash

   jf job info <job_id>

   # Shows:
   # - Job state (WAITING, RUNNING, COMPLETED, FAILED)
   # - Worker assigned
   # - Submission time
   # - Runtime
   # - Error messages (if failed)

View job output:

.. code-block:: bash

   # View job output
   jf job output <job_id>

   # View SIESTA output file
   jf job output <job_id> --file siesta.out

Get results:

.. code-block:: bash

   jf job get <job_id>

   # Returns:
   # - Final structure
   # - Energy
   # - Forces/stresses
   # - All calculation outputs

**Job Management Commands**:

.. code-block:: bash

   # Rerun failed jobs
   jf job rerun <job_id>

   # Stop running job
   jf job stop <job_id>

   # Retry job
   jf job retry <job_id>

   # Unlock stuck job
   jf job unlock <job_id>

   # Delete job
   jf job delete <job_id>

**Best Practices**:

1. **Test locally first**:

   .. code-block:: python

      # Dry-run to validate
      maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
      job = maker.make(structure)
      run_locally(job)  # Check generated files

      # Test with local worker
      submit_flow(job, project="test", worker="local_shell")

      # Then submit to HPC
      submit_flow(job, project="production", worker="slurm_worker")

2. **Monitor runner regularly**:

   .. code-block:: bash

      jf runner status
      tail -f ~/.jfremote/logs/runner.log

3. **Use appropriate workers**:

   * **local_shell**: Testing, quick calculations
   * **slurm_worker/pbs_worker**: Production, expensive calculations

4. **Handle failures**:

   .. code-block:: bash

      # Check failed jobs
      jf job list --state FAILED

      # View error
      jf job info <failed_job_id>

      # Rerun after fixing issue
      jf job rerun <failed_job_id>

**Advanced Features**:

Custom resources:

.. code-block:: python

   # Override default resources
   job_id = submit_flow(
       job,
       project="atomate2siesta",
       worker="slurm_worker",
       resources={
           "nodes": 2,
           "ntasks_per_node": 48,
           "time": "48:00:00",
           "mem_per_cpu": "4GB"
       }
   )

Batch submission:

.. code-block:: python

   # Submit multiple jobs
   job_ids = []
   for structure in structures:
       job = relax_maker.make(structure)
       job_id = submit_flow(job, project="batch", worker="slurm_worker")
       job_ids.append(job_id)

   print(f"Submitted {len(job_ids)} jobs")

**Integration with Database**:

Jobflow-remote automatically stores results in MongoDB (configured in jobstore):

.. code-block:: python

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
   print(f"Completed: {len(docs)}")

   # Query by formula
   si_docs = list(store.query({"formula_pretty": "Si"}))

   # Get energies
   for doc in si_docs:
       energy = doc["output"]["energy"]
       print(f"Energy: {energy:.6f} eV")

**Troubleshooting**:

Runner not starting:

.. code-block:: bash

   # Check configuration
   cat ~/.jfremote/atomate2siesta.yaml

   # Check MongoDB
   mongosh --eval "db.version()"

   # Initialize database
   jf admin reset

   # Check logs
   cat ~/.jfremote/logs/runner.log

Jobs stuck in WAITING:

.. code-block:: bash

   # Check runner status
   jf runner status

   # Start runner if not running
   jf runner start

   # Unlock stuck jobs
   jf job unlock <job_id>

Remote worker connection failed:

.. code-block:: bash

   # Test SSH connection
   ssh username@cluster.edu

   # Check worker configuration
   jf worker list

SIESTA not found on worker:

.. code-block:: yaml

   # Update pre_run in worker config
   workers:
     slurm_worker:
       pre_run: |
         module load siesta/4.1
         export SIESTA_CMD="mpirun siesta < siesta.fdf > siesta.out"
         export SIESTA_PP_PATH=/path/to/pseudos

**Benefits**:

✅ **Modern & Simple**: YAML configuration, comprehensive CLI

✅ **Flexible**: Easy switching between local and HPC workers

✅ **Automatic**: Queue management and dependencies handled

✅ **Integrated**: Built-in MongoDB storage

✅ **Production-ready**: Battle-tested in high-throughput studies

📁 **Location**: ``tutorials/04-infrastructure/02-job-submission/``

⏱️ **Time**: 30 minutes

⭐ **Difficulty**: Intermediate

**CLI Reference**: See :doc:`/siesta/cli-jobflow-remote` for complete CLI documentation

----

Tutorial 14b: FireWorks (Alternative)
======================================

**Note**: FireWorks is an alternative to jobflow-remote. We recommend using jobflow-remote (Tutorial 14) for new projects due to its simpler configuration and modern CLI.

**Learning Objectives**:

* Submit workflows to HPC clusters using FireWorks
* Configure queue adapters for job schedulers
* Manage job dependencies
* Monitor workflow progress

**Key Concepts**:

* **FireWorks**: Workflow execution engine
* **Queue adapters**: Interface to job schedulers
* **Launch directories**: Organized calculation folders
* **Job dependencies**: Automatic workflow orchestration

**FireWorks Setup**:

**1. Install FireWorks**:

.. code-block:: bash

   pip install fireworks

**2. Configure LaunchPad** (``my_launchpad.yaml``):

.. code-block:: yaml

   host: localhost
   port: 27017
   name: fireworks
   username: null
   password: null
   authsource: null
   ssl_ca_certs: null

**3. Configure Queue Adapter** (``my_qadapter.yaml``):

For SLURM:

.. code-block:: yaml

   _fw_name: CommonAdapter
   _fw_q_type: SLURM
   rocket_launch: rlaunch -c /path/to/config singleshot
   nodes: 2
   ntasks_per_node: 24
   walltime: 24:00:00
   queue: normal
   account: myproject
   pre_rocket: |
       module load siesta
       export SIESTA_CMD="srun siesta"

For PBS:

.. code-block:: yaml

   _fw_name: CommonAdapter
   _fw_q_type: PBS
   rocket_launch: rlaunch -c /path/to/config singleshot
   nodes: 2
   ppnode: 24
   walltime: '24:00:00'
   queue: normal
   account: myproject

**Workflow Submission**:

.. code-block:: python

   from fireworks import LaunchPad
   from jobflow.managers.fireworks import flow_to_workflow

   # Load LaunchPad
   lpad = LaunchPad.from_file("my_launchpad.yaml")

   # Convert jobflow Flow to FireWorks Workflow
   wf = flow_to_workflow(flow)

   # Add to database
   lpad.add_wf(wf)

**Launching Jobs**:

.. code-block:: bash

   # Submit to queue
   qlaunch -c /path/to/config rapidfire -m 10

   # Or submit individual Fireworks
   qlaunch -c /path/to/config singleshot

**Monitoring**:

.. code-block:: bash

   # Check workflow status
   lpad get_wflows -d more

   # Check Firework status
   lpad get_fws -s READY

   # Rerun failed jobs
   lpad rerun_fws -s FIZZLED

**Advanced Configuration**:

**Resource Optimization**:

.. code-block:: yaml

   # Different resources for different job types
   small_jobs:
       nodes: 1
       ntasks_per_node: 4
       walltime: 2:00:00

   large_jobs:
       nodes: 4
       ntasks_per_node: 24
       walltime: 24:00:00

**Job Dependencies**:

FireWorks automatically handles dependencies based on Flow structure:

.. code-block:: python

   from jobflow import Flow

   # Create dependent jobs
   relax = RelaxMaker().make(structure)
   bands = BandStructureMaker().make(relax.output.structure)

   # FireWorks will ensure relax completes before bands starts
   flow = Flow([relax, bands])

**Best Practices**:

1. **Test locally first**: Verify workflow before HPC submission
2. **Monitor actively**: Check for failures early
3. **Use checkpoints**: Enable restarts for long calculations
4. **Organize outputs**: Use clear directory structure
5. **Document parameters**: Include in job metadata

📁 **Location**: ``tutorials/04-infrastructure/02-job-submission/`` (FireWorks alternative)

⏱️ **Time**: 60 minutes

⭐ **Difficulty**: Advanced

----

Tutorial 15: Custodian Error Handling ⭐ NEW
============================================

**Learning Objectives**:

* Enable automatic error detection and recovery
* Understand custodian library integration
* Configure custom error handlers
* Analyze correction history

**Key Concepts**:

* **Custodian**: MaterialsProject error handling library
* **Error handlers**: Automatic correction strategies
* **Validators**: Output quality checking
* **JSON logging**: Full correction history

**Architecture**:

atomate2siesta uses the battle-tested **MaterialsProject/custodian** library:

* 10+ error types automatically detected
* Progressive correction strategies
* Automatic retry with modified parameters
* Complete audit trail in ``custodian.json``

**Basic Usage**:

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker

   # Enable custodian (default handlers)
   maker = RelaxMaker.fixed_cell_relaxation(
       use_custodian=True,
       custodian_max_errors=5,
   )

   job = maker.make(structure)
   results = run_locally(job, create_folders=True)

   # Check custodian.json for correction history

**Error Types Detected**:

1. **SCF_NOT_CONV** - SCF convergence failure
2. **MEMORY** - Out of memory
3. **TIME_LIMIT** - Job timeout
4. **NUMERICAL** - NaN/Inf in calculations
5. **SINGULAR_OVERLAP** - Singular overlap matrix
6. **NEGATIVE_EIGENVALUES** - Negative overlap eigenvalues
7. **GEOMETRY_OPTIMIZATION** - Relaxation failure
8. **BASIS_GENERATION** - Basis set error
9. **GRID_INTEGRATION** - Real-space grid error
10. **FILE_IO** - File I/O error

**SCF Convergence Handler** (5-Level Strategy):

Automatically tries progressively more aggressive fixes:

.. list-table::
   :header-rows: 1
   :widths: 10 40 50

   * - Level
     - Changes
     - Rationale
   * - 1
     - ``Mixer.Weight = 0.05``, ``Mix.First = True``
     - Gentle reduction, start fresh
   * - 2
     - ``Mixer.Weight = 0.01``, ``History = 5``
     - More conservative mixing
   * - 3
     - ``Mixer.Weight = 0.005``, ``History = 8``, ``Kick = 40``
     - Very conservative + perturbation
   * - 4
     - Switch to **Pulay mixer**, ``History = 10``
     - Try different algorithm
   * - 5
     - Switch to **Broyden mixer**, ``Weight = 0.001``
     - Last resort: different algorithm

**Custom Handlers**:

.. code-block:: python

   from atomate2.siesta.custodian import (
       SCFConvergenceHandler,
       MemoryHandler,
       TimeHandler,
   )

   # More aggressive SCF recovery
   custom_handlers = [
       SCFConvergenceHandler(max_attempts=10),
       MemoryHandler(max_attempts=3),
       TimeHandler(max_attempts=2),
   ]

   maker = RelaxMaker.fixed_cell_relaxation(
       use_custodian=True,
       custodian_handlers=custom_handlers,
       custodian_max_errors=15,
   )

**Custodian Output** (``custodian.json``):

.. code-block:: json

   {
       "jobs": [
           {
               "job": "SiestaJob",
               "cmd": "siesta < siesta.fdf > siesta.out",
               "final": true
           }
       ],
       "corrections": [
           {
               "handler": "SCFConvergenceHandler",
               "level": 1,
               "errors": ["SCF did not converge in 100 SCF steps"],
               "actions": [
                   "Updated SCF.Mixer.Weight to 0.05",
                   "Set SCF.Mix.First to True"
               ]
           },
           {
               "handler": "SCFConvergenceHandler",
               "level": 2,
               "errors": ["SCF did not converge in 100 SCF steps"],
               "actions": [
                   "Updated SCF.Mixer.Weight to 0.01",
                   "Updated SCF.Mixer.History to 5"
               ]
           }
       ],
       "run_statistics": {
           "total_time": 3600.5,
           "wall_time": 3650.2,
           "errors": 2,
           "corrections": 2
       }
   }

**Validation**:

After successful completion, validators check output quality:

.. code-block:: python

   from atomate2.siesta.custodian.validators import (
       SiestaOutputValidator,
       RelaxationValidator,
   )

   # Validators check:
   # - SCF converged properly
   # - Forces/stresses within tolerance
   # - No warnings in output
   # - Files are complete

**Real Example** (SCF convergence issue):

.. code-block:: python

   # test-custodian-relax.py
   from atomate2.siesta.jobs.core import RelaxMaker

   # Intentionally difficult convergence (metallic surface)
   maker = RelaxMaker.fixed_cell_relaxation(
       use_custodian=True,
       user_params={
           "PAO.BasisSize": "DZP",
           "a2s_kpts": [6, 6, 1],  # 2D surface
           "Mesh.Cutoff": "300 Ry",
           # No occupation function → will fail initially
       }
   )

   job = maker.make(surface_structure)
   results = run_locally(job, create_folders=True)

   # Custodian will:
   # 1. Detect SCF failure
   # 2. Try level 1 correction (Mixer.Weight = 0.05)
   # 3. If still fails, try level 2 (Weight = 0.01, History = 5)
   # 4. Continue until success or max_attempts reached
   # 5. Log all corrections in custodian.json

**Benefits**:

✅ **Automatic**: No manual intervention for common errors

✅ **Transparent**: Full history in ``custodian.json``

✅ **Configurable**: Custom handlers and strategies

✅ **Production-tested**: Built on MaterialsProject's proven framework

✅ **Safety limits**: Respects max attempts to prevent infinite loops

**When to Use**:

* **Production calculations**: Always enable for robustness
* **High-throughput**: Essential for unattended workflows
* **Challenging systems**: Metals, surfaces, complex materials
* **HPC workflows**: Automatic recovery saves queue time

**When to Disable**:

* **Testing/debugging**: Want to see raw failures
* **Very simple systems**: Overhead not needed
* **Custom workflows**: Need specific error handling

📁 **Location**: ``tutorials/04-infrastructure/03-error-handling/``

⏱️ **Time**: 45 minutes

⭐ **Difficulty**: Intermediate

📄 **Documentation**:
   * ``README.md`` - Tutorial and examples
   * ``REFACTORING_SUMMARY.md`` - Complete custodian library integration guide (438 lines)
   * ``CUSTODIAN_STRATEGY.md`` - Implementation strategy

----

Production Workflow Examples
=============================

Option 1: jobflow-remote (Recommended)
---------------------------------------

Modern, simple setup with comprehensive CLI:

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker
   from atomate2.siesta.sets.tiers import apply_tier_preset
   from jobflow_remote import submit_flow
   from jobflow import Flow

   # 1. Create maker with error handling
   maker = RelaxMaker.variable_cell_relaxation(
       use_custodian=True,
       custodian_max_errors=10,
   )

   # 2. Apply tier preset
   maker = apply_tier_preset(maker, "high_accuracy_relax")

   # 3. Create workflow
   jobs = [maker.make(s) for s in structures]
   flow = Flow(jobs)

   # 4. Submit to HPC (MongoDB storage is automatic)
   job_id = submit_flow(flow, project="production", worker="slurm_worker")

   print(f"Submitted job: {job_id}")

.. code-block:: bash

   # Monitor from command line
   jf job list
   jf job info <job_id>
   jf job output <job_id>

Option 2: FireWorks (Alternative)
----------------------------------

Traditional workflow engine:

.. code-block:: python

   from fireworks import LaunchPad
   from jobflow.managers.fireworks import flow_to_workflow
   from maggma.stores import MongoStore
   from atomate2.siesta.jobs.core import RelaxMaker
   from atomate2.siesta.sets.tiers import apply_tier_preset

   # 1. Setup database
   store = MongoStore(
       database="production_database",
       collection_name="relaxations",
   )

   # 2. Create maker with error handling
   maker = RelaxMaker.variable_cell_relaxation(
       use_custodian=True,
       custodian_max_errors=10,
   )

   # 3. Apply tier preset
   maker = apply_tier_preset(maker, "high_accuracy_relax")

   # 4. Create workflow
   jobs = [maker.make(s) for s in structures]
   flow = Flow(jobs)

   # 5. Submit to HPC
   lpad = LaunchPad.from_file("my_launchpad.yaml")
   wf = flow_to_workflow(flow, store=store)
   lpad.add_wf(wf)

.. code-block:: bash

   # Launch on cluster
   qlaunch -c /path/to/config rapidfire

Both setups provide:

* ✅ Automatic error recovery
* ✅ HPC job scheduling
* ✅ Database storage
* ✅ Material-specific parameters
* ✅ Full provenance tracking

----

Next Steps
==========

After setting up infrastructure:

1. **Scale up**: Run high-throughput studies
2. **Analyze data**: Query database for trends
3. **Optimize**: Fine-tune queue settings
4. **Monitor**: Set up dashboards and alerts

See :doc:`index` for full tutorial listing.

----

.. tip::

   **Recommended Path for Production**:

   1. **Tutorial 13**: Set up MongoDB database storage
   2. **Tutorial 14**: Configure jobflow-remote for HPC submission (simpler than FireWorks)
   3. **Tutorial 15**: Enable custodian for automatic error recovery

   This combination provides a robust, production-ready infrastructure with minimal configuration.
