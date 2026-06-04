=========================================
Rerunning Failed Jobs in Jobflow-Remote
=========================================

Complete guide to handling failed jobs using ``jf`` CLI commands with project-specific execution.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

When a job fails in jobflow-remote, you can use the ``jf`` CLI to diagnose, modify resources, and rerun the job. All commands support the ``-p PROJECT_NAME`` flag for multi-project setups.

For complete jobflow-remote documentation, see: https://matgenix.github.io/jobflow-remote/

**NEW**: atomate2siesta now provides tools for modifying FDF parameters! See :ref:`fdf-modification-tools` below.

**Standard Jobflow-Remote Limitation**: The ``jf`` CLI **cannot** modify FDF parameters (like ``Mesh.Cutoff``, ``kpts``, ``SCF.Mixer.Weight``) directly. These are baked into the job definition. Use ``atomate2siesta-jobflow-remote job`` commands for FDF modifications.

Quick Reference
===============

.. code-block:: bash

   # Check job status
   jf -p PROJECTNAME job info <db_id>
   jf -p PROJECTNAME job info 70 --full

   # Modify resources (NOT FDF parameters!)
   jf -p PROJECTNAME job set resources <db_id> --nodes 2 --ntasks 64
   jf -p PROJECTNAME job set worker <db_id> --worker new_worker

   # Rerun the job
   jf -p PROJECTNAME job rerun <db_id>

   # For remote errors only
   jf -p PROJECTNAME job retry <db_id>

Understanding Job States
=========================

Job Lifecycle
-------------

.. code-block:: text

   READY → CHECKED_OUT → UPLOADED → SUBMITTED → RUNNING → DOWNLOADED → COMPLETED
     ↓
   FAILED (convergence, parameter issues)
     ↓
   REMOTE_ERROR (network, cluster issues)

**FAILED Jobs**
   Job completed but failed due to calculation issues (SCF convergence, wrong parameters, etc.)

   **Solution**: Use ``jf job rerun``

**REMOTE_ERROR Jobs**
   Job encountered infrastructure issues (network timeout, file transfer failure, cluster problems)

   **Solution**: Use ``jf job retry``

Check Job Status
----------------

.. code-block:: bash

   # List all jobs (default project)
   jf job list

   # List jobs for specific project
   jf -p cesga_production job list

   # Filter by state
   jf -p PROJECTNAME job list --state FAILED
   jf -p PROJECTNAME job list --state RUNNING
   jf -p PROJECTNAME job list --state COMPLETED

   # Show recent jobs
   jf -p PROJECTNAME job list --max-results 20

   # Get detailed job information
   jf -p PROJECTNAME job info 70
   jf -p PROJECTNAME job info 2bc5558c-2faa-4819-9a9c-12d2f13049eb

   # Full details including error messages
   jf -p PROJECTNAME job info 70 --full

   # View job output
   jf -p PROJECTNAME job output 70

Commands for Failed Jobs
=========================

1. jf job info - Diagnose the Failure
--------------------------------------

**Purpose**: Get detailed information about why a job failed.

**Basic Usage:**

.. code-block:: bash

   # Quick info
   jf -p PROJECTNAME job info <db_id>

   # Full details (includes error messages, stack traces)
   jf -p PROJECTNAME job info <db_id> --full

   # Save output for analysis
   jf -p PROJECTNAME job info 70 --full > job_70_error.txt

**Example Output:**

.. code-block:: text

   Job ID: 70
   UUID: 2bc5558c-2faa-4819-9a9c-12d2f13049eb
   Name: [2/39] Final-CuCSN-7l-100-N-Scan_adsorbate
   State: FAILED
   Worker: cesga_worker
   Created: 2025-11-20 18:49

   Error: SCF did not converge in 50 iterations
   Final DM.Tolerance: 2.3e-03 (target: 1.0e-04)

**Common Error Patterns:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Error Message
     - Meaning
   * - ``SCF did not converge``
     - Need different convergence parameters
   * - ``Exceeded memory limit``
     - Need more resources (use ``jf job set resources``)
   * - ``CANCELLED DUE TO TIME LIMIT``
     - Need more walltime (use ``jf job set resources``)
   * - ``kgrid: ERROR. Grid generates no points``
     - K-point issue (requires new flow)
   * - ``PAO.Basis: Unknown basis size``
     - Basis set error (requires new flow)

2. jf job set - Modify Job Resources
-------------------------------------

**Purpose**: Change computational resources before rerunning (memory, cores, walltime, worker).

**Important**: This **ONLY** modifies HPC resources, **NOT** FDF parameters!

set resources
~~~~~~~~~~~~~

Modify computational resources (nodes, cores, memory, walltime).

.. code-block:: bash

   # Basic syntax
   jf -p PROJECTNAME job set resources <db_id> [OPTIONS]

   # Examples:
   # Increase nodes and tasks
   jf -p cesga_production job set resources 70 --nodes 2 --ntasks 64

   # Increase memory per CPU
   jf -p cesga_production job set resources 70 --mem-per-cpu 4GB

   # Increase walltime
   jf -p cesga_production job set resources 70 --time 48:00:00

   # Change partition/queue
   jf -p cesga_production job set resources 70 --partition high_mem

   # Multiple modifications at once
   jf -p cesga_production job set resources 70 \
       --nodes 2 \
       --ntasks 64 \
       --mem-per-cpu 4GB \
       --time 48:00:00

**Available Options** (check ``jf job set resources -h`` for complete list):

* ``--nodes`` - Number of compute nodes
* ``--ntasks`` - Total number of tasks
* ``--mem-per-cpu`` - Memory per CPU
* ``--time`` - Walltime (format: ``HH:MM:SS`` or ``DD-HH:MM:SS``)
* ``--partition`` - Queue/partition name
* ``--qos`` - Quality of Service
* ``--account`` - Account/project for billing

set worker
~~~~~~~~~~

Change which worker executes the job.

.. code-block:: bash

   # Switch to different worker
   jf -p PROJECTNAME job set worker <db_id> --worker <worker_name>

   # Example: Move job to different cluster
   jf -p production job set worker 70 --worker mn5_worker

**When to Use:**

* Original worker is down or overloaded
* Job needs special hardware (GPUs, large memory nodes)
* Testing on different cluster

set exec-config
~~~~~~~~~~~~~~~

Modify execution configuration (advanced).

.. code-block:: bash

   # Set execution config
   jf -p PROJECTNAME job set exec-config <db_id> [OPTIONS]

**When to Use:**

* Change SLURM/PBS submission options
* Modify environment variables
* Adjust pre/post execution scripts

**Note**: Exact options depend on your jobflow-remote configuration. Check ``jf job set exec-config -h``.

3. jf job rerun - Restart Failed Job
-------------------------------------

**Purpose**: Restart a FAILED job from the beginning.

**Key Behavior:**

* Returns job to ``READY`` state
* **Regenerates all input files** from stored job definition
* **By default, deletes the worker directory** (use ``--no-delete`` to preserve)
* **Cannot modify FDF parameters** (they're in the job definition)

**Basic Usage:**

.. code-block:: bash

   # Rerun with fresh directory
   jf -p PROJECTNAME job rerun <db_id>

   # Rerun keeping old directory (for debugging)
   jf -p PROJECTNAME job rerun <db_id> --no-delete

   # Unlock and rerun
   jf -p PROJECTNAME job rerun <db_id> --break-lock

**Examples:**

.. code-block:: bash

   # Example 1: Simple rerun after resource increase
   jf -p cesga_production job set resources 70 --mem-per-cpu 4GB
   jf -p cesga_production job rerun 70

   # Example 2: Move to different worker and rerun
   jf -p cesga_production job set worker 70 --worker high_mem_worker
   jf -p cesga_production job rerun 70

   # Example 3: Keep old files for comparison
   jf -p cesga_production job rerun 70 --no-delete

**Batch Rerun:**

.. code-block:: bash

   # Rerun all failed jobs
   jf -p PROJECTNAME job rerun --state FAILED

   # Rerun failed jobs matching pattern
   jf -p PROJECTNAME job rerun --state FAILED --name "*adsorbate*"

4. jf job retry - Retry After Remote Error
-------------------------------------------

**Purpose**: Retry a job that encountered infrastructure issues (REMOTE_ERROR state).

**When to Use:**

* Network timeouts
* File transfer failures
* Temporary cluster issues
* SSH connection problems

**Basic Usage:**

.. code-block:: bash

   # Retry single job
   jf -p PROJECTNAME job retry <db_id>

   # Retry all remote errors
   jf -p PROJECTNAME job retry --state REMOTE_ERROR

**Example:**

.. code-block:: bash

   # Job failed due to network timeout
   jf -p production job info 145
   # State: REMOTE_ERROR
   # Error: Connection timeout during file upload

   # Retry (returns to previous state before error)
   jf -p production job retry 145

**Difference from rerun:**

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Command
     - When to Use
     - Behavior
   * - ``retry``
     - REMOTE_ERROR (infrastructure)
     - Returns to previous state
   * - ``rerun``
     - FAILED (calculation error)
     - Starts from READY

Workflow Examples
=================

Example 1: SCF Convergence Failure (Cannot Fix with CLI)
---------------------------------------------------------

**Scenario**: Job failed because SCF didn't converge. You need to change ``SCF.Mixer.Weight``.

.. code-block:: bash

   # Step 1: Check the error
   jf -p cesga_production job info 70 --full
   # Output: SCF did not converge in 50 iterations

**⚠️ LIMITATION**: You **cannot** change FDF parameters using ``jf`` commands!

**Solution**: Use ``atomate2siesta-jobflow-remote job modify-db`` to modify FDF parameters (see :ref:`fdf-modification-tools` below).

Example 2: Memory Limit Exceeded (Can Fix with CLI)
----------------------------------------------------

**Scenario**: Job cancelled because it exceeded memory allocation.

.. code-block:: bash

   # Step 1: Check the error
   jf -p cesga_production job info 70
   # Error: Exceeded step memory limit

   # Step 2: Increase memory
   jf -p cesga_production job set resources 70 --mem-per-cpu 4GB

   # Step 3: Rerun
   jf -p cesga_production job rerun 70

   # Step 4: Monitor
   jf -p cesga_production job info 70

Example 3: Walltime Exceeded (Can Fix with CLI)
------------------------------------------------

**Scenario**: Job cancelled due to time limit.

.. code-block:: bash

   # Step 1: Increase walltime
   jf -p cesga_production job set resources 70 --time 72:00:00

   # Step 2: Optionally move to longer queue
   jf -p cesga_production job set resources 70 --partition long

   # Step 3: Rerun
   jf -p cesga_production job rerun 70

Example 4: Network Failure (Can Fix with CLI)
----------------------------------------------

**Scenario**: Job encountered network timeout during file upload.

.. code-block:: bash

   # Step 1: Check state
   jf -p production job list --state REMOTE_ERROR
   # Shows job 145 in REMOTE_ERROR

   # Step 2: Retry (network is fixed)
   jf -p production job retry 145

   # Or retry all remote errors
   jf -p production job retry --state REMOTE_ERROR

Example 5: Wrong Worker (Can Fix with CLI)
-------------------------------------------

**Scenario**: Job submitted to worker without enough memory.

.. code-block:: bash

   # Step 1: Move to high-memory worker
   jf -p production job set worker 70 --worker high_mem_worker

   # Step 2: Rerun on new worker
   jf -p production job rerun 70

Example 6: Batch Fix Multiple Jobs
-----------------------------------

**Scenario**: Multiple jobs in a scan failed with memory issues.

.. code-block:: bash

   # Step 1: List all failed jobs
   jf -p cesga_production job list --state FAILED --name "*adsorbate*"

   # Step 2: Set resources for all matching jobs
   # (Note: May need to run for each job individually)
   for job_id in $(jf -p cesga_production job list --state FAILED --name "*adsorbate*" --output db_id); do
       jf -p cesga_production job set resources $job_id --mem-per-cpu 4GB
   done

   # Step 3: Rerun all failed jobs with pattern
   jf -p cesga_production job rerun --state FAILED --name "*adsorbate*"

Monitoring Jobs After Rerun
============================

Check Job Progress
------------------

.. code-block:: bash

   # Monitor running jobs
   watch -n 60 'jf -p PROJECTNAME job list --state RUNNING --max-results 10'

   # Check specific job status
   jf -p PROJECTNAME job info 70

   # View live output (if supported)
   jf -p PROJECTNAME job output 70

Runner Management
-----------------

Ensure the runner is active to process resubmitted jobs:

.. code-block:: bash

   # Check runner status
   jf -p PROJECTNAME runner status

   # Start runner if not running
   jf -p PROJECTNAME runner start

   # View runner logs
   jf -p PROJECTNAME runner logs

   # Restart runner (if needed)
   jf -p PROJECTNAME runner restart

Flow Management
---------------

.. code-block:: bash

   # List flows
   jf -p PROJECTNAME flow list

   # Get flow information
   jf -p PROJECTNAME flow info <flow_id>

   # Check all jobs in a flow
   jf -p PROJECTNAME job list --flow-id <flow_id>

   # Delete entire failed flow
   jf -p PROJECTNAME flow delete <flow_id>

What You CANNOT Do with CLI
============================

These require creating a new flow with Python:

❌ **Cannot Modify FDF Parameters:**

* ``Mesh.Cutoff``
* ``kpts`` or ``kgrid.Cutoff``
* ``SCF.Mixer.Weight``
* ``SCF.DM.Tolerance``
* ``PAO.BasisSize``
* ``ElectronicTemperature``
* Any SIESTA FDF parameter

**Why?** These are stored in the job's ``function_kwargs`` in the database. The ``jf`` CLI doesn't provide commands to modify function arguments.

❌ **Cannot Modify User Parameters:**

* ``user_params`` dictionary
* Maker-specific settings
* Flow-level parameters

**Solution:** Use ``atomate2siesta-jobflow-remote job modify-db`` to modify FDF parameters (see :ref:`fdf-modification-tools` below).

✅ **CAN Modify via CLI:**

* Computational resources (nodes, cores, memory, walltime)
* Worker assignment
* Execution configuration
* Job priority

Decision Tree
=============

Use this to decide your approach:

.. code-block:: text

   Job Failed
   │
   ├─ Error Type?
   │  │
   │  ├─ Memory/Time Limit Exceeded
   │  │  └─→ Use CLI: jf job set resources + jf job rerun ✅
   │  │
   │  ├─ Network/Infrastructure (REMOTE_ERROR)
   │  │  └─→ Use CLI: jf job retry ✅
   │  │
   │  ├─ Wrong Worker/Queue
   │  │  └─→ Use CLI: jf job set worker + jf job rerun ✅
   │  │
   │  └─ SCF/K-points/FDF Parameters
   │     └─→ Must create new flow with Python ❌ CLI can't help
   │
   └─ Multiple Jobs Failed?
      │
      ├─ Same resource issue
      │  └─→ Use CLI batch commands ✅
      │
      └─ Parameter issues
         └─→ Create new flow with corrected parameters ❌

Multi-Project Workflow
======================

If you manage multiple projects, always use ``-p`` flag:

.. code-block:: bash

   # List all projects
   jf project list

   # Set default project (optional)
   export JOBFLOW_REMOTE_PROJECT=cesga_production

   # Or always specify with -p
   jf -p cesga_production job list
   jf -p mn5_project job info 70
   jf -p local_testing job rerun 45

   # Useful aliases
   alias jfc='jf -p cesga_production'
   alias jfm='jf -p mn5_project'

   # Then use:
   jfc job list
   jfm job info 70

Best Practices
==============

1. Always Diagnose Before Acting
---------------------------------

.. code-block:: bash

   # Get full error details
   jf -p PROJECTNAME job info <db_id> --full > error.txt

   # Review carefully
   grep -i "error\|failed\|cancelled" error.txt

2. Test Resource Changes with One Job First
--------------------------------------------

.. code-block:: bash

   # Modify and rerun one job
   jf -p PROJECTNAME job set resources 70 --mem-per-cpu 4GB
   jf -p PROJECTNAME job rerun 70

   # Wait for completion
   jf -p PROJECTNAME job info 70

   # If successful, apply to others
   jf -p PROJECTNAME job set resources 71 72 73 --mem-per-cpu 4GB
   jf -p PROJECTNAME job rerun 71 72 73

3. Keep Failed Jobs for Reference
----------------------------------

.. code-block:: bash

   # Don't delete failed jobs immediately
   # Keep for comparison with rerun

   # After confirming rerun succeeded, clean up
   jf -p PROJECTNAME flow delete <old_flow_id>

4. Monitor Runner Health
------------------------

.. code-block:: bash

   # Check runner before mass rerun
   jf -p PROJECTNAME runner status

   # Ensure runner is processing jobs
   jf -p PROJECTNAME runner logs | tail -20

5. Use Descriptive Job Names
-----------------------------

When creating flows, use descriptive names:

.. code-block:: python

   # Good naming helps with CLI filtering
   from jobflow_remote import submit_flow

   submit_flow(
       flow,
       worker="cesga_worker",
       name="Cu_CNC_adsorption_scan_corrected_v2"  # Descriptive!
   )

Then filter easily:

.. code-block:: bash

   jf -p production job list --name "*corrected_v2*"

Troubleshooting
===============

"Command not found: jf"
-----------------------

**Cause**: jobflow-remote not installed or not in PATH.

**Solution**:

.. code-block:: bash

   # Install jobflow-remote
   pip install jobflow-remote

   # Verify installation
   jf --version

"Project not found: PROJECTNAME"
---------------------------------

**Cause**: Project not configured.

**Solution**:

.. code-block:: bash

   # List available projects
   jf project list

   # Generate project config
   jf project generate PROJECTNAME

   # Or check config file
   ls ~/.jfremote/

"Cannot modify job in state COMPLETED"
---------------------------------------

**Cause**: Job already finished.

**Solution**: Create new flow instead of trying to modify completed job.

"Job rerun does nothing"
------------------------

**Possible causes**:

1. **Runner not running**:

   .. code-block:: bash

      jf -p PROJECTNAME runner start

2. **Job still locked**:

   .. code-block:: bash

      jf -p PROJECTNAME job rerun <db_id> --break-lock

3. **Job has dependencies**:

   Check flow structure - parent jobs may need to rerun first.

Command Reference
=================

Complete ``jf job`` Command List
---------------------------------

.. code-block:: bash

   # Query commands
   jf -p PROJECTNAME job list [--state STATE] [--name PATTERN]
   jf -p PROJECTNAME job info <db_id> [--full]
   jf -p PROJECTNAME job output <db_id>

   # Modification commands
   jf -p PROJECTNAME job set resources <db_id> [OPTIONS]
   jf -p PROJECTNAME job set worker <db_id> --worker WORKER
   jf -p PROJECTNAME job set exec-config <db_id> [OPTIONS]
   jf -p PROJECTNAME job set priority <db_id> --priority N

   # Execution commands
   jf -p PROJECTNAME job rerun <db_id> [--no-delete] [--break-lock]
   jf -p PROJECTNAME job retry <db_id>

   # State management
   jf -p PROJECTNAME job pause <db_id>
   jf -p PROJECTNAME job play <db_id>

   # Cleanup
   jf -p PROJECTNAME job delete <db_id>

Job States Reference
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - State
     - Meaning
     - Action
   * - ``READY``
     - Waiting to be picked up
     - Wait for runner
   * - ``CHECKED_OUT``
     - Runner preparing files
     - Monitor
   * - ``UPLOADED``
     - Files uploaded to worker
     - Monitor
   * - ``SUBMITTED``
     - Submitted to scheduler
     - Monitor
   * - ``RUNNING``
     - Currently executing
     - Monitor
   * - ``DOWNLOADED``
     - Results retrieved
     - Almost done
   * - ``COMPLETED``
     - Successfully finished
     - Done ✅
   * - ``FAILED``
     - Calculation failed
     - ``rerun`` or new flow
   * - ``REMOTE_ERROR``
     - Infrastructure issue
     - ``retry``
   * - ``PAUSED``
     - Manually paused
     - ``play`` to resume

Related Documentation
=====================

* :doc:`cli-jobflow-remote` - Jobflow-remote CLI setup
* :doc:`troubleshooting` - General troubleshooting
* :doc:`advanced-workflows` - Complex workflow patterns

External Resources
==================

* `Jobflow-Remote Documentation <https://matgenix.github.io/jobflow-remote/>`_
* `Jobflow-Remote GitHub <https://github.com/Matgenix/jobflow-remote>`_
* `Dealing with Errors Guide <https://matgenix.github.io/jobflow-remote/user/errors.html>`_

Summary
=======

**Key Points:**

1. **Use** ``-p PROJECTNAME`` **for all commands** in multi-project setups

2. **CLI can modify**: Resources, worker, exec-config (infrastructure)

3. **CLI cannot modify**: FDF parameters, user_params (calculation settings)

4. **For FAILED jobs**: Use ``jf job rerun`` (after setting resources if needed)

5. **For REMOTE_ERROR jobs**: Use ``jf job retry``

6. **Always diagnose first**: ``jf -p PROJECTNAME job info <db_id> --full``

**Quick Command Template:**

.. code-block:: bash

   # Standard workflow for resource-limited failure
   jf -p PROJECTNAME job info <db_id> --full              # Diagnose
   jf -p PROJECTNAME job set resources <db_id> [OPTIONS]  # Fix resources
   jf -p PROJECTNAME job rerun <db_id>                    # Rerun
   jf -p PROJECTNAME job info <db_id>                     # Monitor

For FDF parameter changes, see :ref:`fdf-modification-tools` below.

.. _fdf-modification-tools:

FDF Parameter Modification Tools (NEW)
=======================================

atomate2siesta provides dedicated commands for modifying SIESTA FDF parameters in failed jobs. These commands work alongside jobflow-remote to enable parameter adjustments without manual MongoDB editing.

Two-Tier Approach
-------------------

1. **inspect** (Read-Only) - View job details and current FDF parameters
2. **modify-db** (Direct Modification) - Modify parameters in MongoDB with safety features

job inspect - View Job Parameters
----------------------------------

**Purpose**: Read-only inspection of job configuration and FDF parameters.

**Basic Usage:**

.. code-block:: bash

   # Basic job info
   atomate2siesta-jobflow-remote -p production job inspect 70

   # Include FDF parameters (requires pymongo)
   atomate2siesta-jobflow-remote -p production job inspect 70 --full

   # Show only FDF parameters
   atomate2siesta-jobflow-remote -p production job inspect 70 --fdf-only

**What It Shows**:

- Job name, state, worker, UUID
- SIESTA FDF input parameters (with ``--full`` or ``--fdf-only``)
- Complete job document structure

**Requirements**:

- ``pymongo`` package for full parameter inspection: ``pip install pymongo``

**Example Output:**

.. code-block:: text

   Inspecting job 70 in project 'production'

   ┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
   ┃ Property     ┃ Value                    ┃
   ┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
   │ Db Id        │ 70                       │
   │ Name         │ RelaxJob                 │
   │ State        │ FAILED                   │
   │ Worker       │ cesga_worker             │
   │ UUID         │ 2bc5558c-2faa-4819...    │
   └──────────────┴──────────────────────────┘

   ╔════════════════════════════════════════╗
   ║ SIESTA FDF Parameters                  ║
   ╚════════════════════════════════════════╝
   kpts: [4, 4, 4]
   Mesh.Cutoff: 300 Ry
   SCF.Mixer.Weight: 0.1
   MaxSCFIterations: 200

job modify-db - Direct Parameter Modification
----------------------------------------------

**Purpose**: Modify SIESTA FDF parameters directly in MongoDB database.

**⚠️ IMPORTANT ⚠️**

- Modifies existing job in database
- Automatic backup created in ``jobs_backup`` collection before modification
- Requires double confirmation before applying changes
- Parameter validation against 456 registered FDF parameters
- Use when you need to change FDF parameters without recreating the entire flow

**When to Use:**

- SCF convergence failures requiring different mixer settings
- K-point or mesh cutoff adjustments
- Changing electronic temperature or other FDF parameters
- The job is in READY or FAILED state

**Basic Usage:**

.. code-block:: bash

   # Modify single parameter (with confirmations)
   atomate2siesta-jobflow-remote -p prod job modify-db 70 --param "kpts=[6,6,1]"

   # Modify multiple parameters
   atomate2siesta-jobflow-remote -p prod job modify-db 70 \
       --param "kpts=[6,6,1]" \
       --param "Mesh.Cutoff=350 Ry" \
       --param "SCF.Mixer.Weight=0.01"

   # Preview changes without applying (dry-run)
   atomate2siesta-jobflow-remote -p prod job modify-db 70 \
       --param "kpts=[8,8,1]" --dry-run

   # Skip confirmations (use with caution!)
   atomate2siesta-jobflow-remote -p prod job modify-db 70 \
       --param "kpts=[8,8,1]" --force

   # Disable backup (not recommended)
   atomate2siesta-jobflow-remote -p prod job modify-db 70 \
       --param "kpts=[6,6,1]" --no-backup

**Parameter Format:**

- Simple values: ``"MaxSCFIterations=200"``
- Lists: ``"kpts=[6,6,6]"``
- Strings with units: ``"Mesh.Cutoff=350 Ry"``
- Booleans: ``"SaveHS=true"``
- Block parameters: ``"DM.InitSpin=['1 +2.0', '2 -2.0']"``
- Multiple: Use ``--param`` flag multiple times

**Modification Workflow:**

1. Command validates parameters against dataclass registry (456 registered parameters)
2. Shows preview of all changes
3. Requests first confirmation ("Do you understand the risks?")
4. Creates timestamped backup in ``jobs_backup`` collection
5. Shows final confirmation with exact parameter changes
6. Updates MongoDB document directly
7. You rerun with ``jf -p prod job rerun 70``

**Built-in Safety Features:**

- **Parameter validation**: Checks against dataclass registry
- **Typo detection**: Suggests similar parameters if typo detected
- **Internal param filtering**: Strips atomate2siesta-specific parameters
- **Double confirmation**: Two prompts before modification
- **Automatic backup**: Timestamped backup in MongoDB (can restore if needed)
- **Preview mode**: ``--dry-run`` flag to see changes without applying
- **Verification**: Shows exactly what will change before modification

**After Modification:**

1. Verify changes applied: ``atomate2siesta-jobflow-remote -p prod job inspect 70 --full``
2. Rerun the job: ``jf -p prod job rerun 70``
3. Monitor progress: ``jf -p prod job info 70``

Parameter Validation
--------------------

All modification commands validate parameters before applying:

**K-Points Validation:**

.. code-block:: bash

   # Valid
   -m "kpts=[4,4,4]"

   # Invalid - not 3 integers
   -m "kpts=[4,4]"  # Error: kpts must be [k1,k2,k3]

**Mesh Cutoff Validation:**

.. code-block:: bash

   # Valid - includes units
   -m "Mesh.Cutoff=300 Ry"

   # Invalid - missing units
   -m "Mesh.Cutoff=300"  # Error: include units (Ry/eV/Ha)

**Common Typo Detection:**

.. code-block:: bash

   # Detected typo
   -m "MeshCutoff=300 Ry"  # Warning: Did you mean "Mesh.Cutoff"?

   # Correct
   -m "Mesh.Cutoff=300 Ry"

Complete Workflow Examples
---------------------------

**Scenario 1: SCF Convergence Failure (Too Coarse K-Points)**

.. code-block:: bash

   # 1. Inspect failed job
   atomate2siesta-jobflow-remote -p prod job inspect 70 --full

   # 2. See current kpts = [2,2,2], need higher density

   # 3. Modify with finer k-points
   atomate2siesta-jobflow-remote -p prod job modify-db 70 \\
       --param "kpts=[6,6,6]" \\
       --param "SCF.Mixer.Weight=0.05"

   # 4. Verify modifications
   atomate2siesta-jobflow-remote -p prod job inspect 70 --full

   # 5. Rerun the job
   jf -p prod job rerun 70

**Scenario 2: Mixer Weight Too Aggressive**

.. code-block:: bash

   # 1. Inspect and see SCF.Mixer.Weight = 0.3 (too high)
   atomate2siesta-jobflow-remote -p prod job inspect 70 --full

   # 2. Reduce mixer weight
   atomate2siesta-jobflow-remote -p prod job modify-db 70 \\
       --param "SCF.Mixer.Weight=0.01" \\
       --param "MaxSCFIterations=300"

   # 3. Rerun the job
   jf -p prod job rerun 70

**Scenario 3: Emergency Database Fix (Use modify-db)**

.. code-block:: bash

   # 1. Inspect job
   atomate2siesta-jobflow-remote -p prod job inspect 70 --full

   # 2. Preview changes
   atomate2siesta-jobflow-remote -p prod job modify-db 70 \\
       --param "kpts=[6,6,1]" \\
       --dry-run

   # 3. Apply changes (with confirmations)
   atomate2siesta-jobflow-remote -p prod job modify-db 70 \\
       --param "kpts=[6,6,1]"

   # 4. Verify modification
   atomate2siesta-jobflow-remote -p prod job inspect 70 --full

   # 5. Rerun with jf
   jf -p prod job rerun 70

Requirements
------------

**Basic Commands** (inspect):

- atomate2siesta package
- jobflow-remote configured
- Project configuration in ``~/.jfremote/``

**Full Functionality** (inspect --full, modify-db):

.. code-block:: bash

   pip install pymongo

**For Direct Modifications**:

- MongoDB access credentials in project config
- Understanding of jobflow-remote internals
- Database backups (highly recommended)

Best Practices
--------------

1. **Always use inspect first**: Understand current parameters before modifying

2. **Use dry-run mode**: Preview changes with ``--dry-run`` before applying

3. **Keep backups enabled**: Database backups created automatically (default)

4. **Test modifications**: Try on single job before batch operations

5. **Verify after modification**: Use ``inspect --full`` to confirm changes

6. **Monitor after rerun**: Check job completes successfully with new parameters

7. **Avoid force mode**: Only use ``--force`` when automation is required

8. **Document changes**: Keep notes on what parameters were changed and why

FDF Modification Summary
------------------------

**Complete Workflow for Modifying FDF Parameters:**

.. code-block:: bash

   # 1. Inspect current parameters
   atomate2siesta-jobflow-remote -p prod job inspect 70 --full

   # 2. Preview changes (optional but recommended)
   atomate2siesta-jobflow-remote -p prod job modify-db 70 \
       --param "kpts=[6,6,1]" --dry-run

   # 3. Apply modifications (with safety features)
   atomate2siesta-jobflow-remote -p prod job modify-db 70 \
       --param "kpts=[6,6,1]"

   # 4. Verify changes were applied
   atomate2siesta-jobflow-remote -p prod job inspect 70 --full

   # 5. Rerun with jobflow-remote
   jf -p prod job rerun 70

   # 6. Monitor execution
   jf -p prod job info 70

**Built-in Safety Features:**

- ✅ Double confirmation prompts (can bypass with ``--force``)
- ✅ Automatic backups in ``jobs_backup`` collection
- ✅ Parameter validation against 456 registered FDF parameters
- ✅ Typo detection with smart suggestions
- ✅ Internal parameter filtering (prevents deserialization errors)
- ✅ Dry-run mode for preview without changes
- ✅ Full verification with ``inspect --full``

**Key Points:**

- Always start with ``job inspect --full`` to see current parameters
- Use ``--dry-run`` to preview changes before applying
- Automatic backup created before modification (restore if needed)
- Supports 456 validated SIESTA FDF parameters
- Works for READY and FAILED jobs only
