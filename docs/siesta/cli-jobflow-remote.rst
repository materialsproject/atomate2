=======================================
Jobflow Remote CLI Reference
=======================================

Comprehensive guide to managing jobflow-remote configurations with ``atomate2siesta-jobflow-remote``.

Overview
========

The ``atomate2siesta-jobflow-remote`` CLI provides comprehensive tools for managing jobflow-remote configurations. It simplifies:

* Installing and setting up jobflow-remote
* Managing multiple project configurations
* Updating MongoDB connection settings
* Adding descriptive comments to configuration files
* Viewing project details and worker information

For complete jobflow-remote documentation, see: https://matgenix.github.io/jobflow-remote/

Key Features
------------

* **Multi-project support**: Manage multiple jobflow-remote projects
* **Safe updates**: Automatic backups before modifications
* **Self-documenting configs**: Add inline comments to YAML files
* **Project discovery**: List and inspect all configured projects
* **Selective updates**: Update specific settings without touching others

Installation
============

The CLI is included with atomate2siesta:

.. code-block:: bash

   pip install atomate2siesta

Verify installation:

.. code-block:: bash

   atomate2siesta-jobflow-remote --help

Available Commands
==================

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Description
   * - ``install``
     - Install jobflow-remote package
   * - ``setup``
     - Generate or update project configuration
   * - ``update``
     - Update existing project configuration
   * - ``info``
     - Show project information and list all projects
   * - ``test``
     - Submit test job to verify setup
   * - ``runner``
     - Display runner management commands

Command Details
===============

1. Install Command
------------------

Install jobflow-remote package (stable or development version).

**Basic Usage:**

.. code-block:: bash

   # Install stable version from PyPI
   atomate2siesta-jobflow-remote install

   # Install development version from GitHub
   atomate2siesta-jobflow-remote install --dev

**Options:**

* ``--dev``: Install latest development version from GitHub

**Example Output:**

.. code-block:: text

   Installing jobflow-remote...
   Installing stable version from PyPI

   Installing to: /path/to/site-packages

   ✓ Installation successful!
   Location: /path/to/site-packages/jobflow_remote/

   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
   ┃               Installation Complete                ┃
   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

   Next Steps:
   1. Check installation: jf --version
   2. Generate project configuration
   3. Check configuration
   4. Initialize database

----

2. Setup Command
----------------

Generate new project configuration or update existing ones.

**Basic Usage:**

.. code-block:: bash

   # Generate default configuration
   atomate2siesta-jobflow-remote setup

   # Custom project name
   atomate2siesta-jobflow-remote setup --project-name my_project

   # Custom MongoDB settings
   atomate2siesta-jobflow-remote setup --database mydb --host localhost --port 27017

   # Update existing configuration
   atomate2siesta-jobflow-remote setup --update --host newhost --port 27018

   # Update without backup
   atomate2siesta-jobflow-remote setup --update --no-backup --database mydb

**Options:**

* ``--project-name TEXT``: Project name (default: "atomate2siesta")
* ``--worker-name TEXT``: Worker name (default: "local_shell")
* ``--database TEXT``: MongoDB database name (default: "atomate2siesta")
* ``--host TEXT``: MongoDB host (default: "localhost")
* ``--port INTEGER``: MongoDB port (default: 27017)
* ``--update``: Update existing config instead of generating new one
* ``--backup/--no-backup``: Create backup before updating (default: True)

**What It Does:**

**New Configuration Mode** (without ``--update``):

1. Runs ``jf project generate <project_name>``
2. Creates ``~/.jfremote/<project_name>.yaml``
3. Displays configuration instructions

**Update Mode** (with ``--update``):

1. Creates timestamped backup of existing config
2. Updates MongoDB settings in:

   * ``queue.store``
   * ``jobstore.docs_store``
   * ``jobstore.additional_stores`` (if present)

3. Preserves all other settings (workers, etc.)

**Example: Generate New Project**

.. code-block:: bash

   $ atomate2siesta-jobflow-remote setup --project-name hpc_project

   Generating jobflow-remote configuration...
   Generating project: hpc_project
   Running: jf project generate hpc_project

   ✓ Project generated successfully!

   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
   ┃            Configuration Generated                 ┃
   ┃                                                    ┃
   ┃ Configuration File: ~/.jfremote/hpc_project.yaml  ┃
   ┃                                                    ┃
   ┃ Key Settings to Configure:                        ┃
   ┃                                                    ┃
   ┃ 1. workers:                                        ┃
   ┃    - Name: local_shell                             ┃
   ┃    - Type: local (for testing) or remote (for HPC) ┃
   ┃    - Scheduler: shell, slurm, pbs, etc.            ┃
   ┃                                                    ┃
   ┃ 2. queue.store:                                    ┃
   ┃    - Database: atomate2siesta                      ┃
   ┃    - Host: localhost                               ┃
   ┃    - Port: 27017                                   ┃
   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

**Example: Update Existing Configuration**

.. code-block:: bash

   $ atomate2siesta-jobflow-remote setup --update --host db.server.com --port 27018

   Updating jobflow-remote configuration...

   ✓ Backup created: ~/.jfremote/atomate2siesta.backup_20251016_143022.yaml

   ✓ Configuration updated successfully!

   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
   ┃                 Update Complete                    ┃
   ┃                                                    ┃
   ┃ Updated Configuration:                             ┃
   ┃ ~/.jfremote/atomate2siesta.yaml                    ┃
   ┃                                                    ┃
   ┃ MongoDB Settings:                                  ┃
   ┃   • Database: atomate2siesta                       ┃
   ┃   • Host: db.server.com                            ┃
   ┃   • Port: 27018                                    ┃
   ┃                                                    ┃
   ┃ Updated Sections:                                  ┃
   ┃   • queue.store                                    ┃
   ┃   • jobstore.docs_store                            ┃
   ┃   • jobstore.additional_stores (if present)        ┃
   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

----

3. Update Command
-----------------

Dedicated command for updating existing configurations with advanced features.

**Basic Usage:**

.. code-block:: bash

   # Update MongoDB settings
   atomate2siesta-jobflow-remote update --database mydb --host localhost --port 27017

   # Add descriptive comments to ALL configuration entries
   atomate2siesta-jobflow-remote update --add-comments

   # Update settings AND add comments
   atomate2siesta-jobflow-remote update --database mydb --add-comments

   # Update specific project
   atomate2siesta-jobflow-remote update --project-name siesta --database newdb

   # Update without backup (not recommended)
   atomate2siesta-jobflow-remote update --no-backup --database mydb

**Options:**

* ``--project-name TEXT``: Project to update (default: "atomate2siesta")
* ``--database TEXT``: MongoDB database name to update
* ``--host TEXT``: MongoDB host to update
* ``--port INTEGER``: MongoDB port to update
* ``--add-comments``: Add descriptive comments to all config entries
* ``--backup/--no-backup``: Create backup before updating (default: True)

**What It Does:**

1. Creates timestamped backup (unless ``--no-backup``)
2. Loads existing configuration
3. Updates specified MongoDB settings (if provided)
4. Optionally rewrites entire file with inline comments
5. Preserves all other configuration (workers, etc.)

**Example: Add Comments to Configuration**

.. code-block:: bash

   $ atomate2siesta-jobflow-remote update --add-comments

   Updating jobflow-remote configuration...

   ✓ Backup created: ~/.jfremote/atomate2siesta.backup_20251016_143045.yaml

   ✓ Configuration updated successfully!

   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
   ┃                 Update Complete                    ┃
   ┃                                                    ┃
   ┃ Updated Configuration:                             ┃
   ┃ ~/.jfremote/atomate2siesta.yaml                    ┃
   ┃                                                    ┃
   ┃ ✓ Descriptive comments added to all configuration  ┃
   ┃   entries                                          ┃
   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

**Before adding comments:**

.. code-block:: yaml

   name: atomate2siesta
   workers:
     example_worker:
       type: remote
       scheduler_type: slurm
       work_dir: /path/to/run/folder
       timeout_execute: 60
   queue:
     store:
       type: MongoStore
       host: localhost
       database: db_name

**After adding comments:**

.. code-block:: yaml

   # Jobflow Remote Configuration File
   # Generated by atomate2siesta-jobflow-remote
   # Documentation: https://matgenix.github.io/jobflow-remote/

   name: atomate2siesta  # Project name for jobflow-remote

   # Worker configurations for job execution
   workers:
     example_worker:
       type: remote  # Worker type: 'local' for testing, 'remote' for HPC
       scheduler_type: slurm  # Scheduler: 'shell', 'slurm', 'pbs', 'sge', 'lsf'
       work_dir: /path/to/run/folder  # Directory where jobs will run on the worker
       timeout_execute: 60  # Timeout in seconds for job execution

   # Queue store configuration (MongoDB)
   queue:

     # MongoDB connection settings for job queue
     store:
       type: MongoStore  # Store type (typically MongoStore)
       host: localhost  # MongoDB server hostname
       database: db_name  # Database name for job queue

----

4. Info Command
---------------

Display jobflow-remote installation status, list all projects, and show detailed project information.

**Basic Usage:**

.. code-block:: bash

   # Show general information and list all projects
   atomate2siesta-jobflow-remote info

   # Show detailed information for specific project
   atomate2siesta-jobflow-remote info --project-name siesta

**Options:**

* ``--project-name TEXT``: Show details for a specific project

**Example: List All Projects**

.. code-block:: text

   $ atomate2siesta-jobflow-remote info

   ╭──────────────────────────────────────────────────────╮
   │            Jobflow Remote Setup Helper               │
   ╰──────────────────────────────────────────────────────╯

   Installation Status:

     ✓ jobflow-remote: 0.1.8

   Available Projects:

    Project Name    Config File                  Workers
    test            ~/.jfremote/test.yaml        2 (example_worker,
                                                 example_local)
    atomate2siesta  ~/.jfremote/atomate2...yaml  1 (example_worker)
    siesta          ~/.jfremote/siesta.yaml      3 (agustina_worker,
                                                 mn5_worker,
                                                 macbook_worker)
    production      ~/.jfremote/production.yaml  2 (hpc_worker,
                                                 local_worker)

   Key Features:
     • Remote Submission - Submit jobs to HPC clusters
     • Queue Management - Automatic job queue handling
     • Worker Support - Multiple worker configurations
     • MongoDB Backend - Persistent job storage

   Quick Commands:
     install - Install jobflow-remote
     setup - Generate project configuration
     update - Update existing configuration
     test - Submit test job
     info - Show this information

----

5. Test Command
---------------

Submit a test job to verify jobflow-remote setup.

**Basic Usage:**

.. code-block:: bash

   atomate2siesta-jobflow-remote test

**What It Does:**

1. Creates simple test flow: ``add(1, 2) + 3``
2. Submits to jobflow-remote
3. Returns job ID
4. Displays commands to check job status

**Example:**

.. code-block:: text

   $ atomate2siesta-jobflow-remote test

   Submitting test job...
   Creating test flow: add(1, 2) + 3

   ✓ Test job submitted successfully!

   Job ID: 1

   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
   ┃              Check Job Status                      ┃
   ┃                                                    ┃
   ┃ 1. List all jobs:                                  ┃
   ┃    jf job list                                     ┃
   ┃                                                    ┃
   ┃ 2. Start runner (if not running):                  ┃
   ┃    jf runner start                                 ┃
   ┃                                                    ┃
   ┃ 3. Check runner status:                            ┃
   ┃    jf runner status                                ┃
   ┃                                                    ┃
   ┃ 4. Get job output:                                 ┃
   ┃    jf job output 1                                 ┃
   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

----

6. Runner Command
-----------------

Display runner management and job control commands.

**Basic Usage:**

.. code-block:: bash

   atomate2siesta-jobflow-remote runner

**What It Shows:**

* Runner daemon commands (start, stop, status, restart, logs)
* Job management commands (list, info, output, cancel, retry)
* Useful tips for common operations

----

Job Management Commands
=======================

The CLI provides powerful tools for inspecting and modifying jobflow-remote jobs,
including SIESTA FDF parameter modification.

7. Job Inspect Command
----------------------

Inspect job details and SIESTA FDF parameters stored in the jobflow-remote database.

**Basic Usage:**

.. code-block:: bash

   # Basic job information
   atomate2siesta-jobflow-remote -p production job inspect 70

   # Include FDF parameters
   atomate2siesta-jobflow-remote -p production job inspect 70 --full

   # Show only FDF parameters
   atomate2siesta-jobflow-remote -p production job inspect 70 --fdf-only

   # Show tier preset defaults
   atomate2siesta-jobflow-remote -p production job inspect 70 --show-all-defaults

   # Show actual generated siesta.fdf file
   atomate2siesta-jobflow-remote -p production job inspect 70 --show-actual-fdf

**Options:**

* ``--full``: Show full job details including FDF parameters
* ``--fdf-only``: Show only FDF parameters
* ``--show-all-defaults``: Show what tier preset contributes (includes default parameters)
* ``--show-actual-fdf``: Show the actual generated siesta.fdf file from job run directory

**What It Shows:**

1. **Basic job information** (always shown):
   - Job name, index, UUID
   - State (READY, RUNNING, COMPLETED, FAILED)
   - Worker assignment
   - Run directory path

2. **FDF parameters** (with ``--full`` or ``--fdf-only``):
   - User-configurable parameters from job definition
   - These are the parameters you can modify with ``job modify-db``
   - Does NOT include SIESTA defaults or auto-generated blocks

3. **Tier defaults** (with ``--show-all-defaults``):
   - Parameters contributed by tier preset (basic, intermediate, advanced, expert)
   - Shows what the tier preset adds beyond user parameters
   - Helps understand complete parameter set

4. **Actual FDF file** (with ``--show-actual-fdf``):
   - Complete generated siesta.fdf file from job run directory
   - Includes all parameters: user params + tier defaults + SIESTA defaults
   - Retrieved via multiple methods (local file, SSH, jobflow-remote download)

**Example Output:**

.. code-block:: text

   $ atomate2siesta-jobflow-remote -p alberto job inspect 70 --full

   Inspecting job 70 in project 'alberto'

   ┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
   ║ Property               ║ Value                                  ║
   ┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
   │ Db Id                  │ 70                                     │
   │ Uuid                   │ 2bc5558c-2faa-4819-9a9c-12d2f13049eb   │
   │ Index                  │ 1                                      │
   │ Name                   │ [2/39] Final-CuCSN-7l-100-N-Scan_...   │
   │ State                  │ READY                                  │
   │ Worker                 │ alberto                                │
   └────────────────────────┴────────────────────────────────────────┘

   ╭──────────────────────── SIESTA FDF Parameters ────────────────────────╮
   │ PAO.BasisSize: DZP                                                    │
   │ Mesh.Cutoff: 300 Ry                                                   │
   │ OccupationFunction: FD                                                │
   │ ElectronicTemperature: 25 K                                           │
   │ SCF.Mixer.Weight: 0.1                                                 │
   │ SCF.DM.Tolerance: 1e-05                                               │
   │ Spin: polarized                                                       │
   ╰───────────────────────────────────────────────────────────────────────╯

   ╭──────────────── ℹ️  Parameter Display Explanation ────────────────────╮
   │ The parameters shown above are user-configurable parameters from      │
   │ the job definition. The actual generated siesta.fdf file contains     │
   │ many more parameters:                                                 │
   │                                                                       │
   │   • SIESTA defaults (marked with '# SIESTA DEFAULT VALUE')            │
   │   • Tier preset contributions (from dataclass modules)                │
   │   • Auto-generated blocks (k-points, structure, etc.)                 │
   │                                                                       │
   │ Options to see more:                                                  │
   │   • --show-all-defaults - Show what tier preset contributes           │
   │   • --show-actual-fdf - Show the actual generated FDF file            │
   ╰───────────────────────────────────────────────────────────────────────╯

   Next steps:
     • Modify parameters: job modify-db (modifies in place)
     • View full params: job inspect --full
     • View tier defaults: job inspect --show-all-defaults
     • View actual FDF: job inspect --show-actual-fdf

----

8. Job Modify-DB Command
-------------------------

Modify job FDF parameters directly in the MongoDB database (RISKY method).

.. warning::

   **⚠️  DANGER - DATABASE MODIFICATION ⚠️**

   This command modifies jobs DIRECTLY in the MongoDB database!
   This can cause:

   * Job execution failures
   * Data corruption
   * Loss of reproducibility
   * Broken workflow tracking

   Use ONLY when you understand the risks and have database backups.

**Basic Usage:**

.. code-block:: bash

   # Modify single parameter with confirmation
   atomate2siesta-jobflow-remote -p alberto job modify-db 70 \
       --param "Spin=polarized"

   # Modify multiple parameters
   atomate2siesta-jobflow-remote -p alberto job modify-db 70 \
       --param "Spin=polarized" \
       --param "Mesh.Cutoff=400 Ry"

   # Skip confirmations (dangerous!)
   atomate2siesta-jobflow-remote -p alberto job modify-db 70 \
       --param "Spin=polarized" --force

   # Disable backup (not recommended)
   atomate2siesta-jobflow-remote -p alberto job modify-db 70 \
       --param "Spin=polarized" --no-backup

**Options:**

* ``-p, --param TEXT``: Parameter to modify (format: ``key=value``). Can be used multiple times.
* ``--force``: Skip safety confirmation prompts (USE WITH EXTREME CAUTION)
* ``--backup / --no-backup``: Create backup before modification (default: True)

**Parameter Syntax:**

**Single-line parameters:**

.. code-block:: bash

   --param "Spin=polarized"
   --param "Mesh.Cutoff=400 Ry"
   --param "PAO.BasisSize=DZP"
   --param "SCF.Mixer.Weight=0.1"

**Block parameters (use Python list syntax):**

.. code-block:: bash

   --param "DM.InitSpin=['1 +2.0', '2 -2.0', '3 +2.0']"
   --param "Geometry.Constraints=['position from 1 to 10']"
   --param "kgrid_Monkhorst_Pack=['4 0 0 0.5', '0 4 0 0.5', '0 0 1 0.0']"

**Numeric and boolean values:**

.. code-block:: bash

   --param "kpts=[4,4,4]"              # List of integers
   --param "SCF.Mixer.Weight=0.005"    # Float
   --param "WriteCoorStep=true"        # Boolean
   --param "MaxSCFIterations=100"      # Integer

**Case Sensitivity:**

Parameter names are **case-insensitive** for validation, but you should use
standard SIESTA capitalization for clarity:

.. code-block:: bash

   # All equivalent and accepted:
   --param "Mesh.Cutoff=400 Ry"  # ✓ Standard (recommended)
   --param "mesh.cutoff=400 Ry"  # ✓ Lowercase
   --param "MESH.CUTOFF=400 Ry"  # ✓ Uppercase

   # Must use dot notation:
   --param "MeshCutoff=400 Ry"   # ✗ Rejected (missing dot)
   # Error: Did you mean: mesh.cutoff?

**What Gets Modified:**

The command updates the MongoDB document at:

.. code-block:: text

   job_doc['job']['function']['@bound']['input_set_generator']['user_params']

**Internal parameters are automatically filtered out:**

* ``tier``, ``xc``, ``mesh_cutoff``, ``kpts``, ``kgrid_cutoff`` (atomate2siesta internal)
* ``a2s_*``, ``atomate2siesta_*`` (prefixed internal parameters)
* ``fdf_arguments`` (handled separately)

These are NOT modified to prevent deserialization errors.

**Parameter Validation:**

All parameters are validated against the **dataclass registry** (456 registered parameters):

.. code-block:: bash

   # Recognized parameters (456 total from 28 dataclass modules):
   --param "Spin=polarized"           # ✓ Valid (SpinSettings)
   --param "Mesh.Cutoff=400 Ry"       # ✓ Valid (RealSpaceGridParameters)
   --param "PAO.BasisSize=DZP"        # ✓ Valid (BasisSetsAndProjectors)

   # Typos detected with suggestions:
   --param "MeshCutoff=400"           # ✗ Invalid
   # Error: Did you mean: mesh.cutoff?

   # Unregistered parameters (warning, but allowed):
   --param "MyCustomParam=value"      # ⚠️  Warning
   # SIESTA will validate at runtime

**Example Workflow:**

.. code-block:: bash

   # Step 1: Inspect current parameters
   atomate2siesta-jobflow-remote -p alberto job inspect 70 --full

   # Step 2: Modify parameters
   atomate2siesta-jobflow-remote -p alberto job modify-db 70 \
       --param "Spin=polarized" \
       --param "Mesh.Cutoff=400 Ry"

   # Output shows:
   # - Parameter validation results
   # - Preview of changes
   # - Confirmation prompts
   # - Backup creation
   # - Update status

   # Step 3: Verify modification
   atomate2siesta-jobflow-remote -p alberto job inspect 70 --full

   # Step 4: Rerun the job
   jf -p alberto job rerun 70

   # Step 5: Monitor execution
   jf -p alberto job info 70

**Example Output:**

.. code-block:: text

   $ atomate2siesta-jobflow-remote -p alberto job modify-db 70 \
       --param "Spin=polarized"

   ╭────────────────────── ⚠️  WARNING ──────────────────────╮
   │ ⚠️  DANGER - DATABASE MODIFICATION ⚠️                    │
   │                                                         │
   │ You are about to DIRECTLY modify a job in the MongoDB   │
   │ database. This can cause:                               │
   │   • Job execution failures                              │
   │   • Data corruption                                     │
   │   • Loss of reproducibility                             │
   │   • Broken workflow tracking                            │
   ╰─────────────────────────────────────────────────────────╯

   Do you understand the risks and wish to proceed? [y/n] (n): y

   Parsing parameter modifications...
   Validating parameters...
   ✓ All parameters valid

   Fetching job details from database...
   ✓ Found 7 original parameters

   New parameter: Spin = polarized

   ┏━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┓
   ┃ Parameter ┃ Original ┃ New       ┃ Change Type ┃
   ┡━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━┩
   │ Spin      │          │ polarized │ ADDED       │
   └───────────┴──────────┴───────────┴─────────────┘

   ╭────────────────── Final confirmation required ──────────────────╮
   │ You are about to modify job 70 in project 'alberto'.           │
   │ This will PERMANENTLY change the job parameters in database.    │
   │                                                                 │
   │ Backup will be created: True                                    │
   ╰─────────────────────────────────────────────────────────────────╯

   Proceed with database modification? [y/n] (n): y

   Modifying database...
   ✓ Backup created in collection 'jobs_backup'
   ✓ Modified 1 document(s)

   ╭─────────────────────── Success ───────────────────────╮
   │ ✓ Database modified successfully                      │
   │                                                       │
   │ Job 70 parameters have been updated.                  │
   │                                                       │
   │ Next steps:                                           │
   │   • Verify parameters: job inspect 70 --full          │
   │   • Rerun job: jf -p alberto job rerun 70             │
   │   • Monitor execution: jf -p alberto job info 70      │
   ╰───────────────────────────────────────────────────────╯

**Safety Features:**

1. **Double confirmation**: Two separate confirmation prompts
2. **Automatic backup**: Creates timestamped backup in ``jobs_backup`` collection
3. **Preview changes**: Shows exactly what will be modified before applying
4. **Parameter validation**: Validates against 456 registered FDF parameters
5. **Internal param filtering**: Prevents corruption by filtering internal parameters
6. **Error detection**: Catches typos and suggests corrections

**Backup Management:**

Backups are stored in MongoDB collection ``jobs_backup``:

.. code-block:: bash

   # View backups (requires MongoDB access)
   mongosh --eval "use atomate2siesta; db.jobs_backup.find({db_id: '70'}).pretty()"

   # Restore from backup (if needed)
   # Contact your database administrator or use MongoDB tools

**When to Use:**

✅ **Safe scenarios:**

* Job is in READY state (not running)
* You have database backups
* You need to fix a parameter typo
* You want to adjust convergence parameters
* You understand MongoDB and jobflow-remote

✗ **Unsafe scenarios:**

* Job is RUNNING or COMPLETED
* No database backups
* You don't understand the parameter system
* You're not sure what you're doing

**Troubleshooting:**

**Error: Unknown FDF parameter(s): mesh_cutoff, tier, xc**

This error occurred in earlier versions when internal parameters weren't filtered.
The current version automatically filters these out.

**Solution:** Update to latest version.

**Error: No documents modified**

The query couldn't find the job in the database.

**Solution:**

.. code-block:: bash

   # Check job exists
   jf -p alberto job info 70

   # Verify project name
   atomate2siesta-jobflow-remote -p alberto job inspect 70

**Error: Duplicate key error in jobs_backup**

The backup collection already has this job (from previous modification).

**Solution:** This is now automatically handled by removing ``_id`` and adding timestamps.

Use Case 9: Inspecting Failed Jobs
-----------------------------------

**Scenario:** A job failed and you need to understand why and fix parameters.

.. code-block:: bash

   # Step 1: Check job status
   jf -p alberto job info 70

   # Step 2: Inspect FDF parameters
   atomate2siesta-jobflow-remote -p alberto job inspect 70 --full

   # Step 3: View actual FDF file to see what was generated
   atomate2siesta-jobflow-remote -p alberto job inspect 70 --show-actual-fdf

   # Step 4: Modify problematic parameters
   atomate2siesta-jobflow-remote -p alberto job modify-db 70 \
       --param "SCF.Mixer.Weight=0.01" \
       --param "MaxSCFIterations=200"

   # Step 5: Rerun the job
   jf -p alberto job rerun 70

Use Case 10: Enabling Spin-Polarized Calculations
--------------------------------------------------

**Scenario:** You need to enable spin polarization for a magnetic system.

.. code-block:: bash

   # Step 1: Inspect current parameters
   atomate2siesta-jobflow-remote -p alberto job inspect 70 --full

   # Step 2: Add spin polarization
   atomate2siesta-jobflow-remote -p alberto job modify-db 70 \
       --param "Spin=polarized"

   # Step 3: Optionally add initial spin configuration
   atomate2siesta-jobflow-remote -p alberto job modify-db 70 \
       --param "Spin=polarized" \
       --param "DM.InitSpin=['1 +2.0', '2 -2.0', '3 +2.0', '4 -2.0']"

   # Step 4: Verify modification
   atomate2siesta-jobflow-remote -p alberto job inspect 70 --fdf-only

   # Step 5: Rerun
   jf -p alberto job rerun 70

Use Case 11: Batch Parameter Modification
------------------------------------------

**Scenario:** You need to modify multiple parameters at once for convergence.

.. code-block:: bash

   # Modify multiple parameters in one command
   atomate2siesta-jobflow-remote -p alberto job modify-db 70 \
       --param "Mesh.Cutoff=400 Ry" \
       --param "kpts=[6,6,1]" \
       --param "SCF.Mixer.Weight=0.01" \
       --param "SCF.DM.Tolerance=1e-6" \
       --param "MaxSCFIterations=150"

   # Verify all changes
   atomate2siesta-jobflow-remote -p alberto job inspect 70 --full

Use Case 12: Comparing User Params vs Tier Defaults
----------------------------------------------------

**Scenario:** You want to understand what the tier preset contributes.

.. code-block:: bash

   # View user parameters
   atomate2siesta-jobflow-remote -p alberto job inspect 70 --fdf-only

   # View tier preset defaults
   atomate2siesta-jobflow-remote -p alberto job inspect 70 --show-all-defaults

   # Compare with actual generated FDF
   atomate2siesta-jobflow-remote -p alberto job inspect 70 --show-actual-fdf

   # This shows three levels:
   # 1. User params (what you can modify)
   # 2. Tier defaults (from dataclass modules)
   # 3. Complete FDF (includes SIESTA defaults)

Common Use Cases
================

Use Case 1: Setting Up a New Project
-------------------------------------

**Scenario:** You need to set up jobflow-remote for the first time.

.. code-block:: bash

   # Step 1: Install jobflow-remote
   atomate2siesta-jobflow-remote install

   # Step 2: Generate configuration
   atomate2siesta-jobflow-remote setup --project-name my_project

   # Step 3: Add descriptive comments to help understand the config
   atomate2siesta-jobflow-remote update --project-name my_project --add-comments

   # Step 4: Edit the configuration file manually
   nano ~/.jfremote/my_project.yaml

   # Step 5: Verify configuration
   jf project check --errors

   # Step 6: Initialize database
   jf admin reset

   # Step 7: Submit test job
   atomate2siesta-jobflow-remote test

   # Step 8: Start runner
   jf runner start

Use Case 2: Managing Multiple HPC Clusters
-------------------------------------------

**Scenario:** You have three HPC systems and want separate configurations for each.

.. code-block:: bash

   # Create configurations for each cluster
   atomate2siesta-jobflow-remote setup --project-name cluster_a
   atomate2siesta-jobflow-remote setup --project-name cluster_b
   atomate2siesta-jobflow-remote setup --project-name cluster_c

   # Add comments to all configs for documentation
   atomate2siesta-jobflow-remote update --project-name cluster_a --add-comments
   atomate2siesta-jobflow-remote update --project-name cluster_b --add-comments
   atomate2siesta-jobflow-remote update --project-name cluster_c --add-comments

   # List all projects
   atomate2siesta-jobflow-remote info

   # View details for specific cluster
   atomate2siesta-jobflow-remote info --project-name cluster_a

   # When switching clusters, use jf with -p flag:
   jf -p cluster_a job list
   jf -p cluster_b runner start

Use Case 3: Migrating MongoDB Database
---------------------------------------

**Scenario:** Your MongoDB server is moving to a new host.

.. code-block:: bash

   # Option 1: Update using setup command
   atomate2siesta-jobflow-remote setup --update \
       --host new.mongodb.server.com \
       --port 27018

   # Option 2: Update using dedicated update command
   atomate2siesta-jobflow-remote update \
       --host new.mongodb.server.com \
       --port 27018

   # Both commands will:
   # - Create backup automatically
   # - Update all MongoDB connections (queue, jobstore, GridFS)
   # - Preserve all other settings

   # Verify the changes
   cat ~/.jfremote/atomate2siesta.yaml

   # Test connection
   jf project check --errors

   # If everything works, reinitialize database
   jf admin reset

   # If there's an issue, restore from backup
   cp ~/.jfremote/atomate2siesta.backup_TIMESTAMP.yaml \
      ~/.jfremote/atomate2siesta.yaml

Best Practices
==============

1. Always Use Backups
----------------------

.. code-block:: bash

   # Backups are enabled by default
   atomate2siesta-jobflow-remote update --database newdb

   # Only disable for testing
   atomate2siesta-jobflow-remote update --no-backup --database testdb

2. Add Comments Early
---------------------

.. code-block:: bash

   # Add comments right after initial setup
   atomate2siesta-jobflow-remote setup
   atomate2siesta-jobflow-remote update --add-comments

   # This makes future manual edits easier

3. Verify After Updates
-----------------------

.. code-block:: bash

   # Always verify configuration after updates
   atomate2siesta-jobflow-remote update --database newdb
   jf project check --errors

   # Check specific project
   jf -p my_project project check --errors

4. Document Worker Configurations
----------------------------------

Use descriptive worker names and add comments:

.. code-block:: yaml

   # Good naming examples:
   # ✓ agustina_slurm, local_shell, mn5_production
   # ✗ worker1, worker2, test

   workers:
     agustina_slurm:  # BIFI cluster, 128GB RAM, 32 cores
       type: remote
       scheduler_type: slurm

5. Use Project-Specific Commands
---------------------------------

.. code-block:: bash

   # Always specify project when working with multiple configs
   jf -p production job list
   jf -p development runner status

   # Or set JOBFLOW_REMOTE_PROJECT environment variable
   export JOBFLOW_REMOTE_PROJECT=production
   jf job list  # Uses production project

Troubleshooting
===============

Configuration file not found
----------------------------

**Error:**

.. code-block:: text

   ✗ Configuration file not found: ~/.jfremote/my_project.yaml
   Generate it first with: atomate2siesta-jobflow-remote setup

**Solution:**

.. code-block:: bash

   # Generate the configuration first
   atomate2siesta-jobflow-remote setup --project-name my_project

   # Or check if it exists under a different name
   ls ~/.jfremote/
   atomate2siesta-jobflow-remote info

jobflow-remote is not installed
--------------------------------

**Error:**

.. code-block:: text

   ✗ jobflow-remote is not installed!
   Install it first with: atomate2siesta-jobflow-remote install

**Solution:**

.. code-block:: bash

   # Install jobflow-remote
   atomate2siesta-jobflow-remote install

   # Verify installation
   pip show jobflow-remote

   # Or install manually
   pip install jobflow-remote

MongoDB connection fails after update
--------------------------------------

**Symptoms:** ``jf project check --errors`` shows connection errors.

**Diagnosis:**

.. code-block:: bash

   # Check current MongoDB settings
   atomate2siesta-jobflow-remote info --project-name atomate2siesta

   # Test MongoDB connection manually
   mongosh --host HOSTNAME --port PORT --eval "db.adminCommand('ping')"

**Solution:**

.. code-block:: bash

   # Restore from backup
   cp ~/.jfremote/atomate2siesta.backup_TIMESTAMP.yaml \
      ~/.jfremote/atomate2siesta.yaml

   # Update with correct settings
   atomate2siesta-jobflow-remote update \
       --host correct.host.com \
       --port 27017 \
       --database correct_db

   # Verify
   jf project check --errors

Summary
=======

The ``atomate2siesta-jobflow-remote`` CLI provides comprehensive tools for managing jobflow-remote configurations and jobs:

**Setup & Configuration Commands:**

* ``install`` - Install jobflow-remote
* ``setup`` - Generate/update project configurations
* ``update`` - Advanced configuration updates with inline comments
* ``info`` - List and inspect all projects
* ``test`` - Submit test job to verify setup
* ``runner`` - Display runner management commands

**Job Management Commands:**

* ``job inspect`` - View job details and FDF parameters
* ``job modify-db`` - Modify job parameters directly in MongoDB (risky)

**Key Features:**

* **Multi-project management**: Handle multiple HPC clusters
* **Automatic backups**: Safe updates with timestamped backups
* **Self-documenting configs**: Inline comments for all settings
* **Safe MongoDB updates**: Update database settings without breaking configs
* **Job parameter inspection**: View FDF params, tier defaults, actual FDF files
* **In-place parameter modification**: Modify running/failed jobs without recreating flows
* **Parameter validation**: 456 registered FDF parameters with typo detection
* **Internal param filtering**: Prevents deserialization errors
* **Case-insensitive**: Accepts Mesh.Cutoff, mesh.cutoff, MESH.CUTOFF

**Best Practices:**

**For Setup:**

* Always use backups (enabled by default)
* Add comments after initial setup for documentation
* Verify configurations after updates with ``jf project check``
* Use descriptive worker names
* Keep recent backups

**For Job Modification:**

* **Inspect before modifying**: Use ``job inspect --full`` to see current params
* **Use backups**: Never disable backup creation
* **Validate changes**: Check with ``job inspect`` after modification
* **Understand risks**: Direct DB modification can cause data corruption
* **Only modify READY jobs**: Don't modify RUNNING or COMPLETED jobs
* **Case doesn't matter**: Use standard capitalization (Mesh.Cutoff) for clarity
* **Check tier defaults**: Use ``--show-all-defaults`` to understand preset contributions

See Also
========

* `Official Documentation <https://matgenix.github.io/jobflow-remote/>`_
* `GitHub Repository <https://github.com/Matgenix/jobflow-remote>`_
* :doc:`cli-database` - Database CLI Reference
* :doc:`cli-cluster-setup` - Cluster Setup CLI
