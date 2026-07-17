=======================================
CLI Tools Overview
=======================================

atomate2siesta provides a comprehensive suite of command-line interface (CLI) tools for workflow generation, database management, cluster setup, and HPC job submission.

Overview
========

The CLI tools integrate seamlessly to provide a complete workflow from script generation to job execution and result storage:

**Information & Discovery**

1. :ref:`info-cli` - Display capabilities and information
2. :ref:`tiers-cli` - Explore tier presets and defaults
3. :ref:`recipe-cli` - Browse Recipe Book workflows

**Workflow Generation**

4. :ref:`workflow-generator-cli` - Generate ready-to-run workflow scripts

**Infrastructure**

5. :ref:`cluster-setup-cli` - Set up remote HPC clusters with SIESTA and dependencies
6. :ref:`jobflow-remote-cli` - Configure job submission and queue management
7. :ref:`database-cli` - Manage MongoDB databases for result storage

**Utilities**

8. :ref:`pseudos-cli` - Manage pseudopotentials and generate basis blocks
9. :ref:`plot-pseudo-cli` - Plot pseudopotential data from PSML files
10. :ref:`structure-info-cli` - Display structure information and symmetry
11. :ref:`structure-manipulation-cli` - Crystallographic operations (scale, supercell, rotate, translate)
12. :ref:`convert-cli` - Convert structure formats between sisl/pymatgen/ASE
13. :ref:`inputs-cli` - Generate SIESTA input files from structure

All CLI tools feature:

* 🎨 **Rich terminal UI** with colored output and progress indicators
* 🔒 **Safe operations** with automatic backups and confirmations
* 📚 **Self-documenting** with comprehensive help messages and examples
* ✅ **Production-ready** with extensive error handling

.. _info-cli:

Information CLI
===============

**Command:** ``atomate2siesta-info``

**Purpose:** Showcase all atomate2siesta capabilities and information

Key Features
------------

* **Complete overview** with quick stats (material-specific presets, recipe book, 13 CLI tools)
* **CLI tools catalog** organized by category
* **Workflow types** organized by category
* **Feature list** with descriptions
* **Quick start examples** with code snippets
* **Version information** and configuration paths

Available Subcommands
---------------------

.. code-block:: bash

   atomate2siesta-info              # Overview (default)
   atomate2siesta-info overview     # Same as default
   atomate2siesta-info tools        # List all 13 CLI tools
   atomate2siesta-info workflows    # Show 13 workflows by category
   atomate2siesta-info workflows <name> --full  # Show full Maker documentation
   atomate2siesta-info features     # Display 10 major features
   atomate2siesta-info examples     # Quick start code examples
   atomate2siesta-info version      # Version and config info

Viewing Detailed Documentation
-------------------------------

The ``--full`` flag displays comprehensive documentation for any Maker or FlowMaker:

.. code-block:: bash

   # View complete docstring for any workflow/maker
   atomate2siesta-info workflows StaticMaker --full
   atomate2siesta-info workflows RelaxMaker --full
   atomate2siesta-info workflows BandStructureMaker --full
   atomate2siesta-info workflows SiestaEosFlowMaker --full

   # Works with all 10 single-job Makers:
   atomate2siesta-info workflows StaticMaker --full
   atomate2siesta-info workflows RelaxMaker --full
   atomate2siesta-info workflows LuaMaker --full
   atomate2siesta-info workflows SocketIOStaticMaker --full
   atomate2siesta-info workflows BandStructureMaker --full
   atomate2siesta-info workflows DOSMaker --full
   atomate2siesta-info workflows PDOSMaker --full
   atomate2siesta-info workflows PhononMaker --full
   atomate2siesta-info workflows OpticalMaker --full
   atomate2siesta-info workflows SiestaPhononMaker --full

**Full Documentation Includes:**

* **Detailed scientific context** - What the calculation does and why
* **Workflow steps** - Step-by-step breakdown of computation
* **Key results** - What outputs are generated
* **Applications** - Real-world use cases
* **Parameters** - All available configuration options
* **Examples** - Practical code snippets
* **Notes** - Best practices and convergence tips

All comprehensive docstrings are displayed in beautiful Rich panels with formatted markdown sections.

Example Output
--------------

.. code-block:: bash

   $ atomate2siesta-info

   ╔═══════════════════════════════════════╗
   ║         atomate2siesta                ║
   ║ Automated SIESTA Workflows            ║
   ║                                       ║
   ║ 🔬 Production-ready DFT workflows     ║
   ║ ⚡ significant code reduction              ║
   ║ 🛠️ 13 CLI tools                       ║
   ║ 📚 Many tutorials                     ║
   ╚═══════════════════════════════════════╝

   📦 Workflows    13+ production workflows
   🍳 Recipes      one-line workflows
   🎯 Tier Presets 32 material-specific presets

.. _tiers-cli:

Tier Presets CLI
================

**Command:** ``atomate2siesta-presets``

**Purpose:** Explore and discover tier presets with automatic detection

Key Features
------------

* **Automatic preset detection** - No manual updates needed!
* **Dynamic tier counting** - Automatically counts new tier levels
* **Material-specific presets** across 10 categories (2d, surface, electronic, phonon, etc.)
* **Tier defaults** - View base parameter sets for all tier levels
* **Category browsing** - Explore presets by category
* **Search functionality** - Find presets by tier level or category
* **Rich terminal UI** with tables and panels

Available Subcommands
---------------------

.. code-block:: bash

   atomate2siesta-presets                      # List all presets (default)
   atomate2siesta-presets list                 # Same as default
   atomate2siesta-presets show <preset>        # Show preset details
   atomate2siesta-presets category <name>      # Show presets in category
   atomate2siesta-presets search --tier basic  # Search by tier level
   atomate2siesta-presets defaults             # Show tier-level base parameters
   atomate2siesta-presets examples             # Usage examples

Automatic Detection
-------------------

**Add new presets** - They appear automatically in CLI!

.. code-block:: python

   # Add to presets/surface.py
   SURFACE_PRESETS = {
       "your_new_preset": {
           "description": "Your description",
           "tier": "intermediate",
           "recommended_params": {...},
       },
   }

**Result:**

.. code-block:: bash

   $ atomate2siesta-presets show your_new_preset
   # ✅ Shows immediately - no CLI updates needed!

**Naming conventions for automatic categorization:**

* ``2d_*`` → 2d category
* ``surface_*`` → surface category
* ``adsorbate_*`` → surface category
* ``magnetic_*`` → magnetic category
* ``phonon_*`` → phonon category
* Contains ``relax`` → structural category

See :ref:`adding-custom-tiers-presets` for full guide.

Example Output
--------------

.. code-block:: bash

   $ atomate2siesta-presets show adsorbate_screening

   ╔═══════════ adsorbate_screening ═══════════╗
   ║ Fast adsorbate screening                  ║
   ║ Tier: basic                               ║
   ╚═══════════════════════════════════════════╝

   Parameters:
      PAO.BasisSize    DZP
      kpts             [4, 4, 1]
      Mesh.Cutoff      200 Ry

   Usage Example:
   from atomate2.siesta.sets.tiers import apply_tier_preset
   maker = apply_tier_preset(maker, 'adsorbate_screening')

.. _recipe-cli:

Recipe Book CLI
===============

**Command:** ``atomate2siesta-recipe``

**Purpose:** Browse and search the Recipe Book (one-line workflows)

Key Features
------------

* Recipes across 6 categories (Complete, Electronic, Mechanical, Thermal, Surface, Convergence)
* **significant code reduction** - Comprehensive workflows in one line
* **Category browsing** - Explore recipes by category
* **Keyword search** - Find recipes by name or property
* **Code examples** - Ready-to-copy usage examples with line numbers
* **Statistics** - View Recipe Book metrics and code reduction percentages

Available Subcommands
---------------------

.. code-block:: bash

   atomate2siesta-recipe                     # List all recipes (default)
   atomate2siesta-recipe list                # Same as default
   atomate2siesta-recipe show <recipe>       # Show recipe details with code
   atomate2siesta-recipe category <name>     # Show recipes in category
   atomate2siesta-recipe search <keyword>    # Search by keyword
   atomate2siesta-recipe stats               # Recipe Book statistics
   atomate2siesta-recipe examples            # Usage examples

Example Output
--------------

.. code-block:: bash

   $ atomate2siesta-recipe show phonon_workflow

   ╔═══════════ phonon_workflow ═══════════════╗
   ║ Phonon calculation with auto plotting    ║
   ║ Category: Thermal Properties              ║
   ║ Code Reduction: high                       ║
   ╚═══════════════════════════════════════════╝

   Usage Example:
    1  from atomate2.siesta.recipes import RecipeBook
    2  from pymatgen.core import Structure
    3
    4  structure = Structure.from_file('structure.cif')
    5  flow = RecipeBook.phonon_workflow(structure)
    6  results = run_locally(flow, create_folders=True)

.. _workflow-generator-cli:

Workflow Script Generator CLI
===============================

**Command:** ``atomate2siesta-maker``

**Purpose:** Generate ready-to-run Python workflow scripts from structure files

Key Features
------------

* **30-second workflow setup** - From structure file to running calculation
* **13 workflow templates** organized in 4 categories (Basic, Vibrational, Mechanical, Advanced)
* **Context-specific help** for each workflow type
* **Tier preset support** for applying material-specific parameters
* **Dry-run mode** for testing workflows without running calculations
* **Remote submission** integration with jobflow-remote
* **Customization sections** with commented examples in generated scripts

Available Workflows
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Workflow
     - Description
   * - **Basic Workflows (4)**
     -
   * - ``relax``
     - Structure relaxation (fixed or variable cell)
   * - ``static``
     - Single-point energy calculation
   * - ``bands``
     - Electronic band structure
   * - ``dos``
     - Density of states
   * - **Vibrational Properties (3)**
     -
   * - ``phonon``
     - Phonon calculation with automatic plotting
   * - ``gruneisen``
     - Grüneisen parameters and thermal expansion
   * - ``qha``
     - Quasi-harmonic approximation
   * - **Mechanical Properties (3)**
     -
   * - ``eos``
     - Equation of state
   * - ``elastic``
     - Elastic constants
   * - ``bulk-modulus``
     - Quick bulk modulus calculation
   * - **Advanced Workflows (3)**
     -
   * - ``neb``
     - Nudged elastic band (transition states)
   * - ``surface``
     - Surface energy calculation
   * - ``adsorption``
     - Adsorption site scanning

Interactive Mode (NEW!)
-----------------------

**Beginner-friendly guided workflow generation:**

.. code-block:: bash

   # Interactive mode with step-by-step prompts
   atomate2siesta-maker --interactive

   # The interactive mode will guide you through:
   # 1. Workflow type selection (13 workflows)
   # 2. Structure file selection
   # 3. Tier preset selection (optional)
   # 4. Execution mode (local/remote/dry-run)
   # 5. Database configuration (optional)
   #    - Choose: jobflow.yaml (automatic) or explicit JobStore
   #    - Configure: host, port, database, collection
   # 6. Workflow-specific parameters
   # 7. Output filename

**Features:**

* **No memorization required** - See all options as you go
* **Input validation** - Validates parameters before generating
* **Sensible defaults** - Suggests common values
* **Beautiful UI** - Colored prompts with arrow key navigation
* **Cancel anytime** - Press Ctrl+C to exit

Quick Examples
--------------

.. code-block:: bash

   # Interactive mode (NEW!) - Easiest way to start
   atomate2siesta-maker --interactive

   # List all available workflows
   atomate2siesta-maker list

   # Basic relaxation
   atomate2siesta-maker relax Si.cif

   # With tier preset
   atomate2siesta-maker relax Si.cif --preset relax_standard

   # Phonon calculation with custom supercell
   atomate2siesta-maker phonon Si.cif --supercell 2 2 2

   # EOS with custom parameters
   atomate2siesta-maker eos Si.cif --number-of-frames 9 --strain-range 0.08

   # NEB transition state search
   atomate2siesta-maker neb initial.cif final.cif --number-of-images 7

   # Surface energy calculation
   atomate2siesta-maker surface bulk.cif --miller-indices 1,1,1

   # Adsorption site scanning
   atomate2siesta-maker adsorption slab.cif --grid-size 4 4 --height 2.0

   # Remote submission with jobflow-remote
   atomate2siesta-maker relax Si.cif --remote --worker hpc_cluster

   # Dry-run mode (preview only)
   atomate2siesta-maker relax Si.cif --dry-run

   # With MongoDB database configuration
   atomate2siesta-maker relax Si.cif --database

   # Custom database settings
   atomate2siesta-maker relax Si.cif --database --db-name my_calculations --db-host db.server.com

   # Database with preset and remote submission
   atomate2siesta-maker relax Si.cif --preset relax_standard --database --remote --worker cluster1

   # Get workflow-specific help
   atomate2siesta-maker relax --help
   atomate2siesta-maker phonon --help
   atomate2siesta-maker neb --help

Common Options
--------------

All workflows support these options:

.. code-block:: bash

   -o, --output PATH       # Output filename (default: <workflow>_<formula>.py)
   --preset TEXT           # Tier preset (e.g., relax_standard, band_structure)
   --dry-run               # Generate script with dry_run=True for testing
   --remote                # Generate script for jobflow-remote submission
   --worker TEXT           # Remote worker name (default: default)

   # Database options
   --database              # Include MongoDB database configuration in script
   --db-host TEXT          # MongoDB host (default: localhost)
   --db-port INTEGER       # MongoDB port (default: 27017)
   --db-name TEXT          # MongoDB database name (default: atomate2siesta)
   --db-collection TEXT    # MongoDB collection name (default: tasks)

Workflow-Specific Options
--------------------------

Each workflow has specialized options:

**Relax:**

.. code-block:: bash

   --cell-type [fixed|variable]  # Cell relaxation type (default: fixed)

**Bands:**

.. code-block:: bash

   --kpath-density INTEGER  # K-points per Å⁻¹ (default: 20)

**Phonon:**

.. code-block:: bash

   --supercell INTEGER...    # Supercell size (3 integers: nx ny nz)
   --min-length FLOAT        # Minimum supercell length in Å (default: 10.0)
   --displacement FLOAT      # Atomic displacement in Å (default: 0.01)
   --custom-params           # Use separate relax/force parameters

**EOS:**

.. code-block:: bash

   --number-of-frames INTEGER  # Volume sampling points (default: 7)
   --strain-range FLOAT        # Strain range ±fraction (default: 0.05)

**Elastic:**

.. code-block:: bash

   --strain-magnitude FLOAT  # Strain magnitude (default: 0.005)

**NEB:**

.. code-block:: bash

   --number-of-images INTEGER         # Intermediate images (default: 5)
   --relax-endpoints                  # Relax initial/final structures first
   --interpolation [idpp|linear]      # Interpolation method (default: idpp)

**Surface:**

.. code-block:: bash

   --slab-directory PATH       # Directory with slab structures (default: ./slabs)
   --miller-indices TEXT       # Miller indices (e.g., 1,0,0 or 1,1,1)
   --relax-slabs              # Relax slabs before energy calculation

**Adsorption:**

.. code-block:: bash

   --grid-size INTEGER...     # Grid size (2 integers: nx ny, default: 3 3)
   --height FLOAT             # Adsorbate height above surface in Å (default: 2.0)
   --miller-indices TEXT      # Miller indices of surface
   --adsorbate PATH           # Path to adsorbate molecule file

Generated Script Features
-------------------------

Each generated script includes:

1. **Header and metadata** - Timestamp, command used, workflow description
2. **Configuration validation** - Checks SIESTA_PP_PATH and SIESTA_CMD
3. **Structure information** - Displays formula, space group, number of atoms
4. **Database configuration** (optional) - MongoDB JobStore setup with SETTINGS.JOB_STORE
5. **Workflow setup** - Maker creation with all specified options
6. **Customization section** - Commented examples for common modifications
7. **Execution code** - Local (run_locally) or remote (submit_flow) execution
8. **Expected outputs** - List of files that will be generated

When ``--database`` is used, the script includes:

.. code-block:: python

   from jobflow import SETTINGS, JobStore
   from maggma.stores import MongoStore

   # Define MongoDB store
   store = MongoStore(
       database="atomate2siesta",
       collection_name="tasks",
       host="localhost",
       port=27017,
   )

   # Create JobStore and set in SETTINGS
   job_store = JobStore(docs_store=store)
   SETTINGS.JOB_STORE = job_store

   print("✓ Database configured:")
   print(f"  Host: localhost")
   print(f"  Database: atomate2siesta")
   print(f"  Collection: tasks")

Example generated script:

.. code-block:: python

   #!/usr/bin/env python
   """
   Generated by atomate2siesta-maker on 2025-11-06
   Command: atomate2siesta-maker relax Si.cif --preset relax_standard

   Structure relaxation workflow for Si
   """

   # Configuration check
   import os
   if "SIESTA_PP_PATH" not in os.environ:
       raise RuntimeError("SIESTA_PP_PATH not set")

   # Imports
   from pymatgen.core import Structure
   from atomate2.siesta.jobs.core import RelaxMaker
   from atomate2.siesta.sets.tiers import apply_tier_preset
   from jobflow import run_locally

   # Load structure
   structure = Structure.from_file("Si.cif")

   # Create workflow
   maker = RelaxMaker.fixed_cell_relaxation()
   maker = apply_tier_preset(maker, "relax_standard")
   job = maker.make(structure)

   # Run workflow
   results = run_locally(job, create_folders=True)

**Status:** ✅ Production-ready

**Benefits:**

* **90% code reduction** - One command replaces 50+ lines of Python
* **Best practices** - Generated code follows recommended patterns
* **Educational** - Scripts teach proper usage through examples
* **Time savings** - 30 seconds from structure to running workflow

.. _cluster-setup-cli:

Remote Cluster Setup CLI
=========================

**Command:** ``atomate2siesta-cluster``

**Purpose:** Automate remote HPC cluster configuration via SSH

Key Features
------------

* SSH-based cluster configuration with multiple authentication methods
* Automatic conda environment creation with all dependencies
* Optional SIESTA installation from conda-forge
* Automatic ``.atomate2.yaml`` configuration file generation
* Environment status checking and verification

Available Commands
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - ``setup``
     - Set up conda environment with atomate2siesta and jobflow-remote
   * - ``status``
     - Check status of remote cluster environment
   * - ``info``
     - Show usage examples and documentation

Quick Examples
--------------

.. code-block:: bash

   # Complete cluster setup with SIESTA installation
   atomate2siesta-cluster setup --host mycluster --ssh-config --git-ssh --install-siesta

   # Check environment status
   atomate2siesta-cluster status --host mycluster --ssh-config

   # View comprehensive documentation
   atomate2siesta-cluster info

**Status:** ✅ Production-ready

**Full Documentation:** :doc:`cli-cluster-setup`

.. _jobflow-remote-cli:

Jobflow Remote Setup CLI
=========================

**Command:** ``atomate2siesta-jobflow-remote``

**Purpose:** Simplified jobflow-remote installation, configuration, and management

Key Features
------------

* One-command installation (stable or development version)
* Multi-project configuration management
* MongoDB connection settings with automatic backups
* Self-documenting configurations with inline comments
* Test job submission for verification
* Comprehensive project discovery and inspection

Available Commands
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - ``install``
     - Install jobflow-remote package
   * - ``setup``
     - Generate or update project configuration
   * - ``update``
     - Update existing configuration with advanced options
   * - ``info``
     - Show installation status and list all projects
   * - ``test``
     - Submit test job to verify setup
   * - ``runner``
     - Display runner and job management commands

Quick Examples
--------------

.. code-block:: bash

   # Install and set up jobflow-remote
   atomate2siesta-jobflow-remote install
   atomate2siesta-jobflow-remote setup --project-name my_project

   # Add descriptive comments to configuration
   atomate2siesta-jobflow-remote update --add-comments

   # Update MongoDB settings
   atomate2siesta-jobflow-remote update --host db.server.com --port 27018

   # List all configured projects
   atomate2siesta-jobflow-remote info

   # Submit test job
   atomate2siesta-jobflow-remote test

**Status:** ✅ Production-ready

**Full Documentation:** :doc:`cli-jobflow-remote`

.. _database-cli:

Database Management CLI
========================

**Command:** ``atomate2siesta-database``

**Purpose:** MongoDB database testing, monitoring, and management

Key Features
------------

* Connection testing with both PyMongo and Maggma
* Recent calculation listing with metadata tables
* Formula-based querying
* Comprehensive database and collection statistics
* Safe deletion with confirmation prompts
* Configuration file generation

Available Commands
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - ``test``
     - Test MongoDB connection and show basic statistics
   * - ``list``
     - List recent calculation results with metadata
   * - ``query``
     - Query calculations by chemical formula
   * - ``stats``
     - Display comprehensive database statistics
   * - ``clear``
     - Clear collection (with safety confirmation)
   * - ``config``
     - Show configuration examples

Quick Examples
--------------

.. code-block:: bash

   # Test MongoDB connection
   atomate2siesta-database test

   # List 20 most recent calculations
   atomate2siesta-database list --limit 20

   # Query all Silicon calculations
   atomate2siesta-database query Si

   # Show comprehensive statistics
   atomate2siesta-database stats

   # Clear test database
   atomate2siesta-database clear --database test_db --force

**Status:** ✅ Production-ready

**Full Documentation:** :doc:`cli-database`

.. _pseudos-cli:

Pseudopotential Management CLI
================================

**Command:** ``atomate2siesta-pseudos``

**Purpose:** Install, inspect, and generate basis blocks from PSML pseudopotentials

Key Features
------------

* **Pseudopotential installation** - Download and install 6 available pseudopotential families
* **Shell information display** - View valence configuration from PSML files
* **PAO.Basis generation** - Generate SIESTA basis blocks from pseudopotentials
* **Multiple basis sizes** - Support for SZ, DZ, DZP, TZ, TZP, TZDP
* **Multiple n-shells** - Generate valence + excited shell configurations
* **Correct rc values** - Descending cutoff radii following SIESTA conventions
* **Rich terminal UI** - Formatted tables and panels

Available Commands
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - ``available``
     - List all available pseudopotential families
   * - ``install``
     - Install pseudopotential family
   * - ``show``
     - Show pseudopotential info and valence shells
   * - ``basis``
     - Generate PAO.Basis block from PSML file
   * - ``plot``
     - Plot pseudopotential properties

Quick Examples
--------------

**Installation:**

.. code-block:: bash

   # List available pseudopotentials
   atomate2siesta-pseudos available

   # Install specific pseudopotential
   atomate2siesta-pseudos install ONCVPSP-PBE-SR-PDv0.4-Standard

   # Install all available pseudopotentials
   atomate2siesta-pseudos install --all

**Pseudopotential Information:**

.. code-block:: bash

   # Show basic pseudopotential info
   atomate2siesta-pseudos show ONCVPSP-PBE-SR-PDv0.4-Standard

   # Show with valence shell information (NEW!)
   atomate2siesta-pseudos show ONCVPSP-PBE-SR-PDv0.4-Standard Zr

   # Output shows:
   # ┏━━━━━━━┳━━━┳━━━┳━━━━━━━━━━━━┓
   # ┃ Shell ┃ n ┃ l ┃ Occupation ┃
   # ┡━━━━━━━╇━━━╇━━━╇━━━━━━━━━━━━┩
   # │ 4s    │ 4 │ s │ 2.00       │
   # │ 4p    │ 4 │ p │ 6.00       │
   # │ 4d    │ 4 │ d │ 2.00       │
   # │ 5s    │ 5 │ s │ 2.00       │
   # └───────┴───┴───┴────────────┘

**Basis Block Generation (NEW!):**

.. code-block:: bash

   # Generate DZP basis for Silicon (default: PSML method)
   atomate2siesta-pseudos basis ONCVPSP-PBE-SR-PDv0.4-Standard Si

   # Custom basis size
   atomate2siesta-pseudos basis ONCVPSP-PBE-SR-PDv0.4-Standard Si --basis-size TZP

   # Multiple n-shells (valence + excited shells)
   atomate2siesta-pseudos basis ONCVPSP-PBE-SR-PDv0.4-Standard Si --n-shells 2

   # Use atomic radius scaling method
   atomate2siesta-pseudos basis ONCVPSP-PBE-SR-PDv0.4-Standard Fe --rc-method scaled

   # Use hydrogenic scaling method
   atomate2siesta-pseudos basis ONCVPSP-PBE-SR-PDv0.4-Standard Cu --rc-method hydrogenic

   # Adjust PSML wavefunction decay threshold
   atomate2siesta-pseudos basis ONCVPSP-PBE-SR-PDv0.4-Standard Si --rc-threshold 0.01

   # Save to file
   atomate2siesta-pseudos basis ONCVPSP-PBE-SR-PDv0.4-Standard Si --output-file Si_basis.fdf

Basis Block Generation Details
-------------------------------

**Supported Basis Sizes:**

* **SZ** - Single-ζ (1 radial function per orbital)
* **DZ** - Double-ζ (2 radial functions per orbital)
* **DZP** - Double-ζ + polarization (2 + p-polarization)
* **TZ** - Triple-ζ (3 radial functions per orbital)
* **TZP** - Triple-ζ + polarization (3 + p-polarization)
* **TZDP** - Triple-ζ + double polarization (3 + p + d)

**Cutoff Radius (rc) Calculation Methods:**

Four methods available for determining rc values (``--rc-method``):

**1. PSML Wavefunction Decay (DEFAULT - BEST)**

* Extracts rc from actual pseudopotential wavefunctions
* Element-specific and orbital-specific (n, l)
* Uses threshold parameter for decay detection (default: 5%)
* Example (Si): 3s → 4.92 bohr, 3p → 5.81 bohr

.. code-block:: bash

   atomate2siesta-pseudos basis PSEUDO El --rc-method psml --rc-threshold 0.05

**2. Atomic Radius Scaling (Periodic Trends)**

* Scales by element's atomic radius relative to Si
* Follows periodic table size trends
* Example: H → 1.14 bohr (small), Fe → 6.36 bohr (large)

.. code-block:: bash

   atomate2siesta-pseudos basis PSEUDO El --rc-method scaled

**3. Hydrogenic Scaling (n²/Z_eff)**

* Uses Slater's effective nuclear charge
* Quantum number dependent
* Example (Si): 3s → 7.43 bohr

.. code-block:: bash

   atomate2siesta-pseudos basis PSEUDO El --rc-method hydrogenic

**4. Fixed Values (Original)**

* Standard SIESTA defaults (not element-specific)
* s=5.00, p=6.00, d=5.50, f=5.00 bohr
* Backward compatible

.. code-block:: bash

   atomate2siesta-pseudos basis PSEUDO El --rc-method fixed

**Comparison Table** (s orbital, first zeta):

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 15

   * - Element
     - PSML
     - Scaled
     - Hydrogenic
     - Fixed
   * - H
     - 2.92
     - 1.14
     - 1.65
     - 5.00
   * - Si
     - 4.92
     - 5.00
     - 7.43
     - 5.00
   * - Fe
     - 3.61
     - 6.36
     - 6.60
     - 5.00

**Descending Zeta Pattern:**

For multiple zeta functions, rc values decrease:

* First zeta: Largest rc (most extended orbital)
* Subsequent zetas: rc × 0.85^i (more localized)
* Example: 5.00 → 4.25 → 3.61 bohr

**Multiple n-Shells:**

Use ``--n-shells 2`` to include excited shell configurations:

.. code-block:: text

   # Single n-shell (default):
   n=3 0 2  # 3s orbital
     5.00 4.25

   # Two n-shells:
   n=3 0 2  # 3s orbital (valence)
     5.00 4.25
   n=4 0 2  # 4s orbital (excited)
     6.00 5.10

Example Output
--------------

.. code-block:: bash

   $ atomate2siesta-pseudos basis ONCVPSP-PBE-SR-PDv0.4-Standard Si --basis-size DZP

   %block PAO.Basis
   Si  3  # Label, l_shells
     n=3 0 2  # 3s orbital
       5.00 4.25  # rc(izeta=1..Nzeta) (Bohr)
     n=3 1 2 P 1  # 3p orbital (with d-polarization)
       6.00 5.10  # rc(izeta=1..Nzeta) (Bohr)
       5.50  # rc for polarization orbital (Bohr)
     n=3 2 1  # 3d orbital (polarization)
       5.50  # rc(izeta=1) (Bohr)
   %endblock PAO.Basis

**Features in Output:**

* Orbital name comments (3s, 3p, 3d)
* Shell type indicators ((excited) for n > valence)
* Polarization type labels (d-polarization, f-polarization)
* Descending rc values for proper convergence

**Status:** ✅ Production-ready

**Benefits:**

* **Quick basis generation** - From pseudopotential to SIESTA input in seconds
* **Correct conventions** - Follows SIESTA best practices automatically
* **Flexible customization** - Control basis size and shell depth
* **Inspect before use** - View valence shells with ``show`` command

.. _plot-pseudo-cli:

Pseudopotential Plotting CLI
==============================

**Command:** ``atomate2siesta-plot-pseudo``

**Purpose:** Generate comprehensive plots from PSML pseudopotential files

Key Features
------------

* **Multiple plot types** - wavefunctions, potentials, 3D potential, occupation, density
* **Flexible output** - Individual or all plots at once
* **Customizable range** - Control radial distance for different plot types
* **Rich visualization** - Uses seaborn and matplotlib for publication-quality plots
* **PSML format support** - XML-based pseudopotential markup language

Available Plot Types
--------------------

.. code-block:: bash

   --plot-type wavefunctions    # Radial wavefunctions for all orbitals
   --plot-type potentials       # Local and non-local pseudopotentials
   --plot-type 3d-potential     # 3D visualization of local potential
   --plot-type occupation       # Valence shell occupations
   --plot-type density          # Radial charge density
   --plot-type all              # Generate all plots (default)

Quick Examples
--------------

.. code-block:: bash

   # Generate all plots for a pseudopotential
   atomate2siesta-plot-pseudo Si.psml

   # Generate only wavefunction plots
   atomate2siesta-plot-pseudo Si.psml --plot-type wavefunctions

   # Custom output directory
   atomate2siesta-plot-pseudo Si.psml --output-dir ./plots

   # Limit radial range to 8 bohr
   atomate2siesta-plot-pseudo Si.psml --r-plot 8.0

Command Options
---------------

.. code-block:: bash

   atomate2siesta-plot-pseudo PSML_FILE [OPTIONS]

   Options:
     --plot-type [wavefunctions|potentials|3d-potential|occupation|density|all]
                                   Type of plot to generate (default: all)
     --output-dir PATH              Output directory for plots (default: .)
     --r-plot FLOAT                 Maximum radial distance in bohr

Generated Outputs
-----------------

Depending on ``--plot-type``, the following files are created:

* ``<element>_wavefunctions.png`` - Radial wavefunctions
* ``<element>_potentials.png`` - Local and non-local potentials
* ``<element>_potential_3d.png`` - 3D potential visualization
* ``<element>_occupation.png`` - Valence shell occupations
* ``<element>_density.png`` - Radial charge density

**Status:** ✅ Production-ready

.. _structure-info-cli:

Structure Information CLI
==========================

**Command:** ``atomate2siesta-structure info``

**Purpose:** Display comprehensive information about structure files

Key Features
------------

* **Crystal symmetry analysis** - Space group, crystal system, point group
* **Lattice parameters** - a, b, c, α, β, γ, volume
* **Atomic composition** - Element counts and percentages
* **Magnetic property detection** - Automatic detection of magnetic elements
* **Per-site magnetic moments** - Display individual atom moments
* **Multi-format support** - CIF, XSF, XV, FDF, POSCAR formats
* **Rich terminal UI** - Colored tables with clear organization

Quick Examples
--------------

.. code-block:: bash

   # Basic structure information
   atomate2siesta-structure info structure.cif

   # Analyze SIESTA FDF file
   atomate2siesta-structure info siesta.fdf

   # High precision symmetry analysis
   atomate2siesta-structure info structure.cif --symprec 0.001

   # Disable magnetic analysis
   atomate2siesta-structure info structure.xsf --no-magnetic

Command Options
---------------

.. code-block:: bash

   atomate2siesta-structure info STRUCTURE_FILE [OPTIONS]

   Options:
     --symprec FLOAT            Symmetry precision (default: 0.01 Å)
     --angle-tolerance FLOAT    Angle tolerance (default: 5.0 degrees)
     --magnetic / --no-magnetic Enable/disable magnetic analysis (default: enabled)

Output Information
------------------

**Crystal System & Symmetry:**

* Chemical formula (reduced and full)
* Space group symbol and number
* Crystal system (cubic, orthorhombic, etc.)
* Point group
* Number of sites (input, primitive, conventional)

**Lattice Parameters:**

* Lattice constants (a, b, c)
* Lattice angles (α, β, γ)
* Unit cell volume

**Atomic Composition:**

* Element counts
* Atomic percentages
* Total number of atoms

**Magnetic Properties** (if enabled):

* Magnetic moment detection status
* Number of magnetic sites
* Magnetic ordering (FM/AFM/non-magnetic)
* Unique magnetic moment values
* Per-site magnetic moments (first 20 shown)

Example Output
--------------

.. code-block:: bash

   $ atomate2siesta-structure info CuNCN.xsf

   ╭─────────────────────────────────────────────────╮
   │ Structure Information: CuNCN.xsf                │
   ╰─────────────────────────────────────────────────╯

   Crystal System & Symmetry
   ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
   ┃ Property            ┃ Value         ┃
   ┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
   │ Formula             │ CuCN2         │
   │ Space Group         │ Pmmn (#59)    │
   │ Crystal System      │ orthorhombic  │
   │ Number of Sites     │ 96            │
   └─────────────────────┴───────────────┘

   Magnetic Properties
   ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
   ┃ Property            ┃ Value              ┃
   ┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
   │ Magnetic Sites      │ 24 / 96            │
   │ Magnetic Ordering   │ Ferromagnetic (FM) │
   │ Unique Moments      │ 0.600 μB           │
   └─────────────────────┴────────────────────┘

**Status:** ✅ Production-ready

.. _structure-manipulation-cli:

Structure Manipulation CLI
===========================

**Command:** ``atomate2siesta-structure {scale|supercell|rotate|translate|slab|vacuum|stack}``

**Purpose:** Crystallographic operations for structure modification and preparation

Overview
--------

The structure manipulation toolkit provides 16 commands organized in 4 tiers:

**Tier 1: Basic Operations (6 commands)**

1. **info** - Structure information and symmetry analysis
2. **convert** - Format conversion (SIESTA ↔ CIF/POSCAR/XSF)
3. **scale** - Lattice parameter scaling (EOS, pressure, strain studies)
4. **supercell** - Supercell generation (phonons, defects, surfaces)
5. **rotate** - Structure rotation (alignment, reorientation, surface cuts)
6. **translate** - Atomic translation (centering, interfaces, heterostructures)

**Tier 2: Surface & 2D Operations (3 commands)**

7. **slab** - Surface slab generation (adsorption, catalysis)
8. **vacuum** - Vacuum spacing control (2D materials, surfaces)
9. **stack** - Layer stacking (heterostructures, multilayers)

**Tier 3: Advanced Atomic Manipulation (4 commands)**

10. **substitute** - Atomic substitution (doping, alloying, defects)
11. **remove** - Atom removal (vacancies, cleanup, proximity-based)
12. **add** - Add atoms/molecules (adsorbates, dopants, oriented placement)
13. **perturb** - Random perturbations (MD initialization, transition states)

**Tier 4: Analysis & Optimization (3 commands)**

14. **compare** - Structure comparison (RMSD, lattice differences, composition)
15. **standardize** - Cell standardization (conventional, primitive, international)
16. **optimize-cell** - Cell optimization (Niggli reduction, orthogonalization)

All commands share common features:

* **Multiple output formats** - CIF, POSCAR, XSF, JSON, FDF, XV
* **Rich comparison tables** - Before/after with changes/multipliers
* **Site property preservation** - Magnetic moments, selective dynamics, etc.
* **Automatic output naming** - prefix_input.ext (e.g., scaled_structure.cif)
* **Comprehensive validation** - Clear error messages and usage tips
* **Beautiful terminal UI** - Color-coded output with progress indicators

scale - Lattice Parameter Scaling
----------------------------------

**Purpose:** Scale lattice parameters for equation of state (EOS) studies, pressure simulations, and strain analysis

**Quick Examples:**

.. code-block:: bash

   # Uniform 5% expansion
   atomate2siesta-structure scale Si.cif --factor 1.05

   # Anisotropic scaling (expand a, keep b, compress c)
   atomate2siesta-structure scale Si.cif --abc 1.05 1.0 0.95

   # Scale to specific volume
   atomate2siesta-structure scale Si.cif --volume 50.0

   # Apply 5% volumetric strain
   atomate2siesta-structure scale Si.cif --strain 0.05

   # Generate EOS series (11 structures from -5% to +5%)
   atomate2siesta-structure scale Si.cif --series --min 0.95 --max 1.05 --steps 11

**Command Options:**

.. code-block:: bash

   atomate2siesta-structure scale STRUCTURE_FILE [OPTIONS]

   Options:
     --factor FLOAT           Uniform scaling factor (e.g., 1.05 for 5% expansion)
     --abc FLOAT FLOAT FLOAT  Anisotropic scaling (a, b, c independent)
     --volume FLOAT           Target volume in Å³
     --strain FLOAT           Volumetric strain (e.g., 0.05 for 5% expansion)
     --series                 Generate series of scaled structures
     --min FLOAT              Minimum scaling factor for series
     --max FLOAT              Maximum scaling factor for series
     --steps INTEGER          Number of steps in series (default: 11)
     -o, --output PATH        Output file (default: scaled_<input>)
     --format [cif|poscar|xsf|json]  Output format (default: cif)

**Use Cases:**

* **EOS workflows** - Generate 11 structures with different volumes
* **Pressure studies** - Simulate crystal under compression/expansion
* **Strain engineering** - Apply controlled strain to structures
* **Lattice optimization** - Find equilibrium lattice constants

**Technical Details:**

* Uniform scaling: V_new = V_orig × factor³
* Volume scaling: Finds factor where V_new = target_volume
* Strain scaling: V_new = V_orig × (1 + strain)
* Series generation: Linear spacing between min and max factors
* Preserves fractional coordinates during scaling

supercell - Supercell Generation
---------------------------------

**Purpose:** Generate supercells for phonon calculations, defect studies, and surface modeling

**Quick Examples:**

.. code-block:: bash

   # 2×2×2 supercell
   atomate2siesta-structure supercell Si.cif --matrix 2 2 2

   # Non-cubic supercell (2×2×1)
   atomate2siesta-structure supercell Si.cif --matrix 2 2 1

   # Automatic sizing for 10 Å minimum length
   atomate2siesta-structure supercell Si.cif --min-length 10.0

   # At least 50 atoms
   atomate2siesta-structure supercell Si.cif --min-atoms 50

   # With phonon computation estimates
   atomate2siesta-structure supercell Si.cif --matrix 3 3 3 --show-estimate

**Command Options:**

.. code-block:: bash

   atomate2siesta-structure supercell STRUCTURE_FILE [OPTIONS]

   Options:
     --matrix INT INT INT     Supercell matrix (nx ny nz for diagonal)
     --min-length FLOAT       Minimum supercell length in Å (automatic sizing)
     --min-atoms INTEGER      Minimum number of atoms (automatic sizing)
     --preserve-magmom        Preserve magnetic moments (default: True)
     --show-estimate          Show phonon/NEB computation estimates
     -o, --output PATH        Output file (default: supercell_<input>)
     --format [cif|poscar|xsf|json]  Output format (default: cif)

**Automatic Sizing:**

* **min-length mode**: Calculates nx = ceil(min_length / a), similarly for ny, nz
* **min-atoms mode**: Finds cubic supercell n³ where n³ × N_unit ≥ min_atoms
* **Manual matrix**: Direct specification for precise control

**Computation Estimates** (with ``--show-estimate``):

* **Phonon displacements**: N_atoms × 3 × 2 (±x, ±y, ±z)
* **Estimated time**: ~5 min per displacement (rough estimate)
* **Memory estimate**: ~100 MB per atom for wavefunctions
* **NEB estimates**: 5-7 images, ~10-30 min per image

**Use Cases:**

* **Phonon calculations** - Finite displacement method (typical: 2×2×2 or 3×3×3)
* **Defect studies** - Avoid spurious interactions (typical: 10 Å separation)
* **Surface modeling** - Periodic boundary conditions for slabs
* **Alloying studies** - Create large cells for random substitution

**Technical Details:**

* Diagonal supercell matrices only: [[nx,0,0], [0,ny,0], [0,0,nz]]
* Site properties automatically propagated (magmom, selective_dynamics)
* Pymatgen Structure.make_supercell() backend
* Lattice multipliers: a_new = a × nx, b_new = b × ny, c_new = c × nz
* Volume multiplier: V_new = V_orig × nx × ny × nz

rotate - Structure Rotation
----------------------------

**Purpose:** Rotate structures for surface preparation, alignment, and visualization

**Quick Examples:**

.. code-block:: bash

   # Rotate 45° about z-axis
   atomate2siesta-structure rotate Si.cif --axis z --angle 45

   # Align [111] direction to z-axis
   atomate2siesta-structure rotate Si.cif --align-to-z 1,1,1

   # Euler angle rotation (ZXZ convention)
   atomate2siesta-structure rotate Si.cif --euler 30 45 60

   # Rotate atoms only, keep cell fixed
   atomate2siesta-structure rotate Si.cif --axis z --angle 30 --rotate-atoms-only

   # Show lattice angle changes
   atomate2siesta-structure rotate Si.cif --axis x --angle 90 --show-angles

**Command Options:**

.. code-block:: bash

   atomate2siesta-structure rotate STRUCTURE_FILE [OPTIONS]

   Options:
     --axis [x|y|z]           Rotation axis (use with --angle)
     --angle FLOAT            Rotation angle in degrees (counterclockwise)
     --align-to-x H,K,L       Align Miller index [h,k,l] to x-axis
     --align-to-y H,K,L       Align Miller index [h,k,l] to y-axis
     --align-to-z H,K,L       Align Miller index [h,k,l] to z-axis
     --euler FLOAT FLOAT FLOAT  Euler angles (α, β, γ) in degrees (ZXZ)
     --rotate-cell            Rotate both cell and atoms (default: True)
     --rotate-atoms-only      Rotate only atoms, keep cell fixed
     --show-angles            Show lattice angle changes (α, β, γ)
     -o, --output PATH        Output file (default: rotated_<input>)
     --format [cif|poscar|xsf|json]  Output format (default: cif)

**Rotation Methods:**

* **Axis-angle**: Right-hand rule rotation about Cartesian axis
* **Euler angles**: ZXZ convention (intrinsic rotations)
* **Miller alignment**: Aligns crystallographic direction to Cartesian axis
* **Atoms-only**: Rotates atomic positions, cell unchanged (fractional coords modified)

**Use Cases:**

* **Surface preparation** - Align direction for slab generation (e.g., [111] → z)
* **Grain boundaries** - Controlled misorientation between crystals
* **Visualization** - Standard orientation for publication figures
* **Nanostructure alignment** - Position before stacking/combining

**Technical Details:**

* Uses scipy.spatial.transform.Rotation for robust matrices
* Pymatgen SymmOp for structure transformations
* Volume-preserving rotations
* Site properties preserved during rotation
* Alignment uses Rodrigues rotation formula

translate - Atomic Position Translation
----------------------------------------

**Purpose:** Translate atomic positions for interface alignment, centering, and heterostructure preparation

**Quick Examples:**

.. code-block:: bash

   # Translate all atoms by 2.5 Å in z
   atomate2siesta-structure translate Si.cif --vector 0 0 2.5

   # Fractional translation
   atomate2siesta-structure translate Si.cif --fractional 0 0 0.1

   # Center structure in cell
   atomate2siesta-structure translate Si.cif --center

   # Shift only Cu atoms
   atomate2siesta-structure translate CuO.cif --element Cu --vector 0 0 0.5

   # No wrapping (allow atoms outside cell)
   atomate2siesta-structure translate Si.cif --vector 0 0 5 --no-wrap

   # Show position changes
   atomate2siesta-structure translate Si.cif --fractional 0 0 0.1 --show-before-after

**Command Options:**

.. code-block:: text

   atomate2siesta-structure translate STRUCTURE_FILE [OPTIONS]

   Options:
     --vector FLOAT FLOAT FLOAT      Translation in Cartesian coords (x, y, z in Å)
     --fractional FLOAT FLOAT FLOAT  Translation in fractional coords
     --center                        Center structure (geometric center → [0.5, 0.5, 0.5])
     --element SYMBOL                Translate only this element (e.g., Cu, O)
     --wrap                          Wrap atoms back into cell (default: True)
     --no-wrap                       Do not wrap atoms (allow outside cell)
     --show-before-after             Show positions before/after (first 5 atoms)
     -o, --output PATH               Output file (default: translated_<input>)
     --format [cif|poscar|xsf|json]  Output format (default: cif)

**Translation Modes:**

* **Cartesian**: Direct translation in Å (x, y, z)
* **Fractional**: Translation in crystal coordinates (a, b, c)
* **Centering**: Automatic calculation of shift to center structure
* **Element-selective**: Translate only specified element type

**Wrapping Behavior:**

* **With wrapping** (default): frac_coords = frac_coords - floor(frac_coords) → [0, 1)
* **No wrapping**: Atoms can be outside unit cell (useful for interfaces)

**Use Cases:**

* **Interface alignment** - Position layers correctly for heterostructures
* **Visualization** - Center structure for better viewing
* **Defect studies** - Position dopants/vacancies at specific locations
* **Grain boundaries** - Relative positioning of grains

**Technical Details:**

* Uses pymatgen Structure.translate_sites()
* Centering: shift = [0.5, 0.5, 0.5] - mean(frac_coords)
* Element selection via specie.symbol matching
* Site properties preserved during translation

slab - Surface Slab Generation
--------------------------------

**Purpose:** Generate surface slabs from bulk structures for adsorption and catalysis studies

**Quick Examples:**

.. code-block:: bash

   # Basic (111) surface with 5 layers and 15 Å vacuum
   atomate2siesta-structure slab bulk.cif --miller 1,1,1

   # (100) surface with custom parameters
   atomate2siesta-structure slab bulk.cif --miller 1,0,0 --layers 7 --vacuum 20

   # List all possible terminations first
   atomate2siesta-structure slab bulk.cif --miller 1,1,1 --list-terminations

   # Generate specific termination
   atomate2siesta-structure slab bulk.cif --miller 1,1,1 --termination 1

   # Generate orthogonal cell (for DFT)
   atomate2siesta-structure slab bulk.cif --miller 1,1,1 --orthogonal

   # Generate all low-index surfaces
   atomate2siesta-structure slab bulk.cif --all-surfaces --max-index 1

**Usage:**

.. code-block:: text

   atomate2siesta-structure slab STRUCTURE_FILE [OPTIONS]

**Key Options:**

* ``--miller h,k,l`` - Miller indices (e.g., ``1,1,1`` or ``1,0,0``)
* ``--layers N`` - Minimum number of layers (default: 5)
* ``--vacuum THICKNESS`` - Vacuum thickness in Å (default: 15.0)
* ``--termination N`` - Specific termination index (0-based)
* ``--list-terminations`` - List all possible terminations
* ``--all-terminations`` - Generate all terminations
* ``--all-surfaces`` - Generate all symmetry-unique surfaces
* ``--orthogonal`` - Make slab orthogonal (α=90°, β=90°)
* ``--show-layers`` - Show layer information and positions
* ``--format {cif,poscar,xsf,json,fdf,XV}`` - Output format

**Features:**

* **Miller index specification** - Any crystallographic surface
* **Termination discovery** - Automatic detection of all possible surface cuts
* **Bulk surface generation** - All symmetry-unique surfaces up to max index
* **Orthogonalization** - Creates orthogonal cells for better DFT compatibility
* **Layer analysis** - Layer-by-layer composition and thickness
* **Surface composition** - Top surface species identification

**Use Cases:**

* **Adsorption studies** - Generate surfaces for molecule adsorption
* **Catalysis** - Create catalyst surfaces
* **Surface energy** - Compare different terminations
* **Interface studies** - Prepare substrates for heterostructures

**Technical Details:**

* Uses pymatgen SlabGenerator with automatic termination discovery
* Orthogonalization via Structure.get_orthogonal_c_slab()
* Symmetry analysis with get_symmetrically_distinct_miller_indices()
* Supports FDF and XV formats via sisl integration

vacuum - Vacuum Spacing Control
---------------------------------

**Purpose:** Add vacuum spacing to slabs and 2D materials

**Quick Examples:**

.. code-block:: bash

   # Add 10 Å vacuum with centering
   atomate2siesta-structure vacuum slab.cif --thickness 10.0 --center

   # Add vacuum in a direction
   atomate2siesta-structure vacuum slab.cif --thickness 10.0 --direction a

   # Show layer positions before/after
   atomate2siesta-structure vacuum slab.cif --thickness 10.0 --show-layers

   # Multiple formats
   atomate2siesta-structure vacuum slab.cif --thickness 10.0 --format xsf

**Usage:**

.. code-block:: text

   atomate2siesta-structure vacuum STRUCTURE_FILE [OPTIONS]

**Key Options:**

* ``--thickness ANGSTROM`` - Vacuum thickness to add (required)
* ``--direction {a,b,c}`` - Direction to add vacuum (default: c)
* ``--center`` - Center structure in vacuum space (recommended)
* ``--show-layers`` - Show layer positions before/after
* ``--format {cif,poscar,xsf,json,fdf,XV}`` - Output format
* ``-o OUTPUT`` - Output filename

**Features:**

* **Precise vacuum control** - Add exact thickness in Ångstroms
* **Structure centering** - Automatic centering in vacuum space
* **Multi-direction** - Support for a, b, or c directions
* **Coordinate preservation** - Cartesian coordinates preserved (slab thickness unchanged)
* **Before/after comparison** - Lattice and thickness changes

**Use Cases:**

* **2D materials** - Add vacuum for non-periodic direction
* **Surface calculations** - Ensure sufficient vacuum for DFT
* **Heterostructure preparation** - Adjust vacuum before stacking
* **Convergence testing** - Test vacuum thickness convergence

**Technical Details:**

* Preserves cartesian coordinates while modifying lattice
* Automatically calculates slab thickness and vacuum space
* Creates new Structure with preserved site properties
* Supports all output formats including FDF and XV

stack - Layer Stacking
-----------------------

**Purpose:** Stack layers to create heterostructures and multilayers

**Quick Examples:**

.. code-block:: bash

   # Simple bilayer (same material)
   atomate2siesta-structure stack graphene.cif --spacing 3.35

   # Heterostructure (two materials)
   atomate2siesta-structure stack MoS2.cif WS2.cif --spacing 3.0

   # Multilayer with repetition
   atomate2siesta-structure stack layer.cif --repetitions 5

   # Complex pattern (2x layer1 + 3x layer2)
   atomate2siesta-structure stack layer1.cif layer2.cif --repetitions 2,3

   # Stack along different direction
   atomate2siesta-structure stack slab1.cif slab2.cif --direction a

   # With layer analysis
   atomate2siesta-structure stack layer.cif --spacing 3.0 --show-layers

**Usage:**

.. code-block:: text

   atomate2siesta-structure stack STRUCTURE1 [STRUCTURE2] [OPTIONS]

**Key Options:**

* ``STRUCTURE1`` - First structure file (required)
* ``STRUCTURE2`` - Second structure file (for heterostructures)
* ``--spacing ANGSTROM`` - Spacing between layers (default: 3.0)
* ``--repetitions N`` or ``N,M`` - Repetition pattern
* ``--direction {a,b,c}`` - Stacking direction (default: c)
* ``--center`` - Center stack in cell (default: True)
* ``--show-layers`` - Show layer information after stacking
* ``--format {cif,poscar,xsf,json,fdf,XV}`` - Output format
* ``-o OUTPUT`` - Output filename

**Features:**

* **Multilayer mode** - Repeat same structure (bilayer, trilayer, etc.)
* **Heterostructure mode** - Stack two different materials
* **Flexible patterns** - Custom repetitions (e.g., 2x + 3x)
* **Lattice compatibility** - Warns if mismatch >5%
* **Layer analysis** - Interlayer spacing statistics
* **Automatic centering** - Optional vacuum on both sides

**Use Cases:**

* **Twisted bilayers** - Create bilayer graphene, TMDs
* **van der Waals heterostructures** - MoS2/WS2, graphene/hBN
* **Multilayer films** - Repeated layer patterns
* **Interface studies** - Metal/oxide, semiconductor/insulator

**Technical Details:**

* Preserves in-plane lattice from first structure
* Automatic lattice extension in stacking direction
* Lattice mismatch detection (warns if >5%)
* Interlayer spacing analysis with statistics
* Supports FDF and XV formats via sisl integration

substitute - Atomic Substitution
---------------------------------

**Purpose:** Replace atoms with different elements for doping, alloying, and defect studies

**Quick Examples:**

.. code-block:: bash

   # Complete substitution (all Si → Ge)
   atomate2siesta-structure substitute Si.cif --replace Si:Ge

   # Partial random substitution (25% Si → Ge)
   atomate2siesta-structure substitute Si.cif --replace Si:Ge --fraction 0.25

   # Site-specific substitution
   atomate2siesta-structure substitute Si.cif --replace Si:Ge --sites 0,2,5

   # Multiple random configurations
   atomate2siesta-structure substitute Si.cif --replace Si:Ge --fraction 0.25 --n-configs 5

   # Reproducible random substitution
   atomate2siesta-structure substitute Si.cif --replace Si:Ge --fraction 0.5 --seed 42

**Usage:**

.. code-block:: text

   atomate2siesta-structure substitute STRUCTURE_FILE [OPTIONS]

**Key Options:**

* ``--replace OLD:NEW`` - Element replacement (e.g., ``Si:Ge``, ``Fe:Co``)
* ``--fraction FLOAT`` - Fraction of atoms to substitute (0.0-1.0)
* ``--sites INDICES`` - Comma-separated site indices (0-based)
* ``--n-configs N`` - Number of random configurations (default: 1)
* ``--seed INT`` - Random seed for reproducibility
* ``--format {cif,poscar,xsf,json,fdf,XV}`` - Output format
* ``-o OUTPUT`` - Output filename

**Substitution Modes:**

* **Complete** - Replace all atoms of old element (no ``--fraction`` or ``--sites``)
* **Random partial** - Replace random fraction (``--fraction 0.25``)
* **Site-specific** - Replace specific sites (``--sites 0,2,5``)
* **Multi-configuration** - Generate multiple random configs (``--n-configs 5``)

**Features:**

* **Reproducible randomness** - Use ``--seed`` for deterministic results
* **Site property preservation** - Maintains magnetic moments, etc.
* **Automatic validation** - Checks element exists and indices are valid
* **Before/after comparison** - Shows composition changes

**Use Cases:**

* **Doping studies** - Partial substitution for electronic properties
* **Alloy generation** - Random mixing of elements
* **Defect studies** - Site-specific replacement
* **Statistical sampling** - Multiple random configurations

**Technical Details:**

* Uses random.sample() for unbiased random selection
* Preserves all site properties during substitution
* Automatic file naming: ``substituted_OLD_to_NEW_<input>``
* Multi-config output: ``substituted_OLD_to_NEW_config01_<input>``

remove - Atom Removal
----------------------

**Purpose:** Remove atoms to create vacancies, clean structures, or reduce system size

**Quick Examples:**

.. code-block:: bash

   # Remove all hydrogen atoms
   atomate2siesta-structure remove slab.cif --element H

   # Remove specific sites
   atomate2siesta-structure remove structure.cif --sites 0,5,10

   # Remove atoms near a position (within 2.5 Å)
   atomate2siesta-structure remove structure.cif --near 5.0,5.0,5.0 --radius 2.5

   # Remove using fractional coordinates
   atomate2siesta-structure remove structure.cif --near 0.5,0.5,0.5 --radius 2.5 --fractional

   # Combine element and site removal
   atomate2siesta-structure remove structure.cif --element O --sites 10,11

**Usage:**

.. code-block:: text

   atomate2siesta-structure remove STRUCTURE_FILE [OPTIONS]

**Key Options:**

* ``--element SYMBOL`` - Remove all atoms of this element
* ``--sites INDICES`` - Comma-separated site indices to remove
* ``--near X,Y,Z`` - Remove atoms near position (Cartesian Å or fractional)
* ``--radius FLOAT`` - Radius for proximity-based removal (Å)
* ``--fractional`` - Treat ``--near`` coordinates as fractional
* ``--format {cif,poscar,xsf,json,fdf,XV}`` - Output format
* ``-o OUTPUT`` - Output filename

**Removal Modes:**

* **By element** - Remove all atoms of specified element
* **By sites** - Remove specific site indices
* **By proximity** - Remove atoms within radius of position
* **Combined** - Multiple criteria (union of all matches)

**Features:**

* **Coordinate systems** - Cartesian (Å) or fractional for proximity
* **Multiple criteria** - Combine element, sites, and proximity
* **Automatic validation** - Prevents removing all atoms
* **Before/after statistics** - Shows composition changes

**Use Cases:**

* **Vacancy creation** - Remove atoms for defect studies
* **Adsorbate cleanup** - Remove unwanted molecules from surfaces
* **Proximity cleanup** - Remove atoms near specific positions
* **Structure reduction** - Simplify structures for testing

**Technical Details:**

* Proximity uses ``np.linalg.norm()`` for distance calculation
* Fractional coordinates converted via ``lattice.get_cartesian_coords()``
* Automatic validation prevents empty structures
* Output naming: ``removed_<element/sites/near>_<input>``

add - Add Atoms/Molecules
--------------------------

**Purpose:** Add atoms or molecules to structures for adsorption, doping, and functionalization studies

**Quick Examples:**

.. code-block:: bash

   # Add single oxygen atom at position
   atomate2siesta-structure add structure.cif --atom O --position 5.0,5.0,10.0

   # Add H2O molecule from library
   atomate2siesta-structure add slab.cif --molecule H2O --position 5.0,5.0,15.0

   # Add molecule on top of surface (automatic placement)
   atomate2siesta-structure add slab.cif --molecule H2O --on-top --distance 2.5

   # Add molecule on bottom with custom distance
   atomate2siesta-structure add slab.cif --molecule CO2 --on-bottom --distance 3.0

   # Add with rotation (Euler angles)
   atomate2siesta-structure add slab.cif --molecule H2O --position 5,5,15 --rotate 45,30,0

   # Align molecule to axis
   atomate2siesta-structure add slab.cif --molecule CO2 --position 5,5,15 --align-to z

   # Add molecule from external file
   atomate2siesta-structure add slab.cif --molecule custom.xyz --position 5,5,15

   # Fractional positioning
   atomate2siesta-structure add slab.cif --molecule H2O --position 0.5,0.5,0.9 --fractional

**Usage:**

.. code-block:: text

   atomate2siesta-structure add STRUCTURE_FILE [OPTIONS]

**Key Options:**

* ``--atom SYMBOL`` - Add single atom (e.g., ``O``, ``H``, ``Fe``)
* ``--molecule NAME`` - Add molecule from library or file path
* ``--position X,Y,Z`` - Position in Cartesian (Å) or fractional coords
* ``--fractional`` - Treat position as fractional coordinates
* ``--on-top`` - Place on top of structure (auto z_max + distance)
* ``--on-bottom`` - Place on bottom (auto z_min - distance)
* ``--distance FLOAT`` - Distance from surface for auto placement (Å, default: 2.5)
* ``--rotate ALPHA,BETA,GAMMA`` - Euler angles in degrees (ZYZ convention)
* ``--align-to {x,y,z}`` - Align molecule's principal axis to x/y/z
* ``--format {cif,poscar,xsf,json,fdf,XV}`` - Output format
* ``-o OUTPUT`` - Output filename

**Molecule Library:**

Built-in molecules (accurate bond lengths and angles):

* ``H2`` - Hydrogen (0.74 Å bond)
* ``O2`` - Oxygen (1.21 Å bond)
* ``N2`` - Nitrogen (1.10 Å bond)
* ``H2O`` - Water (0.96 Å OH, 104.5° angle)
* ``CO`` - Carbon monoxide (1.13 Å bond)
* ``CO2`` - Carbon dioxide (1.16 Å CO bonds, linear)
* ``NH3`` - Ammonia (1.01 Å NH, 107° angles, pyramidal)
* ``CH4`` - Methane (1.09 Å CH, tetrahedral)

**Placement Modes:**

* **Manual** - Specify exact Cartesian or fractional position
* **Automatic top** - Place on top (``z_max + distance``) with x-y centering
* **Automatic bottom** - Place on bottom (``z_min - distance``) with x-y centering

**Orientation Control:**

* **Euler rotations** - ZYZ convention: ``--rotate alpha,beta,gamma``
* **Axis alignment** - Align principal axis: ``--align-to {x,y,z}``
* **Combined** - Apply both rotation and alignment sequentially

**Features:**

* **Automatic surface detection** - Finds z_max/z_min and centers in x-y
* **Distance control** - Customizable spacing from surface
* **Molecular orientation** - Full 3D rotation control (Euler + alignment)
* **External molecules** - Load from XYZ, CIF, or other structure files
* **Site property preservation** - Maintains existing properties

**Use Cases:**

* **Adsorption studies** - Add molecules to surfaces with controlled orientation
* **Doping** - Insert atoms at specific positions
* **Functionalization** - Add chemical groups to surfaces
* **Molecule-surface interaction** - Study binding geometries

**Technical Details:**

* Euler rotation matrix: R = Rz(γ) · Ry(β) · Rz(α) (ZYZ convention)
* Axis alignment uses Rodrigues' rotation formula
* Automatic centering: ``x = (x_max + x_min)/2``, ``y = (y_max + y_min)/2``
* Molecule center preserved during rotation/alignment
* External molecules loaded via pymatgen Structure.from_file()
* Output naming: ``added_<atom/molecule>_<input>``

perturb - Random Perturbations
--------------------------------

**Purpose:** Apply random displacements for MD initialization and transition state searches

**Quick Examples:**

.. code-block:: bash

   # Uniform random displacement (0.1 Å amplitude)
   atomate2siesta-structure perturb structure.cif --amplitude 0.1

   # Thermal displacement (300 K, Maxwell-Boltzmann)
   atomate2siesta-structure perturb structure.cif --temperature 300

   # Perturb only specific element
   atomate2siesta-structure perturb structure.cif --amplitude 0.1 --element Si

   # Generate multiple configurations
   atomate2siesta-structure perturb structure.cif --amplitude 0.1 --n-configs 10

   # Reproducible perturbations
   atomate2siesta-structure perturb structure.cif --amplitude 0.1 --seed 42

   # Show displacement statistics
   atomate2siesta-structure perturb structure.cif --temperature 300 --show-stats

**Usage:**

.. code-block:: text

   atomate2siesta-structure perturb STRUCTURE_FILE [OPTIONS]

**Key Options:**

* ``--amplitude FLOAT`` - Uniform random displacement amplitude (Å)
* ``--temperature FLOAT`` - Temperature for thermal displacements (K)
* ``--element SYMBOL`` - Only perturb atoms of this element
* ``--n-configs N`` - Number of configurations to generate (default: 1)
* ``--seed INT`` - Random seed for reproducibility
* ``--show-stats`` - Show displacement statistics (RMS, mean, etc.)
* ``--format {cif,poscar,xsf,json,fdf,XV}`` - Output format
* ``-o OUTPUT`` - Output filename

**Perturbation Modes:**

* **Uniform random** - Isotropic random displacements with uniform amplitude

  * Direction: Uniform on sphere (θ ∈ [0, 2π], cos φ ∈ [-1, 1])
  * Magnitude: Uniform in [0, amplitude]

* **Thermal (Maxwell-Boltzmann)** - Temperature-based displacements

  * Per-component Gaussian: σ = √(k_B·T/m)
  * Mass-dependent: Lighter atoms move more
  * k_B = 8.617×10⁻⁵ eV/K

**Features:**

* **Two perturbation types** - Uniform random or thermal (Maxwell-Boltzmann)
* **Element-selective** - Perturb only specified elements
* **Multi-configuration** - Generate ensembles for sampling
* **Reproducible** - Use ``--seed`` for deterministic results
* **Statistics** - RMS, mean, percentiles of displacements
* **Mass-dependent thermal** - Realistic temperature effects

**Use Cases:**

* **MD initialization** - Generate starting configurations at temperature
* **Transition state search** - Perturb around saddle points
* **Statistical sampling** - Generate ensemble of structures
* **Phonon pre-conditioning** - Small displacements for finite differences

**Technical Details:**

* Uniform: Random direction (uniform on sphere) + random magnitude
* Thermal: Per-axis Gaussian with σ = √(k_B·T/m) × 0.1 (unit conversion)
* Element selection via ``site.specie.symbol`` matching
* Displacement statistics: RMS, mean, std, 50th/90th/99th percentiles
* Automatic naming: ``perturbed_amp<X>_<input>`` or ``perturbed_<T>K_<input>``
* Multi-config: ``perturbed_<mode>_config01_<input>``

compare - Structure Comparison
--------------------------------

**Purpose:** Quantitatively compare two crystal structures

**Quick Examples:**

.. code-block:: bash

   # Basic comparison
   atomate2siesta-structure compare struct1.cif struct2.cif

   # With custom tolerance
   atomate2siesta-structure compare struct1.cif struct2.cif --tolerance 0.1

   # Lattice only (skip site comparison)
   atomate2siesta-structure compare struct1.cif struct2.cif --no-compare-sites

   # Detailed site-by-site comparison
   atomate2siesta-structure compare struct1.cif struct2.cif --verbose

   # Skip RMSD calculation
   atomate2siesta-structure compare struct1.cif struct2.cif --no-calculate-rmsd

**Usage:**

.. code-block:: text

   atomate2siesta-structure compare STRUCTURE1 STRUCTURE2 [OPTIONS]

**Key Options:**

* ``STRUCTURE1`` - First structure file (required)
* ``STRUCTURE2`` - Second structure file (required)
* ``--tolerance FLOAT`` - Tolerance for structure matching (Å, default: 0.01)
* ``--compare-lattice / --no-compare-lattice`` - Compare lattice parameters (default: True)
* ``--compare-sites / --no-compare-sites`` - Compare atomic sites (default: True)
* ``--calculate-rmsd / --no-calculate-rmsd`` - Calculate RMSD (default: True)
* ``--verbose`` - Show detailed site-by-site comparison

**Comparison Categories:**

* **Basic Properties**

  * Number of sites
  * Chemical formula
  * Reduced formula

* **Lattice Parameters**

  * Lattice lengths (a, b, c)
  * Lattice angles (α, β, γ)
  * Volume

* **Atomic Composition**

  * Element-by-element comparison
  * Counts and percentages

* **Site Matching**

  * Automatic site-by-site matching
  * Considers periodic boundary conditions
  * Reports matched/unmatched sites
  * Verbose mode shows detailed site-by-site comparison table

* **RMSD Analysis**

  * Root-mean-square deviation
  * Centered coordinate comparison
  * Automatic site matching by element
  * Tolerance-based pass/fail

**Features:**

* **Tolerance-based matching** - Configurable precision for lattice and site comparison
* **Automatic site pairing** - Finds closest matching sites across periodic boundaries
* **Rich terminal output** - Color-coded tables with match indicators (✓/✗)
* **RMSD calculation** - Centered coordinates with automatic element matching
* **Verbose mode** - Comprehensive site-by-site table showing:

  * All sites (matched and unmatched)
  * Fractional coordinates for both structures
  * Distance between each site pair (Å)
  * Color-coded match status (green/yellow/red)

**Use Cases:**

* **Optimization verification** - Compare before/after optimization
* **Symmetry analysis** - Check standardization results
* **Format conversion** - Verify structure preservation across formats
* **Calculation validation** - Compare input/output structures

**Technical Details:**

* Site matching uses ``np.linalg.norm()`` with periodic boundary conditions
* RMSD uses centered coordinates: ``coords - coords.mean(axis=0)``
* Element matching via ``site.specie`` comparison
* Tolerance applies to both lattice and site distances

standardize - Cell Standardization
------------------------------------

**Purpose:** Convert structures to conventional, primitive, or international standard cells

**Quick Examples:**

.. code-block:: bash

   # Convert to conventional cell
   atomate2siesta-structure standardize structure.cif --conventional

   # Convert to primitive cell
   atomate2siesta-structure standardize structure.cif --primitive

   # International standard setting
   atomate2siesta-structure standardize structure.cif --international

   # Custom symmetry precision
   atomate2siesta-structure standardize structure.cif --primitive --symprec 0.1

   # With before/after comparison
   atomate2siesta-structure standardize structure.cif --conventional --show-before-after

**Usage:**

.. code-block:: text

   atomate2siesta-structure standardize STRUCTURE_FILE [OPTIONS]

**Key Options:**

* ``--conventional`` - Convert to conventional cell (standard crystallographic cell)
* ``--primitive`` - Convert to primitive cell (smallest repeating unit)
* ``--international`` - Use international standard setting
* ``--symprec FLOAT`` - Symmetry precision (Å, default: 0.01)
* ``--angle-tolerance FLOAT`` - Angle tolerance (degrees, default: 5.0)
* ``--show-before-after`` - Show before/after comparison table
* ``-o, --output PATH`` - Output filename (default: <mode>_<input>)
* ``--format {cif,poscar,xsf,json,fdf,XV}`` - Output format (default: cif)

**Standardization Modes:**

* **Conventional Cell**

  * Standard crystallographic cell
  * Larger than primitive (more symmetric)
  * Used in crystallographic databases
  * Generated via ``SpacegroupAnalyzer.get_conventional_standard_structure()``

* **Primitive Cell**

  * Smallest repeating unit
  * Minimum number of atoms
  * Most efficient for calculations
  * Generated via ``SpacegroupAnalyzer.get_primitive_standard_structure()``

* **International Setting**

  * Refined structure in standard orientation
  * Follows International Tables conventions
  * Generated via ``SpacegroupAnalyzer.get_refined_structure()``

**Features:**

* **Symmetry analysis** - Displays space group, crystal system, point group
* **Before/after comparison** - Shows lattice changes and site count
* **Multi-format output** - CIF, POSCAR, XSF, JSON, FDF, XV
* **Configurable precision** - Symmetry tolerance and angle tolerance
* **Automatic validation** - Ensures only one mode selected

**Use Cases:**

* **Database comparison** - Standardize for consistent comparison
* **Calculation efficiency** - Use primitive cell for DFT
* **Symmetry analysis** - Convert to conventional for visualization
* **Literature matching** - International standard for publications

**Technical Details:**

* Uses pymatgen ``SpacegroupAnalyzer`` with spglib backend
* Symmetry precision controls spglib's ``symprec`` parameter
* Angle tolerance for lattice angle symmetry detection
* Site properties preserved during standardization
* Output naming: ``<mode>_<basename>.<ext>``

optimize-cell - Cell Shape Optimization
-----------------------------------------

**Purpose:** Optimize cell shape for better periodic calculations

**Quick Examples:**

.. code-block:: bash

   # Niggli reduction
   atomate2siesta-structure optimize-cell structure.cif --niggli

   # Find orthogonal cell
   atomate2siesta-structure optimize-cell structure.cif --orthogonalize

   # Orthogonalize with custom max atoms
   atomate2siesta-structure optimize-cell structure.cif --orthogonalize --max-atoms 500

   # With before/after comparison
   atomate2siesta-structure optimize-cell structure.cif --niggli --show-before-after

**Usage:**

.. code-block:: text

   atomate2siesta-structure optimize-cell STRUCTURE_FILE [OPTIONS]

**Key Options:**

* ``--niggli`` - Apply Niggli reduction (find most reduced cell)
* ``--orthogonalize`` - Find most orthogonal supercell
* ``--max-atoms INT`` - Maximum atoms for orthogonalization (default: 1000)
* ``--show-before-after`` - Show before/after comparison table
* ``-o, --output PATH`` - Output filename (default: <mode>_<input>)
* ``--format {cif,poscar,xsf,json,fdf,XV}`` - Output format (default: cif)

**Optimization Modes:**

* **Niggli Reduction**

  * Finds most reduced lattice representation
  * Unique reduced form for each lattice
  * Shortest lattice vectors
  * Follows Niggli reduction algorithm
  * Via ``lattice.get_niggli_reduced_lattice()``

* **Orthogonalization**

  * Finds most orthogonal supercell
  * Better for DFT calculations (FFT efficiency)
  * Searches supercells up to max_atoms limit
  * Orthogonality score: Σ(angle - 90°)²
  * Lower score = more orthogonal

**Features:**

* **Automatic search** - Tries multiple supercell matrices (up to 5×5×5)
* **Orthogonality scoring** - Quantitative measure of cell orthogonality
* **Size control** - Limit maximum atoms to avoid huge cells
* **Before/after tables** - Shows lattice changes and orthogonality improvement
* **Smart fallback** - Returns original if no better cell found

**Use Cases:**

* **DFT efficiency** - Orthogonal cells for faster FFT
* **Unique representation** - Niggli cell for database comparison
* **Calculation speedup** - More orthogonal cells converge faster
* **Standard form** - Reduced cells for consistent analysis

**Technical Details:**

* Niggli reduction via pymatgen's ``get_niggli_reduced_lattice()``
* Orthogonalization: exhaustive supercell search with scoring
* Orthogonality score: ``sum((angle - 90)² for angle in [α, β, γ])``
* Search space: all integer matrices [a,b,c] where a×b×c×sites ≤ max_atoms
* Best cell selected by minimum orthogonality score
* Output naming: ``<mode>_<basename>.<ext>``

**Status:** ✅ Complete - All 16 commands production-ready

.. _convert-cli:

Structure Format Conversion CLI
================================

**Command:** ``atomate2siesta-structure convert``

**Purpose:** Convert structure formats between sisl, pymatgen, and ASE with automatic format detection

Key Features
------------

* **Automatic format detection** - Detects input format from file extension (NEW!)
* **Multi-format support** - FDF, XV, CIF, XSF, JSON, pickle
* **Ghost atom handling** - Separate structures with/without ghost atoms
* **Rich terminal output** - Formatted tables with structure information
* **Flexible conversion** - Multiple output formats in one command
* **Metadata extraction** - Species information, atomic numbers, coordinates

Supported Formats
-----------------

**Input (automatic detection):**

* ``.fdf`` - SIESTA FDF input file
* ``.xv`` / ``.XV`` - SIESTA XV restart geometry file
* ``.cif`` - Crystallographic Information File (NEW!)
* ``.xsf`` - XCrySDen Structure File (NEW!)

**Output:**

* CIF - Crystallographic Information File (pymatgen)
* XSF - XCrySDen format (ASE)
* JSON - Structure data with metadata
* Pickle - Python objects (sisl/ASE/pymatgen)
* FDF - SIESTA input (with or without ghost atoms)

Quick Examples
--------------

.. code-block:: bash

   # Automatic format detection - FDF to CIF
   atomate2siesta-structure convert input.fdf --write-cif

   # Automatic format detection - XV to XSF
   atomate2siesta-structure convert structure.XV --write-xsf

   # NEW! CIF to SIESTA FDF (automatic detection)
   atomate2siesta-structure convert structure.cif --write-fdf

   # NEW! XSF to multiple formats (automatic detection)
   atomate2siesta-structure convert structure.xsf --write-cif --write-json

   # Write all formats from FDF
   atomate2siesta-structure convert input.fdf \
       --write-cif \
       --write-xsf \
       --write-json \
       --write-pymatgen-pickle

   # Custom output prefix
   atomate2siesta-structure convert input.fdf --write-cif --output-prefix my_structure
   # Creates: my_structure_no_ghost.cif

   # Write FDF with and without ghost atoms
   atomate2siesta-structure convert input.fdf \
       --write-fdf \
       --write-fdf-no-ghost \
       --output-prefix converted

Command Options
---------------

.. code-block:: bash

   atomate2siesta-structure convert INPUT_FILE [OPTIONS]

   Arguments:
     INPUT_FILE                  Path to structure file (.fdf, .xv, .XV, .cif, or .xsf)
                                 Format automatically detected from extension

   Options:
     --write-xsf                 Write ASE structure to XSF file
     --write-cif                 Write pymatgen structure to CIF file
     --write-json                Write structure data to JSON file
     --write-sisl-pickle         Write sisl structure to pickle
     --write-ase-pickle          Write ASE structures to pickle
     --write-pymatgen-pickle     Write pymatgen structures to pickle
     --write-fdf                 Write sisl structure (with ghost) to FDF
     --write-fdf-no-ghost        Write sisl structure (no ghost) to FDF
     --output-prefix TEXT        Prefix for output files (default: structure)

.. note::
   **Note:** The ``--xv`` flag has been removed. File format is now automatically detected from the file extension.

Generated Outputs
-----------------

Depending on options, creates files with pattern ``<prefix>.<extension>``:

* ``<prefix>.cif`` - Pymatgen structure (no ghost)
* ``<prefix>.xsf`` - ASE structure (no ghost)
* ``<prefix>.json`` - Structure metadata and coordinates
* ``<prefix>_sisl.pkl`` - Sisl geometry object
* ``<prefix>_ase.pkl`` / ``<prefix>_ase_with_ghost.pkl`` - ASE Atoms objects
* ``<prefix>_pymatgen.pkl`` / ``<prefix>_pymatgen_with_ghost.pkl`` - Pymatgen Structure objects
* ``<prefix>.fdf`` / ``<prefix>_no_ghost.fdf`` - SIESTA input files

Terminal Output
---------------

The tool displays formatted tables with:

* Species information (symbol, atomic number, mass)
* Structure statistics (atoms, lattice parameters)
* Ghost atom identification
* Output file paths

**Status:** ✅ Production-ready

.. _inputs-cli:

Data Classes Information CLI
=============================

**Command:** ``atomate2siesta-inputs``

**Purpose:** Explore SIESTA input dataclass structures and parameters

Key Features
------------

* **List all dataclasses** - View all 24 SIESTA input parameter dataclasses
* **Search by keyword** - Find parameters by name, type, or category
* **Show detailed info** - View complete parameter documentation and defaults
* **Rich terminal UI** - Formatted tables with clear organization

Available Commands
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - ``list``
     - List all available data classes
   * - ``search``
     - Search for data class attributes by keyword
   * - ``show``
     - Show detailed information about a specific data class

Quick Examples
--------------

.. code-block:: bash

   # List all available dataclasses
   atomate2siesta-inputs list

   # Search for parameters related to k-points
   atomate2siesta-inputs search kpt

   # Show details of a specific dataclass
   atomate2siesta-inputs show BasisParameters

**Status:** ✅ Production-ready

.. note::
   For generating SIESTA input files directly, use the ``atomate2siesta-maker`` CLI tool
   instead. See :ref:`workflow-generator-cli`.

Complete Workflow Example
==========================

Here's how all CLI tools work together:

Step 1: Generate Workflow Script
----------------------------------

.. code-block:: bash

   # Generate a relaxation workflow script
   atomate2siesta-maker relax Si.cif --preset relax_standard

   # This creates relax_Si.py - inspect before running
   cat relax_Si.py

   # Test with dry-run first
   atomate2siesta-maker relax Si.cif --dry-run
   python relax_Si.py  # Generates inputs without running SIESTA

Step 2: Set Up Remote Cluster
------------------------------

.. code-block:: bash

   # Configure remote HPC cluster with SIESTA
   atomate2siesta-cluster setup \
       --host cluster.university.edu \
       --user myuser \
       -i ~/.ssh/id_rsa \
       --git-ssh \
       --install-siesta

   # Verify setup
   atomate2siesta-cluster status --host cluster.university.edu

Step 3: Configure Jobflow-Remote
---------------------------------

.. code-block:: bash

   # Install jobflow-remote
   atomate2siesta-jobflow-remote install

   # Generate configuration
   atomate2siesta-jobflow-remote setup --project-name hpc_project

   # Add inline documentation
   atomate2siesta-jobflow-remote update --add-comments

   # Test the setup
   atomate2siesta-jobflow-remote test

Step 4: Set Up Database
------------------------

.. code-block:: bash

   # Test MongoDB connection
   atomate2siesta-database test

   # Show configuration examples
   atomate2siesta-database config

   # Monitor calculations
   atomate2siesta-database list --limit 10

Step 5: Submit Calculations
----------------------------

On remote cluster:

.. code-block:: bash

   ssh cluster.university.edu
   conda activate jobflow-remote
   jf runner start -d  # Start runner daemon

On local machine (using maker CLI):

.. code-block:: bash

   # Generate script for remote submission
   atomate2siesta-maker relax Si.cif --remote --worker hpc_cluster

   # Run the generated script (submits to remote)
   python relax_Si.py

Step 6: Monitor Results
------------------------

.. code-block:: bash

   # Check job status on cluster
   jf job list
   jf job info <job_id>

   # Query results in database
   atomate2siesta-database query Si
   atomate2siesta-database stats

Integration with Tutorials
===========================

The CLI tools complement the tutorial series:

* **Tutorial 13**: Database storage configuration and usage
* **Tutorial 14**: HPC job submission workflows

Command Reference Summary
=========================

Workflow Generator
------------------

.. code-block:: bash

   atomate2siesta-maker list
   atomate2siesta-maker <workflow> <structure> [OPTIONS]
   atomate2siesta-maker neb <initial> <final> [OPTIONS]

Common workflows: relax, static, bands, dos, phonon, gruneisen, qha,
                  eos, elastic, bulk-modulus, neb, surface, adsorption

Cluster Setup
-------------

.. code-block:: bash

   atomate2siesta-cluster setup --host <cluster> [OPTIONS]
   atomate2siesta-cluster status --host <cluster>
   atomate2siesta-cluster info

Jobflow-Remote
--------------

.. code-block:: bash

   atomate2siesta-jobflow-remote install [--dev]
   atomate2siesta-jobflow-remote setup [--project-name NAME] [OPTIONS]
   atomate2siesta-jobflow-remote update [--add-comments] [OPTIONS]
   atomate2siesta-jobflow-remote info [--project-name NAME]
   atomate2siesta-jobflow-remote test
   atomate2siesta-jobflow-remote runner

Database
--------

.. code-block:: bash

   atomate2siesta-database test [--host HOST] [--port PORT]
   atomate2siesta-database list [--limit N]
   atomate2siesta-database query <formula>
   atomate2siesta-database stats
   atomate2siesta-database clear [--force]
   atomate2siesta-database config

Common Options
==============

Authentication (Cluster Setup)
------------------------------

.. code-block:: bash

   --ssh-config              # Use SSH config alias
   -i, --identity-file PATH  # SSH private key
   --password                # Prompt for password

Database Connection
-------------------

.. code-block:: bash

   --host TEXT      # MongoDB host (default: localhost)
   --port INTEGER   # MongoDB port (default: 27017)
   --database TEXT  # Database name (default: atomate2siesta)

Project Management
------------------

.. code-block:: bash

   --project-name TEXT  # Project name for jobflow-remote
   --update             # Update existing configuration
   --backup             # Create backup before changes (default: True)

Tips and Best Practices
========================

1. **Start with Cluster Setup**
   Configure your remote cluster first before setting up jobflow-remote

2. **Test Connections First**
   Always use ``test`` commands to verify connectivity before operations

3. **Use SSH Config**
   Set up ``~/.ssh/config`` for easier cluster access:

   .. code-block:: text

      Host mycluster
          HostName cluster.university.edu
          User myuser
          IdentityFile ~/.ssh/id_rsa

4. **Enable Backups**
   Keep automatic backups enabled when updating configurations

5. **Add Comments**
   Use ``--add-comments`` to make configurations self-documenting

6. **Regular Monitoring**
   Use database ``stats`` command to monitor growth and performance

7. **Test Before Production**
   Use test jobs to verify setup before submitting real calculations

Troubleshooting
===============

Connection Issues
-----------------

.. code-block:: bash

   # Test SSH connection manually
   ssh user@cluster

   # Test MongoDB connection
   atomate2siesta-database test

   # Verify jobflow-remote setup
   jf project check --errors

Installation Problems
---------------------

.. code-block:: bash

   # Verify conda installation on cluster
   atomate2siesta-cluster status --host cluster

   # Check jobflow-remote installation
   atomate2siesta-jobflow-remote info

   # Reinstall if needed
   atomate2siesta-jobflow-remote install --dev

Configuration Updates
---------------------

.. code-block:: bash

   # Restore from backup if update fails
   cp ~/.jfremote/project.backup_TIMESTAMP.yaml \
      ~/.jfremote/project.yaml

   # Verify configuration
   jf project check --errors

Getting Help
============

All commands provide comprehensive help:

.. code-block:: bash

   atomate2siesta-cluster --help
   atomate2siesta-jobflow-remote --help
   atomate2siesta-database --help

For detailed documentation, see the individual CLI reference pages:

* :doc:`cli-cluster-setup` - Full cluster setup documentation
* :doc:`cli-jobflow-remote` - Complete jobflow-remote guide
* :doc:`cli-database` - Database management reference

Development History
===================

**October 2025**: Cluster Setup Enhancements

* Added SIESTA installation from conda-forge
* Automatic ``.atomate2.yaml`` configuration generation
* Enhanced error handling with manual fallback instructions

**October 2025**: CLI Tools Introduction

* Database management CLI with 7 commands
* Jobflow-remote setup CLI with 5 commands
* Rich terminal UI implementation
* Production-ready with comprehensive testing

See Also
========

* :doc:`installation` - Installation guide
* :doc:`usage` - Basic usage patterns
* :doc:`tutorials/index` - Tutorial series
* `Jobflow-Remote Documentation <https://matgenix.github.io/jobflow-remote/>`_
* `MongoDB Documentation <https://docs.mongodb.com/>`_
