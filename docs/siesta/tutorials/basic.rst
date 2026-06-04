====================
Basic Learning Paths
====================

Introduction to fundamental atomate2siesta workflow concepts and simple calculations.

----

Tutorial 01: Relaxation Basics
===============================

**Learning Objectives**:

* Understand the Maker pattern
* Create and run a simple relaxation workflow
* Access results from calculations

**Key Concepts**:

* ``RelaxMaker`` class
* Fixed vs. variable cell relaxation
* ``run_locally()`` function
* Result schemas

**Example**:

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker
   from pymatgen.core import Structure
   from jobflow import run_locally

   # Load structure
   structure = Structure.from_file("Si.cif")

   # Create maker
   maker = RelaxMaker.fixed_cell_relaxation()

   # Generate and run job
   job = maker.make(structure)
   results = run_locally(job, create_folders=True)

   # Access results
   print(f"Final energy: {results.output.output.energy} eV")

📁 **Location**: ``tutorials/01-basics/01-relaxation/``

⏱️ **Time**: 10 minutes

⭐ **Difficulty**: Beginner

----

Tutorial 02: Relaxation Parameters
===================================

**Learning Objectives**:

* Customize SIESTA parameters using ``user_params``
* Understand common input parameters
* Use powerups to modify workflows

**Key Concepts**:

* ``user_params`` dictionary
* PAO basis sets (SZ, DZ, DZP, TZP)
* K-point sampling
* Mesh cutoff
* ``update_user_siesta_settings()`` powerup

**Example**:

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker
   from atomate2.siesta.powerups import update_user_siesta_settings

   # Method 1: Direct user_params
   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "PAO.BasisSize": "DZP",
           "a2s_kpts": [4, 4, 4],
           "Mesh.Cutoff": "300 Ry",
       }
   )

   # Method 2: Using powerups
   job = maker.make(structure)
   job = update_user_siesta_settings(job, {
       "SCF.Mixer.Weight": 0.01,
   })

📁 **Location**: ``tutorials/01-basics/02-relaxation-parameters/``

⏱️ **Time**: 15 minutes

⭐ **Difficulty**: Beginner

----

Tutorial 03: Band Structure
============================

**Learning Objectives**:

* Calculate electronic band structures
* Understand band gap determination
* Visualize electronic properties

**Key Concepts**:

* ``BandStructureMaker`` class
* High-symmetry k-path generation
* SCF + non-SCF workflow pattern
* Band gap analysis

**Example**:

.. code-block:: python

   from atomate2.siesta.jobs.core import BandStructureMaker

   maker = BandStructureMaker(
       user_params={
           "PAO.BasisSize": "DZP",
           "a2s_kpts": [6, 6, 6],  # SCF k-points
       }
   )

   flow = maker.make(structure)
   results = run_locally(flow, create_folders=True)

   # Access band gap
   print(f"Band gap: {results.output.band_gap} eV")

📁 **Location**: ``tutorials/01-basics/03-band-structure/``

⏱️ **Time**: 20 minutes

⭐ **Difficulty**: Intermediate

----

Tutorial 04: Lua Scripts
=========================

**Learning Objectives**:

* Use Lua scripting for advanced SIESTA features
* Implement custom constraints
* Access FLOS library capabilities

**Key Concepts**:

* Lua scripting in SIESTA
* FLOS library (Forces, Lattice, Optimization, Structure)
* Custom geometry constraints
* Variable cell relaxation with custom stress control

**Example**:

.. code-block:: python

   lua_script = """
   -- Custom Lua script for constrained relaxation
   local flos = require "flos"

   function siesta_step(siesta)
       -- Custom force/stress manipulation
       local forces = siesta.forces
       -- ... Lua logic ...
   end
   """

   maker = RelaxMaker.variable_cell_relaxation(
       user_params={
           "MD.TypeOfRun": "Lua",
           "LUA.Script": lua_script,
       }
   )

📁 **Location**: ``tutorials/01-basics/04-lua-scripts/``

⏱️ **Time**: 30 minutes

⭐ **Difficulty**: Intermediate

----

Tutorial 05: Multi-Step Workflows
==================================

**Learning Objectives**:

* Compose multi-step workflows
* Pass outputs between jobs
* Use Flow for complex calculations

**Key Concepts**:

* ``Flow`` composition
* Job dependencies
* Output passing (``prev_dir``)
* Workflow orchestration

**Example**:

.. code-block:: python

   from jobflow import Flow
   from atomate2.siesta.jobs.core import RelaxMaker, BandStructureMaker

   # Create jobs
   relax = RelaxMaker.variable_cell_relaxation().make(structure)
   bands = BandStructureMaker().make(relax.output.structure)

   # Compose workflow
   flow = Flow([relax, bands])
   results = run_locally(flow, create_folders=True)

📁 **Location**: ``tutorials/01-basics/05-workflows/``

⏱️ **Time**: 25 minutes

⭐ **Difficulty**: Intermediate

----

Next Steps
==========

After completing the basic tutorials:

1. **Convergence Studies** (Tutorials 06-08) - Learn to optimize parameters
2. **Advanced Workflows** (Tutorials 09-18) - Complex multi-step calculations
3. **Infrastructure** (Tutorials 13-15) - Production setup

See :doc:`index` for full tutorial listing.
