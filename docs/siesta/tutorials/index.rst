==============
Learning Paths
==============

Comprehensive learning paths for mastering atomate2siesta workflows, from basic calculations to advanced
convergence studies and production workflows.

----

📚 Tutorial Organization
========================

The tutorials are organized into four categories based on complexity and topic:

* 🎯 **Basic Tutorials (01-05)**: :doc:`basic` - Fundamental concepts and simple calculations
* 📊 **Convergence Studies (06-08)**: :doc:`convergence` - Systematic parameter optimization
* 🚀 **Advanced Workflows (09-27)**: :doc:`advanced` - Complex multi-step calculations and advanced features
* ⚙️ **Infrastructure (13-15)**: :doc:`infrastructure` - Database, HPC, error handling

----

Quick Start Example
===================

All tutorials follow this consistent pattern:

.. code-block:: python

   from pymatgen.core import Structure
   from atomate2.siesta.jobs.core import RelaxMaker
   from jobflow import run_locally

   # 1. Create or load a structure
   structure = Structure.from_file("material.cif")

   # 2. Create a maker with settings
   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "PAO.BasisSize": "DZP",
           "a2s_kpts": [4, 4, 4],
           "Mesh.Cutoff": "300 Ry",
       }
   )

   # 3. Generate the job
   job = maker.make(structure)

   # 4. Run locally
   results = run_locally(job, create_folders=True)

   # 5. Access results
   print(f"Final energy: {results.output.output.energy} eV")

----

Complete Tutorial Listing
=========================

🎯 Basic Tutorials (01-05)
--------------------------

Introduction to fundamental workflow concepts and simple calculations.

.. list-table::
   :header-rows: 1
   :widths: 15 50 20 15

   * - Tutorial
     - Description
     - Time
     - Difficulty
   * - **01**
     - :doc:`/siesta/tutorials-md/01-basics/01-RelaxMaker/README` - Relaxation basics with default settings
     - 10 min
     - ⭐
   * - **02**
     - :doc:`/siesta/tutorials-md/01-basics/02-BandStructureMaker/README` - Band structure calculations
     - 15 min
     - ⭐
   * - **03**
     - :doc:`/siesta/tutorials-md/01-basics/03-LuaMaker/README` - Lua scripts for advanced features
     - 20 min
     - ⭐⭐
   * - **04**
     - :doc:`/siesta/tutorials-md/01-basics/04-RelaxMaker-StaticMaker/README` - Multi-step workflows
     - 30 min
     - ⭐⭐
   * - **05**
     - :doc:`/siesta/tutorials-md/01-basics/05-DOSMaker/README` - DOS calculations
     - 25 min
     - ⭐⭐

📊 Convergence Studies (06-08)
------------------------------

Systematic parameter optimization for production calculations.

.. list-table::
   :header-rows: 1
   :widths: 15 50 20 15

   * - Tutorial
     - Description
     - Time
     - Difficulty
   * - **06**
     - :doc:`/siesta/tutorials-md/02-workflows/01-convergence/README` - K-points and mesh cutoff convergence
     - 45 min
     - ⭐⭐
   * - **07**
     - :doc:`/siesta/tutorials-md/01-basics/07-PhonopyMaker/README` - Phonon calculations with Phonopy
     - 60 min
     - ⭐⭐⭐
   * - **08**
     - Complete basis set convergence (size, shift, norm)
     - 90 min
     - ⭐⭐⭐

🚀 Advanced Workflows (09-27)
-----------------------------

Complex multi-step calculations for materials properties and advanced features.

**Structural & Mechanical (09-12)**

.. list-table::
   :header-rows: 1
   :widths: 15 50 20 15

   * - Tutorial
     - Description
     - Time
     - Difficulty
   * - **09**
     - Equation of State (EOS) calculations
     - 40 min
     - ⭐⭐
   * - **10**
     - EOS with basis parameter convergence
     - 90 min
     - ⭐⭐⭐
   * - **11**
     - Elastic constants and mechanical properties
     - 60 min
     - ⭐⭐⭐
   * - **12**
     - Nudged Elastic Band (NEB) for transition states
     - 120 min
     - ⭐⭐⭐⭐

**Vibrational & Thermal Properties (16, 20-21)**

.. list-table::
   :header-rows: 1
   :widths: 15 50 20 15

   * - Tutorial
     - Description
     - Time
     - Difficulty
   * - **16**
     - **NEW** Phonon calculations with phonopy integration
     - 60 min
     - ⭐⭐⭐
   * - **20**
     - **NEW** Grüneisen parameters and thermal expansion
     - 90 min
     - ⭐⭐⭐⭐
   * - **21**
     - **NEW** Quasi-harmonic approximation (QHA) thermodynamics
     - 120 min
     - ⭐⭐⭐⭐

**Surface & Adsorption (17, 19)**

.. list-table::
   :header-rows: 1
   :widths: 15 50 20 15

   * - Tutorial
     - Description
     - Time
     - Difficulty
   * - **17**
     - **NEW** Surface energy calculations
     - 90 min
     - ⭐⭐⭐⭐
   * - **19**
     - **NEW** Adsorption site scanning
     - 75 min
     - ⭐⭐⭐

**Advanced Configuration (18, 22)**

.. list-table::
   :header-rows: 1
   :widths: 15 50 20 15

   * - Tutorial
     - Description
     - Time
     - Difficulty
   * - **18**
     - **NEW** Tier-based calculations and presets
     - 30 min
     - ⭐⭐
   * - **22**
     - **NEW** Powerups - Advanced workflow customization
     - 45 min
     - ⭐⭐

**Advanced Features (23-27)** ⭐ NEW

.. list-table::
   :header-rows: 1
   :widths: 15 50 20 15

   * - Tutorial
     - Description
     - Time
     - Difficulty
   * - **23**
     - **NEW** DOS calculations with direct SIESTA FDF format
     - 20 min
     - ⭐⭐
   * - **24**
     - **NEW** Phonon inputs (force constants parameters)
     - 15 min
     - ⭐⭐
   * - **25**
     - **NEW** Optical properties (absorption, dielectric)
     - 25 min
     - ⭐⭐
   * - **26**
     - **NEW** DFT+U calculations for correlated systems
     - 30 min
     - ⭐⭐⭐
   * - **27**
     - **NEW** Charge/dipole/electric field calculations
     - 25 min
     - ⭐⭐⭐

⚙️ Infrastructure (13-15)
-------------------------

Database integration, HPC submission, and error recovery.

.. list-table::
   :header-rows: 1
   :widths: 15 50 20 15

   * - Tutorial
     - Description
     - Time
     - Difficulty
   * - **13**
     - Database storage (MongoDB/Maggma)
     - 45 min
     - ⭐⭐
   * - **14**
     - **RECOMMENDED** HPC job submission with jobflow-remote (SLURM, PBS, SGE, local)
     - 30 min
     - ⭐⭐
   * - **14b**
     - Alternative: FireWorks for HPC submission
     - 60 min
     - ⭐⭐⭐
   * - **15**
     - **NEW** Custodian error handling and recovery
     - 45 min
     - ⭐⭐

----

📖 Recommended Learning Paths
==============================

For Beginners
-------------

Start with fundamental workflow concepts:

1. **Tutorial 01** - Relaxation basics (understand Maker pattern)
2. **Tutorial 02** - Parameter customization (user_params)
3. **Tutorial 03** - Band structure (electronic properties)
4. **Tutorial 05** - Multi-step workflows (Flow composition)

For Convergence Testing
-----------------------

Essential for all production calculations:

1. **Tutorial 06** - K-points and cutoff (most critical parameters)
2. **Tutorial 07** - Basis parameters (quality vs. speed tradeoff)
3. **Tutorial 08** - Complete basis convergence (comprehensive)

Apply converged parameters to all subsequent calculations.

For Production Calculations
---------------------------

Complete workflow development:

1. Complete convergence studies (**Tutorials 06-08**)
2. Apply to production workflows (**Tutorials 09-12**)
3. Set up database storage (**Tutorial 13**)
4. Configure HPC submission with **jobflow-remote** (**Tutorial 14** - recommended) or FireWorks (**Tutorial 14b**)
5. Enable error handling (**Tutorial 15**)

For Specific Properties
-----------------------

**Vibrational Properties**:
   * **Tutorial 16** - Phonon calculations with phonopy
   * **Tutorial 20** - Grüneisen parameters and thermal expansion
   * **Tutorial 21** - Quasi-harmonic approximation (QHA)

**Surface Chemistry**:
   * **Tutorial 17** - Surface energy calculations
   * **Tutorial 19** - Adsorption site scanning

**Transition States**:
   Start with **Tutorial 12** (NEB calculations)

**Mechanical Properties**:
   Start with **Tutorial 11** (Elastic constants)

**Workflow Customization**:
   **Tutorial 22** - Powerups for advanced parameter control

For Advanced Users
------------------

**Tier-Based Configuration**:
   **Tutorial 18** - Material-specific presets and automatic module activation

**Custom Workflows**:
   Combine Makers with powerups, use **Tutorial 05** as starting point

**Error Recovery**:
   **Tutorial 15** - Custodian integration for robust calculations

----

🎓 Tutorial Features
====================

Each tutorial includes:

✅ **Detailed READMEs**
   Comprehensive documentation with theory and best practices

✅ **Multiple Examples**
   Simple and detailed versions for different use cases

✅ **Automatic Plotting**
   Publication-quality plots generated automatically

✅ **Result Summaries**
   Text and JSON outputs for easy analysis

✅ **Timing Analysis**
   Computational cost tracking (Tutorials 07, 10)

✅ **Convergence Criteria**
   Clear guidelines for parameter selection

----

⚡ Quick Reference
==================

Common Calculation Types
------------------------

.. code-block:: python

   # Structure relaxation
   from atomate2.siesta.jobs.core import RelaxMaker
   maker = RelaxMaker.fixed_cell_relaxation()

   # Band structure
   from atomate2.siesta.jobs.core import BandStructureMaker
   maker = BandStructureMaker()

   # Phonons
   from atomate2.siesta.jobs.core import SiestaPhononFlowMaker
   maker = SiestaPhononFlowMaker(min_length=12.0)

   # Surface energy
   from atomate2.siesta.flows.multi_surface import MultiSurfaceEnergyFlowMaker
   maker = MultiSurfaceEnergyFlowMaker(
       miller_indices=[(1,0,0), (1,1,0), (1,1,1)],
       slab_layers=4,
   )

   # Elastic constants
   from atomate2.siesta.flows.elastic import ElasticFlowMaker
   maker = ElasticFlowMaker()

   # Equation of State
   from atomate2.siesta.flows.eos import EOSMaker
   maker = EOSMaker()

Parameter Customization
-----------------------

.. code-block:: python

   # Direct in Maker
   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "PAO.BasisSize": "DZP",
           "a2s_kpts": [6, 6, 6],
           "Mesh.Cutoff": "300 Ry",
       }
   )

   # Using powerups
   from atomate2.siesta.powerups import update_user_siesta_settings
   job = maker.make(structure)
   job = update_user_siesta_settings(job, {
       "SCF.Mixer.Weight": 0.005,
   })

   # Using tier presets
   from atomate2.siesta.sets.tiers import apply_tier_preset
   maker = apply_tier_preset(maker, "surface_metal")

----

💡 Best Practices
=================

1. **Always converge parameters first** (Tutorials 06-08)
2. **Use tier presets** for material-specific settings (Tutorial 18)
3. **Enable custodian** for production runs (Tutorial 15)
4. **Store results in database** for high-throughput (Tutorial 13)
5. **Validate against literature** when available
6. **Document your parameters** in workflow scripts

----

🛠️ Setup Requirements
======================

Install atomate2siesta:

.. code-block:: bash

   pip install -e ".[dev,tests,docs]"

Configure in ``~/.atomate2.yaml``:

.. code-block:: yaml

   SIESTA_CMD: "mpirun -np 4 siesta < siesta.fdf > siesta.out"
   SIESTA_PP_PATH: "/path/to/pseudopotentials"

For database storage (Tutorial 13):

.. code-block:: bash

   pip install maggma pymongo

For HPC submission (Tutorial 14):

.. code-block:: bash

   pip install fireworks

----

📞 Getting Help
===============

* Check individual tutorial READMEs for detailed documentation
* See main documentation: :doc:`/index`
* Report issues: https://github.com/materialsproject/atomate2/issues
* Ask questions: https://github.com/materialsproject/atomate2/discussions

----

🤝 Contributing
===============

To add a new tutorial:

1. Create numbered directory following the organization above
2. Include a comprehensive README.md with:

   * Overview and learning objectives
   * Requirements and setup
   * Step-by-step instructions
   * Expected output
   * Troubleshooting tips

3. Provide both simple and detailed example scripts
4. Update this index with your tutorial entry
5. Add tutorial.rst file to ``docs/source/tutorials/``

----

.. toctree::
   :maxdepth: 1
   :hidden:

   basic
   convergence
   advanced
   infrastructure
