Makers vs FlowMakers
=====================

Overview
--------

Atomate2SIESTA uses two types of workflow builders:

- **Makers** (``*Maker``): Single-job calculations
- **FlowMakers** (``*FlowMaker``): Multi-step workflows that chain multiple jobs together

Understanding when to use each is crucial for building efficient computational workflows.

Viewing Comprehensive Documentation
------------------------------------

All Makers and FlowMakers have comprehensive docstrings viewable via CLI:

.. code-block:: bash

   # View detailed documentation for any Maker or FlowMaker
   atomate2siesta-info workflows StaticMaker --full
   atomate2siesta-info workflows RelaxMaker --full
   atomate2siesta-info workflows BandStructureMaker --full
   atomate2siesta-info workflows SiestaEosFlowMaker --full

Each comprehensive docstring includes:

* **Detailed scientific context** - What the calculation does and why
* **Workflow steps** - Step-by-step breakdown of computation
* **Key results** - What outputs are generated (energies, forces, band gaps, etc.)
* **Applications** - Real-world use cases (catalysis, materials discovery, etc.)
* **Parameters** - All available configuration options
* **Examples** - Practical code snippets
* **Notes** - Best practices, convergence tips, and common pitfalls

All 10 single-job Makers have comprehensive documentation:

.. code-block:: python

   # Available Makers with full documentation:
   StaticMaker          # SCF calculations
   RelaxMaker           # Geometry optimization
   LuaMaker             # Lua scripting for NEB/MD
   SocketIOStaticMaker  # Batch calculations via socket
   BandStructureMaker   # Electronic band structure
   DOSMaker             # Total density of states
   PDOSMaker            # Projected density of states
   PhononMaker          # Force constants calculation
   OpticalMaker         # Optical properties
   SiestaPhononMaker    # Complete phonon workflow

See :ref:`info-cli` in the CLI tools documentation for complete details.

Naming Convention
-----------------

As of v1.0.0, all multi-step workflows use the ``FlowMaker`` suffix for clarity:

**Single-Job Makers** (``*Maker``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These perform a single calculation and return one result:

.. code-block:: python

   from atomate2.siesta.jobs.core import (
       StaticMaker,      # Single SCF calculation
       RelaxMaker,       # Single relaxation
       BandsMaker,       # Single band structure calculation
   )

**Multi-Step FlowMakers** (``*FlowMaker``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These orchestrate multiple calculations in a workflow:

.. code-block:: python

   from atomate2.siesta.flows.phonon import SiestaPhononFlowMaker
   from atomate2.siesta.flows.elastic import ElasticFlowMaker
   from atomate2.siesta.flows.eos import SiestaEosFlowMaker
   from atomate2.siesta.flows.convergence import (
       MeshCutoffConvergenceFlowMaker,
       KpointsConvergenceFlowMaker,
   )

When to Use Each
----------------

Use a **Maker** when you need:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

✓ A single calculation (SCF, relaxation, band structure, etc.)

✓ Building blocks for custom workflows

✓ Maximum control over individual job parameters

Example:

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker

   maker = RelaxMaker.fixed_cell_relaxation()
   job = maker.make(structure)

Use a **FlowMaker** when you need:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

✓ Multi-step workflows with automatic job chaining

✓ Property calculations requiring multiple stages (phonons, EOS, elastic)

✓ Convergence testing across parameter spaces

✓ Complex workflows with data flow between jobs

Example:

.. code-block:: python

   from atomate2.siesta.flows.phonon import SiestaPhononFlowMaker

   maker = SiestaPhononFlowMaker(
       min_length=15.0,
       displacement=0.01
   )
   flow = maker.make(structure)

Parameter Customization
-----------------------

There are three main ways to customize calculations:

1. Constructor Parameters (``user_params``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass parameters directly when creating the maker:

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker

   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "PAO.BasisSize": "DZP",
           "a2s_kpts": [4, 4, 4],
           "Mesh.Cutoff": "300 Ry",
           "SCF.Mixer.Weight": 0.01,
       }
   )

**When to use**: Setting base parameters before applying presets or powerups.

2. Tier Presets
~~~~~~~~~~~~~~~

Apply material-specific parameter sets:

.. code-block:: python

   from atomate2.siesta.sets.tiers import apply_tier_preset

   maker = RelaxMaker.fixed_cell_relaxation()
   maker = apply_tier_preset(maker, "relax_standard")

   # Override specific preset parameters
   maker = apply_tier_preset(
       maker,
       "relax_standard",
       override_params={
           "a2s_kpts": [6, 6, 6],  # Denser k-points
           "Spin": "polarized"  # Add magnetism
       }
   )

**When to use**: Starting from validated parameter sets for specific material types.

.. warning::
   Always use ``override_params`` when modifying preset parameters. Passing
   ``user_params`` to the maker before applying a preset will NOT work - the
   preset will overwrite them!

3. Powerups (Runtime Modifications)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modify jobs or flows after creation:

.. code-block:: python

   from atomate2.siesta.powerups import (
       update_user_siesta_settings,
       add_metadata_to_flow,
   )

   # Create job/flow
   job = maker.make(structure)

   # Apply powerups
   job = update_user_siesta_settings(job, {
       "SCF.Mixer.Weight": 0.005,
       "OccupationFunction": "MP",
   })

   job = add_metadata_to_flow(job, {"project": "catalyst_study"})

**When to use**:

- Modifying jobs after creation
- Conditional parameter updates based on structure analysis
- Adding metadata or tags
- Bulk modifications across multiple jobs in a flow

Comparison Table
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Method
     - Timing
     - Use Case
     - Example
   * - ``user_params``
     - At creation
     - Base parameters
     - Custom basis size
   * - Tier presets
     - At creation
     - Material-specific sets
     - ``relax_standard``
   * - Powerups
     - After creation
     - Runtime modifications
     - Conditional updates

Complete Examples
-----------------

Example 1: Single-Job Maker with Preset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker
   from atomate2.siesta.sets.tiers import apply_tier_preset
   from jobflow import run_locally

   # Create maker and apply preset
   maker = RelaxMaker.fixed_cell_relaxation()
   maker = apply_tier_preset(
       maker,
       "relax_standard",
       override_params={
           "a2s_kpts": [8, 8, 8],  # Override k-points
           "Spin": "polarized"  # Add magnetism
       }
   )

   # Generate and run job
   job = maker.make(structure)
   results = run_locally(job)

Example 2: FlowMaker with Powerups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atomate2.siesta.flows.phonon import SiestaPhononFlowMaker
   from atomate2.siesta.powerups import update_user_siesta_settings
   from atomate2.siesta.sets.tiers import apply_tier_preset

   # Create flow maker with preset
   maker = SiestaPhononFlowMaker(
       min_length=15.0,
       displacement=0.01
   )

   # Apply preset to underlying makers
   maker.static_maker = apply_tier_preset(
       maker.static_maker,
       "phonon_high_accuracy"
   )

   # Create flow
   flow = maker.make(structure)

   # Apply powerup to all jobs in flow
   flow = update_user_siesta_settings(flow, {
       "SCF.Mixer.Weight": 0.005,
       "SCF.MustConverge": True
   })

Example 3: Custom Workflow with Makers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build a custom workflow using individual Makers:

.. code-block:: python

   from jobflow import Flow
   from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker

   # Create makers
   relax = RelaxMaker.fixed_cell_relaxation(
       user_params={"PAO.BasisSize": "SZ"}
   )
   static = StaticMaker.scf(
       user_params={"PAO.BasisSize": "DZP"}
   )

   # Build workflow manually
   relax_job = relax.make(structure)
   static_job = static.make(relax_job.output.structure)

   # Create flow
   flow = Flow([relax_job, static_job])

Example 4: Convergence Testing with FlowMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atomate2.siesta.flows.convergence import MeshCutoffConvergenceFlowMaker
   from atomate2.siesta.jobs.core import StaticMaker

   # Create static maker for convergence tests
   static_maker = StaticMaker.scf(
       user_params={
           "PAO.BasisSize": "DZP",
           "a2s_kpts": [6, 6, 6]
       }
   )

   # Create convergence flow
   maker = MeshCutoffConvergenceFlowMaker(
       static_maker=static_maker,
       mesh_cutoffs=[200, 250, 300, 350, 400],  # Ry
   )

   flow = maker.make(structure)

Common Patterns
---------------

Pattern 1: Preset + Override
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start with a validated preset and override specific parameters:

.. code-block:: python

   from atomate2.siesta.sets.tiers import apply_tier_preset

   maker = RelaxMaker.fixed_cell_relaxation()
   maker = apply_tier_preset(
       maker,
       "surface_metal",
       override_params={
           "a2s_kpts": [12, 12, 1],  # Dense in-plane, sparse out-of-plane
           "OccupationFunction": "MP",
           "ElectronicTemperature": "300 K"
       }
   )

Pattern 2: Conditional Powerups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply powerups based on structure analysis:

.. code-block:: python

   from atomate2.siesta.powerups import update_user_siesta_settings

   job = maker.make(structure)

   # Check if structure contains magnetic elements
   magnetic_elements = {"Fe", "Co", "Ni", "Mn", "Cr"}
   if any(el.symbol in magnetic_elements for el in structure.species):
       job = update_user_siesta_settings(job, {
           "Spin": "polarized",
           "DM.InitSpin": True
       })

Pattern 3: Flow Customization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Customize individual jobs within a FlowMaker:

.. code-block:: python

   from atomate2.siesta.flows.elastic import ElasticFlowMaker
   from atomate2.siesta.sets.tiers import apply_tier_preset

   maker = ElasticFlowMaker()

   # Customize the relaxation maker
   maker.bulk_relax_maker = apply_tier_preset(
       maker.bulk_relax_maker,
       "high_accuracy"
   )

   # Customize the elastic deformation maker
   maker.elastic_relax_maker = apply_tier_preset(
       maker.elastic_relax_maker,
       "high_accuracy"
   )

   flow = maker.make(structure)

Best Practices
--------------

✓ **Use FlowMakers for standard workflows**: Phonons, EOS, elastic constants, etc.

✓ **Use Makers for custom workflows**: When you need fine-grained control

✓ **Start with presets**: Use tier presets as starting points and override as needed

✓ **Use powerups for conditional logic**: Apply powerups when parameters depend on runtime conditions

✓ **Document custom workflows**: Add comments explaining parameter choices

✗ **Don't mix user_params and presets incorrectly**: Use ``override_params`` when modifying presets

✗ **Don't duplicate parameters**: Choose one customization method per parameter

✗ **Don't create FlowMakers manually**: Use the provided FlowMaker classes for standard workflows

Migration Guide
---------------

If you're updating code from before v1.0.0, here's how to migrate:

Old Code (Before v1.0.0)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atomate2.siesta.flows.phonon import SiestaPhononMaker
   from atomate2.siesta.flows.elastic import ElasticMaker
   from atomate2.siesta.flows.convergence import MeshCutoffConvergenceMaker

   phonon = SiestaPhononMaker()
   elastic = ElasticMaker()
   convergence = MeshCutoffConvergenceMaker()

New Code (v1.0.0+)
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atomate2.siesta.flows.phonon import SiestaPhononFlowMaker
   from atomate2.siesta.flows.elastic import ElasticFlowMaker
   from atomate2.siesta.flows.convergence import MeshCutoffConvergenceFlowMaker

   phonon = SiestaPhononFlowMaker()
   elastic = ElasticFlowMaker()
   convergence = MeshCutoffConvergenceFlowMaker()

**Simple find-and-replace**: Add ``Flow`` before ``Maker`` for all multi-step workflows.

See Also
--------

- :doc:`usage` - General usage guide
- :doc:`tier-system` - Tier presets documentation
- :doc:`features` - Advanced features including powerups
- :doc:`recipe-book` - High-level workflow recipes
- :doc:`advanced-workflows` - Complex workflow examples
