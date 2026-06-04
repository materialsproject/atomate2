=====
Usage
=====

This guide demonstrates common usage patterns for atomate2siesta, from basic calculations to advanced workflows.

----

Quick Start
===========

Basic Structure Relaxation
---------------------------

The simplest workflow: relax a crystal structure.

.. code-block:: python

   from pymatgen.core import Structure
   from atomate2.siesta.jobs.core import RelaxMaker
   from jobflow import run_locally

   # Load or create a structure
   structure = Structure.from_file("POSCAR")  # or use Structure API

   # Create a relaxation maker
   maker = RelaxMaker.fixed_cell_relaxation()

   # Generate the job
   job = maker.make(structure)

   # Run locally
   results = run_locally(job, create_folders=True)

   # Access results
   print(f"Final energy: {results[job.uuid][1].output.energy} eV")
   print(f"Optimized structure: {results[job.uuid][1].output.structure}")

----

Configuration
=============

SIESTA Settings File
--------------------

Create ``~/.atomate2siesta.yaml`` with your SIESTA configuration:

.. code-block:: yaml

   # SIESTA executable command
   SIESTA_CMD: "mpirun -np 4 siesta"

   # Pseudopotential directory
   SIESTA_PP_PATH: "/path/to/pseudopotentials"

   # Optional: Database configuration
   MONGO_URI: "mongodb://localhost:27017"
   MONGO_DB: "siesta_calcs"

   # Optional: Workflow execution
   JOBFLOW_REMOTE_CONFIG: "/path/to/jobflow_config.yaml"

Using Environment Variables
----------------------------

Alternatively, set environment variables:

.. code-block:: bash

   export SIESTA_CMD="mpirun -np 4 siesta"
   export SIESTA_PP_PATH="/opt/siesta/pseudos"

Programmatic Configuration
---------------------------

Override settings in Python:

.. code-block:: python

   from atomate2.siesta import SETTINGS

   # Update SIESTA command
   SETTINGS.SIESTA_CMD = "siesta < siesta.fdf > siesta.out"

   # Update pseudopotential path
   SETTINGS.SIESTA_PP_PATH = "/custom/pseudo/path"

----

Common Workflows
================

1. Electronic Structure
-----------------------

**Band Structure Calculation**

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker, BandStructureMaker
   from jobflow import Flow

   # Create makers
   relax_maker = RelaxMaker.fixed_cell_relaxation()
   band_maker = BandStructureMaker.bandstructure_calculation()

   # Create jobs
   relax_job = relax_maker.make(structure)
   band_job = band_maker.make(structure)

   # Combine into flow
   flow = Flow([relax_job, band_job], name="relax_and_bands")

   # Run
   results = run_locally(flow, create_folders=True)

2. Phonon Calculations
-----------------------

**Complete Phonon Workflow with Plotting**

.. code-block:: python

   from atomate2.siesta.jobs.core import SiestaPhononFlowMaker

   # Create phonon maker with automatic plotting
   phonon_maker = SiestaPhononFlowMaker(
       min_length=12.0,          # Minimum supercell length (Å)
       displacement=0.01,         # Atomic displacement (Å)
       generate_plots=True,       # Enable automatic plotting
       plot_band_structure=True,  # Plot phonon bands
       plot_dos=True,            # Plot phonon DOS
       plot_thermal=True,        # Plot thermal properties
       write_summary=True,       # Write text summary
   )

   # Generate and run
   flow = phonon_maker.make(structure)
   results = run_locally(flow, create_folders=True)

   # Plots and summaries saved automatically to working directory

3. Surface Energy
-----------------

**Multi-Surface Energy Comparison**

.. code-block:: python

   from atomate2.siesta.flows.multi_surface import MultiSurfaceEnergyFlowMaker

   # Create multi-surface maker
   surface_maker = MultiSurfaceEnergyFlowMaker(
       miller_indices=[(1,0,0), (1,1,0), (1,1,1)],
       slab_layers=4,
       vacuum_thickness=15.0,
       generate_terminations=True,  # Auto-find terminations
   )

   # Run for bulk structure
   flow = surface_maker.make(bulk_structure)
   results = run_locally(flow, create_folders=True)

   # Results include surface energies and comparison plots

4. Equation of State
--------------------

**EOS with Automatic Fitting**

.. code-block:: python

   from atomate2.siesta.flows.eos import EOSMaker

   eos_maker = EOSMaker(
       n_points=7,              # Number of volume points
       scale_factors=None,      # Auto-generate around equilibrium
   )

   flow = eos_maker.make(structure)
   results = run_locally(flow, create_folders=True)

   # Access fitted parameters
   # B0 (bulk modulus), B0' (pressure derivative), V0, E0

5. Elastic Constants
--------------------

**Full Elastic Tensor Calculation**

.. code-block:: python

   from atomate2.siesta.flows.elastic import ElasticFlowMaker

   elastic_maker = ElasticFlowMaker()

   flow = elastic_maker.make(structure)
   results = run_locally(flow, create_folders=True)

   # Results include: elastic tensor, bulk modulus, shear modulus,
   # Young's modulus, Poisson ratio

----

Parameter Customization
=======================

Method 1: Direct in Maker
--------------------------

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "PAO.BasisSize": "DZP",
           "a2s_kpts": [6, 6, 6],
           "Mesh.Cutoff": "300 Ry",
           "SCF.DM.Tolerance": 1.0e-5,
       }
   )

Method 2: Using Powerups
-------------------------

Apply updates after job creation:

.. code-block:: python

   from atomate2.siesta.powerups import update_user_siesta_settings

   # Create job
   job = maker.make(structure)

   # Apply powerups
   job = update_user_siesta_settings(
       job,
       siesta_updates={
           "SCF.Mixer.Weight": 0.005,
           "OccupationFunction": "MP",
           "ElectronicTemperature": "300 K",
       }
   )

Method 3: Tier Presets
-----------------------

Use material-specific presets:

.. code-block:: python

   from atomate2.siesta.sets.tiers import apply_tier_preset

   # Apply preset for metal surfaces
   maker = apply_tier_preset(maker, "surface_metal")

   # Available presets:
   # - relax_standard
   # - high_accuracy
   # - band_structure_precise
   # - surface_metal
   # - surface_semiconductor
   # - phonon_high_accuracy
   # ... (14 total presets)

----

Advanced Features
=================

Error Handling with Custodian
------------------------------

Enable automatic error recovery:

.. code-block:: python

   from atomate2.siesta.custodian import SCFConvergenceHandler

   # Create maker with custodian enabled
   maker = RelaxMaker.fixed_cell_relaxation(
       use_custodian=True,
       custodian_handlers=[SCFConvergenceHandler(max_attempts=10)],
       custodian_max_errors=15,
   )

   # Jobs will automatically recover from common errors:
   # - SCF convergence failures
   # - Memory issues
   # - Time limits
   # - Basis set problems
   # ... (10+ error types)

Database Storage
----------------

Store results in MongoDB:

.. code-block:: python

   from maggma.stores import MongoStore
   from jobflow import run_locally

   # Create MongoDB store
   store = MongoStore(
       database="siesta_calcs",
       collection_name="tasks",
       host="localhost",
       port=27017,
   )

   # Run with database storage
   results = run_locally(
       flow,
       create_folders=True,
       store=store,
   )

   # Query results later
   docs = list(store.query({"formula_pretty": "Si"}))

HPC Cluster Submission
----------------------

Submit to remote HPC clusters using jobflow-remote:

.. code-block:: python

   from jobflow_remote import submit_flow

   # Define resources
   resources = {
       "nodes": 2,
       "partition": "compute",
       "ntasks": 48,
       "time": "24:00:00",
       "job_name": "siesta_calc",
   }

   # Submit to cluster
   submit_flow(
       flow,
       project="my_project",
       worker="slurm_worker",
       resources=resources,
   )

----

Performance Optimization
========================

Convergence Testing
-------------------

Always test convergence before production runs:

.. code-block:: python

   from atomate2.siesta.flows.convergence import (
       KpointsConvergenceFlowMaker,
       MeshCutoffConvergenceFlowMaker,
   )

   # K-points convergence
   kpts_maker = KpointsConvergenceFlowMaker(
       kpoints_list=[[2,2,2], [4,4,4], [6,6,6], [8,8,8]],
   )

   # Mesh cutoff convergence
   cutoff_maker = MeshCutoffConvergenceFlowMaker(
       mesh_cutoffs=[200, 250, 300, 350, 400],  # in Ry
   )

   # Run both convergence studies
   kpts_flow = kpts_maker.make(structure)
   cutoff_flow = cutoff_maker.make(structure)

   # Plots generated automatically showing convergence

Parallel Execution
------------------

Use MPI for SIESTA calculations:

.. code-block:: yaml

   # In ~/.atomate2siesta.yaml
   SIESTA_CMD: "mpirun -np 16 siesta"

Optimize performance with proper resource allocation.

----

Output and Results
==================

Accessing Job Results
---------------------

.. code-block:: python

   # Run job
   results = run_locally(job, create_folders=True)

   # Get output
   task_doc = results[job.uuid][1].output

   # Access properties
   energy = task_doc.energy                    # Total energy (eV)
   forces = task_doc.output.forces             # Forces (eV/Å)
   stress = task_doc.output.stress             # Stress tensor
   structure = task_doc.structure              # Final structure
   formula = task_doc.formula_pretty          # Chemical formula

Understanding Task Documents
----------------------------

All results are stored as ``SiestaTaskDoc`` objects containing:

- **Input**: Structure, parameters used
- **Output**: Energy, forces, stress, electronic properties
- **Metadata**: Calculation type, SIESTA version, timing
- **Analysis**: Derived properties (bandgap, magnetization, etc.)

Saving Results
--------------

**JSON Export**

.. code-block:: python

   import json

   # Convert to dictionary
   result_dict = task_doc.dict()

   # Save to JSON
   with open("results.json", "w") as f:
       json.dump(result_dict, f, indent=2)

**Structure Export**

.. code-block:: python

   # Save final structure
   task_doc.structure.to(filename="POSCAR_final")

----

Troubleshooting
===============

Common Issues
-------------

**Issue**: SCF not converging

.. code-block:: python

   # Solution: Adjust mixing parameters
   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "SCF.Mixer.Weight": 0.005,  # Slower mixing
           "SCF.Mixer.History": 10,    # Longer history
           "SCF.MaxIterations": 200,   # More iterations
       }
   )

**Issue**: Basis set errors

.. code-block:: python

   # Solution: Use larger basis or adjust PAO parameters
   user_params={
       "PAO.BasisSize": "DZP",
       "PAO.EnergyShift": "0.01 Ry",
       "PAO.SplitNorm": 0.15,
   }

**Issue**: Memory problems

.. code-block:: python

   # Solution: Enable custodian with memory handler
   from atomate2.siesta.custodian import MemoryHandler

   maker = RelaxMaker.fixed_cell_relaxation(
       use_custodian=True,
       custodian_handlers=[MemoryHandler()],
   )

Enable Debug Logging
--------------------

.. code-block:: python

   import logging

   logging.basicConfig(level=logging.DEBUG)
   logger = logging.getLogger("atomate2.siesta")

----

Next Steps
==========

* **Tutorials**: See :doc:`tutorials/index` for 22 comprehensive tutorials
* **Features**: Explore :doc:`features` for detailed feature documentation
* **Advanced Workflows**: Check :doc:`advanced-workflows` for complex calculations
* **API Reference**: Browse :doc:`api/modules` for complete API documentation

For questions and support:

* GitHub Issues: https://github.com/materialsproject/atomate2/issues
* Discussions: https://github.com/materialsproject/atomate2/discussions
