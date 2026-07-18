==================
Advanced Workflows
==================

Complex multi-step calculations for materials properties using atomate2siesta.

----

Overview
========

This page demonstrates how to construct sophisticated workflows by combining multiple
calculation types, using advanced features, and optimizing for production use.

----

Multi-Property Workflows
=========================

Structure → Relax → Properties
-------------------------------

Standard workflow pattern for comprehensive materials characterization:

.. code-block:: python

   from jobflow import Flow
   from atomate2.siesta.jobs.core import (
       RelaxMaker,
       BandStructureMaker,
       SiestaPhononFlowMaker,
   )
   from atomate2.siesta.flows.elastic import ElasticFlowMaker

   # 1. Variable cell relaxation
   relax = RelaxMaker.variable_cell_relaxation(
       user_params={
           "PAO.BasisSize": "DZP",
           "a2s_kpts": [6, 6, 6],
           "Mesh.Cutoff": "400 Ry",
       }
   ).make(structure)

   # 2. Electronic structure
   bands = BandStructureMaker().make(relax.output.structure)

   # 3. Elastic properties
   elastic = ElasticFlowMaker().make(relax.output.structure)

   # 4. Phonon properties
   phonon = SiestaPhononFlowMaker(min_length=12.0).make(relax.output.structure)

   # Combine into workflow
   flow = Flow([relax, bands, elastic, phonon])

High-Throughput Screening
--------------------------

Process multiple structures with same workflow:

.. code-block:: python

   from pymatgen.core import Structure
   from atomate2.siesta.jobs.core import RelaxMaker
   from atomate2.siesta.sets.tiers import apply_tier_preset
   from jobflow import Flow
   from maggma.stores import MongoStore

   # Load structures
   structures = [Structure.from_file(f) for f in structure_files]

   # Create optimized maker
   maker = RelaxMaker.fixed_cell_relaxation()
   maker = apply_tier_preset(maker, "relax_standard")

   # Generate jobs
   jobs = [maker.make(s) for s in structures]
   flow = Flow(jobs)

   # Run with database storage
   store = MongoStore(
       database="screening_db",
       collection_name="relaxations",
   )

   results = run_locally(flow, create_folders=True, store=store)

----

Convergence-Then-Production
============================

Automated Convergence Study
----------------------------

Test parameters, then use converged values for production:

.. code-block:: python

   from atomate2.siesta.flows.convergence import (
       KpointsConvergenceFlowMaker,
       MeshCutoffConvergenceFlowMaker,
   )
   from atomate2.siesta.flows.basis import CompleteBasisConvergenceFlowMaker

   # 1. Convergence studies
   kpts_conv = KpointsConvergenceFlowMaker(
       kpoints_list=[[2,2,2], [4,4,4], [6,6,6], [8,8,8]],
   ).make(structure)

   cutoff_conv = MeshCutoffConvergenceFlowMaker(
       mesh_cutoffs=[200, 300, 400, 500],
   ).make(structure)

   # 2. Review convergence plots and text files
   # Files generated: energy_convergence.png, convergence_comparison.png
   # Use plots to determine converged parameters

   # 3. Production calculation with manually selected converged parameters
   production = RelaxMaker.variable_cell_relaxation(
       user_params={
           "a2s_kpts": [6, 6, 6],  # Selected from convergence plot
           "Mesh.Cutoff": "400 Ry",  # Selected from convergence plot
       }
   ).make(structure)

   # Combine
   flow = Flow([kpts_conv, cutoff_conv, production])

----

Restart Workflows for Efficiency
==================================

SIESTA provides powerful restart capabilities to accelerate multi-stage calculations
by reusing converged results from previous steps. This can reduce total computation
time by 30-60% for common workflows.

Density Matrix (DM) Restart
----------------------------

Reuse converged electronic structure for faster SCF convergence when refining
k-points or tightening tolerances:

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
   from jobflow import Flow

   # Step 1: Coarse calculation (saves DM)
   coarse_maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "a2s_kpts": [2, 2, 2],
           "PAO.BasisSize": "DZ",
           "DM.Tolerance": 1e-4,
           "DM.UseSaveDM": True,  # Save DM for reuse
           "Mesh.Cutoff": "200 Ry",
       },
   )
   coarse_job = coarse_maker.make(structure)

   # Step 2: Fine calculation (reads DM from Step 1)
   fine_maker = StaticMaker.scf(
       user_params={
           "a2s_kpts": [4, 4, 4],  # Refined k-mesh
           "PAO.BasisSize": "DZ",  # MUST match Step 1
           "DM.Tolerance": 1e-5,  # Tighter convergence
           "DM.UseSaveDM": True,  # Read previous DM
           "Mesh.Cutoff": "200 Ry",
       },
       copy_siesta_kwargs={"restart_to_input": True},  # Copy DM file
   )
   fine_job = fine_maker.make(structure, prev_dir=coarse_job.output.dir_name)

   # Create workflow
   workflow = Flow([coarse_job, fine_job], name="DM Restart Workflow")

**Expected Speedup**: 50-70% fewer SCF iterations in Step 2

**Requirements**:

- ``DM.UseSaveDM: True`` in both jobs
- Same basis set (``PAO.BasisSize``) in both jobs
- ``copy_siesta_kwargs={"restart_to_input": True}`` to copy DM file
- Use ``prev_dir`` to link jobs

Geometry (XV) Restart
---------------------

Continue geometry optimization from previous atomic positions for two-stage
relaxation:

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker
   from jobflow import Flow

   # Step 1: Quick relaxation (saves XV)
   quick_maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "a2s_kpts": [4, 4, 4],
           "PAO.BasisSize": "DZ",
           "Mesh.Cutoff": "200 Ry",
           "MD.MaxForceTol": "0.04 eV/Ang",  # Loose tolerance
           "MD.UseSaveXV": True,  # Save XV for reuse
       },
   )
   quick_job = quick_maker.make(structure)

   # Step 2: Fine relaxation (reads XV from Step 1)
   fine_maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "a2s_kpts": [4, 4, 4],
           "PAO.BasisSize": "DZ",
           "Mesh.Cutoff": "200 Ry",
           "MD.MaxForceTol": "0.01 eV/Ang",  # Tight tolerance
           "MD.UseSaveXV": True,  # Read previous XV
       },
       copy_siesta_kwargs={
           "restart_to_input": True,  # Copy DM file
           "additional_siesta_files": ["siesta.XV"],  # Also copy XV file
       },
   )
   fine_job = fine_maker.make(structure, prev_dir=quick_job.output.dir_name)

   # Create workflow
   workflow = Flow([quick_job, fine_job], name="XV Restart Workflow")

**Expected Speedup**: 50-70% fewer relaxation steps in Step 2

**Requirements**:

- ``MD.UseSaveXV: True`` in both jobs
- ``copy_siesta_kwargs`` with **both** ``restart_to_input: True`` and ``additional_siesta_files: ["siesta.XV"]``
- Use ``prev_dir`` to link jobs

Files Copied by ``copy_siesta_kwargs``
---------------------------------------

Understanding what gets copied with different parameters:

**With** ``restart_to_input=True`` **only**:

- ``siesta.DM`` - Density matrix (for SCF restart)
- ``*.STRUCT_OUT`` - Structure output files
- ``*.xyz`` - Trajectory files

**To also copy geometry restart files**, add ``additional_siesta_files``:

.. code-block:: python

   copy_siesta_kwargs={
       "restart_to_input": True,
       "additional_siesta_files": ["siesta.XV"],  # Geometry + velocities
   }

**To copy custom files** (e.g., wavefunctions):

.. code-block:: python

   copy_siesta_kwargs={
       "restart_to_input": True,
       "additional_siesta_files": [
           "siesta.XV",
           "siesta.WFSX",  # Wavefunctions
           "siesta.bands",  # Band structure
       ],
   }

Common Restart Patterns
-----------------------

**1. Coarse → Fine k-points** (DM restart):

   * Quick coarse calculation with minimal k-points
   * Refine with denser k-mesh using previous DM
   * Use case: High-throughput screening with validation

**2. Loose → Tight relaxation** (XV restart):

   * Fast pre-relaxation with loose force tolerance
   * Final relaxation with tight tolerance using previous geometry
   * Use case: Complex materials, surfaces, molecular systems

**3. DFT → DFT+U** (DM restart):

   * Converge without U correction first
   * Add U correction using previous DM as initial guess
   * Use case: Strongly correlated systems

**4. Phonon workflows** (DM + XV restart):

   * Pre-relax structure (saves DM and XV)
   * Supercell force calculations read DM for faster SCF
   * Use case: Phonon calculations with displaced structures

.. seealso::

   Tutorial files in ``tutorials/01-basics/05-workflows/``:

   * ``02_restart_from_dm.py`` - DM restart example
   * ``03_restart_from_xv.py`` - XV restart example

----

Surface Science Workflows
==========================

Multi-Surface Energy Comparison
--------------------------------

Calculate surface energies for multiple Miller indices:

.. code-block:: python

   from atomate2.siesta.flows.multi_surface import MultiSurfaceEnergyFlowMaker
   from atomate2.siesta.jobs.core import StaticMaker
   from atomate2.siesta.sets.tiers import apply_tier_preset

   # Setup makers
   bulk_maker = StaticMaker()
   bulk_maker = apply_tier_preset(bulk_maker, "relax_standard")

   slab_maker = StaticMaker()
   slab_maker = apply_tier_preset(slab_maker, "surface_metal")

   # Multi-surface workflow
   multi_surface = MultiSurfaceEnergyFlowMaker(
       miller_indices=[
           (1, 0, 0),
           (1, 1, 0),
           (1, 1, 1),
           (2, 1, 0),
       ],
       bulk_static_maker=bulk_maker,
       slab_static_maker=slab_maker,
       slab_layers=4,
       vacuum_size=15.0,
       symmetrize=False,  # Explore all terminations
       plot_results=True,
       write_summary=True,
   )

   flow = multi_surface.make(bulk_structure)

Adsorption Energy Calculation
------------------------------

Calculate molecule adsorption on surface:

.. code-block:: python

   from pymatgen.core import Molecule

   # 1. Clean slab energy
   clean_slab = StaticMaker().make(slab_structure)

   # 2. Slab with adsorbate (create manually)
   slab_with_mol = slab_structure.copy()
   # Add molecule to surface...

   # 3. Adsorbate slab energy
   ads_slab = StaticMaker().make(slab_with_mol)

   # 4. Isolated molecule energy
   molecule = Molecule.from_file("molecule.xyz")
   mol_energy = StaticMaker().make(molecule)

   # 5. Calculate adsorption energy
   # E_ads = E(slab+mol) - E(slab) - E(mol)

   flow = Flow([clean_slab, ads_slab, mol_energy])

----

Phonon-Related Workflows
=========================

Temperature-Dependent Properties
---------------------------------

Quasi-harmonic approximation for thermal expansion:

.. code-block:: python

   from atomate2.siesta.flows.eos import EOSMaker
   from atomate2.siesta.flows.phonon import SiestaPhononFlowMaker

   # 1. EOS to get volume range
   eos = EOSMaker(number_of_frames=5).make(structure)

   # 2. Phonons at different volumes
   volumes = eos.output.volumes
   phonon_jobs = []

   for vol in volumes:
       # Scale structure to volume
       scaled_structure = structure.copy()
       scaled_structure.scale_lattice(vol)

       # Phonon calculation
       phonon = SiestaPhononFlowMaker(
           min_length=12.0,
           generate_plots=False,  # Save time
       ).make(scaled_structure)

       phonon_jobs.append(phonon)

   # 3. Analyze thermal properties vs volume
   flow = Flow([eos] + phonon_jobs)

Phonon + Band Structure
------------------------

.. code-block:: python

   from atomate2.siesta.jobs.core import (
       RelaxMaker,
       BandStructureMaker,
       SiestaPhononFlowMaker,
   )

   # Use tier preset for phonons
   from atomate2.siesta.sets.tiers import apply_tier_preset

   relax = RelaxMaker.variable_cell_relaxation()
   relax = apply_tier_preset(relax, "phonon_high_accuracy")

   # Workflow
   relax_job = relax.make(structure)

   bands = BandStructureMaker().make(relax_job.output.structure)

   phonon = SiestaPhononFlowMaker(
       min_length=15.0,
       mesh=(100, 100, 100),
   ).make(relax_job.output.structure)

   flow = Flow([relax_job, bands, phonon])

----

Magnetic Systems
================

Spin-Polarized + DFT+U
----------------------

For strongly correlated magnetic materials:

.. code-block:: python

   from atomate2.siesta.sets.tiers import apply_tier_preset

   maker = RelaxMaker.variable_cell_relaxation()
   maker = apply_tier_preset(maker, "magnetic_correlated")

   # Additional DFT+U parameters
   job = maker.make(structure)
   job = update_user_siesta_settings(job, {
       "DFT.U.Projectors": {
           "Mn": {"l": 2, "U": 4.0},  # U = 4 eV for Mn d-orbitals
           "O": {"l": 1, "U": 0.0},
       },
   })

Magnetic Anisotropy
-------------------

Calculate magnetic anisotropy energy (MAE):

.. code-block:: python

   # 1. Relaxation
   relax = RelaxMaker.variable_cell_relaxation(
       user_params={
           "spin": "polarized",
           "PAO.BasisSize": "DZP",
       }
   ).make(structure)

   # 2. Static calculations with different spin orientations
   spin_x = StaticMaker(user_params={
       "spin": "polarized",
       "SpinConstraint.Theta": 90,
       "SpinConstraint.Phi": 0,
   }).make(relax.output.structure)

   spin_z = StaticMaker(user_params={
       "spin": "polarized",
       "SpinConstraint.Theta": 0,
       "SpinConstraint.Phi": 0,
   }).make(relax.output.structure)

   # MAE = E(spin along x) - E(spin along z)
   flow = Flow([relax, spin_x, spin_z])

----

Reaction Pathways
=================

NEB (Nudged Elastic Band) Calculations
---------------------------------------

Calculate minimum energy paths and transition states using two approaches:

**1. Lua-based NEB (SIESTA native, faster)**

Uses SIESTA's FLOS library for on-the-fly optimization:

.. code-block:: python

   from atomate2.siesta.flows.neb import NebDirectFlowMaker
   from atomate2.siesta.jobs.core import LuaMaker, RelaxMaker

   # Basic: Direct NEB without endpoint relaxation
   maker = NebDirectFlowMaker(
       number_of_images=7,
       relax_endpoints=False,  # Use structures as provided
       interpolation_method="idpp",  # or "linear"
       neb_maker=LuaMaker.neb(
           use_custodian=True,
           user_params={
               "PAO.BasisSize": "DZP",
               "a2s_kpts": [4, 4, 4],
               "Mesh.Cutoff": "300 Ry",
           }
       ),
   )
   flow = maker.make(initial_structure, final_structure)

   # Relax both endpoints (backward compatible)
   maker = NebDirectFlowMaker(
       number_of_images=7,
       relax_endpoints=True,  # or "both" - same behavior
       relax_maker=RelaxMaker.fixed_cell_relaxation(),
   )
   flow = maker.make(initial_structure, final_structure)

   # Selective endpoint relaxation
   # Relax only initial structure (final already optimized)
   maker = NebDirectFlowMaker(
       number_of_images=7,
       relax_endpoints="initial",  # Only relax initial
       relax_maker=RelaxMaker.fixed_cell_relaxation(),
   )
   flow = maker.make(initial_structure, final_structure)

   # Relax only final structure (initial is known state)
   maker = NebDirectFlowMaker(
       number_of_images=7,
       relax_endpoints="final",  # Only relax final
       relax_maker=RelaxMaker.fixed_cell_relaxation(),
   )
   flow = maker.make(initial_structure, final_structure)

   # Different relaxation settings for each endpoint
   # Use case: Quick relax for initial, precise relax for final
   maker = NebDirectFlowMaker(
       number_of_images=7,
       relax_endpoints=True,
       # Quick, coarse relaxation for initial state
       relax_initial_maker=RelaxMaker.fixed_cell_relaxation(
           user_params={
               "PAO.BasisSize": "DZP",
               "a2s_kpts": [2, 2, 2],
               "Mesh.Cutoff": "200 Ry",
           }
       ),
       # Precise, fine relaxation for final state
       relax_final_maker=RelaxMaker.fixed_cell_relaxation(
           user_params={
               "PAO.BasisSize": "TZP",
               "a2s_kpts": [4, 4, 4],
               "Mesh.Cutoff": "400 Ry",
           }
       ),
   )
   flow = maker.make(initial_structure, final_structure)

**2. ASE-based NEB (fully iterative, no FLOS required)**

Uses ASE's NEB implementation with iterative SIESTA force calculations:

.. code-block:: python

   from atomate2.siesta.flows.neb import AseNebFlowMaker
   from atomate2.siesta.jobs.core import StaticMaker, RelaxMaker
   from atomate2.siesta.powerups import update_user_siesta_settings

   # Basic: ASE NEB without endpoint relaxation
   maker = AseNebFlowMaker(
       number_of_images=5,
       relax_endpoints=False,
       optimizer="BFGS",  # or "FIRE"
       fmax=0.05,  # NEB force convergence in eV/Å
       climbing_image=False,
       spring_constant=5.0,  # Spring constant in eV/Å²
       static_maker=StaticMaker(),
   )
   flow = maker.make(initial_structure, final_structure)

   # Apply SIESTA parameters using powerup
   flow = update_user_siesta_settings(flow, {
       "PAO.BasisSize": "DZP",
       "a2s_kpts": [2, 2, 2],
       "Mesh.Cutoff": "200 Ry",
   })

   # Selective endpoint relaxation (same as Lua-based)
   # Supports: False, True, "initial", "final", "both"
   maker = AseNebFlowMaker(
       number_of_images=5,
       relax_endpoints="initial",  # Only relax initial structure
       relax_maker=RelaxMaker.fixed_cell_relaxation(),
       optimizer="BFGS",
       fmax=0.05,
   )
   flow = maker.make(initial_structure, final_structure)

   # Different relaxation for each endpoint
   maker = AseNebFlowMaker(
       number_of_images=5,
       relax_endpoints=True,
       # Separate makers for initial and final
       relax_initial_maker=RelaxMaker.fixed_cell_relaxation(
           user_params={"PAO.BasisSize": "DZP", "a2s_kpts": [2, 2, 2]}
       ),
       relax_final_maker=RelaxMaker.fixed_cell_relaxation(
           user_params={"PAO.BasisSize": "TZP", "a2s_kpts": [4, 4, 4]}
       ),
       optimizer="BFGS",
       fmax=0.05,
   )
   flow = maker.make(initial_structure, final_structure)

**Key features of ASE NEB:**
- Fully iterative optimization with NEB force convergence
- Runs SIESTA static calculations in parallel for each iteration
- No FLOS library dependency (more portable)
- Automatic convergence checking and history tracking
- Flexible endpoint relaxation (initial, final, both, or neither)
- Separate RelaxMaker for each endpoint

**Endpoint Relaxation Options (both Lua and ASE NEB):**
- ``relax_endpoints=False``: No relaxation (use structures as provided)
- ``relax_endpoints=True``: Relax both endpoints (backward compatible)
- ``relax_endpoints="initial"``: Relax only initial structure
- ``relax_endpoints="final"``: Relax only final structure
- ``relax_endpoints="both"``: Relax both (explicit, same as True)
- ``relax_initial_maker``: Custom RelaxMaker for initial endpoint
- ``relax_final_maker``: Custom RelaxMaker for final endpoint

**Comparison:**
- **Lua-based**: Faster (single SIESTA MD run), requires FLOS
- **ASE-based**: More flexible, portable, easier to customize
- **Both support**: Full endpoint relaxation control

----

Large-Scale Systems
===================

OrderN for Large Systems
-------------------------

Linear-scaling DFT for >100 atoms:

.. code-block:: python

   from atomate2.siesta.sets.tiers import apply_tier_preset

   maker = RelaxMaker.fixed_cell_relaxation()
   maker = apply_tier_preset(maker, "large_system")

   # Additional OrderN parameters
   job = maker.make(large_structure)
   job = update_user_siesta_settings(job, {
       "SolutionMethod": "OrderN",
       "ON.MaxNumIter": 1000,
       "ON.eta": 1e-3,
   })

HPC Parallel Optimization
--------------------------

.. code-block:: python

   from atomate2.siesta.sets.tiers import apply_tier_preset

   maker = RelaxMaker.variable_cell_relaxation()
   maker = apply_tier_preset(maker, "parallel_hpc")

   # Optimize for specific cluster
   job = maker.make(structure)
   job = update_user_siesta_settings(job, {
       "Diag.ParallelOverK": True,
       "NumberOfEigenStates": 500,
   })

----

Error-Resilient Production Workflows
=====================================

Full Production Stack
---------------------

Combining all infrastructure features:

.. code-block:: python

   from fireworks import LaunchPad
   from jobflow.managers.fireworks import flow_to_workflow
   from maggma.stores import MongoStore
   from atomate2.siesta.jobs.core import RelaxMaker
   from atomate2.siesta.sets.tiers import apply_tier_preset
   from atomate2.siesta.custodian import SCFConvergenceHandler

   # 1. Database
   store = MongoStore(
       database="production_db",
       collection_name="materials",
   )

   # 2. Custom error handling
   custom_handlers = [
       SCFConvergenceHandler(max_attempts=15),
   ]

   # 3. Maker with all features
   maker = RelaxMaker.variable_cell_relaxation(
       use_custodian=True,
       custodian_handlers=custom_handlers,
       custodian_max_errors=20,
   )

   # 4. Apply preset
   maker = apply_tier_preset(maker, "high_accuracy_relax")

   # 5. Create workflow
   jobs = [maker.make(s) for s in structures]
   flow = Flow(jobs)

   # 6. Submit to HPC
   lpad = LaunchPad.from_file("launchpad.yaml")
   wf = flow_to_workflow(flow, store=store)
   lpad.add_wf(wf)

Automatic Retry Logic
----------------------

.. code-block:: python

   # If calculation fails, automatically restart with looser parameters
   from atomate2.siesta.jobs.core import RelaxMaker

   # First attempt: tight convergence
   relax_tight = RelaxMaker.variable_cell_relaxation(
       user_params={
           "MD.MaxForceTol": "0.005 eV/Ang",
           "SCF.DM.Tolerance": 1e-6,
       },
       use_custodian=True,
   ).make(structure)

   # If fails, use looser (custodian handles automatic retry)
   # Custodian will progressively relax SCF parameters

----

Electrocatalysis Workflows
==========================

ORR, OER, and HER workflows apply the computational hydrogen electrode (CHE) model to
compute reaction free-energy diagrams and overpotentials automatically.

.. code-block:: python

   from atomate2.siesta.flows.electrocatalysis import (
       ORRFlowMaker,            # Oxygen reduction reaction (4 electron)
       OERFlowMaker,            # Oxygen evolution reaction (4 electron)
       HERFlowMaker,            # Hydrogen evolution reaction
       BifunctionalFlowMaker,   # Combined ORR/OER bifunctional gap
   )

   # Screen a catalyst surface; overpotential + free-energy diagram are produced automatically
   flow = ORRFlowMaker().make(slab_structure)

Outputs include free-energy diagrams, overpotential summaries, and (for
``BifunctionalFlowMaker``) the ORR/OER bifunctional gap. The tier presets
``electrocatalysis_dirty/basic/intermediate/gas_phase`` tune cost vs. accuracy.

**See**: ``tutorials/siesta/02-workflows/09-electrocatalysis``

----

Defect Workflows
================

``DefectFlowMaker`` automates vacancy, substitution, antisite, and interstitial defect
calculations, including charge states, chemical potentials, and finite-size corrections.

.. code-block:: python

   from atomate2.siesta.flows.defects import DefectFlowMaker

   maker = DefectFlowMaker(
       include_bandstructure=True,   # per-defect band structure (auto k-path)
       include_pdos=True,            # per-defect PDOS for defect-level analysis
   )
   flow = DefectFlowMaker.from_pristine_structure(bulk_structure)

Defect and host calculations are automatically spin-polarized (``Spin = polarized``) and can
emit band structure + PDOS for defect-level analysis. Surface-aware generators
(``SurfaceVacancyGenerator``, ``SurfaceInterstitialGenerator``,
``SurfaceSubstitutionGenerator``) target slab surfaces for 2D-material and catalysis studies.

**See**: ``tutorials/siesta/02-workflows/08-defects``

----

Surface Energy Convergence
==========================

``SurfaceEnergyConvergenceFlowMaker`` systematically converges surface energy against slab
thickness and/or vacuum thickness, using symmetric slabs with optional termination selection
and chemical-potential handling for non-stoichiometric surfaces.

.. code-block:: python

   from atomate2.siesta.flows.surface import (
       SurfaceEnergyConvergenceFlowMaker,   # convergence modes: layers / vacuum / both
       MultiSurfaceEnergyFlowMaker,         # compare multiple Miller surfaces
       AdsorptionScanFlowMaker,             # grid-based adsorption site scanning
   )

   flow = SurfaceEnergyConvergenceFlowMaker().make(bulk_structure)

``AdsorptionScanFlowMaker`` supports slab-energy reuse (``precalc_slab_energy``) so
multi-adsorbate screening runs a single slab calculation instead of one per adsorbate.

**See**: ``tutorials/siesta/02-workflows/03-surfaces-and-adsorption``

----

Heterostructure Workflows
=========================

``InterfaceFlowMaker`` builds 2D heterostructure interfaces (lattice matching via strain or
supercell mode), optimizes the interlayer distance, and computes the binding energy.

.. code-block:: python

   from atomate2.siesta.flows.heterostructures.interface import InterfaceFlowMaker

   flow = InterfaceFlowMaker().make(bottom_layer, top_layer)

**See**: ``tutorials/siesta/02-workflows/10-heterostructures``

----

Best Practices
==============

Workflow Design
---------------

1. **Start simple, add complexity**

   * Test each step individually
   * Combine when confident

2. **Use tier presets**

   * Material-specific starting points
   * Override as needed

3. **Enable custodian for production**

   * Automatic error recovery
   * Saves time on HPC

4. **Database storage for high-throughput**

   * Programmatic access to results
   * Easy querying and analysis

5. **Document parameters**

   * Include in workflow scripts
   * Note tier/preset used

Parameter Selection
-------------------

1. **Converge first**

   * Run convergence studies once
   * Apply converged parameters to all workflows

2. **Use appropriate tier**

   * intermediate: Most cases
   * advanced: Specialized properties
   * expert: Performance tuning only

3. **Balance accuracy vs. cost**

   * Screening: basic/intermediate
   * Production: intermediate/advanced
   * Benchmarking: advanced/expert

Error Handling
--------------

1. **Always enable custodian for**:

   * HPC batch jobs
   * High-throughput workflows
   * Challenging systems

2. **Monitor custodian.json**:

   * Check corrections applied
   * Identify systematic issues

3. **Adjust max_errors based on**:

   * System difficulty
   * Throughput requirements
   * Available compute time

----

Example: Complete Characterization Workflow
============================================

.. code-block:: python

   from jobflow import Flow
   from maggma.stores import MongoStore
   from atomate2.siesta.jobs.core import (
       RelaxMaker,
       BandStructureMaker,
       SiestaPhononFlowMaker,
   )
   from atomate2.siesta.flows.elastic import ElasticFlowMaker
   from atomate2.siesta.flows.eos import EOSMaker
   from atomate2.siesta.sets.tiers import apply_tier_preset

   # Setup
   store = MongoStore(database="materials_db", collection_name="full_char")

   # 1. Structure relaxation
   relax = RelaxMaker.variable_cell_relaxation(
       use_custodian=True,
   )
   relax = apply_tier_preset(relax, "high_accuracy_relax")
   relax_job = relax.make(structure)

   # 2. Electronic structure
   bands = BandStructureMaker().make(relax_job.output.structure)

   # 3. Mechanical properties
   elastic = ElasticFlowMaker().make(relax_job.output.structure)
   eos = EOSMaker().make(relax_job.output.structure)

   # 4. Vibrational properties
   phonon = SiestaPhononFlowMaker(
       min_length=15.0,
       mesh=(100, 100, 100),
       generate_plots=True,
   ).make(relax_job.output.structure)

   # Combine and run
   flow = Flow([relax_job, bands, elastic, eos, phonon])
   results = run_locally(flow, create_folders=True, store=store)

   # Access results from database
   # All properties stored with full provenance

----

See Also
========

* :doc:`features` - Feature overview
* :doc:`tier-system` - Tier-based configuration
* :doc:`custodian` - Error handling
* :doc:`tutorials/index` - Hands-on tutorials

----

.. tip::

   Start with simple workflows (relax → bands) and gradually add complexity.
   Use tier presets as starting points and customize as needed.
