=======================
Advanced Learning Paths
=======================

Complex multi-step calculations for materials properties including structural, mechanical,
vibrational, and surface properties.

----

Tutorial 16: Phonon Calculations ⭐ NEW
=======================================

**Learning Objectives**:

* Perform phonon calculations with phonopy integration
* Analyze vibrational properties and thermal properties
* Understand supercell convergence
* Interpret phonon band structures and DOS

**Key Concepts**:

* Phonopy force constant calculation
* Supercell generation (automatic and manual)
* Symmetry-reduced displacements
* Thermal properties (heat capacity, entropy, free energy)
* Automatic plotting (4 plot types)

**Workflow Steps**:

1. Structure relaxation (optional)
2. Supercell generation based on ``min_length``
3. Displacement generation (symmetry-reduced)
4. SIESTA force calculations for each displacement
5. Phonopy analysis and property calculation

**Example**:

.. code-block:: python

   from atomate2.siesta.jobs.core import SiestaPhononFlowMaker
   from pymatgen.core import Structure

   structure = Structure.from_file("Si.cif")

   # Automatic supercell + all plots
   maker = SiestaPhononFlowMaker(
       min_length=12.0,          # Supercell ≥ 12 Å
       displacement=0.01,        # Atomic displacement (Å)
       mesh=(50, 50, 50),        # Q-point mesh for DOS
       generate_plots=True,      # Master switch
       plot_band_structure=True,
       plot_dos=True,
       plot_thermal=True,
       write_summary=True,
   )

   flow = maker.make(structure)
   results = run_locally(flow, create_folders=True)

**Output Files**:

* ``phonon_bands.png`` - Phonon dispersion with high-symmetry path
* ``phonon_dos.png`` - Phonon density of states
* ``thermal_properties.png`` - Cv, S, F vs Temperature
* ``phonon_summary.txt`` - Comprehensive text summary

**Convergence Guidelines**:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Parameter
     - Recommendation
     - Criterion
   * - Supercell size
     - Vectors > 10-15 Å
     - Frequencies converged within 0.03 THz
   * - Displacement
     - 0.005-0.02 Å
     - Harmonic approximation valid
   * - Force k-points
     - 8×8×8 minimum
     - Force accuracy < 0.001 eV/Å
   * - Mesh cutoff
     - 400-500 Ry
     - Higher than relaxation

**Interpreting Results**:

* **Positive frequencies**: Normal vibrational modes
* **Imaginary frequencies** (negative): Structural instability

  * Check structure relaxation convergence
  * May indicate phase transition
  * Could be due to insufficient supercell size

* **Thermal properties**:

  * Heat capacity → 3Nk_B per atom (Dulong-Petit limit)
  * Entropy measures vibrational disorder
  * Free energy F = E - TS

📁 **Location**: ``tutorials/05-vibrational-properties/01-phonons/``

⏱️ **Time**: 60 minutes

⭐ **Difficulty**: Advanced

📄 **Documentation**:
   * ``README.md`` - Complete tutorial (400+ lines)
   * ``IMPLEMENTATION_SUMMARY.md`` - Technical details
   * ``PLOTTING_IMPLEMENTATION.md`` - Plotting system

----

Tutorial 17: Surface Energy Calculations ⭐ NEW
===============================================

**Learning Objectives**:

* Calculate surface energies for crystalline materials
* Generate slabs with multiple terminations
* Understand polar vs. non-polar surfaces
* Perform multi-surface comparison studies
* Predict equilibrium crystal morphology

**Key Concepts**:

* Surface energy formula: γ = (E_slab - N×E_bulk) / A
* Layer-based slab specification
* Symmetric vs. asymmetric slabs
* K-point sampling for 2D-periodic systems
* Dipole corrections for polar surfaces
* Wulff construction

**Workflow Architecture**:

**Phase 1**: Slab generation (external Python script)

.. code-block:: bash

   python generate_slabs_refactored.py bulk.fdf \
       --miller 0,0,1 \
       --layers 4 \
       --vacuum 15.0 \
       --no-symmetrize \
       --output-dir slabs_001

**Phase 2**: Energy calculations (atomate2siesta)

.. code-block:: python

   from atomate2.siesta.flows.multi_surface import MultiSurfaceEnergyFlowMaker
   from atomate2.siesta.jobs.core import StaticMaker

   # Setup parameters
   bulk_params = {
       "PAO.BasisSize": "DZP",
       "a2s_kpts": [6, 6, 6],      # 3D periodic
       "Mesh.Cutoff": "100 Ry",
   }

   slab_params = {
       "PAO.BasisSize": "DZP",
       "a2s_kpts": [6, 6, 1],      # 2D periodic (Γ in z)
       "Mesh.Cutoff": "100 Ry",
   }

   # Create workflow
   multi_surface_maker = MultiSurfaceEnergyFlowMaker(
       miller_indices=[(1,0,0), (1,1,0), (1,1,1)],
       bulk_static_maker=StaticMaker(...),
       slab_static_maker=StaticMaker(...),
       slab_layers=4,           # Number of atomic layers
       vacuum_size=15.0,        # Vacuum spacing (Å)
       symmetrize=False,        # Explore all terminations
       formula_units_per_cell=4,
       plot_results=True,
       write_summary=True,
   )

   flow = multi_surface_maker.make(bulk_structure)
   results = run_locally(flow, create_folders=True)

**Output Files**:

* ``multi_surface_comparison.png`` - Two-panel bar/scatter plot
* ``multi_surface_summary.txt`` - Complete analysis
* Individual slab calculation directories

**Convergence Testing**:

1. **Slab thickness** (number of layers)

   * Test: 4, 6, 8, 10 layers
   * Criterion: γ change < 0.01 eV/Ų

2. **Vacuum spacing**

   * Test: 15, 20, 25 Å
   * Criterion: E_slab change < 0.001 eV/atom

3. **K-point sampling** (in-plane)

   * Test: [4,4,1], [6,6,1], [8,8,1], [10,10,1]
   * Criterion: γ change < 0.005 eV/Ų

4. **Mesh cutoff**

   * Use same cutoff for bulk and slab
   * Follow bulk convergence guidelines

**Important Considerations**:

.. warning::

   **Polar Surfaces**:

   Surfaces with net dipole moment (e.g., MgO (001), ZnO (0001)) require special treatment:

   * Use symmetric slabs (same termination both sides)
   * Enable dipole corrections: ``SlabDipoleCorrection True``
   * Ensure stoichiometric slabs

   Asymmetric polar slabs may have:

   * Non-physical electric field
   * Slow convergence with thickness
   * Incorrect surface energies

**Surface Energy Units**:

* eV/Ų (common in DFT)
* J/m² (experimental)
* Conversion: 1 eV/Ų = 16.0218 J/m²

**Example Results** (MgO):

.. code-block:: text

   Surface    γ (eV/Ų)    γ (J/m²)
   (1 0 0)    0.327       5.24
   (1 1 0)    0.302       4.84   ← Global minimum
   (1 1 1)    0.366       5.87

📁 **Location**: ``tutorials/06-surfaces-and-adsorption/01-surface-energy/``

⏱️ **Time**: 90 minutes

⭐ **Difficulty**: Expert

📄 **Documentation**:
   * ``README.md`` - Complete tutorial (450+ lines)
   * ``SURFACE_ENERGY_STRATEGY.md`` - Implementation strategy

----

Tutorial 09-12: Structural & Mechanical Properties
===================================================

Tutorial 09: Equation of State (EOS)
-------------------------------------

**Learning Objectives**:

* Fit equation of state to energy-volume data
* Determine bulk modulus and equilibrium volume
* Understand different EOS models (Birch-Murnaghan, Vinet, etc.)

**Example**:

.. code-block:: python

   from atomate2.siesta.flows.eos import EOSMaker

   maker = EOSMaker(
       number_of_frames=7,  # Volume points
       postprocessor="eos_analysis",
   )

   flow = maker.make(structure)

📁 **Location**: ``tutorials/03-advanced-workflows/01-eos/``

⏱️ **Time**: 40 minutes

⭐ **Difficulty**: Intermediate

Tutorial 10: EOS with Basis Convergence
----------------------------------------

**Learning Objectives**:

* Combine EOS with basis parameter convergence
* Understand basis size effects on bulk modulus
* Perform systematic convergence studies

📁 **Location**: ``tutorials/03-advanced-workflows/02-eos-basis-convergence/``

⏱️ **Time**: 90 minutes

⭐ **Difficulty**: Advanced

Tutorial 11: Elastic Constants
-------------------------------

**Learning Objectives**:

* Calculate full elastic tensor
* Determine mechanical properties (bulk/shear modulus, Poisson ratio)
* Assess mechanical stability

**Example**:

.. code-block:: python

   from atomate2.siesta.flows.elastic import ElasticFlowMaker

   maker = ElasticFlowMaker()
   flow = maker.make(structure)

   # Access results
   elastic_tensor = results.output.elastic_tensor
   bulk_modulus = results.output.bulk_modulus  # GPa
   shear_modulus = results.output.shear_modulus
   poisson_ratio = results.output.poisson_ratio

📁 **Location**: ``tutorials/03-advanced-workflows/03-elastic-constants/``

⏱️ **Time**: 60 minutes

⭐ **Difficulty**: Advanced

Tutorial 12: Nudged Elastic Band (NEB)
---------------------------------------

**Learning Objectives**:

* Calculate transition state pathways
* Determine activation barriers
* Understand reaction mechanisms

**Example**:

.. code-block:: python

   from atomate2.siesta.jobs.core import NEBMaker

   maker = NEBMaker(
       n_images=7,  # Intermediate images
   )

   flow = maker.make(
       structures=[initial_structure, final_structure]
   )

📁 **Location**: ``tutorials/03-advanced-workflows/04-neb/``

⏱️ **Time**: 120 minutes

⭐ **Difficulty**: Expert

----

Tutorial 18: Tier-Based Calculations ⭐ NEW
===========================================

**Learning Objectives**:

* Understand the 4-tier module hierarchy
* Use material-specific presets
* Customize tier configurations
* Optimize workflow performance

**Key Concepts**:

* 4 tiers: basic → intermediate → advanced → expert
* 24 dataclass modules organized by complexity
* 14 material-specific presets
* Automatic module activation
* Parameter merging precedence

**The Four Tiers**:

.. list-table::
   :header-rows: 1
   :widths: 20 15 40 25

   * - Tier
     - Modules
     - Use Case
     - Performance
   * - basic
     - 6
     - Quick tests, debugging
     - ~17 ms
   * - intermediate
     - 12
     - Standard calculations (DEFAULT)
     - ~20 ms
   * - advanced
     - 19
     - Phonons, optical, DFT+U
     - ~22 ms
   * - expert
     - 24
     - Performance tuning, large systems
     - ~23 ms

**Direct Tier Usage**:

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker

   # Use advanced tier for phonon preparation
   maker = RelaxMaker.fixed_cell_relaxation(
       tier="advanced",
       enabled_modules=["phonons"],
       user_params={
           "PAO.BasisSize": "DZP",
           "a2s_kpts": [6, 6, 6],
       }
   )

**Material-Specific Presets**:

.. code-block:: python

   from atomate2.siesta.sets.tiers import apply_tier_preset

   maker = RelaxMaker.fixed_cell_relaxation()

   # Apply surface_metal preset
   maker = apply_tier_preset(maker, "surface_metal")
   # Automatically sets:
   # - tier="intermediate"
   # - OccupationFunction="MP"
   # - ElectronicTemperature="300 K"
   # - SCF.Mixer.Weight=0.005

**Available Presets** (14 total):

* **Structural**: ``basic_relax``, ``relax_standard``, ``high_accuracy_relax``
* **Surface**: ``surface_metal``, ``surface_semiconductor``
* **Magnetic**: ``magnetic_2d``, ``magnetic_correlated``
* **Phonon**: ``phonon_standard``, ``phonon_high_accuracy``
* **Optical**: ``optical_response``, ``band_structure``
* **Performance**: ``large_system``, ``parallel_hpc``
* **Testing**: ``convergence_test``

**Override Preset Parameters**:

.. code-block:: python

   maker = apply_tier_preset(
       maker,
       "phonon_high_accuracy",
       override_params={
           "a2s_kpts": [10, 10, 10],    # Denser than preset
           "Mesh.Cutoff": "600 Ry", # Higher than preset
       }
   )

   # Merging precedence: preset < existing < override

📁 **Location**: ``tutorials/07-advanced-features/01-tier-system/``

⏱️ **Time**: 30 minutes

⭐ **Difficulty**: Intermediate

📄 **Documentation**:
   * ``README.md`` - Complete tutorial (370+ lines)
   * See :doc:`/siesta/tier-system` for full tier system documentation

----

Advanced Features Tutorials ⭐ NEW
===================================

Five comprehensive tutorials demonstrating MEDIUM priority dataclass modules with direct SIESTA FDF format.

Tutorial 23: DOS Calculations
------------------------------

**Learning Objectives**:

* Calculate density of states with StaticMaker and RelaxMaker
* Use direct SIESTA FDF format for ProjectedDensityOfStates
* Understand automatic comment header generation
* Interpret DOS output files

**Key Concepts**:

* Direct SIESTA syntax: ``"ProjectedDensityOfStates": ["EF -10.000 10.000 0.100 200 eV"]``
* Automatic comment headers: ``# DensityOfStatesAndBandStructure Settings``
* Works with all makers (StaticMaker, RelaxMaker, BandStructureMaker)
* Dry-run mode for preview

**Example**:

.. code-block:: python

   from atomate2.siesta.jobs.core import StaticMaker
   from pymatgen.core import Structure

   structure = Structure.from_file("Si.cif")

   maker = StaticMaker.scf(
       user_params={
           "xc": "GGA",
           "a2s_mesh_cutoff": "200 Ry",
           "a2s_kpts": [4, 4, 4],
           "PAO.BasisSize": "DZP",
           "ProjectedDensityOfStates": ["EF -10.000 10.000 0.100 200 eV"],
       },
       dry_run=True,
       dry_run_output_dir="dos_preview"
   )

   job = maker.make(structure)
   results = run_locally(job, create_folders=True)

📁 **Location**: ``tutorials/07-advanced-features/05-dos-calculations/``

⏱️ **Time**: 20 minutes

⭐ **Difficulty**: Intermediate

Tutorial 24: Phonon Inputs
---------------------------

**Learning Objectives**:

* Configure force constants parameters for phonon calculations
* Use MD.TypeOfRun, MD.FCDispl, MD.FCfirst, MD.FClast
* Prepare structures for phonopy post-processing

**Key Parameters**:

* ``MD.TypeOfRun``: "FC" (force constants mode)
* ``MD.FCDispl``: Atomic displacement magnitude (Å)
* ``MD.FCfirst``, ``MD.FClast``: Atom range for displacement

📁 **Location**: ``tutorials/07-advanced-features/06-phonon-inputs/``

⏱️ **Time**: 15 minutes

⭐ **Difficulty**: Intermediate

Tutorial 25: Optical Properties
--------------------------------

**Learning Objectives**:

* Calculate optical absorption and dielectric function
* Configure energy range, broadening, and scissor operator
* Understand band gap corrections
* Interpret optical output files

**Key Parameters**:

* ``OpticalCalculation``: Enable optical calculation
* ``Optical.Energy.Minimum/Maximum``: Energy range for spectrum
* ``Optical.Broaden``: Spectral broadening (eV)
* ``Optical.Scissor``: Band gap correction (eV)

**Output Files**:

* ``siesta.EPSIMG`` - Imaginary part of dielectric function
* ``siesta.EPSREAL`` - Real part of dielectric function

📁 **Location**: ``tutorials/07-advanced-features/07-optical-properties/``

⏱️ **Time**: 25 minutes

⭐ **Difficulty**: Intermediate

Tutorial 26: DFT+U Calculations
--------------------------------

**Learning Objectives**:

* Apply Hubbard U corrections for correlated systems
* Configure DFT+U parameters for transition metals
* Understand projector methods
* Use common U values for specific elements

**Key Parameters**:

* ``LDAU.UseLDAU``: Enable DFT+U
* ``LDAU.UEffective``: Effective U parameter (U-J) in eV
* ``LDAU.JHund``: Hund's exchange J in eV
* ``LDAU.ProjectorMethod``: PulaySanchez or SimplifiedLDA

**Common Systems**:

* NiO (3d transition metal oxide)
* 3d transition metals: Ni, Co, Fe, Mn
* 4f rare earths: Ce, Eu

**Example**:

.. code-block:: python

   # NiO with DFT+U
   user_params = {
       "xc": "GGA",
       "PAO.BasisSize": "DZP",
       "a2s_kpts": [4, 4, 4],
       "Mesh.Cutoff": "300 Ry",
       "SpinPolarized": "true",
       "LDAU.UseLDAU": "true",
       "LDAU.UEffective": "5.3 eV",  # U-J for Ni
       "LDAU.JHund": "0.0 eV",
       "LDAU.ProjectorMethod": "PulaySanchez",
   }

📁 **Location**: ``tutorials/07-advanced-features/08-dftu/``

⏱️ **Time**: 30 minutes

⭐ **Difficulty**: Advanced

Tutorial 27: Charge/Dipole/Electric Field
------------------------------------------

**Learning Objectives**:

* Apply external electric fields to systems
* Calculate charged defect systems
* Enable dipole corrections for slabs
* Understand field strength units and conventions

**Key Parameters**:

* ``ExternalElectricField``: Enable external field
* ``Efield``: Field strength (eV/Ang)
* ``NetCharge``: System net charge (electrons)
* ``SlabDipoleCorrection``: Dipole correction for slabs

**Example Applications**:

* Ferroelectric materials under applied field
* Charged defect calculations (vacancies, interstitials)
* Surface slab calculations with dipole correction

**Example**:

.. code-block:: python

   # External electric field
   user_params = {
       "ExternalElectricField": "true",
       "Efield": "0.01 eV/Ang",  # Field along z-axis
   }

   # Charged system (e.g., positively charged defect)
   user_params = {
       "NetCharge": "+1",  # Remove 1 electron
   }

📁 **Location**: ``tutorials/07-advanced-features/09-charge-dipole-efield/``

⏱️ **Time**: 25 minutes

⭐ **Difficulty**: Advanced

----

Next Steps
==========

After completing advanced tutorials:

1. **Infrastructure** (Tutorials 13-15) - Production deployment
2. **Custom Workflows** - Combine makers with specific needs
3. **High-Throughput** - Database integration and automation

See :doc:`index` for full tutorial listing.
