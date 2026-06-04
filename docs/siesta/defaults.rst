===============================
Default Values Reference
===============================

This page documents all default values used by atomate2siesta. Understanding these defaults helps you make informed decisions about when to use them and when to customize.

.. contents:: Quick Navigation
   :local:
   :depth: 2

Overview
========

atomate2siesta uses carefully chosen defaults that work well for most materials calculations. All defaults can be overridden using ``user_params`` or tier presets.

.. important::

   **Default XC Functional**: atomate2siesta uses **PBE** as the default exchange-correlation functional. PBE (Perdew-Burke-Ernzerhof) is the most widely used GGA functional in materials science.

Exchange-Correlation Functional
=================================

**Location**: ``src/atomate2/siesta/dataclass/exchange_correlation_functionals.py:75``

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Parameter
     - Default Value
     - Description
   * - ``xc_functional``
     - ``GGA``
     - Generalized Gradient Approximation
   * - ``xc_authors``
     - **PBE**
     - Perdew-Burke-Ernzerhof functional
   * - ``xc_use_bsc_cell_xc``
     - ``False``
     - BSC cell XC correction (for vdW functionals)

Why PBE?
--------

PBE was chosen as the default because:

1. **Most widely used**: Standard in materials science community
2. **Excellent benchmarking**: Extensive validation across all material types
3. **General purpose**: Works well for molecules, surfaces, and bulk materials
4. **Broad compatibility**: Most pseudopotential libraries available in PBE

**When to use PBEsol instead**:

* Better lattice constants needed (PBEsol optimized for solids)
* Bulk materials where structural accuracy is critical
* Comparison with PBEsol literature

**When to use other functionals**:

* **revPBE**: Better for surfaces and adsorption energies
* **RPBE**: Improved molecular adsorption
* **BLYP**: Alternative GGA with different properties

**How to change**:

.. code-block:: python

   # Change to PBEsol for better lattice constants
   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "xc_authors": "PBEsol",
       }
   )

   # Or use revPBE for surface calculations
   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "xc_authors": "revPBE",
       }
   )

Pseudopotentials
=================

**Location**: ``src/atomate2/siesta/dataclass/pseudopotentials.py``

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Parameter
     - Default Value
     - Description
   * - ``pseudo_family``
     - ``ONCVPSP``
     - Optimized Norm-Conserving Vanderbilt pseudopotentials
   * - ``pseudo_version``
     - ``0.4``
     - PseudoDojo version 0.4
   * - ``pseudo_quality``
     - ``Standard``
     - Standard accuracy (Stringent also available)
   * - ``pseudo_relativistic``
     - ``FR``
     - Fully Relativistic (SR = Scalar Relativistic)
   * - ``xc_authors`` (inferred)
     - ``PBE``
     - Must match XC functional

**Automatic Path Construction**:

Given ``pseudo_base_path = ~/.siesta/pseudos`` and ``xc_authors = PBE``, the full path is automatically constructed as:

.. code-block:: text

   ~/.siesta/pseudos/ONCVPSP-PBE-FR-PDv0.4-Standard/

**Available Pseudopotentials**:

.. code-block:: bash

   $ atomate2siesta-pseudos available

   Available pseudopotentials:
   - ONCVPSP-PBE-SR-PDv0.4-Standard
   - ONCVPSP-PBE-FR-PDv0.4-Standard
   - ONCVPSP-PBEsol-SR-PDv0.4-Standard
   - ONCVPSP-PBEsol-FR-PDv0.4-Standard
   - ONCVPSP-PBE-SR-PDv0.4-Stringent
   - ONCVPSP-PBE-FR-PDv0.4-Stringent

Basis Sets and Projectors
===========================

**Location**: ``src/atomate2/siesta/dataclass/basis_sets_and_projectors.py``

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Parameter
     - Default Value
     - Description
   * - ``pao_basis_size``
     - ``DZP``
     - Double-Zeta Polarized (standard quality)
   * - ``pao_energy_shift``
     - ``100 meV``
     - Energy shift for basis confinement
   * - ``pao_basis_split_norm``
     - ``0.15``
     - Split norm for multiple-zeta orbitals

**Basis Size Options**:

* ``SZ``: Single-Zeta (minimal, testing only)
* ``DZ``: Double-Zeta (minimum for production)
* **``DZP``**: Double-Zeta Polarized (recommended default)
* ``TZP``: Triple-Zeta Polarized (high accuracy, expensive)

K-Point Sampling
=================

**Location**: ``src/atomate2/siesta/dataclass/kpoint_sampling.py``

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Parameter
     - Default Value
     - Description
   * - ``kpts``
     - **Automatic**
     - Gamma-centered Monkhorst-Pack grid
   * - K-point density
     - ~0.04 Å⁻¹
     - Determined by structure size

**Automatic K-point Generation**:

atomate2siesta automatically determines k-points based on reciprocal lattice vectors unless explicitly specified.

**Common Manual Settings**:

.. code-block:: python

   # For metals (dense mesh needed)
   user_params = {"a2s_kpts": [8, 8, 8]}

   # For semiconductors
   user_params = {"a2s_kpts": [4, 4, 4]}

   # For molecules (gamma point only)
   user_params = {"a2s_kpts": [1, 1, 1]}

Real-Space Grid
================

**Location**: ``src/atomate2/siesta/dataclass/real_space_grid_parameters.py``

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Parameter
     - Default Value
     - Description
   * - ``mesh_cutoff``
     - ``200 Ry``
     - Real-space grid fineness
   * - ``grid_cell_sampling``
     - Not set
     - Alternative to mesh_cutoff

**Recommended Values**:

* Quick/testing: 150-200 Ry
* Production: 250-300 Ry
* High accuracy: 350-400 Ry
* Molecules: 300+ Ry

SCF Convergence
================

**Location**: ``src/atomate2/siesta/dataclass/scf_loop_parameters.py``

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Parameter
     - Default Value
     - Description
   * - ``scf_max_iterations``
     - ``50``
     - Maximum SCF cycles
   * - ``scf_dm_tolerance``
     - ``1.0e-4``
     - Density matrix convergence criterion
   * - ``scf_mixer_weight``
     - ``0.1``
     - Mixing weight for density matrix
   * - ``scf_mixer_history``
     - ``5``
     - Number of previous iterations to mix

**Convergence Difficulties**:

If SCF doesn't converge, custodian will automatically try:

1. Reduce mixing weight (0.1 → 0.05 → 0.01)
2. Increase history (5 → 10)
3. Change mixing method
4. Reduce temperature for Fermi distribution

Spin Settings
==============

**Location**: ``src/atomate2/siesta/dataclass/spin_settings.py``

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Parameter
     - Default Value
     - Description
   * - ``spin``
     - ``non-polarized``
     - Non-magnetic calculation
   * - Auto-override
     - **Yes**
     - Changes to ``polarized`` if magnetic moments detected

**Automatic Magnetic Moment Detection**:

When magnetic moments are present in structure:

.. code-block:: python

   structure.add_site_property("magmom", [4.0, 4.0])  # Fe atoms

   # Automatically sets:
   # - Spin: polarized
   # - DM.InitSpin block generated

Structural Relaxation
======================

**Location**: ``src/atomate2/siesta/dataclass/general_constraints.py``

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Parameter
     - Default Value
     - Description
   * - ``md_max_force_tol``
     - ``0.04 eV/Å``
     - Force convergence criterion
   * - ``md_max_stress_tol``
     - ``1.0 GPa``
     - Stress convergence (variable cell)
   * - ``md_max_cg_displ``
     - ``0.2 Bohr``
     - Maximum displacement per step
   * - ``md_num_cg_steps``
     - ``100``
     - Maximum geometry steps

Electronic Structure Options
==============================

**Location**: ``src/atomate2/siesta/dataclass/electronic_structure_calculation_options.py``

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Parameter
     - Default Value
     - Description
   * - ``electronic_temperature``
     - ``25 meV``
     - Electronic smearing (Fermi-Dirac)
   * - ``occupation_function``
     - ``FD``
     - Fermi-Dirac distribution
   * - ``save_density_matrix``
     - ``True``
     - Save DM for restart

Solver and Performance
=======================

**Location**: ``src/atomate2/siesta/dataclass/solvers_and_performance_options.py``

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Parameter
     - Default Value
     - Description
   * - ``diagonalization_algorithm``
     - System-dependent
     - Automatic selection
   * - ``number_of_eigenvalues``
     - Automatic
     - Based on number of electrons

Configuration File
===================

**Default Location**: ``~/.atomate2.yaml``

**Default Contents** (if file doesn't exist):

.. code-block:: yaml

   # atomate2siesta configuration
   # Note: Package name is "atomate2siesta" (no hyphen)

   # Environment variables (alternative to this file):
   # - SIESTA_PP_PATH: Path to pseudopotentials
   # - SIESTA_CMD: Command to run SIESTA
   # - atomate2_SIESTA_PP_PATH: Same as SIESTA_PP_PATH

   # No other defaults - all calculation defaults in dataclasses

**Configuration Precedence**:

1. **Highest**: User params in code (``user_params`` argument)
2. **Medium**: Environment variables (``SIESTA_PP_PATH``, etc.)
3. **Lowest**: Configuration file (``~/.atomate2.yaml``)
4. **Fallback**: Dataclass defaults (documented on this page)

Tier System Defaults
=====================

The tier system provides 18 material-specific presets organized in 7 categories. Each preset overrides certain defaults for specific use cases.

**Default Tier**: No tier applied unless specified with ``apply_tier_preset()``

**Common Presets**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Preset
     - What it changes
   * - ``relax_standard``
     - Balanced accuracy for relaxation (DZP, moderate k-points)
   * - ``band_structure``
     - High-quality band structures (dense k-path, tight convergence)
   * - ``quick_test``
     - Fast calculations for testing (SZ basis, coarse grids)
   * - ``high_accuracy``
     - Maximum accuracy (TZP, dense grids, tight tolerances)

See :doc:`tier-system` for complete preset documentation.

Dry-Run Mode
=============

**Default**: ``dry_run=False`` (runs actual SIESTA calculation)

**Override**:

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
   # Generates input files but doesn't run SIESTA

**When to use dry-run**:

* Testing workflows before production runs
* Debugging input generation
* Previewing calculations (99.9% time savings)

Custodian Settings
===================

**Default**: ``use_custodian=False`` (direct SIESTA execution)

**Override**:

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(
       use_custodian=True,
       custodian_max_errors=10,
   )

**Default Error Handlers** (when custodian enabled):

1. SCF convergence handler (5-level rescue strategy)
2. Time limit handler
3. Memory limit handler
4. Geometry convergence handler

See :doc:`custodian` for detailed error handling documentation.

How to Override Defaults
==========================

Method 1: User Params (Most Common)
-------------------------------------

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "xc_authors": "PBE",           # Change from PBEsol to PBE
           "PAO.BasisSize": "TZP",        # Change from DZP to TZP
           "a2s_kpts": [6, 6, 6],             # Explicit k-points
           "Mesh.Cutoff": "300 Ry",       # Change from 200 Ry
           "a2s_pseudo_relativistic": "SR",   # Change from FR to SR
       }
   )

Method 2: Tier Presets
-----------------------

.. code-block:: python

   from atomate2.siesta.sets.tiers import apply_tier_preset

   maker = RelaxMaker.fixed_cell_relaxation()
   maker = apply_tier_preset(maker, "high_accuracy")
   # Applies preset that overrides multiple defaults

Method 3: Powerups (After Job Creation)
-----------------------------------------

.. code-block:: python

   from atomate2.siesta.powerups import update_user_siesta_settings

   job = maker.make(structure)
   job = update_user_siesta_settings(job, {
       "SCF.Mixer.Weight": 0.05,
       "ElectronicTemperature": "50 meV",
   })

Method 4: Environment Variables
---------------------------------

.. code-block:: bash

   export SIESTA_PP_PATH=/path/to/pseudos
   export SIESTA_CMD="siesta < siesta.fdf > siesta.out"

Summary Table
==============

**Most Important Defaults to Remember**:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Setting
     - Default
     - Why This Value?
   * - **XC Functional**
     - **PBE**
     - Most widely used in materials science
   * - Basis
     - DZP
     - Good accuracy/cost balance
   * - K-points
     - Automatic
     - Structure-dependent
   * - Mesh Cutoff
     - 200 Ry
     - Fast convergence for testing
   * - Pseudos
     - ONCVPSP-PBE-FR-0.4
     - Matches XC functional
   * - SCF Tolerance
     - 1e-4
     - Production quality
   * - Force Tolerance
     - 0.04 eV/Å
     - Standard for relaxation

See Also
=========

* :doc:`usage` - Basic usage patterns
* :doc:`tier-system` - Material-specific presets
* :doc:`installation` - Configuration setup
* :doc:`troubleshooting` - Common issues with defaults
* :doc:`fdf-parameters` - Complete SIESTA parameter guide
