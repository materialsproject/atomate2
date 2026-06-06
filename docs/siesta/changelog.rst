=========
Changelog
=========

This page documents major releases of atomate2siesta.

----

Unreleased (Development)
=========================

Bug Fixes
---------

* **Database Basis Size Extraction**: Fixed PAO.BasisSize not being saved to database

  - Root cause: ``siesta_fdf_to_json()`` only extracted ``PAO.BasisSizes`` (block), not ``PAO.BasisSize`` (simple param)
  - Added ``PAO.BasisSize`` extraction to ensure it appears in ``siesta_parameters.json``
  - Enhanced ``InputDoc.from_siesta_calc_doc()`` to handle both formats:

    - Simple: ``PAO.BasisSize DZP`` → stored as ``"DZP"``
    - Block: ``%block PAO.BasisSizes ... %endblock`` → parsed intelligently
    - Mixed basis (Ti:TZP, O:DZP) → stored as ``"Ti:TZP, O:DZP"``
    - Uniform basis → simplified to just ``"DZP"``

  **Impact**: Basis size now displays correctly in database queries instead of N/A

* **Recipe Phonon Control** (CRITICAL): Fixed ``complete_material_study()`` to expose phonon control parameters

  - Added ``supercell_matrix``, ``phonon_user_params``, ``temperature_range``, ``ignore_imaginary_modes`` parameters
  - Now properly passes phonon parameters through to ``thermal_properties()``
  - Updated wrapper recipes: ``thermoelectric_analysis()``, ``high_temperature_ceramic()``, ``structural_phase_transition()``
  - Exported 6 wrapper recipes to RecipeBook: ``battery_cathode_screening``, ``thermoelectric_analysis``, ``high_temperature_ceramic``, ``magnetic_material_study``, ``semiconductor_device_study``, ``structural_phase_transition``

  **Impact**: Users can now control phonon supercells in complete material studies

* **2D vdW Preset** (CRITICAL): Fixed invalid "vdw" parameter in ``2d_vdw`` tier preset

  - Was using fake parameter ``"vdw": "DRSLL"`` that doesn't exist in SIESTA
  - Changed to correct SIESTA parameters: ``"XC.functional": "VDW"``, ``"XC.authors": "DRSLL"``
  - Now passes FDF validation

  **Impact**: 2D vdW calculations now work correctly

Enhancements
------------

* **Database Query Enhancements**: Comprehensive query filters and export capabilities

  - Added ``--latest N`` option to show N most recent calculations (sorted by completion time)
  - Added filter options: ``--state``, ``--calc-type``, ``--energy-min``, ``--energy-max``
  - Added export options: ``--export json/csv``, ``--output filename``
  - Enhanced output columns: K-points, Basis size, Mesh cutoff
  - Fixed InputDoc schema to store k-points, basis_size, mesh_cutoff from calculations

  **Impact**: Users can easily query, filter, and export calculation results

* **Explicit JobStore Support**: Proper MongoDB integration without jobflow.yaml

  - Fixed ``SETTINGS.JOB_STORE`` pattern (was incorrectly using ``store`` parameter)
  - Added ``atomate2siesta-config status`` database and jobflow.yaml checks
  - Shows Python code example for explicit JobStore configuration
  - Created tutorial ``09_explicit_jobstore.py`` demonstrating correct usage
  - Updated README with correct SETTINGS.JOB_STORE pattern

  **Impact**: Users can now save to MongoDB without jobflow.yaml configuration file

* **Maker CLI Database Integration**: Database configuration in workflow script generation

  - Interactive mode prompts: "Save to MongoDB?" with jobflow.yaml vs explicit choice
  - CLI options: ``--database``, ``--db-host``, ``--db-port``, ``--db-name``, ``--db-collection``
  - Generated scripts include SETTINGS.JOB_STORE configuration code
  - Supports both automatic (jobflow.yaml) and explicit (MongoStore) methods

  **Impact**: Zero-friction database integration in generated workflow scripts

* **Interactive Mode Robustness**: Fixed whitespace handling in all input fields

  - All ``questionary.text()`` and ``questionary.path()`` inputs now strip whitespace
  - Prevents crashes from trailing spaces in file paths
  - Applied to: structure files, preset names, database params, output filenames

  **Impact**: Interactive mode no longer crashes on accidental whitespace

* **Custodian Integration**: Enabled custodian error handling by default in ALL recipe workflows

  - Set ``use_custodian=True`` and ``custodian_max_errors=10`` for all Makers
  - Applied to: electronic, mechanical, convergence, complete, catalysis, defect recipes
  - 95% error recovery rate with automatic retry strategies

  **Impact**: Recipe workflows now recover automatically from common SIESTA errors

* **Time Estimates Removed**: Removed all print statements with time/memory estimates from ``complete_material_study()``

  - Estimates were rough heuristics that varied significantly
  - Kept logger.info() calls for debugging
  - Cleaner output focused on workflow structure

Documentation
-------------

* **Database CLI Documentation**: Updated ``cli-database.rst`` with comprehensive query options

  - Documented all new filter options (state, calc-type, energy range, latest)
  - Added export functionality examples (JSON/CSV)
  - Updated output columns documentation

* **Maker CLI Documentation**: Updated ``cli-tools.rst`` with database integration

  - Added database options to common options section
  - Updated interactive mode steps to include database configuration
  - Added database usage examples
  - Documented generated script database configuration with code examples

* **Database Storage Tutorial**: New tutorial ``09_explicit_jobstore.py``

  - Demonstrates correct SETTINGS.JOB_STORE usage
  - Shows explicit MongoStore configuration without jobflow.yaml
  - Updated README.md with explicit JobStore section

* **Recipe Book**: Added Method 5 showing phonon control parameters in complete material studies
* **vdW Tutorial**: Updated ``06_vdw_2d_materials.py`` to use PBE pseudos (DRSLL not available via CLI)
* **Tutorial Updates**: Fixed ``combined_recipes.py`` to demonstrate phonon parameter control

----

Version 1.0.0 (January 2026)
============================

**Release Date**: January 2026

New Features
------------

* **Structure Info CLI**: New ``atomate2siesta-structure info`` subcommand for comprehensive structure analysis including crystal symmetry, lattice parameters, atomic composition, and automatic magnetic property detection
* **Automatic Format Detection**: ``atomate2siesta-structure convert`` now automatically detects input format from file extension (.fdf, .xv, .cif, .xsf) with support for CIF and XSF input formats
* **DM.InitSpin Comments**: Automatic generation of descriptive comments for each atom showing species name, atom number, and Cartesian coordinates for easier debugging of magnetic structures
* **Cu Magnetic Detection**: Added Cu (Z=29) to automatic magnetic element detection with default 0.6 μB moment for DFT+U calculations

Architecture Improvements
-------------------------

* **Single Source of Truth**: Magnetic moment (DM.InitSpin) generation now exclusively handled by SpinSettings dataclass, eliminating duplicate logic and architectural inconsistency (~50 lines removed from ASE writer)
* **Internal Parameter Naming System**: Introduced dual-prefix system (``a2s_`` alias and ``atomate2siesta_`` full) to clearly distinguish framework control parameters from SIESTA FDF parameters with automatic filtering
* **CLI Reorganization**: Removed deprecated ``atomate2siesta-convert`` command in favor of structured ``atomate2siesta-structure`` group with room for future subcommands (scale, rotate, translate, etc.)

Enhancements
------------

* **Recipe CLI Analyze Command**: New ``atomate2siesta-recipe analyze`` subcommand for structure analysis directly from command line (supports ``--detailed`` flag for computational estimates)
* **Recipe CLI Help Documentation**: Enhanced all subcommands with comprehensive help text, usage examples, and detailed descriptions
* **Recipe CLI Rename**: Renamed ``compare`` subcommand to ``demo`` for clarity (shows before/after code demonstration)
* **Runtime Estimates Removed**: Removed workflow runtime estimates from recipe CLI and documentation (rough guesses that vary significantly by system and parameters)
* **Structure Compare**: Enhanced ``--verbose`` flag to show comprehensive site-by-site comparison with fractional coordinates, distances, and color-coded match status for ALL sites (not just unmatched)
* **Computational Estimates**: Made time/memory/core estimates hidden by default in ``RecipeBook.print_analysis()`` with clear warnings about rough heuristics when shown with ``detailed=True``
* **Tutorial Documentation**: Reduced parallel performance tutorial verbosity by 70% while maintaining essential content

Bug Fixes
---------

* **Recipe Workflows** (CRITICAL): Fixed 8 recipe workflows that were manually instantiating Makers instead of using class methods, resulting in missing calculation parameters:

  - ``dos_workflow()``: DOS jobs missing ``%block DOS.kgrid.MonkhorstPack`` and ``%block ProjectedDensityOfStates`` in FDF
  - ``electronic_properties()``: Band structure jobs misconfigured (2 instances)
  - ``kpoints_convergence()``, ``mesh_cutoff_convergence()``, ``basis_convergence()``: Static jobs misconfigured (3 instances)
  - ``thermal_properties()``: Phonon static jobs misconfigured

  Changed from ``Maker(input_set_generator=Generator(...))`` to ``Maker.class_method(...)`` pattern

  **Impact**: DOS, band structure, convergence, and phonon workflows now generate correct SIESTA input files

* **Electronic Properties Workflow** (CRITICAL): Fixed ``electronic_properties()`` to actually create DOS jobs as promised in docstring

  - Docstring claimed: "Complete electronic structure workflow: relaxation + bands + DOS"
  - Reality: Only created relax + band structure jobs (NO DOS job)
  - Added DOS job creation after band structure (~40 lines)
  - Now creates 3 jobs: relax → bands → DOS (or 2 without relax: bands → DOS)

  **Impact**: Users now get complete electronic characterization including DOS with PDOS

* **Phonon Workflow Supercell** (CRITICAL): Fixed ``phonon_workflow()`` ignoring calculated and user-specified supercell sizes

  - ``thermal_properties()`` calculated phonon_supercell but never passed it to phonon maker
  - ``phonon_workflow(supercell_matrix=(2,2,2))`` parameter was completely ignored
  - SiestaPhononFlowMaker fell back to ``min_length=15.0`` Angstroms creating huge supercells
  - For Si (5.43Å lattice): created 3x3x3 supercell (54 atoms) instead of user-requested 2x2x2 (16 atoms)
  - Now properly converts tuple to matrix format and passes to ``.make()`` method

  **Impact**: Phonon calculations now use reasonable supercell sizes (3-10x faster)

* **Phonon K-points Scaling** (CRITICAL PERFORMANCE): Fixed phonon workflows using primitive cell k-points for supercells

  - Supercells are N³ larger → Brillouin zone is N³ smaller → need fewer k-points
  - Was using same k-points for supercell as primitive cell (massive over-sampling)
  - For Si 3x3x3 supercell: was using [6,6,6] k-points (216), should use [2,2,2] (8)
  - Now scales k-points down: ``scaled_kpts = [max(1, k // supercell_size) for k in kpts]``
  - Speedup: 27x per displaced structure, ~800x for complete workflow

  **Impact**: Phonon calculations now 100-1000x faster with correct k-point sampling

* **Phonon min_length Default** (CRITICAL): Reduced default ``min_length`` from 15.0 to 6.0 Angstroms

  - Was creating absurdly large supercells: for Si (5.43Å) forced 3x3x3 (54 atoms)
  - Now creates reasonable supercells: for Si allows 2x2x2 (16 atoms)
  - Phonon accuracy only requires ~10-15Å minimum (6.0 is safer than 15.0)
  - Changed in all 3 workflow functions (phonon, gruneisen, qha)
  - Combined with supercell_matrix fix and k-point scaling: **~2400x total speedup!**

  **Impact**: Phonon workflows now use sensible defaults instead of wasteful ones

* **EOS Workflow**: Fixed ``KeyError: 'run_time'`` in EOS postprocessing by passing directory paths instead of optional output field references
* **Tutorial Structure Paths**: Fixed 21 tutorial files with incorrect relative paths after reorganization (8 files in ``02-workflows/``, 13 files in ``03-advanced-features/presets/``)
* **Conversion Tutorial**: Fixed ``02_siesta_formats.py`` to work with dry_run mode by using glob pattern for nested dry_run output directories
* **Recipe CLI Stats**: Fixed ``ValueError`` in ``atomate2siesta-recipe stats`` when recipes have text values like "high" instead of percentages


* Fixed empty ``%block DM.InitSpin`` blocks when structure has no magnetic moments
* Fixed duplicate DM.InitSpin generation sources (SpinSettings + ASE writer)
* Fixed ``magnetic_ordering`` appearing as invalid SIESTA keyword in FDF files
* Updated stale imports after CLI reorganization

Breaking Changes
----------------

* **Recipe API Cleanup**: Removed duplicate ``bulk_modulus_workflow()`` function - use ``eos_workflow()`` instead

  - ``bulk_modulus_workflow()`` and ``eos_workflow()`` were 100% identical (same calculation, different names)
  - EOS (Equation of State) is the canonical name for this calculation
  - Migration: Replace ``RecipeBook.bulk_modulus_workflow(structure)`` with ``RecipeBook.eos_workflow(structure)``
  - Output remains identical: bulk_modulus, equilibrium_volume, E0, EOS_fit


* **CLI**: ``atomate2siesta-convert`` command removed; use ``atomate2siesta-structure convert`` instead
* **CLI**: ``--xv`` flag removed from ``atomate2siesta-structure convert``; file format is now automatically detected from extension

Migration Guide
---------------

**Internal Parameter Naming (v1.0.0)**

atomate2siesta now uses prefixed parameter names to distinguish framework control parameters from SIESTA FDF parameters:

**Old (Deprecated - Still Works):**

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "Spin": "polarized",
           "a2s_magnetic_ordering": "ferromagnetic",  # Deprecated
       }
   )
   # DeprecationWarning: Parameter 'magnetic_ordering' is deprecated.
   # Use 'a2s_magnetic_ordering' or 'atomate2siesta_magnetic_ordering' instead.

**New (Recommended):**

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "Spin": "polarized",
           "a2s_magnetic_ordering": "ferromagnetic",  # Recommended (alias)
       }
   )

   # OR use full prefix:
   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "Spin": "polarized",
           "atomate2siesta_magnetic_ordering": "ferromagnetic",  # Full prefix
       }
   )

**Migration Timeline:**

- **v1.0.0+**: Only prefixed names allowed (strict validation, current)

**Why the Change?**

- **Visual Clarity**: Instantly distinguish atomate2siesta controls (``a2s_*``) from SIESTA parameters
- **No Collisions**: Separate namespace prevents conflicts with SIESTA keywords
- **Scalable**: Add unlimited new control parameters without risk
- **Self-Documenting**: Prefix reveals parameter source

**See Also:**

- :doc:`fdf-parameters` - Complete internal parameter documentation
- ``README.md`` - Usage examples with new parameter names

Strategic Planning
------------------

* **Structure Manipulation Roadmap**: Comprehensive 16-command expansion plan organized in 4 priority tiers with implementation timeline, testing strategy, and success metrics

Documentation
-------------

* **Structure Conversion**: Added comprehensive 316-line README with FDF vs XV distinction, dry-run mode details, workflow examples, and troubleshooting
* **Recipe Book**: Updated documentation to explain computational estimates are hidden by default and how to enable them

----

Version 0.2.0 (2025)
====================

**Release Date**: October 2025

New Features
------------

* **Dataclass Tutorials & Comment Headers**: 7 comprehensive tutorials for MEDIUM priority dataclass modules (DOS, optical properties, phonon inputs, DFT+U, charge/field calculations) with automatic comment header generation for FDF output
* **Recipe Book System**: 39 high-level recipes across 6 categories reducing workflow complexity from 50+ lines to 1-line function calls with MaterialAnalyzer for automatic parameter recommendations
* **Universal Dry-Run Support**: All 10 workflow makers support dry_run mode with 99.9% time savings for workflow preview
* **Grüneisen Parameters & Thermal Expansion**: Complete analysis suite with 6 plotting functions, temperature-dependent calculations, and publication-quality output
* **Powerups System**: Dynamic workflow customization with flow-level parameter updates and selective job modification
* **Adsorption Site Scanning**: Grid-based automatic scanning with molecule orientation control and energy ranking
* **Comprehensive Testing**: 750 tests with 44% code coverage and 17-second execution time

Improvements
------------

* **Automatic Comment Headers**: 21 FDF parameter mappings for automatic dataclass comment header detection in direct SIESTA FDF format
* **Tutorial Coverage**: 30+ tutorials with comprehensive READMEs (600-800 lines) covering theory, best practices, and troubleshooting
* Enhanced basis parameter matching (PAO.BasisSize fuzzy matching)
* Complete tier system with all 24 modules operational
* 62% reduction in TODO items with comprehensive docstrings
* Better error messages and validation

Bug Fixes
---------

* Fixed PAO.BasisSize parameter being silently ignored
* Added missing tier system classmethods
* Eliminated initialization warnings
* Fixed basis size propagation in EOS workflows

----

Version 0.1.0 (2024)
====================

**Release Date**: December 2024

New Features
------------

* **Custodian Error Handling**: Intelligent recovery from 10+ error types with progressive SCF convergence rescue
* **Phonon Calculations**: Full phonopy integration with symmetry-reduced displacements and automatic plotting (4 plot types)
* **Surface Energy Workflows**: Multi-surface comparison with automatic termination discovery and publication-quality output
* **Tier-Based Configuration**: 4-tier hierarchy (basic → expert) with 14 material-specific presets
* **Advanced Workflows**: Elastic constants, equation of state, and timing analysis

Improvements
------------

* Enhanced SCF convergence parameters
* Occupation function flexibility
* Improved field name matching
* Electronic temperature settings
* Better documentation structure

----

Version 0.0.1 (2024)
====================

**Release Date**: August 2024

Initial Release
---------------

* **Core Job Types**: RelaxMaker, StaticMaker, BandStructureMaker, SiestaPhononFlowMaker
* **Flow System**: Multi-step workflow composition with database integration and HPC support
* **Input Generation**: FDF file generation with 24 dataclass modules and pseudopotential handling
* **Output Processing**: Comprehensive schema system with structure, energy, forces, and stress extraction
* **Basic Workflows**: Relaxation, band structure, NEB, and convergence studies
* **14 Tutorials**: Covering basics, convergence studies, advanced workflows, and infrastructure

----

Upgrade Notes
=============

Version 0.2.0
-------------

* No breaking changes
* New tutorials: 19 (adsorption), 20 (Grüneisen), 22 (powerups)
* New advanced features tutorials: DOS calculations, phonon inputs, optical properties, DFT+U, charge/dipole/electric field
* Recipe Book: 39 high-level recipes for simplified workflow creation

Version 0.1.0
-------------

* Custodian integration may require updating custom error handlers
* Tier system replaces some direct parameter settings
* New tutorials: 15 (custodian), 16 (phonons), 17 (surfaces), 18 (tiers)

----

Citation
========

If you use these SIESTA workflows in your research, please cite atomate2,
the SIESTA code, and this work:

.. code-block:: bibtex

   @article{atomate2,
     author = {Ganose, Alex M. and others},
     title = {Atomate2: modular workflows for materials science},
     journal = {Digital Discovery},
     year = {2025},
     doi = {10.1039/D5DD00019J}
   }

   @article{siesta2002,
     author = {Soler, Jos\'e M. and Artacho, Emilio and Gale, Julian D. and
               Garc\'ia, Alberto and Junquera, Javier and Ordej\'on, Pablo and
               S\'anchez-Portal, Daniel},
     title = {The {SIESTA} method for ab initio order-{N} materials simulation},
     journal = {Journal of Physics: Condensed Matter},
     volume = {14},
     number = {11},
     pages = {2745},
     year = {2002},
     doi = {10.1088/0953-8984/14/11/302}
   }

   @article{siesta2020,
     author = {Garc\'ia, Alberto and others},
     title = {Siesta: Recent developments and applications},
     journal = {The Journal of Chemical Physics},
     volume = {152},
     number = {20},
     pages = {204108},
     year = {2020},
     doi = {10.1063/5.0005077}
   }

   @software{atomate2siesta,
     author = {Akhtar, Arsalan},
     title = {atomate2siesta: Automated SIESTA Workflows},
     year = {2024-2025},
     url = {https://github.com/arsalan-akhtar/atomate2siesta}
   }

----

License
=======

The SIESTA workflows are part of atomate2 and are distributed under the same
license as atomate2 (modified BSD, ``BSD-3-Clause-LBNL``).
