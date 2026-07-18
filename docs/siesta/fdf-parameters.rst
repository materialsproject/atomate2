============================
FDF Parameter Customization
============================

This guide explains how to customize SIESTA calculations using FDF (Flexible Data Format) parameters in atomate2siesta.

.. note::
   Complete FDF parameter routing system with 444 registered parameters.
   User-provided FDF parameters correctly override auto-generated values.

----

Overview
========

SIESTA uses FDF (Flexible Data Format) input files to control calculations. In atomate2siesta, you can provide parameters in two ways:

1. **SIESTA FDF Parameters** (no prefix): Actual SIESTA keywords like ``Mesh.Cutoff``, ``PAO.BasisSize``, ``Spin``
2. **atomate2siesta Internal Parameters** (prefixed): Framework controls like ``a2s_magnetic_ordering``

.. note::
   Internal parameter naming system uses ``atomate2siesta_`` and ``a2s_`` prefixes.
   This clearly distinguishes framework controls from SIESTA parameters.

----

Internal Control Parameters
============================

.. versionadded:: 1.0.0

atomate2siesta uses special **internal control parameters** that modify framework behavior
but are NOT written to SIESTA FDF files. These are distinguished by the ``atomate2siesta_``
or ``a2s_`` prefix.

Why the Prefix?
---------------

The prefix makes it immediately clear that these are **framework control parameters**,
not SIESTA keywords. This prevents:

- Accidentally writing internal parameters to FDF files
- Confusion between SIESTA and atomate2siesta parameters
- Collision with future SIESTA keywords

Dual Prefix Support
-------------------

Both prefixes work identically (user choice):

.. code-block:: python

   # Short alias (recommended for brevity)
   user_params = {
       "Spin": "polarized",              # SIESTA parameter (no prefix)
       "a2s_magnetic_ordering": "AFM",   # Internal parameter (alias)
   }

   # Full prefix (more explicit)
   user_params = {
       "Spin": "polarized",                          # SIESTA parameter
       "atomate2siesta_magnetic_ordering": "AFM",    # Internal parameter (full)
   }

Available Internal Parameters (8 Total)
----------------------------------------

atomate2siesta provides **8 framework-specific shortcuts** that control atomate2siesta
behavior (not SIESTA itself). All use the ``a2s_`` or ``atomate2siesta_`` prefix.

**1. a2s_magnetic_ordering** (or **atomate2siesta_magnetic_ordering**)
   Controls automatic DM.InitSpin generation for magnetic calculations.

   **Values**: ``"ferromagnetic"`` (or ``"FM"``), ``"antiferromagnetic"`` (or ``"AFM"``), ``"custom"``

   **Default**: ``"antiferromagnetic"``

   **Example**:

   .. code-block:: python

      from atomate2.siesta.jobs.core import RelaxMaker

      maker = RelaxMaker.fixed_cell_relaxation(
          user_params={
              "Spin": "polarized",                  # SIESTA parameter
              "a2s_magnetic_ordering": "FM",        # Internal control
          }
      )

   **Generated FDF** (DM.InitSpin block auto-generated):

   .. code-block:: text

      Spin    polarized

      %block DM.InitSpin
      1  +2.5  # Fe atom 1 at (0.000, 0.000, 0.000)
      2  +2.5  # Fe atom 2 at (1.435, 1.435, 1.435)
      %endblock DM.InitSpin

   See ``tutorials/07-advanced-features/16-magnetic-calculations/`` for complete examples.

**2. a2s_kpts** (or **atomate2siesta_kpts**)
   Shorthand for k-point grid specification (translated to ``%block kgrid.Monkhorst.Pack``).

   **Example**:

   .. code-block:: python

      user_params = {
          "a2s_kpts": [4, 4, 4],  # Simple 3-element list
      }

   **Alternative**: Use SIESTA parameter directly:

   .. code-block:: python

      user_params = {
          "Mesh.Cutoff": "300 Ry",            # Direct SIESTA parameter
          "%block kgrid.Monkhorst.Pack": [    # Direct SIESTA block
              [4, 0, 0, 0.0],
              [0, 4, 0, 0.0],
              [0, 0, 4, 0.0],
          ],
      }

**3-8. Pseudopotential Parameters**
   Six parameters controlling pseudopotential selection:

   - ``a2s_pseudo_path``: Custom path to pseudopotential directory
   - ``a2s_pseudo_base_path``: Base directory for pseudopotential families
   - ``a2s_pseudo_family``: Family name (e.g., ``"ONCVPSP-PBE-SR"``)
   - ``a2s_pseudo_version``: Version string (e.g., ``"PDv0.4"``)
   - ``a2s_pseudo_quality``: Quality level (e.g., ``"Standard"``)
   - ``a2s_pseudo_relativistic``: Relativistic treatment (``"SR"`` or ``"FR"``)

   **Basic Example**:

   .. code-block:: python

      user_params = {
          "a2s_pseudo_family": "ONCVPSP-PBE-SR",
          "a2s_pseudo_relativistic": "SR",
      }

   **Advanced: Explicit Path with XC Override** (v1.0.0+):

   .. code-block:: python

      user_params = {
          "a2s_pseudo_path": "/path/to/ONCVPSP-PBE-FR-PDv0.4-Standard",  # PBE pseudos
          "xc.authors": "PW91",  # Override XC in FDF (expert use)
      }

   .. warning::
      **XC Mismatch**: When using explicit ``a2s_pseudo_path`` with different ``xc.authors``,
      you'll get an automatic warning about XC functional mismatch. This is intentional for
      advanced users who understand the implications.

      Automatic XC validation checks your pseudopotential files and warns
      if they don't match the FDF XC functional.

   See :doc:`siesta-pseudos` for complete pseudopotential documentation.

Legacy Support (Deprecated)
----------------------------

.. warning::
   As of v1.0.0, unprefixed internal parameter names (like ``magnetic_ordering``) are NO LONGER SUPPORTED.
   You MUST use prefixed versions (``a2s_magnetic_ordering`` or ``atomate2siesta_magnetic_ordering``).

.. code-block:: python

   # ❌ Legacy (NO LONGER WORKS as of v1.0.0)
   user_params = {
       "magnetic_ordering": "AFM",  # Raises ValueError!
   }

   # ✅ Correct (v1.0.0+)
   user_params = {
       "a2s_magnetic_ordering": "AFM",
   }

Migration Timeline
~~~~~~~~~~~~~~~~~~

- **v1.0.0+**: Only prefixed names supported (strict validation)

----

Basic Usage
===========

Using Internal Parameters (Shortcuts)
--------------------------------------

atomate2siesta provides 8 framework-specific shortcuts (``a2s_`` prefix) for parameters
that control atomate2siesta behavior rather than SIESTA itself:

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker

   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "a2s_kpts": [4, 4, 4],                    # K-point grid specification
           "a2s_magnetic_ordering": "AFM",           # Magnetic ordering control
           "a2s_pseudo_family": "ONCVPSP-PBE-SR",   # Pseudopotential selection
       }
   )

.. note::
   For SIESTA FDF parameters like ``PAO.BasisSize``, ``Mesh.Cutoff``, ``PAO.EnergyShift``,
   use the direct FDF parameter names (shown in next section). These do NOT have shortcuts.

Using Direct FDF Parameters
----------------------------

These are the actual SIESTA keywords (case-insensitive):

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "Mesh.Cutoff": "300 Ry",        # Direct SIESTA parameter
           "PAO.BasisSize": "DZP",          # Direct SIESTA parameter
           "XC.functional": "GGA",          # Direct SIESTA parameter
           "XC.authors": "PBE",             # Direct SIESTA parameter
       }
   )

**Result in siesta.fdf**:

.. code-block:: text

   Mesh.Cutoff    300.0 Ry
   PAO.BasisSize  DZP
   XC.functional  GGA
   XC.authors     PBE

----

Block Parameters
================

SIESTA uses blocks for multi-line parameters. atomate2siesta automatically handles block formatting.

K-Point Sampling
----------------

**Method 1: Using kpts shortcut (recommended)**:

.. code-block:: python

   user_params = {
       "a2s_kpts": [4, 4, 4],  # Simple 3-element list
   }

**Generated FDF**:

.. code-block:: text

   #------------------#
   #  KPointSampling
   #------------------#
   %block kgrid.Monkhorst.Pack
   4 0 0 0.0
   0 4 0 0.0
   0 0 4 0.0
   %endblock kgrid.Monkhorst.Pack

**Method 2: Using block format directly**:

.. code-block:: python

   user_params = {
       "%block kgrid.Monkhorst.Pack": [
           [4, 0, 0, 0.0],
           [0, 4, 0, 0.0],
           [0, 0, 4, 0.0],
       ],
   }

**Same result**: Both methods produce identical FDF output!

**Method 3: Using kgrid.Cutoff instead**:

.. code-block:: python

   user_params = {
       "kgrid.Cutoff": "15.0 Ang",  # Alternative k-point method
   }

**Generated FDF**:

.. code-block:: text

   #------------------#
   #  KPointSampling
   #------------------#
   kgrid.Cutoff    15.0 Ang

.. note::
   The system automatically chooses **either** ``kgrid.Cutoff`` **or** ``kgrid.Monkhorst.Pack``, never both.
   User-provided values always take priority over auto-generated ones.

Other Common Blocks
-------------------

**Geometry Constraints**:

.. code-block:: python

   user_params = {
       "%block Geometry.Constraints": [
           "position from 1 to 10",
           "position 20",
       ],
   }

**DM.InitSpin** (Magnetic Moments):

.. code-block:: python

   user_params = {
       "Spin": "polarized",
       "%block DM.InitSpin": [
           "1  +2.5",  # Atom 1: +2.5 μB
           "2  -2.5",  # Atom 2: -2.5 μB
       ],
   }

.. tip::
   For magnetic calculations, use the automatic DM.InitSpin generation:

   .. code-block:: python

      from atomate2.siesta.sets.utils import get_default_initial_magnetic_moments

      magmoms = get_default_initial_magnetic_moments(structure)
      structure.add_site_property("magmom", magmoms)

      maker = RelaxMaker.fixed_cell_relaxation(
          user_params={
              "Spin": "polarized",
              "a2s_magnetic_ordering": "antiferromagnetic",  # or "ferromagnetic"
          }
      )

----

Advanced FDF Block Inputs (fdf_arguments)
==========================================

For advanced SIESTA features not directly supported by atomate2siesta's parameter system,
you can use the ``fdf_arguments`` dictionary to specify **any** FDF parameter or block directly.

When to Use fdf_arguments
--------------------------

Use ``fdf_arguments`` for:

1. **FDF Blocks**: Multi-line SIESTA input blocks (GeometryConstraints, DM.InitSpin, etc.)
2. **Unsupported Parameters**: SIESTA features not yet in atomate2siesta
3. **Direct Control**: Bypass atomate2siesta's parameter processing when needed
4. **Complex Configurations**: Advanced setups requiring precise FDF syntax

.. note::
   Most common parameters have dedicated support in ``user_params``.
   Use ``fdf_arguments`` only when necessary for maintainability.

Basic Syntax
------------

**Rule**: FDF blocks are specified as **lists of strings** in the ``fdf_arguments`` dictionary.

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker

   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           # Regular parameters
           "PAO.BasisSize": "DZP",
           "a2s_kpts": [6, 6, 6],
           "Spin": "polarized",

           # Advanced FDF blocks via fdf_arguments
           "fdf_arguments": {
               # Block format: list of strings
               "DM.InitSpin": [
                   "1  +2.0",  # Atom 1: +2.0 μB
                   "2  -2.0",  # Atom 2: -2.0 μB (antiferromagnetic)
               ],
               "GeometryConstraints": [
                   "position from 1 to 4",  # Fix first 4 atoms
               ],
           },
       }
   )

**Generated FDF**:

.. code-block:: text

   PAO.BasisSize   DZP
   Spin            polarized

   %block kgrid.Monkhorst.Pack
   6 0 0 0.0
   0 6 0 0.0
   0 0 6 0.0
   %endblock kgrid.Monkhorst.Pack

   %block DM.InitSpin
   1  +2.0
   2  -2.0
   %endblock DM.InitSpin

   %block Geometry.Constraints
   position from 1 to 4
   %endblock Geometry.Constraints

Common FDF Blocks
-----------------

1. DM.InitSpin (Magnetic Initialization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set initial spin polarization per atom:

.. code-block:: python

   user_params = {
       "Spin": "polarized",  # Required for magnetic calculations
       "fdf_arguments": {
           "DM.InitSpin": [
               "1  +2.0",   # Atom 1: +2.0 μB (spin up)
               "2  -2.0",   # Atom 2: -2.0 μB (spin down)
           ],
       },
   }

.. tip::
   **Automatic Alternative**: Use structure magnetic moments instead!

   .. code-block:: python

      from atomate2.siesta.sets.utils import get_default_initial_magnetic_moments

      # Automatic magnetic moment detection
      magmoms = get_default_initial_magnetic_moments(structure)
      structure.add_site_property("magmom", magmoms)

      maker = RelaxMaker.fixed_cell_relaxation(
          user_params={
              "Spin": "polarized",
              "a2s_magnetic_ordering": "AFM",  # or "FM"
          }
      )

   See ``tutorials/07-advanced-features/16-magnetic-calculations/`` for complete examples.

2. GeometryConstraints (Fix Atoms)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fix atoms or constrain motion directions:

.. code-block:: python

   user_params = {
       "fdf_arguments": {
           "GeometryConstraints": [
               "position from 1 to 10",   # Fix atoms 1-10 (all directions)
               "position 15 x y",          # Fix atom 15 in x,y only
           ],
       },
   }

**Common Use Cases**:

- **Surface relaxation**: Fix bottom substrate layers
- **Defect studies**: Fix boundary atoms
- **2D materials**: Constrain specific directions

3. ExternalElectricField
~~~~~~~~~~~~~~~~~~~~~~~~~

Apply uniform electric field:

.. code-block:: python

   user_params = {
       "fdf_arguments": {
           "ExternalElectricField": [
               "0.0 0.0 0.1 V/Ang",  # Field in z-direction
           ],
       },
   }

**Units**: ``V/Ang``, ``V/Bohr``, ``Ry/Bohr/e``, ``Har/Bohr/e``

4. ProjectedDensityOfStates (PDOS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate orbital-projected DOS:

.. code-block:: python

   user_params = {
       "fdf_arguments": {
           "ProjectedDensityOfStates": [
               "-15.0  10.0  0.1  250  eV",  # Emin Emax dE nPoints units
           ],
           "PDOS.kgrid_Monkhorst_Pack": [
               "6  0  0  0.0",
               "0  6  0  0.0",
               "0  0  6  0.0",
           ],
       },
   }

5. BandLines (Custom K-Point Paths)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define custom k-point path for band structure:

.. code-block:: python

   user_params = {
       "fdf_arguments": {
           "BandLinesScale": ["pi/a"],
           "%block BandLines": [
               "1   0.000  0.000  0.000  \\Gamma",
               "   40  0.500  0.000  0.500  X",
               "   40  0.500  0.250  0.750  W",
               "   40  0.000  0.000  0.000  \\Gamma",
           ],
       },
   }

6. MD.TargetStress (NPT Ensemble)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Target stress tensor for variable-cell MD:

.. code-block:: python

   user_params = {
       "fdf_arguments": {
           "MD.TargetStress": [
               "0.5 0.5 0.5 0.0 0.0 0.0 GPa",  # Hydrostatic pressure
           ],
       },
   }

**Stress components**: (Sxx, Syy, Szz, Sxy, Sxz, Syz)

Combining Multiple Blocks
--------------------------

You can specify multiple FDF blocks in one calculation:

.. code-block:: python

   n_atoms = len(structure)
   n_fixed = 4  # Bottom layer

   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "PAO.BasisSize": "DZP",
           "a2s_kpts": [6, 6, 6],
           "Spin": "polarized",

           "fdf_arguments": {
               # Electric field
               "ExternalElectricField": ["0.0 0.0 0.1 V/Ang"],

               # Fix bottom layer
               "GeometryConstraints": [f"position from 1 to {n_fixed}"],

               # Initialize spin (only free atoms)
               "DM.InitSpin": [
                   f"{i+1}  2.0" for i in range(n_fixed, n_atoms)
               ],

               # PDOS calculation
               "ProjectedDensityOfStates": ["-15.0  10.0  0.1  250  eV"],
           },
       }
   )

.. warning::
   **Block Compatibility**: Most blocks can coexist, but be careful with:

   - ✅ Compatible: Constraints + Electric field
   - ✅ Compatible: Spin + PDOS
   - ⚠️ Careful: BandLines + PDOS (both define k-points, use separate calculations)
   - ⚠️ Careful: MD.TargetStress + GeometryConstraints (constraints limit cell motion)

Common Errors and Solutions
----------------------------

**Error 1: Block not appearing in FDF**

.. code-block:: python

   # ❌ WRONG: String instead of list
   "fdf_arguments": {
       "ExternalElectricField": "0.0 0.0 0.1 V/Ang"  # Missing brackets!
   }

   # ✅ CORRECT: List of strings
   "fdf_arguments": {
       "ExternalElectricField": ["0.0 0.0 0.1 V/Ang"]
   }

**Error 2: Atom indexing (0 vs 1-based)**

.. code-block:: python

   # ❌ WRONG: Python 0-indexing
   "GeometryConstraints": ["position from 0 to 7"]

   # ✅ CORRECT: SIESTA 1-indexing
   "GeometryConstraints": ["position from 1 to 8"]

**Error 3: Missing units**

.. code-block:: python

   # ❌ WRONG: No units specified
   "ExternalElectricField": ["0.0 0.0 0.1"]

   # ✅ CORRECT: Units included
   "ExternalElectricField": ["0.0 0.0 0.1 V/Ang"]

Validation and Debugging
-------------------------

**1. Always use dry-run first**:

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(
       dry_run=True,  # Preview FDF without running
       user_params={...}
   )

**2. Check generated FDF file**:

.. code-block:: bash

   # Find the generated input
   find . -name "siesta.fdf" -exec cat {} \;

   # Look for your blocks
   grep -A5 "ExternalElectricField" job_*/siesta.fdf

**3. Verify block syntax**:

Expected format in FDF:

.. code-block:: text

   %block BlockName
     data line 1
     data line 2
   %endblock BlockName

Complete Reference
-------------------

For comprehensive examples of all FDF blocks:

- **Tutorial**: ``tutorials/07-advanced-features/02-fdf-block-inputs/``
- **Examples**: 7 complete tutorials covering all major block types
- **Documentation**: 1,050 lines covering syntax, validation, and best practices

----

Complete Example
================

Combining Multiple Parameters
------------------------------

.. code-block:: python

   from pymatgen.core import Structure
   from atomate2.siesta.jobs.core import RelaxMaker
   from jobflow import run_locally

   structure = Structure.from_file("POSCAR")

   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           # K-point sampling
           "a2s_kpts": [6, 6, 6],

           # Basis and mesh
           "PAO.BasisSize": "DZP",
           "Mesh.Cutoff": "500 Ry",
           "PAO.EnergyShift": "0.01 Ry",

           # Exchange-correlation
           "XC.functional": "GGA",
           "XC.authors": "PBE",

           # SCF convergence
           "SCF.Mixer.Method": "Pulay",
           "SCF.Mixer.Weight": 0.1,
           "DM.NumberPulay": 8,
           "ElectronicTemperature": "300 K",

           # Spin polarization
           "Spin": "polarized",

           # van der Waals (vdW-DF functional, Dion et al. DRSLL)
           "XC.functional": "VDW",
           "XC.authors": "DRSLL",
       }
   )

   job = maker.make(structure)
   results = run_locally(job, create_folders=True)

**Generated siesta.fdf** (excerpt):

.. code-block:: text

   #---------------------------#
   #  BasisSetsAndProjectors
   #---------------------------#
   PAO.BasisSize   DZP
   PAO.EnergyShift 0.01 Ry

   #------------------------------#
   #  ExchangeCorrelationFunctionals
   #------------------------------#
   XC.functional   GGA
   XC.authors      PBE

   #---------------#
   #  SpinSettings
   #---------------#
   Spin    polarized

   #------------------#
   #  KPointSampling
   #------------------#
   %block kgrid.Monkhorst.Pack
   6 0 0 0.0
   0 6 0 0.0
   0 0 6 0.0
   %endblock kgrid.Monkhorst.Pack

   #---------------------------#
   #  RealSpaceGridParameters
   #---------------------------#
   Mesh.Cutoff     500.0 Ry

   #---------------------#
   #  SCFLoopParameters
   #---------------------#
   SCF.Mixer.Method        Pulay
   SCF.Mixer.Weight        0.1
   DM.NumberPulay          8
   ElectronicTemperature   300.0 K

----

Parameter Validation
====================

Unknown Parameter Detection
---------------------------

atomate2siesta validates all FDF parameters against a registry of 444 known SIESTA keywords:

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "Mesh.Cutoff": "300 Ry",        # ✅ Known parameter
           "%block kgrid.monkhorst.pack": [[4,0,0,0.0], ...],  # ✅ Known
           "PAO.BasisSize": "DZP",          # ✅ Known
           "%block 1kgrid.monkhorst.pack": [...],  # ❌ Typo!
       }
   )

**Error message** (with colored output):

.. code-block:: text

   Unknown FDF parameter(s): %block 1kgrid.monkhorst.pack  ← highlighted in yellow

   These parameters are not registered in the FDF registry.

   To fix this:
     1. Check spelling against SIESTA manual (case-insensitive)
     2. Allow unknown parameters with force_unknown=True:
        • RelaxMaker(user_params={...}, force_unknown=True)
        • update_user_siesta_settings(flow, {...}, force_unknown=True)

Allowing Unknown Parameters
----------------------------

If you need to use a parameter not in the registry:

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "CustomParameter": "value",  # Not in registry
       },
       force_unknown=True,  # ← Allow unknown parameters
   )

.. warning::
   Use ``force_unknown=True`` only when necessary. Unknown parameters bypass validation
   and may cause SIESTA errors if misspelled.

XC Functional Validation
-------------------------

.. versionadded:: 1.0.0

atomate2siesta automatically validates that your XC functional (``XC.Functional`` and ``XC.Authors``)
matches the pseudopotential files. This prevents silent inconsistencies that can affect results.

**How it works**:

1. Reads XC information from ``.psml`` files
2. Compares with FDF ``XC.Authors`` parameter
3. Warns if mismatch detected (always shown, regardless of verbosity)

**Example Warning**:

.. code-block:: text

   ⚠️  XC Functional Mismatch Warning
   The XC functional in your FDF does not match the pseudopotentials:

     • Si: FDF has PW91, pseudo has PBE (GGA -- Perdew-Burke-Ernzerhof)

   This may lead to inconsistent results. Consider either:
     1. Using pseudopotentials matching XC.Authors = PW91
     2. Changing XC.Authors to match your pseudopotentials

**When you might see this**:

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "a2s_pseudo_path": "/path/to/ONCVPSP-PBE-FR-PDv0.4-Standard",  # PBE pseudos
           "xc.authors": "PW91",  # Explicit override - MISMATCH!
       }
   )

**Resolution**:

1. **Match pseudos to XC** (recommended):

   .. code-block:: python

      user_params = {
          "a2s_pseudo_path": "/path/to/ONCVPSP-PW91-FR-PDv0.4-Standard",  # Now matches!
          "xc.authors": "PW91",
      }

2. **Match XC to pseudos**:

   .. code-block:: python

      user_params = {
          "a2s_pseudo_path": "/path/to/ONCVPSP-PBE-FR-PDv0.4-Standard",
          "xc.authors": "PBE",  # Now matches!
      }

3. **Advanced users**: If you intentionally want a mismatch (e.g., for testing), you can
   proceed - the warning is informational only and doesn't stop execution.

.. note::
   **Supported formats**: Currently validates PSML files (ONCVPSP format).
   The XC information is extracted from the ``<exchange-correlation>`` XML element.

----

Advanced Features
=================

Overriding Tier Presets
------------------------

When using tier presets, use ``override_params`` to modify preset values:

.. code-block:: python

   from atomate2.siesta.sets.tiers import apply_tier_preset

   maker = RelaxMaker.fixed_cell_relaxation()
   maker = apply_tier_preset(
       maker,
       "relax_standard",
       override_params={
           "a2s_kpts": [8, 8, 8],          # Override preset k-points
           "Mesh.Cutoff": "600 Ry",    # Override preset mesh cutoff
           "Spin": "polarized",         # Add new parameter
       },
   )

.. important::
   **Always use override_params** when modifying preset parameters!

   ❌ **Wrong**: Passing ``user_params`` to maker before applying preset

   .. code-block:: python

      # This DOESN'T work - preset overwrites user_params!
      maker = RelaxMaker.fixed_cell_relaxation(
          user_params={"a2s_kpts": [8, 8, 8]}  # Gets overwritten!
      )
      maker = apply_tier_preset(maker, "relax_standard")

   ✅ **Correct**: Using override_params

   .. code-block:: python

      maker = RelaxMaker.fixed_cell_relaxation()
      maker = apply_tier_preset(
          maker, "relax_standard",
          override_params={"a2s_kpts": [8, 8, 8]}  # Correctly overrides!
      )

Using Powerups
--------------

Modify parameters after job creation:

.. code-block:: python

   from atomate2.siesta.powerups import update_user_siesta_settings

   job = maker.make(structure)

   # Update parameters for this specific job
   job = update_user_siesta_settings(
       job,
       {
           "SCF.Mixer.Weight": 0.005,
           "ElectronicTemperature": "500 K",
       }
   )

----

FDF Registry
============

Viewing Available Parameters
-----------------------------

Use the CLI to explore all 444 registered FDF parameters:

.. code-block:: bash

   # List all dataclass modules
   atomate2siesta-inputs list

   # Show parameters for a specific module
   atomate2siesta-inputs show KPointSampling

   # Search for parameters by keyword
   atomate2siesta-inputs search "mesh"

**Example output**:

.. code-block:: text

   Data Class: KPointSampling
   --------------------------------------------------
   Attributes:

   - k_points: List[Tuple[int, int, int]]
     SIESTA keyword: %block kgrid_Monkhorst_Pack
     Description: Dimensions of the Monkhorst-Pack k-point grid

   - kgrid_cutoff: Optional[float]
     SIESTA keyword: kgrid.Cutoff
     Description: Real-space cutoff for automatic k-point generation

Complete Parameter List
------------------------

atomate2siesta registers parameters from 30 dataclass modules:

- **BasisSetsAndProjectors**: PAO.BasisSize, PAO.EnergyShift, PAO.SplitNorm, etc.
- **KPointSampling**: kgrid.Cutoff, %block kgrid.Monkhorst.Pack
- **RealSpaceGridParameters**: Mesh.Cutoff, MeshSubDivisions, etc.
- **SCFLoopParameters**: SCF.Mixer.*, DM.NumberPulay, ElectronicTemperature, etc.
- **SpinSettings**: Spin, %block DM.InitSpin, FixSpin, etc.
- **ExchangeCorrelationFunctionals**: XC.functional, XC.authors, etc.
- And 24 more modules...

See :doc:`siesta-inputs` for complete documentation.

----

Best Practices
==============

1. **Use Direct FDF Parameters**: For SIESTA parameters, use FDF names directly

   ✅ ``"Mesh.Cutoff": "300 Ry"`` (direct SIESTA parameter)

   ✅ ``"PAO.BasisSize": "DZP"`` (direct SIESTA parameter)

   ✅ ``"PAO.EnergyShift": "0.01 Ry"`` (direct SIESTA parameter)

   ❌ ``"a2s_mesh_cutoff": "300 Ry"`` (doesn't exist - never implemented)

2. **Case-Insensitive**: SIESTA and atomate2siesta handle case automatically

   All equivalent: ``"Mesh.Cutoff"``, ``"mesh.cutoff"``, ``"MESH.CUTOFF"``

3. **Blocks Are Automatic**: Just provide the data, formatting is handled

   .. code-block:: python

      # You write:
      "%block kgrid.Monkhorst.Pack": [[4,0,0,0.0], [0,4,0,0.0], [0,0,4,0.0]]

      # siesta.fdf gets:
      # %block kgrid.Monkhorst.Pack
      # 4 0 0 0.0
      # 0 4 0 0.0
      # 0 0 4 0.0
      # %endblock kgrid.Monkhorst.Pack

4. **Test Your Parameters**: Use dry-run mode to preview FDF files

   .. code-block:: python

      maker = RelaxMaker.fixed_cell_relaxation(
          dry_run=True,  # Generate inputs without running SIESTA
          user_params={...}
      )

5. **Validation Helps**: Don't disable it unless necessary

   ✅ Let validation catch typos

   ❌ Don't use ``force_unknown=True`` by default

----

Advanced Basis Set Customization
=================================

.. versionadded:: 1.0.0

atomate2siesta provides powerful tools for customizing basis sets beyond simple global settings.
This section covers species variants, per-atom control, and programmatic PAO.Basis generation.

Species Variants
-------------------------------

The **dict format** for ``%block PAO.BasisSizes`` enables species variants like ``O_surface``, ``O_bulk``, ``O_ghost``:

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker

   # Assign species labels to structure
   species_labels = []
   for site in structure:
       element = site.species_string
       z = site.frac_coords[2]
       if z > 0.7:
           species_labels.append(f"{element}_surface")
       else:
           species_labels.append(f"{element}_bulk")

   structure.add_site_property("species_label", species_labels)

   # Different basis for each variant
   user_params = {
       "%block PAO.BasisSizes": {
           "Ti_surface": "TZP",    # High accuracy for surface
           "Ti_bulk": "DZP",       # Standard for bulk
           "O_surface": "TZP",     # High accuracy for surface
           "O_bulk": "DZ",         # Efficient for bulk
       },
       "Mesh.Cutoff": "300 Ry",
   }

   maker = RelaxMaker.fixed_cell_relaxation(user_params=user_params)

**Use Cases**:

- **Surface calculations**: Different basis for surface vs bulk atoms
- **Adsorption studies**: High accuracy for adsorbate + interface
- **BSSE corrections**: Ghost atoms (Z < 0) for counterpoise method

See ``tutorials/01-basics/07-basis-set-customization/08-10`` for detailed examples.

Per-Atom Basis Control
---------------------------------

.. versionadded:: 1.0.0

For atom-level precision control, use the **per-atom basis helpers**:

**Method 1: Direct Specification** (1-indexed like SIESTA):

.. code-block:: python

   from atomate2.siesta.sets.utils import apply_per_atom_basis

   # Specify basis for individual atoms
   per_atom_basis = {
       1: "TZP",   # Atom 1 (surface) - highest accuracy
       2: "TZP",   # Atom 2 (subsurface)
       3: "DZP",   # Atom 3 (bulk)
       # Unspecified atoms use fallback
   }

   # Generate species labels and basis dict
   species_labels, pao_basissizes = apply_per_atom_basis(
       structure,
       per_atom_basis,
       fallback_basis="DZ"
   )

   # Add to structure
   structure.add_site_property("species_label", species_labels)

   # Use in RelaxMaker
   maker = RelaxMaker(user_params={'%block PAO.BasisSizes': pao_basissizes})

**Method 2: Grouped Specification** (layer-based):

.. code-block:: python

   from atomate2.siesta.sets.utils import create_per_atom_basis_dict

   # Define logical groups
   atom_groups = {
       "surface": ([1, 2, 3], "TZP"),      # Atoms 1-3: surface
       "subsurface": ([4, 5, 6], "DZP"),   # Atoms 4-6: subsurface
       "bulk": ([7, 8, 9], "DZ"),          # Atoms 7-9: bulk
   }

   species_labels, pao_basissizes = create_per_atom_basis_dict(
       structure, atom_groups
   )

**Features**:

- Atom-level precision control (even for same element)
- Automatic species label generation (e.g., ``Ti``, ``Ti_dzp``, ``Ti_dz``)
- Fallback basis for unspecified atoms
- Validation (checks indices, no overlaps)

**Use Cases**:

- Surface slabs with atom-level precision
- Defects (high accuracy around defect site)
- Dopants (special treatment for specific atoms)
- Layer-based systems (automatic grouping)

See ``tutorials/01-basics/07-basis-set-customization/03_1-03_2`` for detailed examples.

PAO.Basis Helper Functions
-------------------------------------

.. versionadded:: 1.0.0

For **programmatic generation** of custom ``%block PAO.Basis`` specifications:

.. code-block:: python

   from atomate2.siesta.sets.utils.basis_builder import create_pao_basis

   # Define custom orbital specifications
   basis_spec = {
       "Ti": {
           "shells": [
               {"n": 4, "l": 0, "nzeta": 2, "rc": [6.0, 0.0]},  # 4s: 2 zeta
               {
                   "n": 3,
                   "l": 2,
                   "nzeta": 2,
                   "rc": [7.0, 0.0],
                   "polarization": True,  # Add polarization orbital
               },
           ]
       },
       "O_surface": {  # Works with species variants!
           "shells": [
               {"n": 2, "l": 0, "nzeta": 2, "rc": [6.0, 0.0]},
               {
                   "n": 2,
                   "l": 1,
                   "nzeta": 2,
                   "rc": [7.0, 0.0],
                   "polarization": True,
                   "split_norm": 0.25,  # Advanced PAO flags
               },
           ]
       },
   }

   # Generate PAO.Basis block (returns list format)
   pao_basis = create_pao_basis(basis_spec)

   # Use with RelaxMaker
   maker = RelaxMaker(user_params={
       '%block PAO.Basis': pao_basis,
       'PAO.BasisSize': 'DZP',  # Fallback for other species
   })

**Supported PAO Flags**:

- ``polarization``: Add l+1 polarization orbitals
- ``split_norm``: Control second zeta generation (0.0-1.0)
- ``soft_conf``: Soft confinement potential
- ``charge_conf``: Charge confinement for excited states
- ``filteret``, ``screen``, ``delta``, ``contraction``

**Features**:

- Programmatic generation (no manual FDF formatting)
- Validation (nzeta vs rc length, l range, split_norm)
- Type-safe dataclass-based system
- Species variants support
- Returns list format (ready for ``user_params``)

**Use Cases**:

- Custom cutoff radii optimization
- Species variants with different orbitals
- Advanced basis set development
- Combine with per-atom helpers for ultimate control

See ``tutorials/01-basics/07-basis-set-customization/11`` for detailed examples.

Complete Workflow Example
--------------------------

Combining all features for maximum control:

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker
   from atomate2.siesta.sets.utils import create_per_atom_basis_dict
   from atomate2.siesta.sets.utils.basis_builder import create_pao_basis

   # Step 1: Define atom groups by layer
   atom_groups = {
       "surface": ([1, 2], "TZP"),
       "bulk": ([3, 4], "DZP"),
   }

   # Step 2: Generate species labels and basis sizes
   species_labels, pao_basissizes = create_per_atom_basis_dict(
       structure, atom_groups
   )
   structure.add_site_property("species_label", species_labels)

   # Step 3: Define custom orbitals for surface atoms
   custom_orbitals = {
       "Ti": {  # Surface Ti with custom orbitals
           "shells": [
               {"n": 4, "l": 0, "nzeta": 2, "rc": [6.5, 0.0]},
               {"n": 3, "l": 2, "nzeta": 2, "rc": [7.5, 0.0],
                "polarization": True},
           ]
       },
   }
   pao_basis = create_pao_basis(custom_orbitals)

   # Step 4: Combine everything
   user_params = {
       "%block PAO.BasisSizes": pao_basissizes,  # From per-atom helper
       "%block PAO.Basis": pao_basis,            # Custom orbitals
       "Mesh.Cutoff": "300 Ry",
   }

   maker = RelaxMaker.fixed_cell_relaxation(user_params=user_params)

This provides:

- ✅ Atom-level basis size control
- ✅ Custom orbital specifications
- ✅ Species variants
- ✅ Validation and type safety

----

Troubleshooting
===============

Parameters Not Applied
----------------------

**Problem**: User parameters don't appear in siesta.fdf

**Solution**: Check that you're using correct parameter names:

.. code-block:: bash

   # Find the correct parameter name
   atomate2siesta-inputs search "cutoff"

**Common mistakes**:

- ❌ ``"mesh_cutoff"`` or ``"a2s_mesh_cutoff"`` (never existed)
- ✅ ``"Mesh.Cutoff": "300 Ry"`` (correct SIESTA FDF parameter)
- ✅ ``"a2s_kpts": [4, 4, 4]`` (framework shortcut - valid)
- ✅ ``"%block kgrid.Monkhorst.Pack": [...]`` (direct FDF block - also valid)

Unexpected Defaults
-------------------

**Problem**: Auto-generated values instead of user values

**Example**: User provides ``kpts: [2,2,2]`` but gets 18x18x18 in siesta.fdf

**Solution**: This was fixed in v1.0.0. Update to latest version:

.. code-block:: bash

   pip install --upgrade atomate2siesta

The FDF regeneration system ensures user values override auto-generated ones.

----

Parameter Units Reference
==========================

.. versionadded:: 1.0.0

All FDF parameters with physical dimensions now have explicit unit information in their metadata.
This helps with parameter validation, documentation, and unit conversion.

Unit System Overview
--------------------

SIESTA uses the following unit conventions:

**Energy Units**:
   - ``Ry`` (Rydberg) - Default SIESTA energy unit (1 Ry = 13.6057 eV)
   - ``eV`` (electron volt) - Common for small energies, tolerances
   - ``meV`` (millielectron volt) - Very small energy scales
   - ``Hartree`` - Atomic units (1 Hartree = 2 Ry = 27.211 eV)

**Length Units**:
   - ``Ang`` (Angstrom) - Default SIESTA length unit (10⁻¹⁰ m)
   - ``Bohr`` (Bohr radius) - Atomic length unit (1 Bohr = 0.529177 Ang)

**Time Units**:
   - ``fs`` (femtoseconds) - Molecular dynamics timesteps (10⁻¹⁵ s)
   - ``ps`` (picoseconds) - Longer MD simulations (10⁻¹² s)

**Temperature Units**:
   - ``K`` (Kelvin) - Absolute temperature

**Pressure Units**:
   - ``GPa`` (Gigapascal) - Stress/pressure (10⁹ Pa)
   - ``Ry/Bohr³`` - Atomic pressure units
   - ``eV/Ang³`` - Alternative pressure unit

**Force Units**:
   - ``eV/Ang`` - Energy gradient (force)
   - ``Ry/Bohr`` - Atomic force units

**Composite Units**:
   - ``Ry*fs²`` - Mass parameters (Nosé-Hoover, Parrinello-Rahman)
   - ``Ry/rad`` - Angular force tolerance

Accessing Unit Information
---------------------------

The :class:`~atomate2.siesta.dataclass.base.FDFDataclass` base class provides helper methods
to query unit information:

.. code-block:: python

   from atomate2.siesta.dataclass.real_space_grid_parameters import RealSpaceGridParameters

   # Get unit for a specific field
   unit = RealSpaceGridParameters.get_field_unit("mesh_cutoff")
   print(unit)  # Output: "Ry"

   # Get unit for an FDF parameter
   unit = RealSpaceGridParameters.get_fdf_parameter_unit("Mesh.Cutoff")
   print(unit)  # Output: "Ry"

   # Get all fields with units
   fields_with_units = RealSpaceGridParameters.get_all_fields_with_units()
   for field_name, metadata in fields_with_units.items():
       print(f"{field_name}: {metadata['unit']} - {metadata['description']}")

   # Output:
   # mesh_cutoff: Ry - Sets the energy cutoff that determines...
   # eggbox_scale: eV - Energy scale for eggbox correction...

Parameters by Unit Type
-----------------------

**Energy Parameters** (Ry, eV):
   - ``Mesh.Cutoff`` (Ry) - Real-space grid energy cutoff
   - ``PAO.EnergyShift`` (Ry) - Basis set confinement energy
   - ``PAO.EnergyCutoff`` (Ry) - PAO filtering cutoff
   - ``SCF.H.Tolerance`` (eV) - Hamiltonian convergence tolerance
   - ``SCF.EDM.Tolerance`` (eV) - Energy-density-matrix tolerance
   - ``ElectronicTemperature`` (K) - Electronic smearing temperature
   - ``ON.Eta`` (eV) - Linear-scaling localization radius
   - ``BulkBiasVoltage`` (eV) - Transport bias voltage

**Length Parameters** (Ang, Bohr):
   - ``MD.MaxDispl`` (Bohr) - Maximum atomic displacement per MD step
   - ``PAO.SoftInnerRadius`` (Bohr) - Soft confinement inner radius
   - ``RMaxRadialGrid`` (Bohr) - Maximum radius for radial grids
   - ``MM.Cutoff`` (Bohr) - Molecular mechanics cutoff
   - ``DFTD3.2BodyCutoff`` (Bohr) - Grimme D3 2-body cutoff
   - ``ON.RcLWF`` (Bohr) - Localized wavefunction cutoff radius

**Time Parameters** (fs):
   - ``MD.LengthTimeStep`` (fs) - Molecular dynamics timestep
   - ``MD.TauRelax`` (fs) - Nose-Hoover thermostat relaxation time
   - ``TDED.TimeStep`` (fs) - Real-time TDDFT timestep

**Temperature Parameters** (K):
   - ``MD.InitialTemperature`` (K) - Initial MD temperature
   - ``MD.TargetTemperature`` (K) - Target temperature for thermostat
   - ``ElectronicTemperature`` (K) - Electronic occupation smearing

**Pressure/Stress Parameters** (GPa):
   - ``MD.MaxStressTol`` (GPa) - Maximum stress tolerance for relaxation
   - ``MD.TargetPressure`` (GPa) - Target pressure for NPT dynamics
   - ``BasisPressure`` (GPa) - Basis set confinement pressure

**Force Parameters** (eV/Ang, Ry/Bohr):
   - ``MD.MaxForceTol`` (eV/Ang) - Maximum force tolerance
   - ``ZM.ForceTolLength`` (Ry/Bohr) - Z-matrix bond force tolerance
   - ``ZM.ForceTolAngle`` (Ry/rad) - Z-matrix angle force tolerance

**Composite Units**:
   - ``MD.NoseMass`` (Ry*fs²) - Nosé-Hoover thermostat mass
   - ``MD.ParrinelloRahmanMass`` (Ry*fs²) - Barostat mass parameter
   - ``MD.BulkModulus`` (Ry/Bohr³) - Bulk modulus for cell optimization

Example: Using Unit Information
--------------------------------

.. code-block:: python

   from atomate2.siesta.dataclass.basis_sets_and_projectors import BasisSetsAndProjectors

   # Query all parameters with units
   basis_params = BasisSetsAndProjectors.get_all_fields_with_units()

   # Display parameter information
   for field_name, info in basis_params.items():
       print(f"Parameter: {info['SIESTA keyword']}")
       print(f"  Field: {field_name}")
       print(f"  Unit: {info['unit']}")
       print(f"  Description: {info['description'][:60]}...")
       print()

   # Output:
   # Parameter: PAO.EnergyShift
   #   Field: pao_energy_shift
   #   Unit: Ry
   #   Description: The energy shift that determines the cutoff radii of t...
   #
   # Parameter: PAO.BasisPressure
   #   Field: basis_pressure
   #   Unit: GPa
   #   Description: An alternative confinement scheme applying an effectiv...

Implementation Details
----------------------

Units are stored in the ``metadata`` dictionary of each :class:`dataclasses.field`:

.. code-block:: python

   from dataclasses import dataclass, field
   from atomate2.siesta.dataclass.base import FDFDataclass

   @dataclass
   class MeshParameters(FDFDataclass):
       mesh_cutoff: float = field(
           default=100.0,
           metadata={
               "description": "Energy cutoff for real-space grid",
               "SIESTA keyword": "Mesh.Cutoff",
               "unit": "Ry",  # ✨ Unit information
           },
       )

       pao_energy_shift: float = field(
           default=0.02,
           metadata={
               "description": "PAO confinement energy shift",
               "SIESTA keyword": "PAO.EnergyShift",
               "unit": "Ry",  # ✨ Unit information
           },
       )

**51 parameters** across **9 dataclass modules** now include explicit unit metadata.

----

See Also
========

- :doc:`usage` - General usage guide
- :doc:`tier-system` - Tier-based parameter presets
- :doc:`siesta-inputs` - Complete FDF parameter reference
- :doc:`troubleshooting` - Detailed troubleshooting guide
- :doc:`tutorials/index` - Tutorials with parameter examples

----

**Questions or Issues?**

- GitHub Issues: https://github.com/materialsproject/atomate2/issues
- SIESTA Manual: https://docs.siesta-project.org/
