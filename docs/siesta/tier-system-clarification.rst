========================================
Tier System: Critical Clarifications
========================================

.. important::
   **This document clarifies the distinction between tier levels (for module activation)
   and tier defaults (parameter sets). Read this if you encounter errors like
   "Invalid tier" or "object has no attribute '_md_relaxation_module'".**

----

Two Separate Concepts
======================

The tier system has **two separate but related concepts** that are often confused:

1. **Tier Levels** (Module Activation)
---------------------------------------

These control **which dataclass modules are initialized**:

.. code-block:: python

   # Valid tier levels for module activation:
   tier = "basic"          # 6 modules
   tier = "intermediate"   # 12 modules (basic + 6 more)
   tier = "advanced"       # 19 modules (intermediate + 7 more)
   tier = "expert"         # 24 modules (all modules)
   tier = "all"            # Same as expert

**Used in**:

- ``SiestaInputGenerator(tier="basic")``
- ``RelaxMaker.fixed_cell_relaxation(tier="intermediate")``
- Module filtering in registry

**Defined in**: ``src/atomate2/siesta/dataclass/registry.py`` → ``tier_hierarchy`` dict

2. **TIER_DEFAULTS** (Parameter Presets)
-----------------------------------------

These provide **quick starting parameters** for common calculation types:

.. code-block:: python

   TIER_DEFAULTS = {
       "basic_dirty": {"PAO.BasisSize": "SZ", "a2s_kpts": [2,2,2], ...},
       "basic": {"PAO.BasisSize": "DZP", "a2s_kpts": [3,3,3], ...},
       "basic_slab": {"PAO.BasisSize": "DZP", "a2s_kpts": [2,2,1], ...},
       "basic_slab_dirty": {...},
       "intermediate": {...},
       "advanced": {...},
       "expert": {...},
       "ultra": {...},
   }

**Used in**:

- Applied automatically when you set ``tier=`` parameter in base class ``__post_init__()``
- Only affects USER PARAMETERS, not module activation

**Defined in**: ``src/atomate2/siesta/sets/tiers/defaults.py`` → ``TIER_DEFAULTS`` dict

----

The Confusion Explained
=======================

Why ``tier="basic_dirty"`` Fails
---------------------------------

❌ **This will fail**:

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(tier="basic_dirty")
   # Error: Invalid tier 'basic_dirty'. Must be one of ['basic', 'intermediate', 'advanced', 'expert', 'all']

**Why**: ``"basic_dirty"`` exists in ``TIER_DEFAULTS`` (for parameters) but NOT in
``tier_hierarchy`` (for module activation).

**The Fix** (Current System):

.. code-block:: python

   # Option 1: Use a valid tier level
   maker = RelaxMaker.fixed_cell_relaxation(tier="basic")  # ✅ Works

   # Option 2: Use TIER_DEFAULTS for parameters only
   # (parameters are applied automatically in __post_init__)

**Future Fix** (Recommended):

Add extended tier mappings to ``tier_hierarchy`` in ``registry.py``:

.. code-block:: python

   tier_hierarchy = {
       # Core tier levels
       "basic": ["basic"],
       "intermediate": ["basic", "intermediate"],
       "advanced": ["basic", "intermediate", "advanced"],
       "expert": ["basic", "intermediate", "advanced", "expert"],
       "all": ["basic", "intermediate", "advanced", "expert"],

       # Extended tiers (map to core levels for module activation)
       "basic_dirty": ["basic"],      # Use basic modules
       "basic_slab": ["basic"],       # Use basic modules
       "basic_slab_dirty": ["basic"], # Use basic modules
       "ultra": ["basic", "intermediate", "advanced", "expert"],  # All modules
   }

Why ``RelaxMaker`` Fails with ``tier="basic"``
-----------------------------------------------

❌ **This will fail**:

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(tier="basic")
   job = maker.make(structure)
   # Error: 'RelaxSetGenerator' object has no attribute '_md_relaxation_module'

**Why**:

1. ``RelaxSetGenerator`` REQUIRES the ``md_relaxation`` module (MolecularDynamicsAndRelaxation)
2. ``md_relaxation`` is registered in **intermediate tier**, NOT basic tier
3. When ``tier="basic"``, only 6 modules are activated
4. ``md_relaxation`` is NOT in those 6 modules
5. When ``get_parameter_updates()`` tries to access ``self._md_relaxation_module``, it doesn't exist

**The Fix**:

.. code-block:: python

   # Option 1: Use intermediate tier (includes md_relaxation)
   maker = RelaxMaker.fixed_cell_relaxation(tier="intermediate")  # ✅ Works

   # Option 2: Explicitly enable md_relaxation with basic tier
   maker = RelaxMaker.fixed_cell_relaxation(
       tier="basic",
       enabled_modules=["md_relaxation"],  # ✅ Force-enable this module
   )

**Root Cause**: ``md_relaxation`` should probably be in ``basic`` tier since
``RelaxSetGenerator`` (a fundamental class) requires it.

----

Module Requirements by Maker
=============================

Critical Module Dependencies
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 20 20

   * - Maker/Generator
     - Required Module
     - Current Tier
     - Minimum tier=
   * - ``RelaxSetGenerator``
     - ``md_relaxation``
     - intermediate
     - ``intermediate``
   * - ``BandStructureSetGenerator``
     - ``dos_bands``
     - advanced
     - ``advanced``
   * - ``StaticMaker``
     - (none required)
     - -
     - ``basic``
   * - ``PhonopyMaker``
     - ``phonons``
     - advanced
     - ``advanced``

**Recommendation**: Always use at least ``tier="intermediate"`` for RelaxMaker.

----

How Tier Parameter Works
=========================

Step-by-Step Flow
-----------------

When you write:

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(
       tier="intermediate",
       user_params={"PAO.BasisSize": "TZP"}
   )

**What happens**:

1. **Parameter passed to InputSetGenerator**:

   .. code-block:: python

      RelaxSetGenerator(tier="intermediate", user_params={...})

2. **In __post_init__() of SiestaInputGenerator**:

   a. **Load TIER_DEFAULTS**:

      .. code-block:: python

         from atomate2.siesta.sets.tiers import TIER_DEFAULTS
         tier_defaults = TIER_DEFAULTS["intermediate"]
         # tier_defaults = {"PAO.BasisSize": "DZP", "a2s_kpts": [4,4,4], ...}

   b. **Merge with user_params** (user_params take precedence):

      .. code-block:: python

         merged = {**tier_defaults, **user_params}
         # Result: {"PAO.BasisSize": "TZP", "a2s_kpts": [4,4,4], ...}

3. **Later, in get_input_set() → _get_input_parameters()**:

   a. **Get modules for tier**:

      .. code-block:: python

         modules = get_modules_for_tier("intermediate")
         # Returns 12 modules: basic (6) + intermediate (6)

   b. **Initialize modules** (including ``md_relaxation``):

      .. code-block:: python

         for module in modules:
             instance = module.setup_method(user_params)
             setattr(self, f"_{module.instance_attribute}_module", instance)
             # Creates: self._md_relaxation_module

   c. **Call get_parameter_updates()**:

      .. code-block:: python

         # RelaxSetGenerator.get_parameter_updates()
         relaxation_instance = self._md_relaxation_module  # ✅ Now exists!

----

Common Errors and Solutions
============================

Error 1: "Invalid tier '...'"
------------------------------

**Error**:

.. code-block:: text

   ValueError: Invalid tier 'basic_dirty'. Must be one of ['basic', 'intermediate', 'advanced', 'expert', 'all']

**Cause**: Using TIER_DEFAULTS key as tier level parameter.

**Solution**:

.. code-block:: python

   # ❌ Wrong
   maker = RelaxMaker.fixed_cell_relaxation(tier="basic_dirty")

   # ✅ Correct
   maker = RelaxMaker.fixed_cell_relaxation(tier="basic")

**Future**: Extended tier names will be added to ``tier_hierarchy`` mapping.

Error 2: "object has no attribute '_md_relaxation_module'"
-----------------------------------------------------------

**Error**:

.. code-block:: text

   AttributeError: 'RelaxSetGenerator' object has no attribute '_md_relaxation_module'

**Cause**: Using ``tier="basic"`` with ``RelaxMaker``, but ``md_relaxation`` module
is in ``intermediate`` tier.

**Solution 1** (Recommended):

.. code-block:: python

   # Use intermediate tier (includes md_relaxation)
   maker = RelaxMaker.fixed_cell_relaxation(tier="intermediate")

**Solution 2**:

.. code-block:: python

   # Force-enable md_relaxation with basic tier
   maker = RelaxMaker.fixed_cell_relaxation(
       tier="basic",
       enabled_modules=["md_relaxation"],
   )

Error 3: Module Not Activated
------------------------------

**Symptom**: Powerup doesn't work for advanced parameters.

**Example**:

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(tier="basic")
   job = maker.make(structure)

   # Try to set phonon parameter
   job = update_user_siesta_settings(job, {"MD.FCDispl": "0.02 Bohr"})
   # Parameter ignored - phonons module not active!

**Cause**: ``phonons`` module is in ``advanced`` tier, not activated with ``tier="basic"``.

**Solution**:

.. code-block:: python

   # Option 1: Use advanced tier
   maker = RelaxMaker.fixed_cell_relaxation(tier="advanced")

   # Option 2: Enable phonons explicitly
   maker = RelaxMaker.fixed_cell_relaxation(
       tier="basic",
       enabled_modules=["phonons"],
   )

----

Complete Reference
==================

Valid Tier Levels (Module Activation)
--------------------------------------

.. code-block:: python

   # These are VALID for tier= parameter (module activation)
   "basic"          # 6 modules
   "intermediate"   # 12 modules
   "advanced"       # 19 modules
   "expert"         # 24 modules (all)
   "all"            # Same as expert

TIER_DEFAULTS Keys (Parameter Sets)
------------------------------------

.. code-block:: python

   # These provide DEFAULT PARAMETERS only
   # (NOT all are valid for tier= parameter yet)
   "basic_dirty"      # Minimal parameters (SZ, [2,2,2], 50 Ry)
   "basic"            # Standard basic (DZP, [3,3,3], 150 Ry)
   "basic_slab"       # 2D slab (DZP, [2,2,1], 150 Ry)
   "basic_slab_dirty" # Minimal slab (SZ, [1,1,1], 150 Ry)
   "intermediate"     # Standard (DZP, [4,4,4], 200 Ry)
   "advanced"         # High quality (TZP, [6,6,6], 300 Ry)
   "expert"           # Expert (TZP, [8,8,8], 400 Ry)
   "ultra"            # Ultra high (TZP, [8,8,8], 800 Ry)

Modules by Tier Level
----------------------

**Basic (6 modules)**:

- ``pseudopotentials``
- ``basis_sets_and_projectors``
- ``xc_functional``
- ``kpoints``
- ``mesh_cutoff``
- ``general_system``

**Intermediate (+6 modules = 12 total)**:

- ``spin``
- ``scf_loop_parameters``
- ``electronic_structure_calculation_options``
- **``md_relaxation``** ← Required by RelaxMaker
- ``constraints``
- ``lua_scripting``

**Advanced (+7 modules = 19 total)**:

- ``phonons``
- ``optical``
- ``dos_bands`` ← Required by BandStructureSetGenerator
- ``dftu``
- ``charge_dipole``
- ``grids_advanced``
- ``denchar``

**Expert (+5 modules = 24 total)**:

- ``parallel``
- ``solvers``
- ``efficiency``
- ``hamiltonian_overlap``
- ``netcdf``

----

Best Practices
==============

Choosing the Right Tier
------------------------

**For RelaxMaker**:

✅ **Use ``tier="intermediate"`` or higher**

- Includes required ``md_relaxation`` module
- Standard SCF convergence control
- Spin polarization support

❌ **Don't use ``tier="basic"``**

- Missing ``md_relaxation`` → errors
- Unless you explicitly enable it

**For StaticMaker**:

✅ **``tier="basic"`` works fine**

- No special module requirements
- Good for quick calculations

**For Specialized Calculations**:

- Phonons: ``tier="advanced"`` (needs ``phonons`` module)
- Band structure: ``tier="advanced"`` (needs ``dos_bands`` module)
- DFT+U: ``tier="advanced"`` (needs ``dftu`` module)
- HPC tuning: ``tier="expert"`` (needs ``parallel``, ``solvers`` modules)

Using TIER_DEFAULTS
-------------------

**Current behavior**: TIER_DEFAULTS parameters are automatically applied
based on ``tier=`` parameter:

.. code-block:: python

   # This automatically gets TIER_DEFAULTS["intermediate"] parameters
   maker = RelaxMaker.fixed_cell_relaxation(tier="intermediate")

   # Default params: {"PAO.BasisSize": "DZP", "a2s_kpts": [4,4,4], "Mesh.Cutoff": "200 Ry"}

**Override defaults**:

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(
       tier="intermediate",
       user_params={
           "PAO.BasisSize": "TZP",  # Override default "DZP"
           "Mesh.Cutoff": "400 Ry",  # Override default "200 Ry"
       }
   )

----

Planned Improvements
====================

1. **Fix ``tier_hierarchy`` in registry.py**

   Add extended tier mappings so ``tier="basic_dirty"`` etc. work for module activation.

2. **Move ``md_relaxation`` to basic tier**

   Since ``RelaxMaker`` requires it, it should be in basic tier.

3. **Add Maker-specific tier validation**

   ``RelaxMaker.fixed_cell_relaxation()`` should warn or error if ``tier="basic"`` without
   ``enabled_modules=["md_relaxation"]``.

4. **Better error messages**

   When ``_md_relaxation_module`` is missing, suggest using ``tier="intermediate"`` or higher.

----

See Also
========

- :doc:`tier-system` - Main tier system documentation
- :doc:`cli-tools` - ``atomate2siesta-presets`` CLI reference
- :doc:`troubleshooting` - General troubleshooting guide
