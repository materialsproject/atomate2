========================================
TIER_DEFAULTS: Complete Explanation
========================================

.. important::
   **This document explains EXACTLY how TIER_DEFAULTS works with concrete examples.**

----

What is TIER_DEFAULTS?
=======================

``TIER_DEFAULTS`` is a Python dictionary that provides **starting parameter values**
for each tier level. Think of it as a "quick start template" for common calculation types.

**Location**: ``src/atomate2/siesta/sets/tiers/defaults.py``

**Content** (6 entries):

.. code-block:: python

   TIER_DEFAULTS = {
       "dirty": {
           "PAO.BasisSize": "SZ",      # Single-zeta (fastest, lowest quality)
           "a2s_kpts": [1, 1, 1],      # Minimal k-points
           "Mesh.Cutoff": "50 Ry",     # Low cutoff
       },
       "basic": {
           "PAO.BasisSize": "DZP",     # Double-zeta polarized
           "a2s_kpts": [3, 3, 3],
           "Mesh.Cutoff": "150 Ry",
       },
       "intermediate": {
           "PAO.BasisSize": "DZP",
           "a2s_kpts": [6, 6, 6],      # Denser k-points
           "Mesh.Cutoff": "200 Ry",
       },
       "advanced": {
           "PAO.BasisSize": "TZP",     # Triple-zeta polarized
           "a2s_kpts": [6, 6, 6],
           "Mesh.Cutoff": "300 Ry",    # Higher cutoff
       },
       "expert": {
           "PAO.BasisSize": "TZP",
           "a2s_kpts": [8, 8, 8],      # Very dense k-points
           "Mesh.Cutoff": "400 Ry",
       },
       "ultra": {
           "PAO.BasisSize": "TZDP",    # Triple-zeta double polarized
           "a2s_kpts": [10, 10, 10],   # Extremely dense
           "Mesh.Cutoff": "800 Ry",    # Very high cutoff (benchmark quality)
       },
   }

----

When Are TIER_DEFAULTS Used?
=============================

TIER_DEFAULTS are applied **AUTOMATICALLY** in ``SiestaInputGenerator.__post_init__()``
when you create a Maker with a ``tier=`` parameter.

Flow Diagram
------------

.. code-block:: text

   User writes:
   ┌──────────────────────────────────────────────────────────┐
   │ maker = RelaxMaker.fixed_cell_relaxation(               │
   │     tier="intermediate",                                 │
   │     user_params={"PAO.BasisSize": "TZP"}                │
   │ )                                                        │
   └──────────────────────────────────────────────────────────┘
                           ↓

   Step 1: Create RelaxSetGenerator
   ┌──────────────────────────────────────────────────────────┐
   │ RelaxSetGenerator(                                       │
   │     tier="intermediate",                                 │
   │     user_params={"PAO.BasisSize": "TZP"}                │
   │ )                                                        │
   └──────────────────────────────────────────────────────────┘
                           ↓

   Step 2: __post_init__() is called automatically
   ┌──────────────────────────────────────────────────────────┐
   │ # Load TIER_DEFAULTS                                     │
   │ from atomate2.siesta.sets.tiers import TIER_DEFAULTS    │
   │                                                          │
   │ # Get defaults for "intermediate"                        │
   │ tier_defaults = TIER_DEFAULTS["intermediate"]           │
   │ # = {"PAO.BasisSize": "DZP",                            │
   │ #    "a2s_kpts": [6, 6, 6],                             │
   │ #    "Mesh.Cutoff": "200 Ry"}                           │
   └──────────────────────────────────────────────────────────┘
                           ↓

   Step 3: Merge with user_params (user takes precedence)
   ┌──────────────────────────────────────────────────────────┐
   │ merged = {**tier_defaults, **user_params}                │
   │                                                          │
   │ Result:                                                  │
   │ {                                                        │
   │     "PAO.BasisSize": "TZP",        ← USER OVERRIDE!     │
   │     "a2s_kpts": [6, 6, 6],         ← from tier defaults │
   │     "Mesh.Cutoff": "200 Ry",       ← from tier defaults │
   │ }                                                        │
   └──────────────────────────────────────────────────────────┘
                           ↓

   Step 4: Parameters ready for use
   ┌──────────────────────────────────────────────────────────┐
   │ self.user_params = merged                                │
   │                                                          │
   │ # These will be used to create SIESTA FDF file          │
   └──────────────────────────────────────────────────────────┘

----

Concrete Examples
=================

Example 1: Using tier defaults only
------------------------------------

**Code**:

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker

   maker = RelaxMaker.fixed_cell_relaxation(tier="basic")

**What happens**:

1. ``tier="basic"`` → Load ``TIER_DEFAULTS["basic"]``
2. No ``user_params`` provided
3. Final parameters:

.. code-block:: python

   {
       "PAO.BasisSize": "DZP",
       "a2s_kpts": [3, 3, 3],
       "Mesh.Cutoff": "150 Ry",
   }

Example 2: Overriding tier defaults
------------------------------------

**Code**:

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(
       tier="basic",
       user_params={
           "PAO.BasisSize": "TZP",    # Override tier default
           "Spin": "polarized",        # Add new parameter
       }
   )

**What happens**:

1. ``tier="basic"`` → Load ``TIER_DEFAULTS["basic"]``

   .. code-block:: python

      tier_defaults = {
          "PAO.BasisSize": "DZP",
          "a2s_kpts": [3, 3, 3],
          "Mesh.Cutoff": "150 Ry",
      }

2. Merge with user_params (user wins):

   .. code-block:: python

      merged = {**tier_defaults, **user_params}

3. Final parameters:

   .. code-block:: python

      {
          "PAO.BasisSize": "TZP",      # ← USER overrode "DZP"
          "a2s_kpts": [3, 3, 3],       # ← From tier defaults
          "Mesh.Cutoff": "150 Ry",     # ← From tier defaults
          "Spin": "polarized",         # ← USER added
      }

Example 3: Using tier="dirty" for quick tests
----------------------------------------------

**Code**:

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(tier="dirty")

**What happens**:

1. ``tier="dirty"`` → Load ``TIER_DEFAULTS["dirty"]``
2. Final parameters (minimal settings):

.. code-block:: python

   {
       "PAO.BasisSize": "SZ",       # Single-zeta (fastest)
       "a2s_kpts": [1, 1, 1],       # Gamma point only
       "Mesh.Cutoff": "50 Ry",      # Low accuracy
   }

**Use case**: Quick workflow testing, debugging, structure validation

Example 4: Using tier="ultra" for benchmarks
---------------------------------------------

**Code**:

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(tier="ultra")

**What happens**:

1. ``tier="ultra"`` → Load ``TIER_DEFAULTS["ultra"]``
2. Final parameters (very high quality):

.. code-block:: python

   {
       "PAO.BasisSize": "TZDP",     # Triple-zeta double polarized
       "a2s_kpts": [10, 10, 10],    # Very dense k-mesh
       "Mesh.Cutoff": "800 Ry",     # Very high cutoff
   }

**Use case**: Benchmark calculations, publication-quality results, convergence testing

----

Key Points
==========

✅ **Automatic Application**

   TIER_DEFAULTS are applied automatically in ``__post_init__()``.
   You don't need to do anything special.

✅ **User Parameters Win**

   If you provide ``user_params``, they override tier defaults.

   .. code-block:: python

      # Tier says DZP, user says TZP → TZP wins
      maker = RelaxMaker.fixed_cell_relaxation(
          tier="basic",                    # PAO.BasisSize = "DZP"
          user_params={"PAO.BasisSize": "TZP"}  # TZP wins!
      )

✅ **Tier Name = Two Functions**

   The ``tier`` parameter does **TWO things**:

   1. **Module activation** (via ``tier_hierarchy`` in ``registry.py``)

      - ``tier="basic"`` → Activate 6 basic modules
      - ``tier="intermediate"`` → Activate 13 modules
      - ``tier="dirty"`` → Activate 6 basic modules (maps to "basic")
      - ``tier="ultra"`` → Activate all 28 modules (maps to "expert")

   2. **Default parameters** (via ``TIER_DEFAULTS`` in ``defaults.py``)

      - ``tier="basic"`` → Load ``TIER_DEFAULTS["basic"]`` parameters
      - ``tier="dirty"`` → Load ``TIER_DEFAULTS["dirty"]`` parameters

✅ **Only 3 Parameters**

   TIER_DEFAULTS only set **3 core parameters**:

   - ``PAO.BasisSize`` - Basis set size
   - ``a2s_kpts`` - k-point sampling
   - ``Mesh.Cutoff`` - Real-space grid cutoff

   All other parameters use dataclass defaults or must be set by user.

----

Where TIER_DEFAULTS Are Used
=============================

**1. In SiestaInputGenerator.__post_init__()** (MAIN USE)

   **File**: ``src/atomate2/siesta/sets/base.py:535-549``

   **Code**:

   .. code-block:: python

      if self.tier in TIER_DEFAULTS:
          tier_defaults = OrderedDict(TIER_DEFAULTS[self.tier])
          if self.user_params:
              merged_params = OrderedDict(tier_defaults)
              merged_params.update(self.user_params)
              self.user_params = merged_params
          else:
              self.user_params = tier_defaults

   **When**: Every time you create a Maker with ``tier=`` parameter

**2. In CLI (atomate2siesta-presets defaults)**

   **File**: ``src/atomate2/siesta/cli/tiers.py:398``

   **Code**:

   .. code-block:: python

      for tier_name in tier_order:
          params = TIER_DEFAULTS[tier_name]
          console.print(f"Tier: {tier_name}")
          # Display parameters...

   **When**: When you run ``atomate2siesta-presets defaults``

----

Tier Name Meanings
==================

.. list-table::
   :header-rows: 1
   :widths: 15 25 20 40

   * - Tier Name
     - Basis / k-pts / Cutoff
     - Speed
     - Use Case
   * - ``dirty``
     - SZ / [1,1,1] / 50 Ry
     - Very Fast ⚡⚡⚡
     - Quick testing, workflow debugging
   * - ``basic``
     - DZP / [3,3,3] / 150 Ry
     - Fast ⚡⚡
     - Initial relaxations, structure checks
   * - ``intermediate``
     - DZP / [6,6,6] / 200 Ry
     - Medium ⚡
     - Standard production calculations
   * - ``advanced``
     - TZP / [6,6,6] / 300 Ry
     - Slow 🐌
     - High-quality results
   * - ``expert``
     - TZP / [8,8,8] / 400 Ry
     - Very Slow 🐌🐌
     - Publication quality
   * - ``ultra``
     - TZDP / [10,10,10] / 800 Ry
     - Extremely Slow 🐌🐌🐌
     - Benchmark, convergence tests

----

Common Confusion: TIER_DEFAULTS vs TIER_PRESETS
================================================

**TIER_DEFAULTS** (Simple, 3 parameters)

.. code-block:: python

   # Just 3 core parameters
   TIER_DEFAULTS["basic"] = {
       "PAO.BasisSize": "DZP",
       "a2s_kpts": [3, 3, 3],
       "Mesh.Cutoff": "150 Ry",
   }

**TIER_PRESETS** (Complex, many parameters + module control)

.. code-block:: python

   # Many parameters + module activation
   TIER_PRESETS["relax_standard"] = {
       "description": "Standard production relaxation",
       "tier": "intermediate",           # ← Which tier to use
       "enabled_modules": [],            # ← Module control
       "recommended_params": {
           "PAO.BasisSize": "DZP",
           "a2s_kpts": [4, 4, 4],
           "Mesh.Cutoff": "300 Ry",
           "MD.MaxForceTol": "0.02 eV/Ang",  # ← Extra parameters
           "SCF.DM.Tolerance": "1e-4",       # ← Extra parameters
           # ... many more ...
       },
   }

**When to use each**:

- **TIER_DEFAULTS**: Quick start, minimal customization

  .. code-block:: python

     maker = RelaxMaker.fixed_cell_relaxation(tier="basic")

- **TIER_PRESETS**: Material-specific, comprehensive settings

  .. code-block:: python

     from atomate2.siesta.sets.tiers import apply_tier_preset
     maker = RelaxMaker.fixed_cell_relaxation()
     maker = apply_tier_preset(maker, "relax_standard")

----

Testing TIER_DEFAULTS
======================

You can inspect what parameters are loaded:

.. code-block:: python

   from atomate2.siesta.sets.tiers import TIER_DEFAULTS
   from pprint import pprint

   # See all tier defaults
   pprint(TIER_DEFAULTS)

   # See specific tier
   print(TIER_DEFAULTS["intermediate"])
   # Output:
   # {
   #     'PAO.BasisSize': 'DZP',
   #     'a2s_kpts': [6, 6, 6],
   #     'Mesh.Cutoff': '200 Ry'
   # }

   # Check if tier exists
   if "dirty" in TIER_DEFAULTS:
       print("dirty tier exists!")

----

See Also
========

- :doc:`tier-system` - Main tier system documentation
- :doc:`tier-system-clarification` - Tier levels vs tier defaults
- :doc:`cli-tools` - Using ``atomate2siesta-presets`` CLI
