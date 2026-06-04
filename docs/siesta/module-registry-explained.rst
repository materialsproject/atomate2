=======================================
Module Registry System: Complete Guide
=======================================

.. important::
   **This document explains the complete module registration and tier activation system.**

----

What is the Module Registry?
=============================

The **module registry** is a central database that tracks all 28 dataclass modules used to organize SIESTA input parameters. It enables:

1. **Automatic module discovery** - No manual imports needed
2. **Tier-based activation** - Load only the modules you need
3. **Dependency management** - Modules initialize in the correct order
4. **Material-specific presets** - Quick setup for common calculation types

**Location**: ``src/atomate2/siesta/dataclass/registry.py``

----

Three Key Dictionaries
======================

The tier system uses **3 interconnected dictionaries**:

1. MODULE_REGISTRY - All Available Modules
-------------------------------------------

**Location**: ``registry.py:77``

**Type**: ``dict[str, DataclassModule]``

**Purpose**: Global registry storing metadata for all 28 dataclass modules

**Example Entry**:

.. code-block:: python

   MODULE_REGISTRY = {
       "pseudopotentials": DataclassModule(
           name="pseudopotentials",
           module_path="atomate2.siesta.dataclass.pseudopotentials",
           class_name="Pseudopotentials",
           setup_method="setup_pseudos",
           fdf_attribute="pseudo_path",
           tier="basic",
           category="electronic",
           priority=5,
           description="Pseudopotential file paths and species definitions",
       ),
       "md_relaxation": DataclassModule(
           name="md_relaxation",
           tier="intermediate",  # ← This module only loads for intermediate+
           ...
       ),
       # ... 26 more modules
   }

**How It's Built**: Automatically filled by calling ``register_module()`` 28 times

2. tier_hierarchy - Module Activation Mapping
----------------------------------------------

**Location**: ``registry.py:180-192``

**Type**: ``dict[str, list[str]]``

**Purpose**: Maps tier names → which module tiers to activate

**Complete Definition**:

.. code-block:: python

   tier_hierarchy = {
       # Core tier levels (module activation hierarchy)
       "basic": ["basic"],                                    # 6 modules
       "intermediate": ["basic", "intermediate"],             # 13 modules
       "advanced": ["basic", "intermediate", "advanced"],     # 19 modules
       "expert": ["basic", "intermediate", "advanced", "expert"],  # 28 modules
       "all": ["basic", "intermediate", "advanced", "expert"],     # 28 modules (same as expert)

       # Extended tier levels (map to core for module activation)
       "dirty": ["basic"],                                    # 6 modules (minimal)
       "ultra": ["basic", "intermediate", "advanced", "expert"],   # 28 modules (all)
   }

**Example Usage**:

.. code-block:: python

   # User specifies tier="intermediate"
   allowed_tiers = tier_hierarchy["intermediate"]  # → ["basic", "intermediate"]

   # Load all modules where module.tier in ["basic", "intermediate"]
   modules = {
       name: module
       for name, module in MODULE_REGISTRY.items()
       if module.tier in allowed_tiers
   }
   # Result: 13 modules loaded (6 basic + 7 intermediate)

3. TIER_DEFAULTS - Parameter Presets
-------------------------------------

**Location**: ``src/atomate2/siesta/sets/tiers/defaults.py:16-48``

**Type**: ``dict[str, dict[str, Any]]``

**Purpose**: Provides starting parameter values for each tier

**Complete Definition**:

.. code-block:: python

   TIER_DEFAULTS: dict[str, dict[str, Any]] = {
       "dirty": {
           "PAO.BasisSize": "SZ",       # Single-zeta (fastest)
           "a2s_kpts": [1, 1, 1],       # Gamma point only
           "Mesh.Cutoff": "50 Ry",      # Low accuracy
       },
       "basic": {
           "PAO.BasisSize": "DZP",      # Double-zeta polarized
           "a2s_kpts": [3, 3, 3],
           "Mesh.Cutoff": "150 Ry",
       },
       "intermediate": {
           "PAO.BasisSize": "DZP",
           "a2s_kpts": [6, 6, 6],
           "Mesh.Cutoff": "200 Ry",
       },
       "advanced": {
           "PAO.BasisSize": "TZP",      # Triple-zeta polarized
           "a2s_kpts": [6, 6, 6],
           "Mesh.Cutoff": "300 Ry",
       },
       "expert": {
           "PAO.BasisSize": "TZP",
           "a2s_kpts": [8, 8, 8],
           "Mesh.Cutoff": "400 Ry",
       },
       "ultra": {
           "PAO.BasisSize": "TZDP",     # Triple-zeta double polarized
           "a2s_kpts": [10, 10, 10],
           "Mesh.Cutoff": "800 Ry",     # Benchmark quality
       },
   }

----

How Modules Are Registered
===========================

All 28 modules are registered in ``registry.py`` starting at line 265.

The Registration Process
-------------------------

**Step 1: Define Metadata**

Each module is registered with a ``register_module()`` call:

.. code-block:: python

   register_module(
       name="pseudopotentials",           # Unique identifier
       module_path="atomate2.siesta.dataclass.pseudopotentials",
       class_name="Pseudopotentials",     # Dataclass name
       setup_method="setup_pseudos",      # Initialization method
       fdf_attribute="pseudo_path",       # Where FDF params are stored
       tier="basic",                      # ← Which tier this belongs to
       category="electronic",             # Functional category
       priority=5,                        # Load order (lower = earlier)
       description="Pseudopotential file paths and species definitions",
   )

**Step 2: Add to MODULE_REGISTRY**

The ``register_module()`` function creates a ``DataclassModule`` object and adds it to ``MODULE_REGISTRY``:

.. code-block:: python

   def register_module(name, module_path, class_name, setup_method, ...):
       """Register a dataclass module in the global registry."""
       module = DataclassModule(
           name=name,
           module_path=module_path,
           class_name=class_name,
           setup_method=setup_method,
           tier=tier,
           category=category,
           priority=priority,
           description=description,
       )
       MODULE_REGISTRY[name] = module  # ← Add to global registry

**Step 3: Automatic Discovery**

When you create a Maker, the system automatically:

1. Looks up ``tier`` in ``tier_hierarchy``
2. Gets list of allowed module tiers
3. Filters ``MODULE_REGISTRY`` to only include those tiers
4. Initializes matching modules in priority order

All 28 Registered Modules
--------------------------

**TIER 1: BASIC (6 modules)**

.. code-block:: python

   register_module(name="pseudopotentials", tier="basic", priority=5)
   register_module(name="basis_sets", tier="basic", priority=10)
   register_module(name="xc_functional", tier="basic", priority=15)
   register_module(name="kpoints", tier="basic", priority=20)
   register_module(name="mesh_cutoff", tier="basic", priority=25)
   register_module(name="scf", tier="basic", priority=30)

**TIER 2: INTERMEDIATE (7 more modules)**

.. code-block:: python

   register_module(name="chemical_analysis", tier="intermediate")
   register_module(name="constraints", tier="intermediate")
   register_module(name="electronic_structure", tier="intermediate")
   register_module(name="lua_scripting", tier="intermediate")
   register_module(name="md_relaxation", tier="intermediate")
   register_module(name="scf_loop", tier="intermediate")
   register_module(name="spin", tier="intermediate")

**TIER 3: ADVANCED (9 more modules)**

.. code-block:: python

   register_module(name="auxiliary_force_field", tier="advanced")
   register_module(name="charge_dipole", tier="advanced")
   register_module(name="denchar", tier="advanced")
   register_module(name="dftu", tier="advanced")
   register_module(name="dos_bands", tier="advanced")
   register_module(name="grids_advanced", tier="advanced")
   register_module(name="optical", tier="advanced")
   register_module(name="phonons", tier="advanced")
   register_module(name="wannier90", tier="advanced")

**TIER 4: EXPERT (6 more modules)**

.. code-block:: python

   register_module(name="efficiency", tier="expert")
   register_module(name="hamiltonian_overlap", tier="expert")
   register_module(name="netcdf", tier="expert")
   register_module(name="parallel", tier="expert")
   register_module(name="rttddft", tier="expert")
   register_module(name="solvers", tier="expert")

----

Complete Flow: User Code → Module Activation
=============================================

Step-by-Step Example
--------------------

**User Code**:

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker

   maker = RelaxMaker.fixed_cell_relaxation(tier="intermediate")

**What Happens Internally**:

1. **Maker Creation** (``jobs/core.py``)

   .. code-block:: python

      @classmethod
      def fixed_cell_relaxation(cls, tier="intermediate", ...):
          return RelaxSetGenerator(tier="intermediate", ...)

2. **Generator Initialization** (``sets/core.py``)

   .. code-block:: python

      @dataclass
      class RelaxSetGenerator(SiestaInputGenerator):
          def __post_init__(self):
              # Force-enable md_relaxation module
              if self.enabled_modules is None:
                  self.enabled_modules = ["md_relaxation"]

              # Call parent's __post_init__
              super().__post_init__()

3. **Load Tier Defaults** (``sets/base.py:535``)

   .. code-block:: python

      def __post_init__(self):
          from atomate2.siesta.sets.tiers import TIER_DEFAULTS

          if self.tier in TIER_DEFAULTS:
              tier_defaults = TIER_DEFAULTS[self.tier]
              # tier_defaults = {"PAO.BasisSize": "DZP", "a2s_kpts": [6,6,6], ...}

              # Merge with user_params (user wins)
              merged = {**tier_defaults, **self.user_params}
              self.user_params = merged

4. **Activate Modules** (``dataclass/registry.py:201``)

   .. code-block:: python

      def get_modules_by_tier(tier: str):
          # Look up tier in tier_hierarchy
          tier_hierarchy = {
              "intermediate": ["basic", "intermediate"],  # ← User's tier
              ...
          }

          allowed_tiers = tier_hierarchy["intermediate"]
          # allowed_tiers = ["basic", "intermediate"]

          # Filter MODULE_REGISTRY
          return {
              name: module
              for name, module in MODULE_REGISTRY.items()
              if module.tier in allowed_tiers
          }
          # Returns 13 modules (6 basic + 7 intermediate)

5. **Initialize Modules** (``sets/base.py:600+``)

   .. code-block:: python

      # For each module in priority order:
      for module_name in ["pseudopotentials", "basis_sets", ..., "vdw"]:
          module_metadata = MODULE_REGISTRY[module_name]

          # Import the class
          from atomate2.siesta.dataclass.pseudopotentials import Pseudopotentials

          # Call setup method
          module_instance = Pseudopotentials.setup_pseudos(
              user_params=self.user_params,
              structure=structure,
          )

          # Store instance
          self.pseudopotentials = module_instance

**Result**: 13 modules initialized with tier="intermediate" parameter defaults

Visual Flow Diagram
-------------------

.. code-block:: text

   User Code:
   maker = RelaxMaker.fixed_cell_relaxation(tier="intermediate")
        ↓

   1. Create RelaxSetGenerator(tier="intermediate")
        ↓

   2. __post_init__() called automatically
        ↓

   3. Load TIER_DEFAULTS
        from atomate2.siesta.sets.tiers import TIER_DEFAULTS
        tier_defaults = TIER_DEFAULTS["intermediate"]
        → {"PAO.BasisSize": "DZP", "a2s_kpts": [6,6,6], "Mesh.Cutoff": "200 Ry"}
        ↓

   4. Merge with user_params
        merged = {**tier_defaults, **user_params}
        self.user_params = merged
        ↓

   5. Activate modules (get_modules_by_tier)
        tier_hierarchy["intermediate"] → ["basic", "intermediate"]
        Filter MODULE_REGISTRY → 13 modules
        ↓

   6. Initialize modules in priority order
        pseudopotentials (priority=5)
        basis_sets (priority=10)
        xc_functional (priority=15)
        kpoints (priority=20)
        mesh_cutoff (priority=25)
        scf (priority=30)
        md_relaxation (priority=35)
        spin (priority=40)
        output (priority=45)
        electron_density (priority=50)
        molecular_dynamics (priority=55)
        constraints (priority=60)
        vdw (priority=65)
        ↓

   7. Modules ready to generate FDF parameters

----

Tier Comparison Table
======================

.. list-table::
   :header-rows: 1
   :widths: 12 20 15 15 38

   * - Tier Name
     - Module Activation
     - # Modules
     - Parameters
     - Use Case
   * - ``dirty``
     - basic only
     - 6
     - SZ / [1,1,1] / 50 Ry
     - Quick testing, workflow debugging
   * - ``basic``
     - basic
     - 6
     - DZP / [3,3,3] / 150 Ry
     - Initial relaxations, structure checks
   * - ``intermediate``
     - basic + intermediate
     - 13
     - DZP / [6,6,6] / 200 Ry
     - Standard production calculations
   * - ``advanced``
     - basic + intermediate + advanced
     - 22
     - TZP / [6,6,6] / 300 Ry
     - High-quality results
   * - ``expert``
     - all 4 tiers
     - 28
     - TZP / [8,8,8] / 400 Ry
     - Publication quality
   * - ``ultra``
     - all 4 tiers (same as expert)
     - 28
     - TZDP / [10,10,10] / 800 Ry
     - Benchmark, convergence tests
   * - ``all``
     - all 4 tiers (same as expert)
     - 28
     - (no defaults)
     - Load all modules, use maker defaults

----

Common Questions
================

Q: What's the difference between "dirty" and "basic"?
------------------------------------------------------

**Module Activation**: Both use **basic modules only** (6 modules)

**Parameters**: Different defaults from TIER_DEFAULTS

- ``tier="dirty"``: SZ basis, [1,1,1] k-points, 50 Ry cutoff → Very fast, low quality
- ``tier="basic"``: DZP basis, [3,3,3] k-points, 150 Ry cutoff → Fast, reasonable quality

**Example**:

.. code-block:: python

   # Both load the same 6 modules, but different parameter defaults
   maker1 = RelaxMaker.fixed_cell_relaxation(tier="dirty")   # SZ/[1,1,1]/50 Ry
   maker2 = RelaxMaker.fixed_cell_relaxation(tier="basic")   # DZP/[3,3,3]/150 Ry

Q: What's the difference between "expert" and "ultra"?
-------------------------------------------------------

**Module Activation**: Both use **all modules** (28 modules)

**Parameters**: Different defaults from TIER_DEFAULTS

- ``tier="expert"``: TZP basis, [8,8,8] k-points, 400 Ry cutoff → Publication quality
- ``tier="ultra"``: TZDP basis, [10,10,10] k-points, 800 Ry cutoff → Benchmark quality

**Example**:

.. code-block:: python

   # Both load all 28 modules, but different parameter defaults
   maker1 = RelaxMaker.fixed_cell_relaxation(tier="expert")  # TZP/[8,8,8]/400 Ry
   maker2 = RelaxMaker.fixed_cell_relaxation(tier="ultra")   # TZDP/[10,10,10]/800 Ry

Q: Can I use "dirty" tier with RelaxMaker?
-------------------------------------------

**Yes!** (as of v1.0.0)

``RelaxSetGenerator`` automatically enables the ``md_relaxation`` module even if it's not in the tier's module list.

.. code-block:: python

   # This works! (md_relaxation auto-enabled)
   maker = RelaxMaker.fixed_cell_relaxation(tier="dirty")

Before the fix, this would fail with:
``'RelaxSetGenerator' object has no attribute '_md_relaxation_module'``

Q: Why does tier="basic" fail with some Makers?
------------------------------------------------

Some Makers require modules that are only in higher tiers.

**Examples**:

- ``RelaxMaker`` → needs ``md_relaxation`` (intermediate tier) - but auto-enabled now!
- ``PhononMaker`` → needs ``phonons`` module (advanced tier)
- ``OpticalMaker`` → needs ``optical`` module (advanced tier)

**Solution 1**: Use appropriate tier

.. code-block:: python

   maker = PhononMaker.from_phonopy_yaml(tier="advanced")  # ✅ phonons module included

**Solution 2**: Manually enable modules

.. code-block:: python

   maker = PhononMaker.from_phonopy_yaml(
       tier="basic",
       enabled_modules=["phonons"],  # Manually add missing module
   )

Q: How do I see which modules are loaded?
------------------------------------------

Check the parameter evolution log file:

.. code-block:: bash

   cat parameter_evolution.log

Or use the registry directly:

.. code-block:: python

   from atomate2.siesta.dataclass.registry import get_modules_by_tier

   modules = get_modules_by_tier("intermediate")
   print(f"Loaded {len(modules)} modules:")
   for name in modules:
       print(f"  - {name}")

Output::

   Loaded 13 modules:
     - pseudopotentials
     - basis_sets
     - xc_functional
     - kpoints
     - mesh_cutoff
     - scf
     - md_relaxation
     - spin
     - output
     - electron_density
     - molecular_dynamics
     - constraints
     - vdw

----

Code Locations Reference
=========================

**Module Registry System**:

- ``src/atomate2/siesta/dataclass/registry.py:77`` - ``MODULE_REGISTRY`` definition
- ``src/atomate2/siesta/dataclass/registry.py:180-192`` - ``tier_hierarchy`` definition
- ``src/atomate2/siesta/dataclass/registry.py:265-620`` - All 28 ``register_module()`` calls

**Tier Defaults**:

- ``src/atomate2/siesta/sets/tiers/defaults.py:16-48`` - ``TIER_DEFAULTS`` definition

**Tier Application**:

- ``src/atomate2/siesta/sets/base.py:519-549`` - ``__post_init__()`` applies tier defaults
- ``src/atomate2/siesta/sets/base.py:535`` - Loads ``TIER_DEFAULTS``
- ``src/atomate2/siesta/sets/base.py:539-540`` - Merges tier defaults with user params

**Module Activation**:

- ``src/atomate2/siesta/dataclass/registry.py:150-206`` - ``get_modules_by_tier()`` function

**RelaxMaker Auto-Enable**:

- ``src/atomate2/siesta/sets/core.py:46-60`` - ``RelaxSetGenerator.__post_init__()`` auto-enables ``md_relaxation``

----

See Also
========

- :doc:`tier-system` - Main tier system documentation
- :doc:`tier-system-clarification` - Tier levels vs tier defaults distinction
- :doc:`tier-defaults-explained` - Detailed explanation of TIER_DEFAULTS
- :doc:`cli-tools` - Using ``atomate2siesta-presets`` CLI
