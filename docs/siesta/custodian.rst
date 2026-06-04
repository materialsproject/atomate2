===========================
Custodian Error Handling
===========================

Automatic error detection and recovery using the MaterialsProject/custodian library.

.. versionadded:: 2024
   Refactored to use custodian library as foundation

----

Overview
========

The custodian error handling system provides automatic detection and recovery from common
SIESTA calculation failures, built on the battle-tested **MaterialsProject/custodian**
library framework.

Key Features
------------

✅ **10+ error types** automatically detected

✅ **Progressive correction strategies** with increasing aggressiveness

✅ **Automatic JSON logging** (``custodian.json``) with full history

✅ **MSONable serialization** for jobflow compatibility

✅ **Validation framework** for output quality checking

✅ **Safety limits** to prevent infinite retry loops

----

Quick Start
===========

Basic Usage
-----------

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker

   # Enable custodian (default handlers)
   maker = RelaxMaker.fixed_cell_relaxation(
       use_custodian=True,
       custodian_max_errors=5,
   )

   job = maker.make(structure)
   results = run_locally(job, create_folders=True)

   # Check custodian.json for correction history

Custom Handlers
---------------

.. code-block:: python

   from atomate2.siesta.custodian import (
       SCFConvergenceHandler,
       MemoryHandler,
       TimeHandler,
   )

   # More aggressive SCF recovery
   custom_handlers = [
       SCFConvergenceHandler(max_attempts=10),
       MemoryHandler(max_attempts=3),
       TimeHandler(max_attempts=2),
   ]

   maker = RelaxMaker.fixed_cell_relaxation(
       use_custodian=True,
       custodian_handlers=custom_handlers,
       custodian_max_errors=15,
   )

----

Error Detection
===============

Supported Error Types (10 Total)
---------------------------------

1. **SCF_NOT_CONV**

   * Pattern: ``"SCF did not converge"``
   * Cause: Electronic convergence failure
   * Handler: ``SCFConvergenceHandler`` (5-level strategy)

2. **MEMORY**

   * Pattern: ``"Out of memory"``
   * Cause: Insufficient memory allocation
   * Handler: ``MemoryHandler``

3. **TIME_LIMIT**

   * Pattern: ``"TIME LIMIT"``
   * Cause: Job exceeded walltime
   * Handler: ``TimeHandler`` (restart from DM)

4. **NUMERICAL**

   * Pattern: ``"NaN"`` or ``"Inf"``
   * Cause: Numerical instability
   * Handler: ``NumericalHandler``

5. **SINGULAR_OVERLAP**

   * Pattern: ``"Singular overlap matrix"``
   * Cause: Overlap matrix not positive definite
   * Handler: ``SCFConvergenceHandler``

6. **NEGATIVE_EIGENVALUES**

   * Pattern: ``"Negative eigenvalues in overlap"``
   * Cause: Overlap matrix stability issue
   * Handler: ``SCFConvergenceHandler``

7. **GEOMETRY_OPTIMIZATION**

   * Pattern: ``"Geometry relaxation failed"``
   * Cause: Structure optimization did not converge
   * Handler: Custom relaxation handler

8. **BASIS_GENERATION**

   * Pattern: ``"Error in basis set generation"``
   * Cause: PAO basis construction failed
   * Handler: ``NumericalHandler``

9. **GRID_INTEGRATION**

   * Pattern: ``"Grid integration error"``
   * Cause: Real-space grid issues
   * Handler: ``NumericalHandler``

10. **FILE_IO**

    * Pattern: ``"Error opening/reading file"``
    * Cause: File system issues
    * Handler: Generic retry

----

SCF Convergence Handler
========================

The most important handler, with a **5-level progressive correction strategy**:

Level 1: Gentle Reduction
--------------------------

.. code-block:: text

   Changes:
   - SCF.Mixer.Weight = 0.05
   - SCF.Mix.First = True

   Strategy: Reduce mixing weight, start fresh mixing history

Level 2: More Conservative
---------------------------

.. code-block:: text

   Changes:
   - SCF.Mixer.Weight = 0.01
   - SCF.Mixer.History = 5

   Strategy: Further reduce mixing, shorter history

Level 3: Very Conservative + Perturbation
------------------------------------------

.. code-block:: text

   Changes:
   - SCF.Mixer.Weight = 0.005
   - SCF.Mixer.History = 8
   - SCF.Mixer.Kick = 40

   Strategy: Very small mixing + perturbation to escape local minimum

Level 4: Change Algorithm (Pulay)
----------------------------------

.. code-block:: text

   Changes:
   - SCF.Mixer.Method = Pulay
   - SCF.Mixer.History = 10

   Strategy: Try Pulay mixer instead of default

Level 5: Last Resort (Broyden)
-------------------------------

.. code-block:: text

   Changes:
   - SCF.Mixer.Method = Broyden
   - SCF.Mixer.Weight = 0.001

   Strategy: Try Broyden mixer with very small mixing

Example
-------

.. code-block:: python

   from atomate2.siesta.custodian import SCFConvergenceHandler

   # Customize max attempts per level
   handler = SCFConvergenceHandler(
       max_attempts=10,  # Try up to 10 times
   )

----

Custodian Output
================

custodian.json Structure
-------------------------

After each run, ``custodian.json`` contains complete audit trail:

.. code-block:: json

   {
       "jobs": [
           {
               "job": "SiestaJob",
               "cmd": "siesta < siesta.fdf > siesta.out",
               "final": true
           }
       ],
       "corrections": [
           {
               "handler": "SCFConvergenceHandler",
               "level": 1,
               "errors": ["SCF did not converge in 100 SCF steps"],
               "actions": [
                   "Updated SCF.Mixer.Weight to 0.05",
                   "Set SCF.Mix.First to True"
               ],
               "timestamp": "2024-10-09T10:15:30"
           },
           {
               "handler": "SCFConvergenceHandler",
               "level": 2,
               "errors": ["SCF did not converge in 100 SCF steps"],
               "actions": [
                   "Updated SCF.Mixer.Weight to 0.01",
                   "Updated SCF.Mixer.History to 5"
               ],
               "timestamp": "2024-10-09T10:45:20"
           }
       ],
       "run_statistics": {
           "total_time": 3600.5,
           "wall_time": 3650.2,
           "errors": 2,
           "corrections": 2,
           "final_state": "completed"
       }
   }

Analyzing Results
-----------------

.. code-block:: python

   import json

   with open("custodian.json") as f:
       custodian_data = json.load(f)

   # Check if corrections were applied
   if custodian_data["corrections"]:
       print(f"Applied {len(custodian_data['corrections'])} corrections")
       for corr in custodian_data["corrections"]:
           print(f"  - {corr['handler']} level {corr['level']}")
   else:
       print("No corrections needed - clean run!")

----

Architecture
============

Built on Custodian Library
---------------------------

atomate2siesta uses ``custodian>=2024.4.18`` as foundation:

**Key Classes**:

* ``Custodian`` - Main orchestrator (from custodian library)
* ``Job`` - Job execution wrapper (inherit from ``custodian.custodian.Job``)
* ``ErrorHandler`` - Error detection and correction (inherit from ``custodian.custodian.ErrorHandler``)
* ``Validator`` - Output validation (inherit from ``custodian.custodian.Validator``)

**Benefits**:

* ~660 lines of retry logic removed (handled by library)
* Battle-tested framework from MaterialsProject
* Automatic JSON logging and tracking
* MSONable serialization for jobflow

Module Structure
----------------

.. code-block:: text

   custodian/
   ├── jobs.py               # SiestaJob (inherits custodian.custodian.Job)
   ├── fdf_utils.py          # FDF file reading/writing
   ├── handlers/             # Error handlers
   │   ├── base.py          # Re-exports from custodian library
   │   ├── scf.py           # SCFConvergenceHandler (5 levels)
   │   ├── memory.py        # MemoryHandler (4 levels)
   │   ├── time.py          # TimeHandler (restart from DM)
   │   └── numerical.py     # NumericalHandler (tolerances)
   ├── errors/               # Error detection
   │   ├── patterns.py      # SIESTA_ERROR_PATTERNS (10 patterns)
   │   └── detection.py     # detect_error(), check_for_errors()
   └── validators/           # Output validation
       ├── siesta.py        # SiestaOutputValidator
       ├── relaxation.py    # RelaxationValidator
       └── bandstructure.py # BandStructureValidator

----

Advanced Usage
==============

Custom Error Patterns
---------------------

Add custom error detection:

.. code-block:: python

   from atomate2.siesta.custodian.errors import ErrorPattern, ErrorType

   custom_pattern = ErrorPattern(
       pattern=r"Custom error message",
       error_type=ErrorType.SCF_NOT_CONV,
       severity="high",
   )

Writing Custom Handlers
-----------------------

.. code-block:: python

   from custodian.custodian import ErrorHandler

   class MyCustomHandler(ErrorHandler):
       def __init__(self, max_attempts=3):
           self.max_attempts = max_attempts
           self.n_applied_corrections = 0

       def check(self):
           """Check if error occurred."""
           # Return error dict if found, None otherwise
           pass

       def correct(self):
           """Apply correction."""
           # Modify input files
           # Return dict with correction details
           pass

Custom Validators
-----------------

.. code-block:: python

   from custodian.custodian import Validator

   class MyValidator(Validator):
       def check(self):
           """Validate output quality."""
           # Return True if valid, False otherwise
           pass

----

Production Best Practices
==========================

When to Enable Custodian
-------------------------

✅ **Always enable for**:

* Production calculations
* High-throughput workflows
* HPC batch jobs
* Challenging systems (metals, surfaces)

❌ **Disable for**:

* Testing/debugging (want to see raw failures)
* Very simple systems (overhead not needed)
* Custom error handling workflows

Configuration Guidelines
------------------------

**Standard Production**:

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(
       use_custodian=True,
       custodian_max_errors=5,
   )

**High-Throughput (Conservative)**:

.. code-block:: python

   custom_handlers = [
       SCFConvergenceHandler(max_attempts=15),
       MemoryHandler(max_attempts=5),
       TimeHandler(max_attempts=3),
   ]

   maker = RelaxMaker.fixed_cell_relaxation(
       use_custodian=True,
       custodian_handlers=custom_handlers,
       custodian_max_errors=25,
   )

**Testing (Aggressive)**:

.. code-block:: python

   # Very aggressive recovery for difficult systems
   maker = RelaxMaker.fixed_cell_relaxation(
       use_custodian=True,
       custodian_max_errors=50,
   )

Monitoring
----------

.. code-block:: bash

   # Check for custodian runs
   find . -name "custodian.json"

   # Count corrections applied
   jq '.corrections | length' custodian.json

   # List correction types
   jq '.corrections[] | .handler' custodian.json

----

Validation Testing
==================

The custodian system has been validated with:

**Test Results**:

✅ 10 SCF failures detected and corrected

✅ 5 correction levels applied progressively

✅ FDF file modifications confirmed

✅ Safety limits respected (max_attempts enforced)

✅ JSON logging complete and accurate

**Example Test Case**:

Metallic surface with intentional SCF convergence difficulty:

* Initial run: SCF fails (no occupation function)
* Level 1 correction: Reduce mixer weight
* Level 2 correction: Further reduce + history adjustment
* Result: Converges after 2 corrections

----

Troubleshooting
===============

Common Issues
-------------

**Problem**: Custodian keeps retrying but never succeeds

**Solution**:
   * Check ``custodian.json`` to see what corrections were tried
   * May need to adjust initial parameters (not just corrections)
   * Consider if system is fundamentally problematic

**Problem**: Corrections not being applied

**Solution**:
   * Verify ``use_custodian=True`` is set
   * Check that handlers are correctly registered
   * Look for handler initialization errors in logs

**Problem**: custodian.json not created

**Solution**:
   * Custodian may not have been invoked
   * Check that calculation actually ran
   * Verify file permissions

----

See Also
========

* :doc:`features` - Overview of custodian features
* :doc:`tutorials/infrastructure` - Tutorial 15 (custodian tutorial)
* `REFACTORING_SUMMARY.md` - Complete custodian library integration guide (438 lines)
* MaterialsProject custodian: https://github.com/materialsproject/custodian

----

.. note::

   The custodian system was refactored in 2024 to use the
   MaterialsProject/custodian library as foundation, removing ~660 lines of
   custom retry logic while adding battle-tested reliability.
