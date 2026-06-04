=======================
Output Schemas & Models
=======================

Understanding atomate2siesta's data models and output schemas.

----

Overview
========

atomate2siesta uses a hierarchical schema system to organize calculation data. Understanding
these schemas is essential for:

* Accessing calculation results
* Building custom workflows
* Storing data in databases
* Analyzing convergence studies

The schema hierarchy flows from low-level calculation details to high-level task summaries,
providing both detailed information and convenient summary data.

----

Schema Hierarchy
================

The data models are organized in 3 levels:

1. **CalculationOutput** - Raw calculation results (lowest level)
2. **Calculation** - Calculation document with inputs and outputs
3. **OutputDoc** - Summary of calculation results (highest level)
4. **SiestaTaskDoc** - Complete task document (workflow level)

.. code-block:: text

    SiestaTaskDoc (Task level)
    ├── output: OutputDoc (Summary)
    │   ├── energy: float (Total energy)
    │   ├── efermi: float (Fermi energy) ✨ NEW
    │   ├── bandgap: float
    │   ├── forces: list[Vector3D]
    │   └── structure: Structure
    └── calcs_reversed: list[Calculation] (Detailed)
        └── output: CalculationOutput (Raw)
            ├── total_energy: float
            ├── efermi: float
            ├── forces: list[Vector3D]
            └── stress: Matrix3D

----

Core Schema Classes
===================

CalculationOutput
-----------------

**Location**: ``src/atomate2/siesta/schemas/calculation.py``

**Purpose**: Stores raw SIESTA calculation outputs directly parsed from output files.

**Key Fields**:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``total_energy``
     - ``float``
     - Final total DFT energy in eV
   * - ``efermi``
     - ``float``
     - Fermi energy in eV (parsed from ``siesta.out``)
   * - ``structure``
     - ``Structure``
     - Final atomic structure from ``siesta.XV``
   * - ``forces``
     - ``list[Vector3D]``
     - Forces on each atom in eV/Å
   * - ``stress``
     - ``Matrix3D``
     - Stress tensor on unit cell (3×3 matrix)
   * - ``bandgap``
     - ``float``
     - Band gap in eV (from ``.EIG`` file if available)
   * - ``cbm``
     - ``float``
     - Conduction band minimum in eV
   * - ``vbm``
     - ``float``
     - Valence band maximum in eV

**Parsing Source**:

* Total energy: ``siesta.out`` via ``sisl.stdoutSileSiesta.read_energy()['total']``
* Fermi energy: ``siesta.out`` via ``sisl.stdoutSileSiesta.read_energy()['fermi']``
* Structure: ``siesta.XV`` via ``sisl.xvSileSiesta.read_geometry()``
* Band gap: ``siesta.EIG`` via custom EIG parser
* Forces: ``siesta.out`` via ``sisl.stdoutSileSiesta.read_force()``
* Stress: ``siesta.out`` via ``sisl.stdoutSileSiesta.read_stress()``

**Example Access**:

.. code-block:: python

    from jobflow import run_locally
    from atomate2.siesta.jobs.core import StaticMaker

    maker = StaticMaker.scf()
    job = maker.make(structure)
    results = run_locally(job)

    # Access via calcs_reversed
    task_doc = results[job.uuid][1].output
    calc_output = task_doc.calcs_reversed[-1].output  # CalculationOutput

    # Raw calculation data
    energy = calc_output.total_energy        # -229.887354 eV
    efermi = calc_output.efermi              # -3.748021 eV
    forces = calc_output.forces              # [(0.0, 0.0, 0.0), ...]
    bandgap = calc_output.bandgap            # 0.612 eV (for Si)

----

Calculation
-----------

**Location**: ``src/atomate2/siesta/schemas/calculation.py``

**Purpose**: Complete calculation document containing inputs, outputs, and metadata.

**Key Fields**:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Type
     - Description
   * - ``dir_name``
     - ``str``
     - Directory containing calculation files
   * - ``siesta_version``
     - ``str``
     - SIESTA version used (e.g., "5.0.0")
   * - ``input``
     - ``CalculationInput``
     - Input parameters and settings
   * - ``output``
     - ``CalculationOutput``
     - Calculation results (see above)
   * - ``completed_at``
     - ``str``
     - Timestamp of calculation completion
   * - ``task_name``
     - ``str``
     - Name of the calculation task

**Example Access**:

.. code-block:: python

    calc = task_doc.calcs_reversed[-1]  # Last (most recent) calculation

    # Metadata
    print(f"Directory: {calc.dir_name}")
    print(f"SIESTA version: {calc.siesta_version}")
    print(f"Completed: {calc.completed_at}")

    # Results
    energy = calc.output.total_energy
    efermi = calc.output.efermi

----

OutputDoc
---------

**Location**: ``src/atomate2/siesta/schemas/task.py``

**Purpose**: High-level summary of calculation results for convenient access.

**Key Fields**:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``structure``
     - ``Structure``
     - Final atomic structure
   * - ``trajectory``
     - ``Sequence[Structure]``
     - Trajectory of structures (for relaxations)
   * - ``energy``
     - ``float``
     - Final total DFT energy in eV
   * - ``efermi``
     - ``float``
     - Fermi energy in eV
   * - ``bandgap``
     - ``float``
     - DFT band gap in eV
   * - ``cbm``
     - ``float``
     - Conduction band minimum in eV
   * - ``vbm``
     - ``float``
     - Valence band maximum in eV
   * - ``forces``
     - ``list[Vector3D]``
     - Atomic forces in eV/Å
   * - ``stress``
     - ``Matrix3D``
     - Cell stress tensor

**Properties**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Property
     - Description
   * - ``energy_per_atom``
     - Total energy divided by number of atoms (eV/atom)

**Example Access**:

.. code-block:: python

    # Convenient summary access
    output = task_doc.output  # OutputDoc

    # Commonly used fields
    energy = output.energy                # -229.887354 eV
    efermi = output.efermi                # -3.748021 eV ✨ NEW
    bandgap = output.bandgap              # 0.612 eV
    structure = output.structure          # Final Structure
    energy_per_atom = output.energy_per_atom  # -114.943677 eV/atom

**Design Pattern**:

``OutputDoc`` provides a **simplified interface** to the most commonly accessed data.
For detailed or intermediate calculation data, use ``calcs_reversed``:

.. code-block:: python

    # Summary access (recommended for most use cases)
    energy = task_doc.output.energy
    efermi = task_doc.output.efermi

    # Detailed access (for advanced use cases)
    calc_output = task_doc.calcs_reversed[-1].output  # CalculationOutput
    detailed_efermi = calc_output.efermi  # Same value

----

SiestaTaskDoc
-------------

**Location**: ``src/atomate2/siesta/schemas/task.py``

**Purpose**: Complete task document containing all calculation information and metadata.

**Key Fields**:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Type
     - Description
   * - ``dir_name``
     - ``str``
     - Task directory path
   * - ``output``
     - ``OutputDoc``
     - Summary of calculation results (see above)
   * - ``structure``
     - ``Structure``
     - Input structure
   * - ``state``
     - ``TaskState``
     - Task execution state (SUCCESS, FAILED, etc.)
   * - ``calcs_reversed``
     - ``list[Calculation]``
     - List of calculations (most recent first)
   * - ``nsites``
     - ``int``
     - Number of atoms in structure
   * - ``elements``
     - ``list[str]``
     - List of elements present
   * - ``composition``
     - ``Composition``
     - Chemical composition
   * - ``formula_pretty``
     - ``str``
     - Pretty-printed formula (e.g., "Si₂")
   * - ``custodian``
     - ``list[dict]``
     - Custodian error handling logs (if used)

**Example Access**:

.. code-block:: python

    from jobflow import run_locally
    from atomate2.siesta.jobs.core import RelaxMaker

    maker = RelaxMaker.fixed_cell_relaxation()
    job = maker.make(structure)
    results = run_locally(job)

    # Get task document
    task_doc = results[job.uuid][1].output  # SiestaTaskDoc

    # High-level summary
    print(f"Formula: {task_doc.formula_pretty}")
    print(f"Number of atoms: {task_doc.nsites}")
    print(f"Elements: {task_doc.elements}")
    print(f"Task state: {task_doc.state}")

    # Results (via OutputDoc)
    print(f"Final energy: {task_doc.output.energy} eV")
    print(f"Fermi energy: {task_doc.output.efermi} eV")
    print(f"Band gap: {task_doc.output.bandgap} eV")

    # Detailed calculation info
    calc = task_doc.calcs_reversed[-1]  # Most recent calculation
    print(f"Calculation directory: {calc.dir_name}")
    print(f"SIESTA version: {calc.siesta_version}")

----

Fermi Energy in Schemas
========================

The Fermi energy (``efermi``) is now available at multiple levels in the schema hierarchy:

Access Patterns
---------------

**1. Summary Access (Recommended)**:

.. code-block:: python

    # Most convenient - available directly in OutputDoc
    task_doc = results[job.uuid][1].output
    efermi = task_doc.output.efermi  # -3.748021 eV ✨ NEW

**2. Detailed Access**:

.. code-block:: python

    # From CalculationOutput via calcs_reversed
    calc_output = task_doc.calcs_reversed[-1].output
    efermi = calc_output.efermi  # Same value

**3. Convergence Workflows**:

.. code-block:: python

    from atomate2.siesta.flows.convergence import KpointsConvergenceFlowMaker

    flow = KpointsConvergenceFlowMaker(kpoints_list=[[2,2,2], [4,4,4]])
    workflow = flow.make(structure)
    results = run_locally(workflow)

    # Fermi energy automatically collected for convergence analysis
    # Generated files include:
    # - convergence_kpoints.png (3-panel plot with Fermi energy)
    # - convergence_kpoints.txt (table with Fermi E column)

Parsing Details
---------------

**Source**: SIESTA outputs Fermi energy in ``siesta.out``:

.. code-block:: text

    siesta: Fermi energy =      -3.748021 eV

**Parser**: Uses sisl library:

.. code-block:: python

    import sisl
    siesta_output = sisl.get_sile("siesta.out")
    electronic_output = siesta_output.read_energy()
    efermi = electronic_output["fermi"]  # -3.748021

**Availability**:

* ✅ SIESTA always outputs Fermi energy (metals and insulators)
* ✅ Parsed for all calculation types (static, relaxation, bands, etc.)
* ✅ Available in all schema levels (CalculationOutput and OutputDoc)

Units
-----

All Fermi energies are in **electron volts (eV)**.

Physical Interpretation
-----------------------

* **Metals**: Fermi level where occupation function = 0.5
* **Insulators/Semiconductors**: Chemical potential (typically near VBM or between VBM/CBM)
* **Convergence**: Fermi energy should converge alongside total energy

----

Common Access Patterns
=======================

Single Calculation
------------------

.. code-block:: python

    from jobflow import run_locally
    from atomate2.siesta.jobs.core import StaticMaker

    maker = StaticMaker.scf()
    job = maker.make(structure)
    results = run_locally(job)

    # Get task document
    task_doc = results[job.uuid][1].output

    # Access results
    energy = task_doc.output.energy          # Total energy
    efermi = task_doc.output.efermi          # Fermi energy ✨
    bandgap = task_doc.output.bandgap        # Band gap
    structure = task_doc.output.structure    # Final structure
    forces = task_doc.output.forces          # Atomic forces

Relaxation Workflow
-------------------

.. code-block:: python

    from atomate2.siesta.jobs.core import RelaxMaker

    maker = RelaxMaker.fixed_cell_relaxation()
    job = maker.make(structure)
    results = run_locally(job)

    task_doc = results[job.uuid][1].output

    # Initial vs final structure
    initial_structure = task_doc.structure           # Input structure
    final_structure = task_doc.output.structure      # Relaxed structure
    trajectory = task_doc.output.trajectory          # All intermediate steps

    # Energies
    final_energy = task_doc.output.energy
    final_efermi = task_doc.output.efermi

    # Access individual relaxation steps
    for i, calc in enumerate(reversed(task_doc.calcs_reversed)):
        print(f"Step {i}: E={calc.output.total_energy} eV, Ef={calc.output.efermi} eV")

Convergence Workflows
---------------------

.. code-block:: python

    from atomate2.siesta.flows.convergence import MeshCutoffConvergenceFlowMaker

    flow = MeshCutoffConvergenceFlowMaker(
        mesh_cutoffs=[200, 250, 300, 350, 400]
    )
    workflow = flow.make(structure)
    results = run_locally(workflow)

    # Collection job stores convergence data
    collect_job_uuid = workflow.jobs[-2].uuid
    convergence_data = results[collect_job_uuid][1].output

    # Access collected data
    parameters = convergence_data["parameters"]        # ["200Ry", "250Ry", ...]
    energies = convergence_data["energies"]            # [-229.8, -229.85, ...]
    fermi_energies = convergence_data["fermi_energies"]  # [-3.74, -3.75, ...] ✨

    # Plot job generates PNG and TXT files with Fermi energy included

Multi-Job Flows
---------------

.. code-block:: python

    from atomate2.siesta.flows.core import DoubleRelaxFlowMaker

    flow = DoubleRelaxFlowMaker()
    workflow = flow.make(structure)
    results = run_locally(workflow)

    # Access first relaxation
    relax1_uuid = workflow.jobs[0].uuid
    relax1_doc = results[relax1_uuid][1].output
    energy1 = relax1_doc.output.energy
    efermi1 = relax1_doc.output.efermi

    # Access second relaxation
    relax2_uuid = workflow.jobs[1].uuid
    relax2_doc = results[relax2_uuid][1].output
    energy2 = relax2_doc.output.energy
    efermi2 = relax2_doc.output.efermi

----

Schema Evolution
================

Version History
---------------

**v1.0.0**:

* Initial schema implementation
* ``efermi`` field available in both ``CalculationOutput`` and ``OutputDoc``
* Fermi energy accessible at summary level
* Convergence workflows automatically collect and plot Fermi energy
* Text file output includes Fermi energy column

Backward Compatibility
----------------------

The schema changes are **backward compatible**:

* Old code accessing ``calcs_reversed[-1].output.efermi`` still works
* New code can use the simpler ``output.efermi``
* All existing workflows continue to function

Migration Guide
---------------

**Old pattern** (still works):

.. code-block:: python

    calc_output = task_doc.calcs_reversed[-1].output
    efermi = calc_output.efermi

**New pattern** (recommended):

.. code-block:: python

    efermi = task_doc.output.efermi  # ✨ Simpler!

----

Advanced Usage
==============

Custom Data Extraction
----------------------

Extract specific data from multiple calculations:

.. code-block:: python

    def extract_convergence_data(results: dict, job_uuids: list) -> dict:
        """Extract energy and Fermi energy from multiple jobs."""
        data = {
            "energies": [],
            "fermi_energies": [],
            "bandgaps": [],
        }

        for uuid in job_uuids:
            task_doc = results[uuid][1].output
            data["energies"].append(task_doc.output.energy)
            data["fermi_energies"].append(task_doc.output.efermi)
            data["bandgaps"].append(task_doc.output.bandgap)

        return data

Database Storage
----------------

Store task documents in MongoDB:

.. code-block:: python

    from atomate2.siesta.schemas.task import SiestaTaskDoc
    from pymongo import MongoClient

    # Store in database
    client = MongoClient("mongodb://localhost:27017")
    db = client["siesta_calculations"]
    collection = db["tasks"]

    # Insert task document
    task_dict = task_doc.dict()
    collection.insert_one(task_dict)

    # Query by Fermi energy
    metallic_systems = collection.find({"output.efermi": {"$gt": 0}})

    # Query by band gap
    insulators = collection.find({"output.bandgap": {"$gt": 1.0}})

Schema Validation
-----------------

Pydantic models provide automatic validation:

.. code-block:: python

    from atomate2.siesta.schemas.task import OutputDoc

    # Valid data
    valid_output = OutputDoc(
        energy=-229.887,
        efermi=-3.748,
        bandgap=0.612,
        structure=structure,
    )

    # Pydantic validates types automatically
    try:
        invalid_output = OutputDoc(
            energy="not a number",  # ❌ Type error
            efermi=-3.748,
        )
    except ValueError as e:
        print(f"Validation error: {e}")

----

Reference
=========

Schema Module Locations
-----------------------

.. code-block:: text

    src/atomate2/siesta/schemas/
    ├── __init__.py           # Schema exports
    ├── calculation.py        # CalculationOutput, Calculation
    └── task.py               # OutputDoc, SiestaTaskDoc

Key Imports
-----------

.. code-block:: python

    from atomate2.siesta.schemas.calculation import (
        CalculationOutput,
        Calculation,
    )

    from atomate2.siesta.schemas.task import (
        OutputDoc,
        SiestaTaskDoc,
        InputDoc,
    )

External Dependencies
---------------------

* **Pydantic**: Data validation and settings management
* **pymatgen**: Structure objects and materials science data types
* **sisl**: SIESTA file parsing
* **emmet-core**: Base model schemas (from Materials Project)

----

See Also
========

* :doc:`usage` - Basic usage patterns
* :doc:`advanced-workflows` - Multi-step workflow examples
* :doc:`cli-database` - Database integration
* :doc:`troubleshooting` - Common issues and solutions
* :doc:`api/modules` - Complete API reference
