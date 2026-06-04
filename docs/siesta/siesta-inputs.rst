SIESTA Data Classes CLI Documentation
=======================================

``siesta-inputs`` is a command-line interface (CLI) for managing and displaying informatiton about Siesta dataclasses. The following commands are available:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Overview
--------

This Python script provides a command-line interface (CLI) built with the ``click`` library to inspect and query data classes from the ``atomate2.siesta.sets.dataclasses`` module. These data classes represent various parameters and settings for SIESTA simulations, integrated within the Atomate2 workflow framework for Siesta extension. The tool leverages ``dataclasses`` for structured parameter handling, drawing from libraries like ``sisl``, ``ase``, and ``pymatgen`` for materials science computations.

The CLI allows users to:
- List available data classes.
- Display detailed information about a specific data class, including attributes, types, defaults, descriptions, SIESTA keywords, and units.
- Search for attributes across all data classes by keyword, with options for exact matching.

This tool is useful for developers and users working on Siesta workflows in Atomate2, enabling quick reference to parameter metadata without diving into source code.

Requirements
------------

- Python 3.12 or compatible.
- Installed packages: ``click``, ``atomate2`` (with Siesta extension), ``sisl``, ``ase``, ``pymatgen``.
- No additional installations needed beyond the Atomate2 Siesta setup.

Usage
-----

Run the script as a standalone CLI:

.. code-block:: bash

   python siesta_dataclasses_cli.py [COMMAND] [OPTIONS]

Commands
--------

list
^^^^

List all available SIESTA data classes.

**Usage:**

.. code-block:: bash

   python siesta_dataclasses_cli.py list

**Output Example:**

Available SIESTA Data Classes:
- BasisSetsAndProjectors
- ChargeDipoleElectricField
- ChemicalAnalysis
- ...

show
^^^^

Show detailed information about a specific data class.

**Arguments:**

- ``CLASS_NAME``: The name of the data class (required). Use the ``list`` command to view available classes.

**Options:**

- ``-c, --complete``: Show complete detailed info, including descriptions and SIESTA keywords.
- ``-s, --siesta``: Show SIESTA keywords info only.
- ``-u, --unit``: Show units of keywords.
- If no options are provided, displays basic attribute info (name, type, default).

**Usage Examples:**

Basic info:

.. code-block:: bash

   python siesta_dataclasses_cli.py show GeneralSystemDescriptors

With complete details:

.. code-block:: bash

   python siesta_dataclasses_cli.py show GeneralSystemDescriptors --complete

**Output Example (Basic):**

Data Class: GeneralSystemDescriptors
--------------------------------------------------
Description:
No docstring available.

Attributes:
- system_name: str (default: 'None')
- number_of_species: int (default: 'None')
- ...

**Output Example (Complete):**

Data Class: GeneralSystemDescriptors
--------------------------------------------------
Description:
No docstring available.

Attributes:

- system_name: str (default: 'None')
  SIESTA keyword: SystemName
  Description: No description available.

- number_of_species: int (default: 'None')
  SIESTA keyword: NumberOfSpecies
  Description: No description available.

...

search
^^^^^^

Search for data class attributes by keyword in name, type, default, description, or SIESTA keyword. The search is case-insensitive.

**Arguments:**

- ``KEYWORD``: The search term (required).

**Options:**

- ``-r, --restrict``: Restrict search to exact word matches (using regex word boundaries).

**Usage Examples:**

Basic search:

.. code-block:: bash

   python siesta_dataclasses_cli.py search spin

Exact match:

.. code-block:: bash

   python siesta_dataclasses_cli.py search spin --restrict

**Output Example:**

Found 3 attribute(s) matching 'spin':

- SpinSettings:
- spin_polarized: bool (default: False)
  SIESTA keyword: SpinPolarized
  Description: No description available.

- SpinSettings:
- non_collinear_spin: bool (default: False)
  SIESTA keyword: NonCollinearSpin
  Description: No description available.

...

Data Classes Overview
---------------------

The following data classes are defined in ``atomate2.siesta.sets.dataclasses`` and accessible via this CLI:

- **GeneralSystemDescriptors**: General descriptors for the system.
- **Pseudopotentials**: Settings for pseudopotentials.
- **BasisSetsAndProjectors**: Basis sets and projector configurations.
- **StructuralInformationVersion1**: Structural info (version 1).
- **StructuralInformationVersion2**: Structural info (version 2).
- **KPointSampling**: K-point sampling parameters.
- **ExchangeCorrelationFunctionals**: XC functionals.
- **SpinSettings**: Spin-related settings.
- **SCFLoopParameters**: Self-consistent field loop params.
- **RealSpaceGridParameters**: Real-space grid settings.
- **HamiltonianAndOverlapParameters**: Hamiltonian and overlap params.
- **ElectronicStructureCalculationOptions**: Electronic structure options.
- **SolversAndPerformanceOptions**: Solvers and performance tweaks.
- **DensityOfStatesAndBandStructure**: DOS and band structure.
- **ChemicalAnalysis**: Chemical analysis tools.
- **OpticalProperties**: Optical properties calculations.
- **Wannier90**: Wannier90 interface.
- **ChargeDipoleElectricField**: Charge, dipole, and electric field.
- **Grids**: Grid-related settings.
- **AuxiliaryForceField**: Auxiliary force fields.
- **ParallelOptions**: Parallelization options.
- **EfficiencyOptions**: Efficiency optimizations.
- **Denchar**: Density characterization.
- **NetcdfOptions**: NetCDF output options.
- **MolecularDynamicsAndRelaxation**: MD and relaxation.
- **ExternalControlAndScripting**: External control and scripting.
- **GeneralConstraints**: General constraints.
- **PhononCalculations**: Phonon calculations.
- **DFTU**: DFT+U settings.
- **RTTDDFT**: Real-time TDDFT.

Each data class uses Python dataclasses with metadata for SIESTA keywords, units, and descriptions.

Implementation Notes
--------------------

- **Formatting Defaults**: Default values are formatted for display, handling ``None``, callables (e.g., ``list`` or ``dict`` factories), and empty collections.
- **Metadata Usage**: Fields in dataclasses include metadata like ``'description'``, ``'SIESTA keyword'``, and ``'Unit'`` for enhanced output.
- **Search Logic**: Uses regex for restricted searches; otherwise, substring matching. Matches are collected across all classes and displayed with context.
- **Color Output**: Utilizes ``click.style`` for colored terminal output (e.g., cyan for headers, green for field names).

For contributing or extending, refer to the Atomate2 Siesta extension source. This CLI can be integrated into larger workflows using ``sisl``, ``ase``, and ``pymatgen`` for Siesta input generation.
