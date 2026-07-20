==============================
Pseudopotential Management CLI
==============================

``atomate2siesta-pseudos`` is a command-line interface for managing SIESTA pseudopotential
repositories. It supports downloading, installing, exploring, and generating basis sets
from PSML pseudopotential files.

CLI Overview
============

Usage
-----

.. code-block:: console

   atomate2siesta-pseudos [OPTIONS] COMMAND [ARGS]...

Options
-------

- ``--version``: Show the version and exit.
- ``--help``: Show this message and exit.

Commands
--------

- **available**: Show available pseudopotential repositories with download info.
- **basis**: Generate PAO.Basis block from PSML pseudopotential file.
- **element**: Find all pseudos in the installed tables for a given element.
- **install**: Install pseudopotential repositories by name(s).
- **list**: List pseudopotential repos with installation status.
- **plot**: Generate plots for a specified element in a pseudopotential repository.
- **show**: Show info on pseudopotential table(s) and valence shells.
- **uninstall**: Uninstall pseudopotential repositories by name(s).


Commands in Detail
==================

available
---------

Show all available pseudopotential repositories with download information.

.. code-block:: console

   atomate2siesta-pseudos available

Example output shows repository names, XC functionals, relativity type, and element coverage.


install
-------

Install pseudopotential repositories by name.

.. code-block:: console

   # Install a specific repository
   atomate2siesta-pseudos install ONCVPSP-PBE-SR-PDv0.4-Standard

   # Install all available repositories
   atomate2siesta-pseudos install --all

   # Install from local file only (no download)
   atomate2siesta-pseudos install ONCVPSP-PBE-SR-PDv0.4-Standard --local-only


uninstall
---------

Uninstall pseudopotential repositories.

.. code-block:: console

   # Uninstall a specific repository
   atomate2siesta-pseudos uninstall ONCVPSP-PBE-SR-PDv0.4-Standard

   # Uninstall with force (no confirmation)
   atomate2siesta-pseudos uninstall ONCVPSP-PBE-SR-PDv0.4-Standard --force

   # Uninstall all repositories
   atomate2siesta-pseudos uninstall --all


list
----

List all pseudopotential repositories with installation status.

.. code-block:: console

   atomate2siesta-pseudos list

Shows a table with repository name, XC functional, relativity type, version,
and installation status.


show
----

Show detailed information about a pseudopotential repository, including
valence shell configuration for specific elements.

.. code-block:: console

   # Show repository info
   atomate2siesta-pseudos show ONCVPSP-PBE-SR-PDv0.4-Standard

   # Show valence shells for a specific element
   atomate2siesta-pseudos show ONCVPSP-PBE-SR-PDv0.4-Standard Si


element
-------

Find all pseudopotentials available for a given element across installed repositories.

.. code-block:: console

   # Find by element symbol
   atomate2siesta-pseudos element Si

   # Find by atomic number
   atomate2siesta-pseudos element 14


basis
-----

Generate PAO.Basis block from PSML pseudopotential file. This is useful for
creating custom basis set specifications.

.. code-block:: console

   # Basic usage (DZP basis)
   atomate2siesta-pseudos basis ONCVPSP-PBE-SR-PDv0.4-Standard Si

   # Specify basis size
   atomate2siesta-pseudos basis ONCVPSP-PBE-SR-PDv0.4-Standard Fe --basis-size TZP

   # Save to file
   atomate2siesta-pseudos basis ONCVPSP-PBE-SR-PDv0.4-Standard O --output-file O.basis

   # Include excited shells
   atomate2siesta-pseudos basis ONCVPSP-PBE-SR-PDv0.4-Standard Ti --n-shells 2

Options:

- ``--basis-size``: PAO.BasisSize to generate (SZ, DZ, DZP, TZ, TZP, TZDP). Default: DZP
- ``--n-shells``: Number of n-shells per l (1=valence only, 2=valence+excited)
- ``--rc-method``: Method to determine cutoff radii:

  - ``psml``: From wavefunction decay in PSML file (default)
  - ``scaled``: Scaled by atomic radius
  - ``hydrogenic``: n²/Z_eff scaling
  - ``fixed``: Standard fixed values

- ``--rc-threshold``: Threshold for PSML wavefunction decay (default: 0.05 = 5%)
- ``--output-file``: Output file path (default: stdout)


plot
----

Generate visualization plots for pseudopotential data.

.. code-block:: console

   # Generate all plots
   atomate2siesta-pseudos plot ONCVPSP-PBE-SR-PDv0.4-Standard Si

   # Generate specific plot type
   atomate2siesta-pseudos plot ONCVPSP-PBE-SR-PDv0.4-Standard Si --plot-type wavefunctions

   # Specify output directory
   atomate2siesta-pseudos plot ONCVPSP-PBE-SR-PDv0.4-Standard Si --output-dir ./plots

   # Limit radial range
   atomate2siesta-pseudos plot ONCVPSP-PBE-SR-PDv0.4-Standard Si --r-plot 5.0

Plot types:

- ``wavefunctions``: Radial projector functions
- ``potentials``: Local and semilocal potentials (polar plot)
- ``3d-potential``: 3D surface plot of local potential
- ``occupation``: Valence electron occupation heatmap
- ``density``: Radial density (projector magnitude squared)
- ``all``: Generate all plot types (default)


Standalone Plotting Tool
========================

For direct plotting of PSML files without using the repository system:

.. code-block:: console

   atomate2siesta-plot-pseudo <file.psml> [OPTIONS]

Options:

- ``--plot-type``: Type of plot (wavefunctions, potentials, 3d-potential, occupation, density, all)
- ``--output-dir``: Output directory for plots
- ``--r-plot``: Maximum radial distance in bohr


Available Pseudopotential Repositories
======================================

The following repositories are available for installation:

1. **ONCVPSP-PBE-SR-PDv0.4-Standard**: PBE scalar-relativistic (most common)
2. **ONCVPSP-PBE-FR-PDv0.4-Standard**: PBE fully-relativistic
3. **ONCVPSP-PBEsol-SR-PDv0.4-Standard**: PBEsol scalar-relativistic
4. **ONCVPSP-PBEsol-FR-PDv0.4-Standard**: PBEsol fully-relativistic
5. **ONCVPSP-LDA-SR-PDv0.4-Standard**: LDA scalar-relativistic
6. **ONCVPSP-LDA-FR-PDv0.4-Standard**: LDA fully-relativistic

All repositories use the PSML format and are sourced from the PseudoDojo project.


Quick Start Examples
====================

.. code-block:: shell

   # 1. List available repositories
   atomate2siesta-pseudos available

   # 2. Install the standard PBE repository
   atomate2siesta-pseudos install ONCVPSP-PBE-SR-PDv0.4-Standard

   # 3. Check installation status
   atomate2siesta-pseudos list

   # 4. View valence configuration for Silicon
   atomate2siesta-pseudos show ONCVPSP-PBE-SR-PDv0.4-Standard Si

   # 5. Generate a TZP basis block for Silicon
   atomate2siesta-pseudos basis ONCVPSP-PBE-SR-PDv0.4-Standard Si --basis-size TZP

   # 6. Generate visualization plots
   atomate2siesta-pseudos plot ONCVPSP-PBE-SR-PDv0.4-Standard Si --output-dir ./Si_plots
