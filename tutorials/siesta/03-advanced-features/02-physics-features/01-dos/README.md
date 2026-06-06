# Density of States (DOS) Calculations

This tutorial demonstrates how to calculate density of states in atomate2siesta using **direct SIESTA FDF format**.

## Overview

The `ProjectedDensityOfStates` block in SIESTA generates:

- **siesta.DOS**: Total electronic density of states
- **siesta.PDOS**: Projected DOS (orbital-resolved on atoms)
- **siesta.PDOS.xml**: XML format for visualization

## Key Features

✅ **Direct SIESTA syntax**: Use native FDF block format
✅ **Automatic comment headers**: Dataclass sections labeled in output
✅ **Works with all makers**: StaticMaker, RelaxMaker, etc.
✅ **Dry-run support**: Preview input files without calculation

## Examples

### 01_basic_dos.py
Basic DOS calculation with StaticMaker using direct FDF format.

### 02_relax_with_dos.py
DOS calculation during geometry optimization with RelaxMaker.

## Quick Start

### Basic DOS Calculation

```python
from atomate2.siesta.jobs.core import StaticMaker
from pymatgen.core import Structure

# Create structure
structure = Structure.from_file("structure.cif")

# Direct SIESTA FDF format
user_params = {
    "xc.functional": "GGA",
    "xc.authors": "PBE",
    "Mesh.Cutoff": "200 Ry",
    "a2s_kpts": [4, 4, 4],
    "PAO.BasisSize": "DZP",

    # ProjectedDensityOfStates block (requires %block prefix)
    "%block ProjectedDensityOfStates": ["-10.0 10.0 0.1 200 eV"],
}

# Create maker with dry_run
maker = StaticMaker.scf(user_params=user_params, dry_run=True)
job = maker.make(structure)

# Run locally
from jobflow import run_locally
results = run_locally(job, create_folders=True)
```

This generates in `siesta.fdf`:

```
#--------------------------------------------#
#  DensityOfStatesAndBandStructure Settings
#--------------------------------------------#
%block ProjectedDensityOfStates
EF -10.000 10.000 0.100 200 eV
%endblock ProjectedDensityOfStates
```

## ProjectedDensityOfStates Block Format

The SIESTA format for DOS is:

```
%block ProjectedDensityOfStates
EF  Emin  Emax  dE  nE  units
%endblock ProjectedDensityOfStates
```

Where:
- **EF**: Energy relative to Fermi level
- **Emin**: Minimum energy (e.g., -10.000)
- **Emax**: Maximum energy (e.g., 10.000)
- **dE**: Energy spacing (e.g., 0.100)
- **nE**: Number of energy points (e.g., 200)
- **units**: Energy units (eV)

### Examples

Narrow energy range (around Fermi level):
```python
"ProjectedDensityOfStates": ["EF -5.000 5.000 0.050 200 eV"]
```

Wide energy range (full valence/conduction):
```python
"ProjectedDensityOfStates": ["EF -20.000 20.000 0.200 200 eV"]
```

High resolution (fine energy spacing):
```python
"ProjectedDensityOfStates": ["EF -10.000 10.000 0.025 800 eV"]
```

## Output Files

When you run the calculation (not dry-run), SIESTA generates:

- **siesta.DOS** - Total density of states (energy vs DOS)
- **siesta.PDOS** - Projected DOS on atoms/orbitals
- **siesta.PDOS.xml** - XML format for visualization tools

## Usage with Different Makers

### StaticMaker (SCF only)
```python
from atomate2.siesta.jobs.core import StaticMaker

maker = StaticMaker.scf(
    user_params={"ProjectedDensityOfStates": ["EF -10.000 10.000 0.100 200 eV"]}
)
```

### RelaxMaker (geometry optimization)
```python
from atomate2.siesta.jobs.core import RelaxMaker

maker = RelaxMaker.fixed_cell_relaxation(
    user_params={"ProjectedDensityOfStates": ["EF -10.000 10.000 0.100 200 eV"]}
)
```

### BandStructureMaker (bands + DOS)
```python
from atomate2.siesta.jobs.core import BandStructureMaker

maker = BandStructureMaker(
    user_params={"ProjectedDensityOfStates": ["EF -10.000 10.000 0.100 200 eV"]}
)
```

## See Also

- **Band structure calculations**: `tutorials/01-basics/03-band-structure/`
- **SIESTA Manual**: Section on ProjectedDensityOfStates
- **Dataclass reference**: `src/atomate2/siesta/dataclass/density_of_states_and_band_structure.py`
