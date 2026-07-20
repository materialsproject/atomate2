# Dataclass Modules

This directory contains **28 registered dataclass modules** for organizing SIESTA FDF parameters, plus 3 utility modules.

## Quick Start

```python
# Import from package level (recommended)
from atomate2.siesta.dataclass import (
    RealSpaceGridParameters,
    BasisSetsAndProjectors,
    MolecularDynamicsAndRelaxation,
)

# Or import from specific module (still works)
from atomate2.siesta.dataclass.real_space_grid_parameters import RealSpaceGridParameters

# Query unit information
unit = RealSpaceGridParameters.get_field_unit("mesh_cutoff")
print(unit)  # Output: "Ry"
```

## Module Organization

### 28 Registered Modules (4 Tiers)

**Basic Tier (7)** - Always enabled:
- `general_system_descriptors` - System name, label, PAO precision
- `pseudopotentials` - Pseudopotential paths and settings
- `basis_sets_and_projectors` - PAO basis configuration
- `kpoint_sampling` - k-point grids and paths
- `exchange_correlation_functionals` - XC functional selection
- `spin_settings` - Spin polarization and magnetic moments
- `real_space_grid_parameters` - Mesh cutoff and grid settings

**Intermediate Tier (6)**:
- `scf_loop_parameters` - SCF convergence and mixing
- `electronic_structure_calculation_options` - Fermi, DOS, etc.
- `molecular_dynamics_and_relaxation` - MD and geometry optimization
- `general_constraints` - Geometry constraints
- `external_control_and_scripting` - Lua scripting interface
- `chemical_analysis` - Mulliken populations, COOP

**Advanced Tier (9)**:
- `phonon_calculations` - Force constants and phonons
- `optical_properties` - Optical and dielectric properties
- `density_of_states_and_band_structure` - DOS and band structure
- `dftu` - DFT+U parameters
- `charge_dipole_electric_field` - Electric fields and dipole corrections
- `grids` - Advanced grid output (density, potential)
- `wannier90` - Wannier90 interface
- `auxiliary_force_field` - MM potentials, Grimme D3
- `denchar` - Charge density plotting

**Expert Tier (6)**:
- `parallel_options` - MPI and parallelization
- `solvers_and_performance_options` - Diagonalization methods
- `efficiency_options` - Memory and I/O optimization
- `hamiltonian_and_overlap_parameters` - H/S matrix settings
- `netcdf_options` - NetCDF output control
- `rttddft` - Real-time TDDFT

### Utility Modules (3)

- `base` - FDFDataclass base class with helper methods
- `structural_information` - Structure handling (ASE-based)
- `units` - Unit definitions and conversions
- `_metadata` - **Central metadata (author, copyright, license)**

## Features

### 1. Single Source of Truth for Metadata

All modules import metadata from `_metadata.py`:

```python
from atomate2.siesta.dataclass._metadata import (
    __author__,
    __copyright__,
    __license__,
)
```

**To update author/copyright information**: Edit `_metadata.py` once, changes apply to all 31 modules.

### 2. Unit Metadata (51 parameters)

Parameters with physical units include metadata:

```python
mesh_cutoff: float = field(
    default=100.0,
    metadata={
        "description": "Energy cutoff for real-space grid",
        "SIESTA keyword": "Mesh.Cutoff",
        "unit": "Ry",  # ✨ Unit information
    },
)
```

**Helper methods**:
```python
# Get unit for a field
unit = RealSpaceGridParameters.get_field_unit("mesh_cutoff")

# Get unit for FDF parameter
unit = RealSpaceGridParameters.get_fdf_parameter_unit("Mesh.Cutoff")

# Get all fields with units
fields = RealSpaceGridParameters.get_all_fields_with_units()
```

### 3. Package-Level Imports

Import any dataclass from package root:

```python
from atomate2.siesta.dataclass import (
    RealSpaceGridParameters,
    BasisSetsAndProjectors,
    FDFDataclass,  # Base class
)
```

### 4. __all__ Exports

Every module declares its public API:

```python
__all__ = ["RealSpaceGridParameters"]
```

## Updating Metadata

### Single File to Edit

**File**: `_metadata.py`

```python
# Author information
__author__ = "Your Name"
__email__ = "your.email@example.com"
__maintainer__ = "Your Name"

# Copyright and license
__copyright__ = "Copyright (c) 2024-2025, Your Name"
__license__ = "Modified BSD"
__version__ = "1.1.0"

# Project information
__project__ = "atomate2siesta"
__url__ = "https://github.com/your-org/atomate2siesta"
```

**Changes automatically propagate** to all 31 dataclass modules!

### Verify Changes

Run the test suite to verify metadata is consistent across all modules:

```bash
pytest tests/siesta/dataclass/
```

## Adding New Modules

When creating a new dataclass module:

1. **Import metadata** (after module docstring):
```python
"""
Module docstring.
"""

from __future__ import annotations

# Metadata
from atomate2.siesta.dataclass._metadata import (
    __author__,
    __copyright__,
    __license__,
)

__all__ = ["YourClassName"]
```

2. **Inherit from FDFDataclass**:
```python
from atomate2.siesta.dataclass.base import FDFDataclass

@dataclass
class YourClass(FDFDataclass):
    ...
```

3. **Register in** `__init__.py`:
```python
from atomate2.siesta.dataclass.your_module import YourClassName

__all__ = [
    # ... existing classes
    "YourClassName",
]
```

4. **Register in** `registry.py`:
```python
register_module(
    name="your_module",
    module_path="atomate2.siesta.dataclass.your_module",
    class_name="YourClassName",
    setup_method="setup_your_class",
    tier="intermediate",
    priority=50,
)
```

## Statistics

- **Total modules**: 31 (28 registered + 3 utility)
- **Parameters with units**: 51 across 9 modules
- **Tiers**: 4 (basic, intermediate, advanced, expert)
- **Helper methods**: 3 (unit extraction)
- **Import styles**: 2 (package-level + explicit)
- **Metadata sources**: 1 (_metadata.py - single source of truth)

## See Also

- `registry.py` - Module registration and tier system
- `base.py` - FDFDataclass base class with helper methods
- `_metadata.py` - Central metadata (author, copyright, etc.)
- `__init__.py` - Package-level imports
- `../sets/base.py` - Auto-detection of modules from user parameters
