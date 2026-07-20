# Phonon Input Parameters

Direct SIESTA FDF format for phonon-related parameters. These control force constants calculation used in phonon workflows.

## Overview

Phonon calculations in SIESTA use molecular dynamics (MD) mode for force constants:

- **MD.TypeOfRun = FC**: Force constants calculation mode
- **MD.FCDispl**: Atomic displacement magnitude
- **MD.FCfirst**: First atom to displace
- **MD.FClast**: Last atom to displace

## Quick Start

```python
from atomate2.siesta.jobs.core import StaticMaker

user_params = {
    "MD.TypeOfRun": "FC",
    "MD.FCDispl": "0.04 Bohr",
    "MD.FCfirst": 1,
    "MD.FClast": 2,
}

maker = StaticMaker.scf(user_params=user_params, dry_run=True)
```

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| MD.TypeOfRun | str | "CG" | Type of MD run (use "FC" for phonons) |
| MD.FCDispl | str | "0.04 Bohr" | Displacement for force constants |
| MD.FCfirst | int | 1 | First atom index to displace |
| MD.FClast | int | N | Last atom index to displace |

## Examples

### Small displacement (high-precision phonons)
```python
"MD.FCDispl": "0.02 Bohr"
```

### Large displacement (faster convergence)
```python
"MD.FCDispl": "0.08 Bohr"
```

### Displace only certain atoms
```python
"MD.FCfirst": 1,
"MD.FClast": 4,  # Only atoms 1-4
```

## Notes

- For full phonon workflows, use `PhonopyMaker` from `atomate2.siesta.jobs.phonopy`
- These parameters are automatically set by phonon workflows
- Manual control useful for testing or custom phonon calculations

## See Also

- **Phonon workflows**: `tutorials/05-vibrational-properties/01-phonons/`
- **SIESTA Manual**: Section on Force Constants
- **Dataclass**: `src/atomate2/siesta/dataclass/phonon_calculations.py`
