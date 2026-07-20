# Optical Properties Calculations

Direct SIESTA FDF format for optical properties calculations including dielectric function, absorption, and refractive index.

## Overview

SIESTA calculates optical properties from electronic structure:

- **Dielectric function** ε(ω): Real and imaginary parts
- **Absorption coefficient** α(ω)
- **Refractive index** n(ω)
- **Reflectivity** R(ω)

## Quick Start

```python
from atomate2.siesta.jobs.core import StaticMaker

user_params = {
    "OpticalCalculation": "true",
    "Optical.Energy.Minimum": "0.0 eV",
    "Optical.Energy.Maximum": "10.0 eV",
    "Optical.Broaden": "0.1 eV",
    "kpts": [8, 8, 8],  # Dense k-grid required
}

maker = StaticMaker.scf(user_params=user_params, dry_run=True)
```

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| OpticalCalculation | bool | false | Enable optical calculation |
| Optical.Energy.Minimum | str | "0.0 eV" | Minimum photon energy |
| Optical.Energy.Maximum | str | "10.0 eV" | Maximum photon energy |
| Optical.Broaden | str | "0.1 eV" | Spectral broadening |
| Optical.Scissor | str | "0.0 eV" | Band gap correction |
| Optical.NumberOfBands | int | All | Number of bands to include |
| Optical.PolarizationType | str | "unpolarized" | Polarization (unpolarized/polarized) |

## Examples

### UV-Vis spectrum (0-5 eV)
```python
user_params = {
    "OpticalCalculation": "true",
    "Optical.Energy.Minimum": "0.0 eV",
    "Optical.Energy.Maximum": "5.0 eV",
    "Optical.Broaden": "0.05 eV",
}
```

### Wide energy range with band gap correction
```python
user_params = {
    "OpticalCalculation": "true",
    "Optical.Energy.Minimum": "0.0 eV",
    "Optical.Energy.Maximum": "20.0 eV",
    "Optical.Scissor": "0.6 eV",  # Correct LDA/GGA band gap underestimation
}
```

### High resolution spectrum
```python
user_params = {
    "OpticalCalculation": "true",
    "Optical.Energy.Minimum": "0.0 eV",
    "Optical.Energy.Maximum": "10.0 eV",
    "Optical.Broaden": "0.02 eV",  # Sharp features
    "kpts": [12, 12, 12],  # Very dense k-grid
}
```

## Output Files

SIESTA generates:

- **siesta.EPSIMG** - Imaginary part of dielectric function
- **siesta.EPSREAL** - Real part of dielectric function

## Important Notes

1. **Dense k-grids required**: Optical properties need much denser k-point sampling than ground state calculations (typically 8×8×8 or higher)

2. **Band gap correction**: LDA/GGA underestimate band gaps. Use `Optical.Scissor` to shift conduction bands

3. **Convergence**: Test convergence with respect to:
   - k-point density
   - Number of unoccupied bands
   - Energy cutoff

## See Also

- **SIESTA Manual**: Section on Optical Properties
- **Dataclass**: `src/atomate2/siesta/dataclass/optical_properties.py`
