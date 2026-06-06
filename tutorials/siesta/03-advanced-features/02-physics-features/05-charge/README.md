# Charge, Dipole, and Electric Field Calculations

Direct SIESTA FDF format for charged systems, external fields, and dipole corrections.

## Overview

This module covers three related features:

1. **Charged Systems**: Net charge on the system (ions, defects)
2. **External Electric Fields**: Applied electric fields
3. **Dipole Corrections**: Corrections for slab dipoles

## Quick Start

### External Electric Field
```python
user_params = {
    "ExternalElectricField": "true",
    "Efield": "0.01 eV/Ang",
}
```

### Charged System
```python
user_params = {
    "NetCharge": "+2.0",  # Remove 2 electrons
}
```

### Slab Dipole Correction
```python
user_params = {
    "SlabDipoleCorrection": "true",
}
```

## Examples

### 01_external_electric_field.py
Apply external electric field to a slab geometry.

### 02_charged_system.py
Simulate a charged defect (vacancy with +2 charge).

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| NetCharge | str | "0.0" | Net system charge (electrons removed/added) |
| ExternalElectricField | bool | false | Enable external field |
| Efield | str | "0.0 eV/Ang" | Field strength |
| SlabDipoleCorrection | bool | false | Dipole correction for slabs |

## Common Use Cases

### 1. Field Effect in 2D Materials
```python
user_params = {
    "ExternalElectricField": "true",
    "Efield": "0.01 eV/Ang",  # Moderate field
    "kpts": [8, 8, 1],  # 2D k-sampling
}
```

### 2. Charged Defects (Vacancies, Interstitials)
```python
user_params = {
    "NetCharge": "+1.0",  # Positively charged defect
    "kpts": [2, 2, 2],  # Large supercell needed
}
```

### 3. Asymmetric Slab (Surface Dipole)
```python
user_params = {
    "SlabDipoleCorrection": "true",  # Correct artificial field
}
```

### 4. Ferroelectric Switching
```python
user_params = {
    "ExternalElectricField": "true",
    "Efield": "0.05 eV/Ang",  # High field for switching
}
```

### 5. Ionization Energy Calculation
```python
# Neutral system
neutral_params = {"NetCharge": "0.0"}

# Ionized system (remove 1 electron)
ion_params = {"NetCharge": "+1.0"}

# IE = E(ion) - E(neutral)
```

## Important Notes

### Charged Systems

1. **Background charge**: SIESTA adds uniform background charge for neutrality
2. **Large supercells needed**: Minimize interaction between periodic images
3. **Finite-size corrections**: May need correction schemes (Makov-Payne, etc.)

### Electric Fields

1. **Field direction**: Typically along non-periodic direction (z for slabs)
2. **Units**: eV/Ang or V/Ang
3. **Convergence**: Strong fields may require:
   - Smaller mixing: `"SCF.Mixer.Weight": 0.01`
   - More iterations: `"MaxSCFIterations": 200`

### Dipole Corrections

1. **When to use**: Asymmetric slabs with net dipole moment
2. **Effect**: Removes spurious electric field from periodic boundary conditions
3. **Combine with vacuum**: Need sufficient vacuum (>10 Å)

## Field Strength Guidelines

| Application | Field Strength | Notes |
|-------------|----------------|-------|
| Weak perturbation | 0.001-0.01 eV/Ang | Linear response regime |
| Moderate field | 0.01-0.1 eV/Ang | Typical simulations |
| Strong field | 0.1-1.0 eV/Ang | Breakdown, switching |

## See Also

- **SIESTA Manual**: Sections on NetCharge, External Electric Field, Slab Dipole Correction
- **Dataclass**: `src/atomate2/siesta/dataclass/charge_dipole_electric_field.py`
