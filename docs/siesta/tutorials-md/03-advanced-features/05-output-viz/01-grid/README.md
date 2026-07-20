# Tutorial 10: Grid Output Configuration

**Level**: Advanced
**Time**: 20 minutes
**Dataclass Module**: `Grids` (ADVANCED tier)

## Overview

This tutorial demonstrates how to configure grid-based output for charge densities, potentials, and other electronic structure quantities using the `Grids` dataclass module.

## What You'll Learn

- How to save various charge densities and potentials to grid files
- Configure grid output file formats (binary, formatted, netcdf, cube)
- Enable specialized analyses like Bader charge analysis
- Use direct SIESTA FDF parameter names or Python attributes

## Background

SIESTA can output various electronic structure quantities on real-space grids for visualization and post-processing. The `Grids` module provides control over:

- **Charge densities**: Total, deformation, initial
- **Potentials**: Electrostatic, XC, total Kohn-Sham, neutral atom
- **Special outputs**: Ionic charge, total charge, Bader charge analysis

## Available Grid Output Parameters

### Charge Densities

| Parameter | Description | Default |
|-----------|-------------|---------|
| `save_rho` | Total electron charge density | False |
| `save_delta_rho` | Deformation charge density (SCF - atomic) | False |
| `save_initial_charge_density` | Initial non-SCF charge density | False |

### Potentials

| Parameter | Description | Default |
|-----------|-------------|---------|
| `save_rho_xc` | Exchange-correlation potential | False |
| `save_electrostatic_potential` | Total electrostatic potential (Hartree + external) | False |
| `save_neutral_atom_potential` | Superposition of neutral-atom potentials | False |
| `save_total_potential` | Total Kohn-Sham potential (electrostatic + XC) | False |

### Charge Distributions

| Parameter | Description | Default |
|-----------|-------------|---------|
| `save_ionic_charge` | Ionic charge distribution | False |
| `save_total_charge` | Total charge (electronic + ionic) | False |

### Special Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `save_grid_func_format` | File format: "binary", "formatted", "netcdf", "cube" | "binary" |
| `save_bader_charge` | Enable Bader charge analysis | False |
| `analyze_charge_density_only` | Only perform charge analysis then stop | False |

## Tutorial Examples

### Example 1: Basic Charge Density Output

Save total and deformation charge densities in binary format.

**File**: `01_basic_charge_density.py`

### Example 2: Potentials for Visualization

Save various potentials in NetCDF format for visualization tools.

**File**: `02_potentials_netcdf.py`

### Example 3: Bader Charge Analysis

Enable Bader charge analysis for charge transfer calculations.

**File**: `03_bader_analysis.py`

## Running the Examples

```bash
cd tutorials/07-advanced-features/10-grid-output

# Run any example
python3 01_basic_charge_density.py
python3 02_potentials_netcdf.py
python3 03_bader_analysis.py
```

## Output Files

Grid files are written in the calculation directory:

- **Binary format**: `.RHO`, `.DRHO`, `.VH`, `.VT` files
- **Formatted format**: `.RHO.ascii`, `.DRHO.ascii`, etc.
- **NetCDF format**: `.RHO.nc`, `.DRHO.nc`, etc.
- **Cube format**: `.RHO.cube`, `.DRHO.cube` (Gaussian cube format)

## Visualization Tools

Grid files can be visualized with:

- **XCrySDen**: Reads all SIESTA formats
- **VESTA**: Reads cube files
- **VMD**: Reads cube files
- **Python**: Use `pymatgen` or `ASE` to read grid files

## Best Practices

### 1. File Format Selection

```python
# Binary: Fastest, smallest files
user_params = {"SaveGridFunc.Format": "binary"}

# NetCDF: Portable, includes metadata
user_params = {"SaveGridFunc.Format": "netcdf"}

# Cube: Compatible with many visualization tools
user_params = {"SaveGridFunc.Format": "cube"}
```

### 2. Disk Space Considerations

Grid files can be large! For a 50x50x50 grid:
- Binary: ~1 MB per quantity
- Formatted: ~5 MB per quantity
- NetCDF: ~1.5 MB per quantity (with compression)

### 3. Common Visualization Tasks

**Charge Transfer Analysis**:
```python
# Save deformation charge density
user_params = {
    "SaveDeltaRho": True,
    "SaveGridFunc.Format": "cube"
}
```

**Electrostatic Potential Mapping**:
```python
# Save electrostatic potential
user_params = {
    "SaveElectrostaticPotential": True,
    "SaveGridFunc.Format": "netcdf"
}
```

**Bader Charge Analysis**:
```python
# Enable Bader analysis
user_params = {
    "SaveBaderCharge": True,
    "SaveRho": True  # Required for Bader
}
```

## Parameter Naming Flexibility

The module accepts both SIESTA FDF and Python parameter names:

```python
# SIESTA FDF names (recommended for clarity)
user_params = {
    "SaveRho": True,
    "SaveDeltaRho": True,
    "SaveGridFunc.Format": "netcdf"
}

# Python attribute names
user_params = {
    "save_rho": True,
    "save_delta_rho": True,
    "save_grid_func_format": "netcdf"
}

# Mixed (also works)
user_params = {
    "SaveRho": True,
    "save_delta_rho": True,
    "SaveGridFunc.Format": "netcdf"
}
```

## Common Issues

### Issue 1: Large Grid Files

**Problem**: Grid files consuming too much disk space

**Solution**:
- Use binary format instead of formatted
- Only save needed quantities
- Consider coarser mesh cutoff for visualization

### Issue 2: Bader Analysis Not Working

**Problem**: Bader charge analysis produces no output

**Solution**:
```python
# Must enable both SaveBaderCharge AND SaveRho
user_params = {
    "SaveBaderCharge": True,
    "SaveRho": True
}
```

### Issue 3: Visualization Tool Can't Read Files

**Problem**: XCrySDen/VESTA can't open grid files

**Solution**: Use appropriate format
- XCrySDen: Use binary or formatted (default SIESTA)
- VESTA/VMD: Use cube format

## Integration with Workflows

### With RelaxMaker

```python
from atomate2.siesta.jobs.core import RelaxMaker

maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "SaveRho": True,
        "SaveDeltaRho": True,
        "SaveGridFunc.Format": "netcdf"
    }
)
```

### With StaticMaker

```python
from atomate2.siesta.jobs.core import StaticMaker

maker = StaticMaker(
    user_params={
        "SaveElectrostaticPotential": True,
        "SaveTotalPotential": True,
        "SaveGridFunc.Format": "cube"
    }
)
```

## Performance Considerations

### Grid Output vs Calculation Time

Grid output adds minimal computational cost (<1% typically), but I/O time can be significant:

- **Binary format**: Fastest I/O
- **NetCDF format**: Moderate I/O, best compression
- **Formatted/Cube**: Slowest I/O, largest files

### When to Use Grid Output

**Always save**:
- For publication-quality charge density plots
- When performing Bader/Mulliken analysis

**Optional**:
- For routine calculations (can regenerate)
- When disk space is limited

## Advanced Usage

### Analyze Charge Density Only

Stop after charge density analysis without full SCF:

```python
user_params = {
    "AnalyzeChargeDensityOnly": True,
    "SaveRho": True
}
```

Useful for:
- Post-processing existing density matrix
- Quick charge analysis
- Debugging charge distributions

## Further Reading

- **SIESTA Manual**: Section 6.24 - Output of charge densities and potentials
- **Bader Analysis**: http://theory.cm.utexas.edu/henkelman/code/bader/
- **XCrySDen**: http://www.xcrysden.org/
- **Cube Format**: https://gaussian.com/cubegen/

## Next Steps

- **Tutorial 11**: Denchar visualization configuration
- **Tutorial 05**: DOS calculations (complementary electronic structure output)
- **Tutorial 07**: Optical properties (another ADVANCED tier module)

## Summary

You've learned how to:
- ✅ Configure grid-based output for charge densities and potentials
- ✅ Select appropriate file formats for visualization
- ✅ Enable Bader charge analysis
- ✅ Use both SIESTA FDF and Python parameter names
- ✅ Integrate grid output with atomate2 workflows

Grid output is essential for visualizing electronic structure and performing charge transfer analysis!
