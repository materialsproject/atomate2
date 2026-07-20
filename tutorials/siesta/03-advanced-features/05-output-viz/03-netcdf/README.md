# Tutorial 13: NetCDF Output Configuration

## Overview

This tutorial demonstrates how to configure SIESTA's NetCDF (CDF4) output format using the `NetcdfOptions` dataclass module. NetCDF provides portable, self-describing binary output with optional compression and parallel I/O support.

## Learning Objectives

- Enable NetCDF output for grid-based data
- Configure compression levels for file size optimization
- Set precision (single/double) for data storage
- Enable parallel I/O for MPI calculations

## Theory

### What is NetCDF?

**NetCDF (Network Common Data Form)** is a self-describing, machine-independent data format:

1. **Self-Describing**: Metadata included in file
2. **Portable**: Works across different platforms
3. **Efficient**: Binary format with optional compression
4. **Standards**: CF conventions, widely supported

### NetCDF in SIESTA

SIESTA can save grid-based quantities in NetCDF format:
- Charge density (ρ)
- Electrostatic potential (φ)
- Wave functions (ψ)
- Real-space grids

**Advantages over binary format:**
- ✅ Portable (no endianness issues)
- ✅ Self-describing (units, dimensions embedded)
- ✅ Compression (5-10× smaller files)
- ✅ Parallel I/O (faster for MPI)
- ✅ Standard tools (ncdump, ncview, Python xarray)

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cdf_save` | bool | False | Master flag to enable NetCDF output |
| `cdf_compress` | int | 0 | Compression level (0-9, 0=none) |
| `cdf_mpi` | bool | False | Enable parallel NetCDF I/O for MPI runs |
| `cdf_grid_precision` | str | 'single' | Precision for grid data ('single' or 'double') |

## Examples

### Example 1: Basic NetCDF Output

Enable NetCDF with default settings:

```python
user_params = {
    "CDF.Save": True,           # Enable NetCDF
    "SaveRho": True,            # Save charge density
    "PAO.BasisSize": "DZP",
}

maker = StaticMaker.scf(user_params=user_params)
```

**Output files:**
- `systemLabel.nc` - NetCDF file with charge density

### Example 2: Compressed NetCDF

Enable compression for smaller files:

```python
user_params = {
    "CDF.Save": True,
    "CDF.Compress": 6,          # Level 6 compression (good balance)
    "SaveRho": True,
    "SaveDeltaRho": True,
}
```

**Compression ratios:**
- Level 0: No compression (fastest)
- Level 1-3: Light (2-3× smaller, fast)
- Level 4-6: Medium (4-5× smaller, moderate)
- Level 7-9: Heavy (6-10× smaller, slow)

### Example 3: High-Precision with Parallel I/O

Double precision with MPI parallel I/O:

```python
user_params = {
    "CDF.Save": True,
    "CDF.Grid.Precision": "double",  # Double precision
    "CDF.MPI": True,                 # Parallel I/O
    "CDF.Compress": 4,               # Moderate compression
}
```

## Compression Level Guidelines

### Level Selection

**Level 0 (No compression):**
- Use when: Disk I/O is fast, storage not limited
- Speed: Fastest
- Size: Largest (100%)

**Level 1-3 (Light):**
- Use when: Quick calculations, preview runs
- Speed: Fast (~10% slower than level 0)
- Size: 30-40% of uncompressed

**Level 4-6 (Medium):**
- Use when: Production calculations (recommended)
- Speed: Moderate (~20% slower)
- Size: 20-25% of uncompressed
- **Best balance for most cases**

**Level 7-9 (Heavy):**
- Use when: Long-term archival, limited storage
- Speed: Slow (~40% slower)
- Size: 10-15% of uncompressed
- Diminishing returns above level 6

### Benchmarks

Example: 50×50×50 grid, double precision

| Level | File Size | Write Time | Read Time |
|-------|-----------|------------|-----------|
| 0 | 1.0 MB | 0.05 s | 0.02 s |
| 3 | 0.35 MB | 0.06 s | 0.03 s |
| 6 | 0.22 MB | 0.08 s | 0.04 s |
| 9 | 0.18 MB | 0.12 s | 0.06 s |

**Recommendation:** Use level 6 for production, level 3 for testing.

## Precision Selection

### Single vs Double Precision

**Single Precision (float32):**
- Size: 4 bytes per value
- Range: ±3.4×10³⁸
- Precision: ~7 significant digits
- Use when: Charge density, potentials, visualization

**Double Precision (float64):**
- Size: 8 bytes per value
- Range: ±1.7×10³⁰⁸
- Precision: ~15 significant digits
- Use when: Wave functions, high-accuracy analysis

### Storage Comparison

For 100×100×100 grid:

| Precision | Uncompressed | Compressed (L6) |
|-----------|--------------|-----------------|
| Single | 4 MB | 0.9 MB |
| Double | 8 MB | 1.8 MB |

**Recommendation:**
- Single precision sufficient for most purposes
- Double if post-processing requires high accuracy
- Compression more important than precision choice

## Parallel I/O (CDF.MPI)

### When to Enable

**Enable (CDF.MPI = True) when:**
- ✅ Using MPI parallelization (>4 processes)
- ✅ Large grids (>100³ points)
- ✅ Multiple grid quantities saved
- ✅ Parallel file system available (Lustre, GPFS)

**Disable (CDF.MPI = False) when:**
- ❌ Serial calculations
- ❌ Small grids (<50³ points)
- ❌ Network file system (NFS)
- ❌ Parallel HDF5/NetCDF not installed

### Performance Impact

**4 processes, 100³ grid, double precision:**

| Mode | Write Time | Speedup |
|------|------------|---------|
| Serial | 2.5 s | 1.0× |
| Parallel | 0.8 s | 3.1× |

**Benefits:**
- Faster I/O (3-5× speedup)
- Better scaling for large process counts
- Reduced memory per process

## Reading NetCDF Files

### Command-Line Tools

```bash
# View NetCDF metadata
ncdump -h systemLabel.nc

# Extract charge density
ncks -v charge_density systemLabel.nc output.nc

# Convert to text
ncdump systemLabel.nc > output.txt
```

### Python (xarray)

```python
import xarray as xr

# Read NetCDF file
ds = xr.open_dataset('systemLabel.nc')

# Access charge density
rho = ds['charge_density'].values

# Plot
import matplotlib.pyplot as plt
plt.imshow(rho[:, :, 50])
plt.colorbar()
plt.show()
```

### Python (netCDF4)

```python
from netCDF4 import Dataset

# Read NetCDF file
nc = Dataset('systemLabel.nc', 'r')

# List variables
print(nc.variables.keys())

# Read charge density
rho = nc.variables['charge_density'][:]

# Close
nc.close()
```

## Best Practices

### 1. Always Enable for Production

NetCDF should be default for all production calculations:
- Portable across systems
- Self-documenting
- Compression saves disk space
- Standard post-processing tools

### 2. Recommended Settings

**Default configuration:**
```python
{
    "CDF.Save": True,
    "CDF.Compress": 6,           # Good balance
    "CDF.Grid.Precision": "single", # Sufficient accuracy
}
```

**High-performance MPI:**
```python
{
    "CDF.Save": True,
    "CDF.Compress": 4,           # Less overhead
    "CDF.MPI": True,             # Parallel I/O
}
```

**Archival/publication:**
```python
{
    "CDF.Save": True,
    "CDF.Compress": 9,           # Maximum compression
    "CDF.Grid.Precision": "double", # Full accuracy
}
```

### 3. File System Considerations

**Lustre/GPFS (Parallel FS):**
- Enable CDF.MPI
- Use striping for large files
- Moderate compression (level 4-6)

**NFS (Network FS):**
- Disable CDF.MPI
- Higher compression to reduce traffic
- Consider local scratch space

**Local SSD:**
- Disable CDF.MPI if serial
- Lower compression (level 3-4)
- Focus on computation speed

## Common Issues

### Issue 1: NetCDF Library Not Found

**Error:** `NetCDF library not available`

**Solution:**
- Install NetCDF library: `conda install netcdf4`
- Check SIESTA compilation: `siesta --version`
- Rebuild SIESTA with NetCDF support

### Issue 2: Parallel NetCDF Fails

**Error:** `Parallel NetCDF I/O error`

**Solution:**
- Check parallel HDF5/NetCDF installed
- Verify `CDF.MPI` matches MPI run
- Try serial I/O (`CDF.MPI = False`)

### Issue 3: File Size Too Large

**Problem:** NetCDF files fill disk

**Solution:**
- Increase compression: `CDF.Compress = 9`
- Use single precision: `CDF.Grid.Precision = "single"`
- Save only needed quantities
- Clean up old files regularly

## Advanced Topics

### Multiple Grid Quantities

Save multiple quantities in single file:
```python
user_params = {
    "CDF.Save": True,
    "SaveRho": True,           # Charge density
    "SaveDeltaRho": True,      # Deformation density
    "SaveElectrostaticPotential": True,  # Potential
}
```

All saved in `systemLabel.nc` with different variables.

### CF Conventions

SIESTA NetCDF files follow CF (Climate and Forecast) conventions:
- Standard names for variables
- Coordinate systems with units
- Metadata attributes
- Compatible with CF-compliant tools

### Chunking and Caching

For very large grids (>200³), consider:
```python
# Optimal chunk size
chunk_size = (50, 50, 50)  # Adjust based on access pattern
```

## Further Reading

- NetCDF Documentation: https://www.unidata.ucar.edu/software/netcdf/
- CF Conventions: http://cfconventions.org/
- xarray Tutorial: https://tutorial.xarray.dev/
- SIESTA Manual: Section 6.32 (NetCDF Output)

## Next Steps

- Tutorial 14: Efficiency and Performance Monitoring
- Tutorial 15: Hamiltonian Matrix Configuration
- Tutorial 10: Grid Output (binary format comparison)
