# Tutorial: Adsorption Site Scanning

**Category**: 06-surfaces-and-adsorption
**Difficulty**: Advanced
**Time**: ~5 min (dry-run), ~2-4 hours (full scan)

---

## Overview

Automated scanning of adsorption sites on surface slabs to identify optimal binding locations for adsorbates. This tutorial demonstrates grid-based exploration with energy calculations, molecule orientation control, and result visualization.

This consolidates 8 example scripts into a single configurable workflow.

---

## What You'll Learn

- Grid-based adsorption site scanning
- Adsorption energy calculation (E_ads = E_total - E_slab - E_adsorbate)
- Molecule orientation control (5 different modes)
- Top/bottom surface placement
- Two-stage workflows (scan + optimization)
- Result visualization and analysis
- High-throughput site screening

---

## Prerequisites

- **Required**: [17-surface-energy](../01-surface-energy/) completed
- **Required**: Understanding of surface slab models
- **Recommended**: Familiarity with adsorption chemistry
- **Recommended**: pymatgen molecule/structure manipulation

---

## Key Concepts

### Adsorption Energy

$$
E_{ads} = E(slab + adsorbate) - E(clean slab) - E(isolated adsorbate)
$$

- **E_ads < 0**: Favorable adsorption (exothermic)
- **E_ads > 0**: Unfavorable adsorption (endothermic)
- **More negative**: Stronger binding

**Typical Ranges**:
- Physisorption: 0.0 to -0.5 eV
- Chemisorption: -0.5 to -5.0 eV

### Grid-Based Scanning

```
1. Define 2D grid over surface unit cell
   └─ grid_size=(Nx, Ny) → Nx × Ny sites

2. Place adsorbate at each grid point
   └─ height above surface (e.g., 2.0 Å)

3. Calculate total energy at each site
   └─ E_total for slab+adsorbate

4. Compute adsorption energy
   └─ E_ads = E_total - E_slab - E_adsorbate

5. Rank sites by E_ads
   └─ Most negative = strongest binding
```

---

## Tutorial Files

This directory contains 11 comprehensive examples covering different aspects of adsorption scanning:

### Basic Examples

**01_grid_scan.py** - Basic grid-based adsorption scanning
- Simple 3×3 grid scan
- Automatic site discovery
- Energy landscape visualization
- Good starting point

**02_with_presets.py** - Using tier presets for simplified setup
- Demonstrates tier system (`tier="dirty"` for fast testing)
- Automatic parameter selection
- Comparison of different tier levels

**03_custom_parameters.py** - Full parameter customization
- Custom SIESTA settings for slab and adsorbate
- Grid size control
- Height scanning options
- Production-ready example

### Remote/HPC Examples

**04_custom_parameters_remote.py** - Remote execution with jobflow-remote
- HPC cluster submission
- Resource specification
- Same workflow as 03 but for clusters

**05_dynamic_resources_remote.py** - Dynamic per-job resource allocation
- Different resources for different jobs
- Small molecules: 1 core (prevents "too many processors" error)
- Large systems: 24+ cores
- Optimal resource usage

**06_simple_resource_control.py** - Simplified resource control
- Easy resource specification
- Good for standard HPC usage

**07_easiest_resource_control.py** - Easiest resource control method
- Minimal code for resource allocation
- Beginner-friendly

### Advanced Parameter Control

**08_tier_with_powerups.py** - Tier system with powerups
- Apply tier globally
- Override specific parameters with powerups
- Demonstrates tier + customization workflow

**09_tier_with_powerups_after_make.py** - Post-workflow powerups
- Apply powerups after workflow creation
- Modify existing workflows
- Maximum flexibility

### Height Scanning

**10_height_scanning.py** - 3D potential energy surface scanning
- Scan multiple heights above surface
- Full 3D energy landscape
- Optimal height determination

**11_height_scanning_tier.py** - Height scanning with tier presets
- Combines height scanning with tier system
- Fast parameter selection
- Production-ready example

---

## Quick Start

```bash
# 1. Start with basic grid scan (recommended)
python 01_grid_scan.py

# 2. Inspect output
ls job_*/
# Should see: clean slab, isolated adsorbate, site_* jobs

# 3. Try tier presets for fast testing
python 02_with_presets.py

# 4. Custom parameters for production
python 03_custom_parameters.py

# 5. For HPC clusters
python 04_custom_parameters_remote.py

# 6. Advanced: Height scanning (3D potential surface)
python 10_height_scanning.py

# Analyze results
open adsorption_sites.png        # Energy heatmap
cat adsorption_summary.txt        # Best sites ranked
open best_adsorption_structure.cif  # Optimal structure
```

---

## Expected Output

### Dry-Run Mode

```
✅ Dry-run complete!

📁 Workflow Structure:
  1. Clean Slab Calculation (1 job)
  2. Isolated Adsorbate Calculation (1 job)
  3. Adsorption Site Grid (N jobs, where N = grid_size)
  4. Analysis (Automatic)

💡 Total jobs: 2 + grid_size (e.g., 11 for 3×3 grid)
```

### Local Mode

```
✅ Adsorption scan complete!

📊 Output Files:
  - adsorption_sites.png: 2D heatmap
  - adsorption_summary.txt: Best sites ranked
  - adsorption_scan.csv: Complete data
```

**adsorption_sites.png**: 2D contour plot showing energy landscape across surface

**adsorption_summary.txt**:
```
Adsorption Site Ranking:
Rank | Site   | x (frac) | y (frac) | E_ads (eV)
 1   | Site_5 | 0.333    | 0.333    | -2.45
 2   | Site_2 | 0.000    | 0.333    | -2.31
 3   | Site_8 | 0.667    | 0.333    | -2.28
...
```

---

## Analyzing Results

### Identifying Binding Sites

Common high-symmetry sites:
- **Top site**: Above surface atom
- **Bridge site**: Between two surface atoms
- **Hollow site**: Center of 3-4 surface atoms (fcc/hcp)

### Energy Landscape Interpretation

**Smooth landscape** (ΔE < 0.2 eV):
- Weak site preference
- Mobile adsorbate
- All sites roughly equivalent

**Rugged landscape** (ΔE > 0.5 eV):
- Strong site preference
- Localized binding
- Site-specific chemistry

**Barrier height** (energy difference between sites):
- < 0.1 eV: Very mobile, fast diffusion
- 0.1-0.3 eV: Moderate mobility
- > 0.5 eV: Localized, slow diffusion

### Validation Checks

✓ **Energy scale**: Does E_ads make chemical sense?
✓ **Site preference**: Does best site match expected chemistry?
✓ **Comparison**: Literature or experimental data available?
✓ **Convergence**: Try higher grid resolution (3×3 → 5×5 → 7×7)

---

## Molecule Orientation Control

### Parameters

**target_vector** (3D vector):
```python
[0, 0, 1]  # Perpendicular to surface
[1, 0, 0]  # Parallel to surface (along x)
[1, 0, 1]  # Tilted 45° from perpendicular
```

**extra_rotation** (float, degrees):
```python
extra_rotation=30.0  # Additional 30° rotation
rotation_axis=[0, 0, 1]  # Around z-axis
```

**placement** (str):
```python
placement="top"     # Default: top surface
placement="bottom"  # Bottom surface
```

**plane_atoms** (list of int):
```python
plane_atoms=[0, 1, 2]  # For planar molecules (benzene, graphene)
# Ensures plane parallel to surface
```

### Examples

**Linear molecule (CO)**:
```python
# Perpendicular
target_vector=[0, 0, 1]

# Tilted
target_vector=[1, 0, 1]

# Flat with 30° rotation
target_vector=[1, 0, 0], extra_rotation=30.0, rotation_axis=[0, 0, 1]
```

**Planar molecule (benzene)**:
```python
plane_atoms=[0, 1, 2]  # First 3 ring carbons
# Automatically orients ring parallel to surface
```

---

## Parameter Selection Guide

### Grid Size

- **Testing**: (2,2) or (3,3) - ~15-30 min
- **Standard**: (5,5) or (7,7) - ~1-3 hours
- **High resolution**: (10,10) - ~5-10 hours
- **Publication**: (15,15) or higher

**Trade-off**: Resolution vs computational cost

### Height

- **Typical**: 2.0-3.0 Å above surface
- **Small molecules** (CO, H2): 2.0 Å
- **Large molecules** (benzene): 3.0-4.0 Å
- **Rule**: Start at typical bond distance + 0.5 Å

### SIESTA Parameters

**For Slab**:
```python
{
    "PAO.BasisSize": "DZP",  # Or TZP for high accuracy
    "Mesh.Cutoff": "300 Ry",  # Well converged!
    "kpts": [6, 6, 1],  # Dense in-plane, sparse in z
}
```

**For Adsorbate**:
```python
{
    "PAO.BasisSize": "DZP",  # Same as slab!
    "Mesh.Cutoff": "300 Ry",  # Same as slab! (CRITICAL)
}
```

⚠️ **CRITICAL**: Cutoff must be identical for slab and adsorbate!

**For Metals**:
```python
{
    "OccupationFunction": "MP",
    "ElectronicTemperature": "300 K",
    "kpts": [12, 12, 1],  # Denser k-points
}
```

---

## Two-Stage Workflow

### Why Use Two Stages?

**Stage 1 (Scan)**:
- Fast screening with fixed geometries
- Many sites evaluated quickly
- Identifies promising regions

**Stage 2 (Optimization)**:
- Geometry relaxation at best site
- More accurate final energy
- Typically lowers E_ads by 0.1-0.5 eV

### Workflow

```python
# Stage 1: Coarse scan
scan_maker = AdsorptionScanMaker(grid_size=(3, 3))
scan_flow = scan_maker.make(slab, adsorbate)

# Stage 2: Optimize at best site
opt_maker = AdsorptionOptimizationMaker(
    fix_slab=True  # Only relax adsorbate
)
# opt_flow = opt_maker.make(best_site)  # From scan results
```

### Best Practices

1. Run coarse scan (3×3) first
2. Identify top 2-3 sites
3. Run fine scan (5×5) around promising regions
4. Optimize geometry at best site
5. Compare initial vs optimized E_ads

---

## Common Issues

### Issue 1: Adsorbate Overlaps with Slab

**Symptoms**: Very positive E_ads, SCF fails

**Solutions**:
1. Increase `height` (try 3.0 or 4.0 Å)
2. Check vacuum thickness (> 15 Å)
3. Visualize initial structure

### Issue 2: Unphysical Energies

**Symptoms**: E_ads >> 0 or << -10 eV

**Solutions**:
1. ⚠️ **Check cutoff consistency** (slab and adsorbate must match!)
2. Verify basis set consistency
3. Ensure slab convergence (thickness, k-points)
4. Check for SCF failures

### Issue 3: All Sites Similar

**Symptoms**: Energy range < 0.1 eV

**Solutions**:
1. Increase grid resolution (3×3 → 5×5 → 10×10)
2. Try different orientation
3. Reduce initial height slightly
4. May be genuine (homogeneous surface)

### Issue 4: Long Calculation Time

**Solutions**:
1. Start with (2,2) or (3,3) grid
2. Use coarser k-points [2,2,1] for testing
3. Submit to cluster (RUN_MODE="submit")
4. Reduce basis to SZP for initial tests

### Issue 5: No Visualization

**Symptoms**: Missing .png files

**Solutions**:
1. Ensure `plot_results=True`
2. Install matplotlib: `pip install matplotlib`
3. Manually plot from adsorption_scan.csv

### Issue 6: "Invalid tier" Error for Preset Names

**Symptoms**:
```
ValueError: Invalid tier 'relax_dirty'. Must be one of ['basic', 'intermediate',
'advanced', 'expert', 'all', 'dirty', 'ultra']
```

**Cause**: The `tier` parameter accepts tier **levels**, not preset **names**.

**Current Workaround**:
```python
# ❌ FAILS: Preset name
tier="relax_dirty"

# ✅ WORKS: Use tier level
tier="dirty"  # Minimal settings for fast testing
tier="basic"  # Standard production

# Future: Per-maker presets planned
maker_presets={
    "slab_static_maker": "surface_metal",
    "adsorbate_static_maker": "molecular_standard",
}  # Not yet supported
```

**Tier Level Guide**:
- `tier="dirty"`: Fast testing (SZ, 50 Ry, 1×1×1 k-points) - ~5-10 min/site
- `tier="basic"`: Standard (DZP, 100 Ry, 2×2×2 k-points) - ~20-30 min/site
- `tier="intermediate"`: Production (DZP, 300 Ry, 4×4×4 k-points) - ~1-2 hours/site

**Future Enhancement**: Preset name resolution and per-maker preset configuration are planned.

---

## Convergence Tests

Always test convergence of:

1. ✅ **Slab thickness**: Number of layers (3-7 typical)
2. ✅ **Vacuum thickness**: > 15 Å
3. ✅ **K-point mesh**: Especially for metals
4. ✅ **Cutoff energy**: 300-500 Ry (must be same for slab and adsorbate!)
5. ✅ **Grid resolution**: Compare 3×3 vs 5×5 vs 7×7
6. ✅ **Initial height**: Try ±0.5 Å variation

---

## Advanced Topics

### Multiple Orientations

Scan same adsorbate with different orientations:
```python
# Compare perpendicular vs tilted vs flat
orientations = [[0,0,1], [1,0,1], [1,0,0]]
for orient in orientations:
    scan_maker = AdsorptionScanMaker(target_vector=orient)
    # ... run scan
```

### Coverage Dependence

Study lateral interactions:
```python
# Create supercell
slab_2x2 = slab.make_supercell([2,2,1])
# Place multiple adsorbates
# Scan remaining sites
```

### Reaction Intermediates

Map out reaction pathway:
```python
# Scan different species
intermediates = [CO, CO2, HCOO, COOH]
for species in intermediates:
    scan_maker = AdsorptionScanMaker()
    # ... scan each intermediate
```

---

## Tips for Success

✅ **Start small**: (2,2) grid, minimal basis for testing
✅ **Cutoff consistency**: Same for slab and adsorbate (CRITICAL!)
✅ **Visualize first**: Check structures before long calculations
✅ **Use custodian**: Automatic error handling (use_custodian=True)
✅ **For metals**: MP occupation, dense k-points
✅ **Validate**: Compare E_ads with literature/experiment
✅ **Document**: Keep notes on parameter effects
✅ **Save results**: Adsorption scans are expensive!

---

## Best Practices

**Workflow Design**:
1. Dry-run with (2,2) grid first
2. Local test with (3,3) grid
3. Production with (5,5) or higher
4. Always optimize at best site

**Parameter Selection**:
- Testing: SZP, 100 Ry, [2,2,1], (2,2) grid
- Standard: DZP, 300 Ry, [6,6,1], (5,5) grid
- High accuracy: TZP, 500 Ry, [10,10,1], (10,10) grid

**Resource Management**:
1. Estimate time: ~20-30 min per site
2. Grid (3,3) = ~3-5 hours total
3. Grid (5,5) = ~10-15 hours total
4. Use HPC cluster for large grids

**HPC Submission (jobflow-remote)**:
- See `05_dynamic_resources_remote.py` for per-job resource allocation
- Small molecules: 1 core (prevents "too many processors" error)
- Medium systems: 8-12 cores
- Large systems: 24+ cores

---

## Next Steps

After completing this tutorial:

1. **Understand workflow**: Review dry-run output
2. **Test small grid**: Run (2,2) locally to verify setup
3. **Try orientations**: Experiment with different ORIENTATION_MODE
4. **Production run**: Use (5,5) grid with DZP/300 Ry
5. **Optimize**: Run geometry relaxation at best site
6. **Compare**: Literature or experimental E_ads values
7. **Advanced**: Coverage dependence, coadsorption, reactions

---

*Back to [06-surfaces-and-adsorption](../README.md) | [Main Tutorial Index](../../README.md)*
