# 06: Surfaces and Adsorption

**Focus**: Surface chemistry, catalysis, and adsorption site screening

**Difficulty**: Advanced

**Prerequisites**:
- Completed [01-basics](../../01-basics/) and [02-convergence](../../02-workflows/01-convergence/)
- Understanding of surface science concepts
- Familiarity with Miller indices and slab models
- Converged bulk calculation parameters

---

## Tutorials in This Category

### [01-MultiSurfaceEnergyFlowMaker](01-MultiSurfaceEnergyFlowMaker/)
**Description**: Surface energy calculations with automatic slab generation and termination analysis
**Difficulty**: Advanced
**Time**: ~10 min (dry-run), ~2-4 hours (full calculation)
**Key Concepts**: Miller indices, slab models, surface energy, terminations, vacuum spacing, symmetric slabs
**📖 [Complete Tutorial Guide →](01-MultiSurfaceEnergyFlowMaker/README.md)**

### [02-AdsorptionScanFlowMaker](02-AdsorptionScanFlowMaker/)
**Description**: Grid-based adsorption site screening with automatic structure generation
**Difficulty**: Advanced
**Time**: ~5 min (dry-run), ~4-12 hours (full calculation)
**Key Concepts**: Adsorption sites, grid scanning, adsorbate orientation, binding energy, coverage effects
**📖 [Complete Tutorial Guide →](02-AdsorptionScanFlowMaker/README.md)**

---

## Learning Path

**Sequential approach**:

1. **Foundation**: [01-MultiSurfaceEnergyFlowMaker](01-MultiSurfaceEnergyFlowMaker/) - Understand surface models
2. **Application**: [02-AdsorptionScanFlowMaker](02-AdsorptionScanFlowMaker/) - Screen adsorption sites

**Why this order?**
- Need converged slab model before adding adsorbates
- Surface energy provides reference for adsorption energies

---

## Surface Science Basics

### Surface Energy

Surface energy (γ) measures the cost of creating a surface:

```
γ = (E_slab - N × E_bulk) / (2 × A)
```

Where:
- E_slab = Total energy of slab
- E_bulk = Energy per atom in bulk
- N = Number of atoms in slab
- A = Surface area
- Factor of 2 = Two surfaces (top + bottom)

**Typical values**:
- Metals: 1-3 J/m² (low surface energy)
- Ionic crystals: 0.5-2 J/m²
- Covalent materials: 2-4 J/m²

### Adsorption Energy

Adsorption energy (E_ads) measures binding strength:

```
E_ads = E_slab+ads - (E_slab + E_adsorbate)
```

**Sign convention**:
- E_ads < 0: Exothermic adsorption (stable)
- E_ads > 0: Endothermic adsorption (unstable)

---

## Critical Parameters

### Slab Construction

1. **Number of layers**:
   - Minimum: 3-4 layers (quick test)
   - Recommended: 5-7 layers (publication)
   - Large systems: 3-4 may suffice

2. **Vacuum spacing**:
   - Minimum: 15 Å (prevent slab-slab interaction)
   - Recommended: 20 Å (safe for all cases)
   - Check: Plot electron density decay

3. **Symmetry**:
   - Symmetric slabs: Both surfaces identical
   - Asymmetric slabs: Different terminations (needs dipole correction)

4. **K-points**:
   - In-plane: Dense (6×6 or higher for metals)
   - Out-of-plane: 1 (periodic in x,y only)

### Adsorption Sites

1. **Grid resolution**:
   - Coarse scan: 3×3 or 4×4 grid
   - Fine scan: 5×5 or 6×6 grid
   - Ultra-fine: 10×10 grid (computationally expensive)

2. **Height above surface**:
   - Typical: 1.5-2.5 Å
   - Depends on adsorbate size
   - Auto-optimized in relaxation

3. **Adsorbate orientation**:
   - Important for molecules (H2O, CO2, etc.)
   - Use `target_vector` to specify orientation
   - Test multiple orientations for complex molecules

---

## Quick Start

### Surface Energy Calculation

```python
from atomate2.siesta.flows.surface import SurfaceEnergyMaker

# Single surface
surface = SurfaceEnergyMaker(
    miller_indices=(1, 1, 1),
    min_slab_size=10.0,  # Å
    min_vacuum_size=20.0,  # Å
    dry_run=True  # Preview slab structure
)

workflow = surface.make(bulk_structure)

# Generated:
# - Slab structure files
# - Surface energy vs termination
# - Convergence plots (layers, vacuum)
```

### Multi-Surface Comparison

```python
from atomate2.siesta.flows.surface import MultiSurfaceEnergyFlowMaker

# Compare multiple surfaces
multi_surface = MultiSurfaceEnergyFlowMaker(
    miller_indices=[(1,0,0), (1,1,0), (1,1,1)],
    slab_layers=5,
    vacuum_size=20.0,
    dry_run=True
)

workflow = multi_surface.make(bulk_structure)

# Generated:
# - Surface energy comparison plot
# - Wulff construction (equilibrium shape)
# - Summary table
```

### Adsorption Site Scanning

```python
from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker

# Grid-based scanning
scan = AdsorptionScanFlowMaker(
    grid_size=(5, 5),  # 5×5 grid = 25 sites
    height=2.0,  # Å above surface
    placement="top",  # or "bottom"
    dry_run=True  # Preview all sites
)

workflow = scan.make(slab_structure, adsorbate_molecule)

# Generated:
# - Binding energy heatmap
# - Top 5 binding sites
# - Structure files for all sites
# - Energy summary table
```

---

## Common Issues

### Surface Energy Issues

**Issue 1**: "Surface energy is negative"
**Cause**: Bulk energy reference is wrong
**Solution**:
```python
# Use well-converged bulk calculation
# Ensure same parameters (k-points, cutoff, basis)
```

**Issue 2**: "Different terminations have huge energy difference"
**Cause**: Polar surfaces (dipole moment)
**Solution**:
```python
# Use symmetric slabs
# Or add dipole correction
```

**Issue 3**: "Surface energy doesn't converge with layers"
**Cause**: Insufficient slab thickness
**Solution**:
```python
# Increase min_slab_size (12-15 Å)
# Test convergence: 3, 5, 7, 9 layers
```

### Adsorption Issues

**Issue 1**: "Adsorbate falls through slab"
**Cause**: Too close to surface initially
**Solution**:
```python
# Increase initial height (2.5-3.0 Å)
# Check geometry optimization convergence
```

**Issue 2**: "All sites have similar energy"
**Cause**: Grid too coarse, missing site-specific features
**Solution**:
```python
# Increase grid resolution (5×5 → 7×7)
# Check if surface is too symmetric
```

**Issue 3**: "Adsorption energies are all positive (unstable)"
**Cause**: Wrong adsorbate structure or surface termination
**Solution**:
```python
# Check adsorbate geometry (relaxed in vacuum)
# Try different surface terminations
# Verify calculation setup (spin-polarized if needed)
```

**Issue 4**: "Scan takes too long (25+ calculations)"
**Cause**: Large grid size
**Solution**:
```python
# Start with coarse grid (3×3)
# Use dry-run to preview
# Submit to HPC cluster
```

### Tier System Issues

**Issue 5**: "Invalid tier 'relax_dirty' or 'surface_metal'"
**Symptoms**:
```
ValueError: Invalid tier 'relax_dirty'. Must be one of ['basic', 'intermediate',
'advanced', 'expert', 'all', 'dirty', 'ultra']
```

**Cause**: The `tier` parameter currently accepts tier **levels**, not preset **names**.

**Current Workaround**:
```python
# ❌ FAILS: Preset name
flow = MultiSurfaceEnergyFlowMaker(
    tier="relax_dirty"  # Error: "relax_dirty" is a preset name
)

# ✅ WORKS: Use tier level
flow = MultiSurfaceEnergyFlowMaker(
    tier="dirty"  # Valid tier level
)

# ✅ ALSO WORKS: Use tier level equivalent
flow = MultiSurfaceEnergyFlowMaker(
    tier="basic"  # "relax_dirty" preset uses "basic" tier internally
)
```

**Tier Level Guide**:
- `tier="dirty"`: Fast testing (SZ, 50 Ry, 1×1×1 k-points) - ~5-10 min per calculation
- `tier="basic"`: Standard (DZP, 100 Ry, 2×2×2 k-points) - ~20-30 min per calculation
- `tier="intermediate"`: Production (DZP, 300 Ry, 4×4×4 k-points) - ~1-2 hours per calculation

**Future Enhancement**: Automatic preset name resolution and per-maker preset configuration
is planned. This will allow:
```python
# Future (not yet supported):
flow = MultiSurfaceEnergyFlowMaker(
    tier="relax_dirty",  # Auto-resolves to tier level
    maker_presets={
        "bulk_static_maker": "relax_bulk_metal",
        "slab_static_maker": "surface_metal",
    }
)
```

**Note**: This limitation applies to all FlowMakers (MultiSurfaceEnergyFlowMaker,
AdsorptionScanFlowMaker, etc.). See the individual tutorial READMEs for more details.

---

## Best Practices

### Surface Preparation

1. **Relax bulk first**:
   ```python
   # Get equilibrium lattice constant
   # Use for slab construction
   ```

2. **Test slab convergence**:
   ```python
   # Layers: 3, 5, 7, 9
   # Vacuum: 15, 20, 25 Å
   # Use minimal surface first
   ```

3. **Check symmetry**:
   ```python
   # Use symmetric slabs when possible
   # Verify with visualization (VESTA)
   ```

4. **Optimize k-points**:
   ```python
   # Dense in-plane for metals
   # k_z = 1 (surface periodicity)
   ```

### Adsorption Studies

1. **Start with dry-run**:
   ```python
   # Preview all adsorption sites
   # Verify grid spacing makes sense
   ```

2. **Test adsorbate orientation**:
   ```python
   # For molecules: try different orientations
   # Use target_vector parameter
   ```

3. **Incremental complexity**:
   ```python
   # Start: Single atom adsorbate
   # Then: Simple molecule (CO)
   # Finally: Complex molecule (CH3OH)
   ```

4. **Analyze systematically**:
   ```python
   # Plot binding energy heatmap
   # Identify site types (top, bridge, hollow)
   # Compare with literature
   ```

---

## Validation and Analysis

### Surface Energy Validation

1. **Compare with experiments**: Wulff construction vs crystal habit
2. **Compare with literature**: DFT values (within ±10-20%)
3. **Check trends**: Lower-index surfaces typically lower energy

### Adsorption Energy Validation

1. **Compare with TPD**: Temperature-programmed desorption
2. **Compare with literature**: Similar systems
3. **Check physical sense**:
   - Metals: E_ads ~ -0.5 to -3 eV (chemisorption)
   - Physisorption: E_ads ~ -0.1 to -0.5 eV

---

## Common Adsorption Sites

### FCC (111) Surface
- **Top**: Above single surface atom
- **Bridge**: Between two atoms
- **Hollow (FCC)**: Above subsurface atom
- **Hollow (HCP)**: No subsurface atom below

### FCC (100) Surface
- **Top**: Above single atom
- **Bridge**: Between two atoms
- **Hollow**: Four-fold hollow site

### BCC (110) Surface
- **Top**: Above single atom
- **Long bridge**: Along close-packed rows
- **Short bridge**: Perpendicular to rows
- **Hollow**: Pseudo-three-fold

---

## Expected Results

### Surface Energy Calculations
- Surface energy vs termination
- Wulff construction (equilibrium shape)
- Layer convergence plots
- Vacuum convergence plots

### Adsorption Scanning
- Binding energy heatmap (2D map)
- Top 5-10 most stable sites
- Site classification (top/bridge/hollow)
- Energy distribution histogram
- Structure files for all sites

---

## Advanced Topics

### Charged Systems
```python
# Add electrons for negatively charged adsorbates
# Requires: Jellium background or explicit counter-ion
```

### Spin-Polarized Calculations
```python
# Essential for: O2, radical adsorbates, magnetic surfaces
maker = RelaxMaker(user_params={"spin": "polarized"})
```

### Solvation Effects
```python
# Implicit solvation models
# Or explicit water layers
```

---

## Next Steps

After mastering surfaces and adsorption:
- Combine with [05-vibrational-properties](../06-vibrational-properties/) for surface phonons
- Explore [07-advanced-features](../../03-advanced-features/) for workflow automation
- Study reaction mechanisms (NEB on surfaces)

---

*Back to [Main Tutorial Index](../README.md)*
