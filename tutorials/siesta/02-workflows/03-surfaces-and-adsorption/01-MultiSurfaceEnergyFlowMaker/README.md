# Tutorial: Surface Energy Calculations

**Category**: 06-surfaces-and-adsorption
**Difficulty**: Intermediate
**Time**: ~5 min (dry-run), ~10-15 min (basic), ~1-3 hours (multi-surface)

---

## Overview

Calculate surface energies for all terminations of crystalline surfaces using automatic slab generation. Compare multiple Miller indices to identify the most stable surface and predict equilibrium crystal shape (Wulff construction).

This tutorial consolidates surface energy calculation examples into a single configurable workflow with 3 comprehensive examples.

---

## What You'll Learn

- Surface energy formula: γ = (E_slab - N×E_bulk) / A
- Automatic slab generation and termination discovery
- Multi-surface comparison for Wulff construction
- K-point sampling for 2D periodic systems (slabs)
- Convergence testing (slab thickness, vacuum, k-points)
- Polar surface considerations

---

## Prerequisites

- **Required**: [01-relaxation](../../01-basics/01-relaxation/) completed
- **Recommended**: Understanding of crystal surfaces and Miller indices
- **Recommended**: Basic knowledge of surface science

---

## Key Concepts

### Surface Energy

**Definition**:

$$
\gamma = \frac{E_{\text{slab}} - N \times E_{\text{bulk}}}{A}
$$

**Components**:
- **E_slab**: Total energy of the slab
- **E_bulk**: Energy per formula unit of bulk material
- **N**: Number of formula units in the slab
- **A**: Surface area
  - Asymmetric slabs: A = area (1 surface)
  - Symmetric slabs: A = 2 × area (both surfaces identical)

**Units**:
- eV/Ų (electron-volts per square Angstrom)
- J/m² (Joules per square meter)
- Conversion: 1 eV/Ų = 16.0218 J/m²

**Physical Interpretation**:
- Lower γ → More stable surface
- Energy cost to create surface area
- Determines equilibrium crystal morphology (Wulff construction)
- Varies with Miller index and termination

### Automatic Slab Generation

The workflow uses `MultiSurfaceEnergyMaker` which automatically:
1. Generates slabs for each Miller index using pymatgen
2. Discovers all unique terminations
3. Calculates energies for bulk and all slabs
4. Computes surface energies
5. Generates comparison plots and summaries

**No external slab generation needed** - everything is integrated!

### Symmetric vs Asymmetric Slabs

**Asymmetric Slabs** (`symmetrize=False`):
- Different terminations on top and bottom
- Explores all possible terminations
- Surface energy: γ = (E_slab - N×E_bulk) / A (1 surface)
- May have net dipole moment (polar surfaces)
- More terminations to explore

**Symmetric Slabs** (`symmetrize=True`):
- Same termination on both sides (top = bottom)
- Surface energy: γ = (E_slab - N×E_bulk) / (2×A) (factor of 2!)
- No net dipole (better for polar surfaces)
- Fewer unique terminations
- Cleaner calculation

**Recommendation**: Start with asymmetric (`symmetrize=False`) to explore all terminations, then use symmetric for production calculations of polar surfaces.

### K-Point Sampling for Slabs

Slabs are **2D periodic** (periodic in xy, non-periodic in z):

```python
# Bulk (3D periodic)
bulk_params = {"kpts": [6, 6, 6]}  # Dense in all directions

# Slab (2D periodic)
slab_params = {"kpts": [6, 6, 1]}  # Dense in xy, Γ-point in z
```

**Why Γ-point in z?**
- Large vacuum spacing → no periodicity in z-direction
- Dense sampling only needed in periodic directions (xy)
- Saves computational time significantly
- Critical for accurate slab energies!

---

## Tutorial Examples

### MultiSurfaceEnergyFlowMaker_01_single.py - Single Surface Energy

```bash
python MultiSurfaceEnergyFlowMaker_01_single.py
```

**Features**:
- Calculate energy for a single Miller index: (100)
- Learn workflow basics
- Minimal parameters (fast, ~10-15 min)
- ~2-3 calculations (bulk + terminations)
- Good starting point for testing

**Use Cases**:
- Learn surface energy workflow
- Quick validation before production runs
- Test parameter settings

**Parameters**:
```python
miller_indices=[(1, 0, 0)]     # Single surface
slab_layers=4                   # Number of atomic layers
vacuum_size=15.0                # Vacuum spacing (Å)
```

### MultiSurfaceEnergyFlowMaker_02_multiple.py - Multiple Surface Orientation Comparison

```bash
python MultiSurfaceEnergyFlowMaker_02_multiple.py
```

**Features**:
- Compare multiple surface orientations
- Miller indices: (100), (110), (111)
- Identifies most stable surface
- Production-ready parameters
- ~6-10 calculations (30-60 min)

**Use Cases**:
- Wulff construction (equilibrium crystal shape)
- Surface energy anisotropy studies
- Predict crystal morphology
- Identify catalytically active surfaces

**Parameters**:
```python
miller_indices=[
    (1, 0, 0),  # {100} family - cubic faces
    (1, 1, 0),  # {110} family - rectangular faces
    (1, 1, 1),  # {111} family - triangular faces
]
slab_layers=4                   # Increase for convergence
vacuum_size=15.0                # Vacuum spacing (Å)
symmetrize=False                # Explore all terminations
plot_results=True               # Generate comparison plots
write_summary=True              # Write text summary
```

**Output**:
- Surface energies for each orientation (eV/Ų)
- Comparison plots and rankings
- Most stable surface identification

### MultiSurfaceEnergyFlowMaker_03_with_custodian.py - Automatic Error Handling

```bash
python MultiSurfaceEnergyFlowMaker_03_with_custodian.py
```

**Features**:
- Automatic SCF convergence recovery
- Geometry optimization error handling
- Progressive correction strategies
- Production-ready with robustness

**Flow-Level Custodian**:
```python
flow = MultiSurfaceEnergyFlowMaker(
    miller_indices=[(1, 1, 1)],
    use_custodian=True,         # Enable automatic error handling
    custodian_max_errors=10,    # Allow up to 10 corrections
    tier="dirty",               # Fast tier for testing
)
```

**Benefits**:
- Custodian settings automatically propagate to bulk AND slab makers
- No need to configure each maker individually
- high automatic recovery rate
- Saves time on difficult calculations

**Important Note on Tier Parameter**:
```python
# ✅ CORRECT: Use tier level
tier="dirty"           # Valid tier level

# ✅ ALSO CORRECT: Use preset name (auto-resolved to tier level)
tier="relax_dirty"     # Preset name → extracts "basic" tier

# ❌ WRONG: This will fail
tier="surface_metal"   # Not yet supported (planned for future)
```

**Current Limitation**: The `tier` parameter accepts tier levels (`"basic"`, `"intermediate"`, `"advanced"`, `"expert"`, `"dirty"`, `"ultra"`) and will auto-resolve preset names to their tier levels. Per-maker preset configuration is planned for a future release.

### MultiSurfaceEnergyFlowMaker_04_with_preset.py - Using Tier Presets

```bash
python MultiSurfaceEnergyFlowMaker_04_with_preset.py
```

**Features**:
- Demonstrate tier preset system
- Apply optimized parameters automatically
- Compare different tier levels

**Tier Levels**:
```python
# Testing (very fast, ~5-10 min)
tier="dirty"           # Minimal: SZ basis, 50 Ry cutoff, 1×1×1 k-points

# Standard (production, ~30-60 min)
tier="basic"           # Basic: DZP basis, 100 Ry cutoff, 2×2×2 k-points

# High accuracy (publication, ~2-4 hours)
tier="intermediate"    # Intermediate: DZP basis, 300 Ry cutoff, 4×4×4 k-points
```

**Future Enhancement**: Per-maker presets will allow different settings for bulk vs slab:
```python
# Planned for future release:
maker_presets={
    "bulk_static_maker": "relax_bulk_metal",
    "slab_static_maker": "surface_metal",
}
```

---

## Quick Start

```bash
# 1. Start with single surface (recommended)
# Edit MultiSurfaceEnergyFlowMaker_01_single.py: uncomment dry_run=True to preview first
python MultiSurfaceEnergyFlowMaker_01_single.py

# 2. Inspect dry-run output (if enabled)
ls dry_run_output/job_*/
# Should see: 1 bulk + ~2 slab calculations with SIESTA input files

# 3. Run actual calculation (WARNING: ~10-15 minutes)
# Edit MultiSurfaceEnergyFlowMaker_01_single.py: comment out dry_run=True
python MultiSurfaceEnergyFlowMaker_01_single.py

# 4. Compare multiple surfaces (WARNING: ~30-60 minutes)
# Edit MultiSurfaceEnergyFlowMaker_02_multiple.py: adjust miller_indices if needed
python MultiSurfaceEnergyFlowMaker_02_multiple.py
```

---

## Expected Output

### Dry-Run Mode

```
✅ Dry-run complete!

📁 Preview files in: preview_output/job_*/inputs/

💡 Workflow structure:
  • Bulk calculation
  • Slab calculations (all terminations discovered automatically)
  • Surface energy analysis

📋 Next steps:
  1. Check workflow structure
  2. Verify slab generation
  3. Set RUN_MODE = 'local'
```

### Local Mode (Basic Example)

```
✅ Calculated 1 surface(s)

  Surface    Lowest γ (eV/Ų)   Lowest γ (J/m²)   Best Term.      # Terms
  --------------------------------------------------------------------------------
  (0 0 1)            0.0524           0.84        Mg              2

OUTPUT FILES:
  • surface_energy_001.png         - Termination comparison plot
  • surface_energy_001_summary.txt - Detailed analysis
  • job_*/                         - SIESTA calculation directories

WORKFLOW COMPLETE!

📊 Recommended next steps:
   1. Inspect surface_energy_001.png
   2. Read surface_energy_001_summary.txt
   3. Compare with literature (~1.0-1.2 J/m² for MgO (001))
```

### Local Mode (Multi-Surface Example)

```
✅ Calculated 7 surfaces

  Surface    Lowest γ (eV/Ų)   Lowest γ (J/m²)   Best Term.      # Terms
  --------------------------------------------------------------------------------
  (1 0 0)            0.0524           0.84        Mg              2
  (0 1 0)            0.0524           0.84        Mg              2
  (0 0 1)            0.0524           0.84        Mg              2
  (1 1 0)            0.0498           0.80        Mixed           3
  (1 0 1)            0.0498           0.80        Mixed           3
  (0 1 1)            0.0498           0.80        Mixed           3
  (1 1 1)            0.0875           1.40        Mg              2

================================================================================
GLOBAL MINIMUM (Most Stable Surface)
================================================================================
  Surface:         (1 1 0)
  Termination:     Mixed
  Surface energy:  0.0498 eV/Ų (0.80 J/m²)

OUTPUT FILES:
  • surface_energy_*.png         - Comparison plots (one per Miller index)
  • surface_energy_*_summary.txt - Detailed summaries
  • multi_surface_summary.txt    - Global comparison (all surfaces)
  • job_*/                       - SIESTA calculation directories

📊 Use data for Wulff construction to predict equilibrium crystal shape
```

---

## MgO (001) Test Case

MgO (001) is a classic test case for surface energy calculations:

**Structure**:
- Rock salt structure (Fm-3m)
- Lattice parameter: a = 4.257 Å
- (001) surface: alternating Mg²⁺ and O²⁻ layers

**Surface Properties**:
- **Polar surface** (Type 3 in Tasker classification)
- Two terminations: Mg-terminated and O-terminated
- Different energies due to polarity

**Literature Values**:
- Experimental: ~1.0-1.2 J/m² (Tasker, 1979)
- DFT (PW91): ~1.1 J/m² (de Leeuw et al., 1999)
- DFT (PBE): ~1.2 J/m² (Hernandez et al., 2011)

**Expected Results**:
- MgO (001): ~1.0 J/m² (may vary with SIESTA functional)
- MgO (110): ~0.8 J/m² (more stable)
- MgO (111): ~1.5 J/m² (less stable, polar)

---

## Analyzing Results

### Typical Surface Energy Ranges

**By Material Type**:
- Ionic crystals (NaCl, MgO): 0.5-2.0 J/m²
- Metals (Al, Cu, Au): 1.0-3.0 J/m²
- Semiconductors (Si, GaAs): 1.5-4.0 J/m²
- Hard materials (Diamond, SiC): 3.0-10.0 J/m²

**By Miller Index** (for cubic systems):
- Low-index (100), (110), (111): Typically most stable
- High-index (211), (311), etc.: Higher energy (stepped surfaces)
- General trend: γ(110) < γ(100) < γ(111) for many FCC metals

### Validation Checks

✓ **All γ positive**: Negative surface energy indicates error!
✓ **Symmetry consistency**: γ(100) ≈ γ(010) ≈ γ(001) for cubic systems
✓ **Literature agreement**: Compare with experimental/DFT values
✓ **Convergence**: γ changes < 1% when increasing parameters
✓ **Physical trends**: Lower index → lower energy (generally)

### Surface Energy Interpretation

```
γ < 0.5 J/m²:   Very stable (likely calculation error if not metal)
0.5-2.0 J/m²:   Normal range for most materials
2.0-5.0 J/m²:   High energy (polar surfaces, high-index)
> 5.0 J/m²:     Very high (check convergence!)
```

---

## Convergence Testing

Always test convergence of:

### 1. Slab Thickness (Number of Layers)

**Test**: 4, 6, 8, 10 layers

**Criterion**: γ changes by < 0.01 eV/Ų (< 0.16 J/m²)

**Why it matters**: Too thin → bulk region not established, surface atoms interact through slab

**How to test**:
```python
# Run convergence example with different test_layers values
test_layers = 4   # First run
test_layers = 6   # Second run
test_layers = 8   # Third run
# Compare surface energies
```

**Typical convergence**:
- Metals: 5-7 layers
- Ionic crystals: 4-6 layers
- Covalent materials: 6-10 layers

### 2. Vacuum Spacing

**Test**: 15, 20, 25 Å

**Criterion**: E_slab changes by < 0.001 eV/atom

**Why it matters**: Insufficient vacuum → slabs interact between periodic images

**How to test**:
```python
# Modify vacuum_size in workflow configuration
vacuum_size = 15.0  # First run
vacuum_size = 20.0  # Second run
vacuum_size = 25.0  # Third run
```

**Recommendation**: Use 20 Å for production calculations

### 3. K-Point Sampling (In-Plane)

**Test**: [4,4,1], [6,6,1], [8,8,1], [10,10,1]

**Criterion**: γ changes by < 0.005 eV/Ų (< 0.08 J/m²)

**Why it matters**: Surface electronic structure may require denser sampling than bulk

**How to test**:
```python
# Modify slab_params['kpts']
slab_params = {"kpts": [4, 4, 1]}  # First run
slab_params = {"kpts": [6, 6, 1]}  # Second run
slab_params = {"kpts": [8, 8, 1]}  # Third run
```

**Recommendation**: [6,6,1] standard, [8,8,1] high accuracy

### 4. Mesh Cutoff

**Test**: 300, 400, 500 Ry

**Criterion**: Same as bulk convergence

**Important**: Use **same cutoff** for bulk and slab calculations!

### 5. Basis Set Parameters

**PAO.EnergyShift**: 0.01 Ry (standard), 0.005 Ry (high accuracy)
**PAO.BasisSize**: DZP (minimum), TZP (high accuracy)

**Criterion**: Same as bulk convergence

**Important**: Use **same basis** for bulk and slab!

---

## Common Issues

### Issue 1: Negative Surface Energies

**Symptoms**: γ < 0 (physically impossible!)

**Causes**:
1. Bulk energy too high or slab energy too low
2. Different SIESTA parameters for bulk vs slab
3. Poor SCF convergence
4. Wrong formula_units_per_cell

**Solutions**:
1. Use **identical** SIESTA parameters for bulk and slab (except k-points!)
2. Check SCF convergence: grep "SCF Convergence" job_*/output
3. Verify formula_units_per_cell matches bulk structure
4. Tighten DM.Tolerance (1.0e-5 or smaller)
5. Increase MaxSCFIterations (200 → 500)

### Issue 2: Very Different Energies for Similar Terminations

**Symptoms**: Large γ spread for same Miller index

**Causes**:
1. Slab too thin (surface atoms interact)
2. Vacuum too small (slabs interact)
3. Polar surface with dipole issues

**Solutions**:
1. Increase slab thickness: 4 → 6 → 8 layers
2. Increase vacuum: 15 → 20 → 25 Å
3. For polar surfaces, use symmetric slabs: `symmetrize=True`
4. Consider dipole corrections (SIESTA: SlabDipoleCorrection)

### Issue 3: Surface Energy Doesn't Converge with Thickness

**Symptoms**: γ keeps changing significantly with more layers

**Causes**:
1. Bulk region not established
2. Polar surface (net dipole)
3. Surface reconstruction

**Solutions**:
1. Use more layers (8-10 minimum)
2. For polar surfaces, use symmetric slabs
3. Consider adding surface relaxation (slab_relax_maker)
4. Check if surface reconstructs (compare relaxed vs unrelaxed)

### Issue 4: SCF Not Converging for Slabs

**Symptoms**: SCF cycles exceed MaxSCFIterations

**Causes**:
1. Electronic structure more complex than bulk
2. Polar surface (difficult charge distribution)
3. Poor initial guess

**Solutions**:
1. Increase MaxSCFIterations: 200 → 500
2. Add electronic temperature:
   ```python
   "OccupationFunction": "MP",
   "ElectronicTemperature": "1000 K",
   ```
3. Adjust mixer:
   ```python
   "SCF.Mixer.Weight": 0.005,
   "SCF.Mixer.Method": "Pulay",
   "SCF.Mixer.History": 6,
   ```
4. For difficult cases:
   ```python
   "SCF.Mixer.Kick": 50,
   "SCF.Mixer.Kick.Weight": 0.05,
   ```

### Issue 5: "No terminations found" Error

**Symptoms**: Workflow fails during slab generation

**Causes**:
1. Invalid Miller index for the structure
2. Slab generation parameters too restrictive
3. Structure format issue

**Solutions**:
1. Verify Miller index is compatible with lattice
2. Try different slab_layers values
3. Check structure file is valid (view in VESTA)
4. Try `symmetrize=True` for simple cases

### Issue 6: "Invalid tier 'relax_dirty'" Error

**Symptoms**:
```
ValueError: Invalid tier 'relax_dirty'. Must be one of ['basic', 'intermediate',
'advanced', 'expert', 'all', 'dirty', 'ultra']
```

**Cause**: The `tier` parameter currently only accepts tier **levels**, not preset **names**.

**Current Workaround**:
```python
# ❌ FAILS: Preset name not directly supported yet
tier="relax_dirty"

# ✅ WORKS: Use the tier level instead
tier="dirty"  # Equivalent to relax_dirty preset

# Or use the tier level from any preset:
# "relax_dirty" preset uses tier="basic" internally
tier="basic"
```

**Tier Level to Preset Mapping**:
- `tier="dirty"` ≈ relax_dirty preset (minimal settings)
- `tier="basic"` ≈ standard presets (DZP, 100 Ry)
- `tier="intermediate"` ≈ high-accuracy presets (DZP, 300 Ry)

**Future Enhancement**: Automatic preset name resolution and per-maker presets planned.

---

## Best Practices

### Workflow Design

**Progressive Testing**:
1. Dry-run first with basic example
2. Run basic example to verify workflow
3. Test convergence systematically
4. Production run with converged parameters
5. Multi-surface comparison for complete analysis

**Parameter Consistency**:
- ✓ Same SIESTA parameters for bulk and slab (except k-points!)
- ✓ Same basis set, cutoff, DM.Tolerance, etc.
- ✓ Only difference: slab uses [n,n,1] k-points

### Parameter Selection

**Testing** (quick validation, ~10-15 min):
- PAO.BasisSize: DZP
- Mesh.Cutoff: 100 Ry
- K-points: [2,2,2] bulk, [2,2,1] slab
- Slab layers: 4
- Vacuum: 15 Å

**Standard** (production, ~30 min - 1 hour per surface):
- PAO.BasisSize: DZP
- Mesh.Cutoff: 300 Ry
- K-points: [6,6,6] bulk, [6,6,1] slab
- Slab layers: 5-6
- Vacuum: 20 Å

**High Accuracy** (publication, ~2-4 hours per surface):
- PAO.BasisSize: TZP
- Mesh.Cutoff: 500 Ry
- K-points: [8,8,8] bulk, [8,8,1] slab
- Slab layers: 8-10
- Vacuum: 25 Å

### Slab Generation

**Asymmetric vs Symmetric**:
- Start with `symmetrize=False` to discover all terminations
- Use `symmetrize=True` for polar surfaces (production)
- Asymmetric good for exploration, symmetric for accuracy

**Slab Thickness**:
- Minimum: 4 layers (testing only)
- Standard: 5-6 layers
- High accuracy: 8-10 layers
- Always test convergence!

**Vacuum Spacing**:
- Minimum: 15 Å
- Recommended: 20 Å
- High accuracy: 25 Å
- Check convergence with vacuum test

---

## Advanced Topics

### Wulff Construction

**Purpose**: Predict equilibrium crystal shape from surface energies

**Method**:
1. Calculate γ for all relevant Miller indices (multi_surface example)
2. For each surface, construct plane at distance γ from origin
3. Inner envelope of planes = equilibrium shape

**Implementation** (pymatgen):
```python
from pymatgen.analysis.wulff import WulffShape

# Collect surface energies from multi_surface results
miller_indices = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,1,1)]
surface_energies = [0.84, 0.84, 0.84, 0.80, 1.40]  # Example (J/m²)

# Create Wulff shape
wulff = WulffShape(structure.lattice, miller_indices, surface_energies)

# Analyze
print(f"Surface area: {wulff.surface_area:.2f} Ų")
print(f"Volume: {wulff.volume:.2f} ų")
print(f"Effective radius: {wulff.effective_radius:.2f} Å")

# Visualize
wulff.show()
```

### Surface Relaxation

**Why Needed**: Surface atoms relax to minimize energy (different from bulk positions)

**Implementation**:
```python
from atomate2.siesta.jobs.core import RelaxMaker

# Create slab relaxation maker
slab_relax_maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "PAO.BasisSize": "DZP",
        "Mesh.Cutoff": "300 Ry",
        "kpts": [6, 6, 1],
        # Add constraints to fix bottom layers if desired
    }
)

# Use in workflow
workflow = MultiSurfaceEnergyMaker(
    ...,
    slab_relax_maker=slab_relax_maker,  # Enable relaxation
)
```

**Important**: Relaxation increases computation time significantly!

### Dipole Corrections

**When Needed**: Polar surfaces with asymmetric slabs

**SIESTA Implementation**:
```python
slab_params = {
    ...,
    "SlabDipoleCorrection": True,  # Enable dipole correction
}
```

**Effect**: Removes artificial electric field from net dipole moment

### High-Pressure Surfaces

**Motivation**: Surface stability changes under pressure

**Implementation**:
```python
# Calculate bulk energy at different pressures (EOS)
# Calculate surface energies at each pressure
# Track γ(P) for each Miller index
# Predict pressure-induced surface reconstructions
```

---

## Tips for Success

✅ **Start with dry_run**: Verify workflow structure before expensive calculations

✅ **Test basic example first**: Ensure everything works before multi-surface

✅ **Same parameters for bulk and slab**: Except k-points ([n,n,n] vs [n,n,1])

✅ **Slab k-points critical**: Dense in-plane [6,6,1], Γ-point in z

✅ **Test convergence systematically**: Thickness, vacuum, k-points

✅ **Compare with literature**: Validate against experimental/DFT values

✅ **Use multi-surface for Wulff**: Need all Miller indices for equilibrium shape

✅ **Polar surfaces need care**: Use symmetric slabs or dipole corrections

✅ **Check SCF convergence**: Slabs may need more iterations than bulk

✅ **Document parameters**: Record all settings for reproducibility

---

## Next Steps

After completing this tutorial:

1. **Basic Workflow**:
   → Run basic example with dry_run
   → Execute basic example locally
   → Analyze output plots and summaries
   → Compare with literature values

2. **Convergence Testing**:
   → Test slab thickness (convergence example)
   → Test vacuum spacing
   → Test k-point mesh (in-plane)
   → Establish converged parameters

3. **Multi-Surface Analysis**:
   → Run multi_surface example
   → Identify most stable surface
   → Construct Wulff shape
   → Predict crystal morphology

4. **Advanced Applications**:
   → Add surface relaxation
   → Calculate work functions
   → Study adsorption (tutorial 02-adsorption-scanning)
   → Surface reconstruction analysis
   → Pressure-dependent surface stability

5. **Other Materials**:
   → Metals (Cu, Au, Pt)
   → Semiconductors (Si, GaAs, ZnO)
   → 2D materials (graphene, MoS₂)
   → Ionic crystals (NaCl, LiF)

---

## References

1. **Tasker, P. W.** (1979). "The stability of ionic crystal surfaces." *J. Phys. C: Solid State Phys.* 12, 4977. [DOI:10.1088/0022-3719/12/22/036](https://doi.org/10.1088/0022-3719/12/22/036)

2. **Fiorentini, V. & Methfessel, M.** (1996). "Extracting convergent surface energies from slab calculations." *J. Phys.: Condens. Matter* 8, 6525. [DOI:10.1088/0953-8984/8/36/005](https://doi.org/10.1088/0953-8984/8/36/005)

3. **de Leeuw, N. H. et al.** (1999). "Surface structure and morphology of magnesium oxide." *J. Phys. Chem. B* 103, 1270. [DOI:10.1021/jp983239z](https://doi.org/10.1021/jp983239z)

4. **Sun, W. et al.** (2016). "The thermodynamic scale of inorganic crystalline metastability." *Sci. Adv.* 2, e1600225. [DOI:10.1126/sciadv.1600225](https://doi.org/10.1126/sciadv.1600225)

5. **Stekolnikov, A. A. et al.** (2002). "Absolute surface energies of group-IV semiconductors: Dependence on orientation and reconstruction." *Phys. Rev. B* 65, 115318. [DOI:10.1103/PhysRevB.65.115318](https://doi.org/10.1103/PhysRevB.65.115318)

6. **Wulff, G.** (1901). "Zur Frage der Geschwindigkeit des Wachstums und der Auflösung der Kristallflächen." *Z. Kristallogr.* 34, 449.

---

*Back to [06-surfaces-and-adsorption](../README.md) | [Main Tutorial Index](../../README.md)*
