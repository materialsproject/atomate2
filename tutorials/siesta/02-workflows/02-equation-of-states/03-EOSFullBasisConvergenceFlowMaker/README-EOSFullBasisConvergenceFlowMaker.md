# Tutorial: EOS Full Basis Parameter Optimization

**Category**: 02-equation-of-states
**Difficulty**: Advanced
**Time**: ~5 min (dry-run), ~2-6 hours (full optimization)

---

## Overview

Comprehensive basis parameter optimization using equation of state (EOS) calculations. Tests ALL combinations of basis sizes, PAO.EnergyShift, and PAO.SplitNorm to find optimal parameters for accurate bulk properties.

This tutorial focuses on **complete parameter space exploration**, testing every combination of basis parameters with EOS fitting.

---

## What You'll Learn

- Full basis parameter optimization with EOS
- Testing PAO.EnergyShift and PAO.SplitNorm combinations
- Comparing EOSBasisConvergenceFlowMaker vs EOSFullBasisConvergenceFlowMaker
- Automatic error handling with Custodian for parameter sweeps
- Finding optimal parameters for each basis set
- Bulk modulus convergence with basis parameters

---

## Prerequisites

- **Required**: [01-RelaxMaker](../../../01-basics/01-RelaxMaker/) completed
- **Required**: [01-SiestaEosFlowMaker](../01-SiestaEosFlowMaker/) - Basic EOS workflow
- **Recommended**: [02-EOSBasisConvergenceFlowMaker](../02-EOSBasisConvergenceFlowMakere/) - Basis-only testing
- **Recommended**: Understanding of PAO parameters (PAO.EnergyShift, PAO.SplitNorm)

---

## Key Concepts

### EOSFullBasisConvergenceFlowMaker vs EOSBasisConvergenceFlowMaker

**EOSBasisConvergenceFlowMaker** (Simpler):
- Tests only basis sets with FIXED parameters
- Example: DZ (ES=0.01), DZP (ES=0.01), TZP (ES=0.005)
- Result: 3 EOS calculations
- Use when: You want to compare basis sizes only

**EOSFullBasisConvergenceFlowMaker** (This Tutorial):
- Tests ALL parameter combinations
- Example: 2 basis × 3 ES × 2 SN = 12 EOS calculations
- Result: Optimal parameters for EACH basis set
- Use when: You need to optimize PAO parameters

### PAO.EnergyShift (Energy Shift)

**Purpose**: Controls basis orbital confinement radius

**Units**: Ry (Rydberg)

**Effect**:
- Lower values (0.005-0.01 Ry): Larger orbitals, better accuracy, higher cost
- Higher values (0.02-0.05 Ry): Smaller orbitals, faster, less accurate
- Typical range: 0.01-0.02 Ry

**Convergence**: Test 3-5 values to find optimal balance

### PAO.SplitNorm (Split Norm)

**Purpose**: Threshold for generating multiple-zeta orbitals

**Units**: Dimensionless (0-1)

**Effect**:
- Lower values (0.10-0.15): More aggressive splitting, larger basis
- Higher values (0.20-0.30): Less splitting, smaller basis
- Typical range: 0.15-0.25

**Convergence**: Test 2-3 values per basis size

### Computational Cost

**Example**: 2 basis × 3 energy_shifts × 2 split_norms × 5 volumes
- Total SIESTA runs: 2 × 3 × 2 × 5 = **60 calculations**
- Estimated time: 2-6 hours (system dependent)

---

## Workflow Structure

```
EOSFullBasisConvergenceFlowMaker
├── Initial structure relaxation (optional)
├── Parameter combination 1 (DZ, ES=0.01, SN=0.15)
│   ├── Volume 1 (scale=0.96)
│   ├── Volume 2 (scale=0.98)
│   ├── Volume 3 (scale=1.00)
│   ├── Volume 4 (scale=1.02)
│   ├── Volume 5 (scale=1.04)
│   └── EOS fit (B₀, V₀, B'₀)
├── Parameter combination 2 (DZ, ES=0.01, SN=0.20)
│   └── ... (5 volumes + EOS fit)
├── ... (remaining combinations)
└── Analysis (optimal parameters per basis)
```

---

## Quick Start

### Basic Example

```python
from atomate2.siesta.flows.eos import EOSFullBasisConvergenceFlowMaker
from pymatgen.core import Structure
from jobflow import run_locally

# Load structure
structure = Structure.from_file("Si.cif")

# Create full parameter optimization workflow
flow = EOSFullBasisConvergenceFlowMaker(
    dry_run=False,
    basis_sizes=["DZ", "DZP"],  # Test 2 basis sizes
    energy_shifts=[0.01, 0.015, 0.02],  # Test 3 PAO.EnergyShift values (Ry)
    split_norms=[0.15, 0.20],  # Test 2 PAO.SplitNorm values
    linear_strain=(-0.04, 0.04),  # ±4% volume strain
    number_of_frames=5,  # 5 volume points per EOS
    a2s_kpts=[4, 4, 4],  # K-point grid
)

# Run
workflow = flow.make(structure)
results = run_locally(workflow, create_folders=True)
```

**Total calculations**: 2 × 3 × 2 × 5 = **60 SIESTA runs**

---

## Configuration Options

### Parameter Ranges

```python
# Conservative (fast testing)
basis_sizes=["DZ", "DZP"]
energy_shifts=[0.01, 0.015]  # 2 values
split_norms=[0.15, 0.20]  # 2 values
# Total: 2 × 2 × 2 = 8 combinations

# Standard (recommended)
basis_sizes=["DZ", "DZP", "TZP"]
energy_shifts=[0.01, 0.015, 0.02]  # 3 values
split_norms=[0.15, 0.20]  # 2 values
# Total: 3 × 3 × 2 = 18 combinations

# High accuracy (comprehensive)
basis_sizes=["DZ", "DZP", "TZP"]
energy_shifts=[0.005, 0.01, 0.015, 0.02]  # 4 values
split_norms=[0.10, 0.15, 0.20, 0.25]  # 4 values
# Total: 3 × 4 × 4 = 48 combinations
```

### Volume Sampling

```python
# Fast (fewer volumes)
linear_strain=(-0.03, 0.03)
number_of_frames=5  # 5 volumes

# Standard (recommended)
linear_strain=(-0.04, 0.04)
number_of_frames=7  # 7 volumes

# High accuracy (more volumes)
linear_strain=(-0.06, 0.06)
number_of_frames=9  # 9 volumes
```

### K-Point Settings

```python
# Testing
a2s_kpts=[4, 4, 4]

# Standard
a2s_kpts=[6, 6, 6]

# High accuracy
a2s_kpts=[8, 8, 8]
```

---

## Output

### Per-Combination Results

Each parameter combination produces:
- **EOS plot**: Energy vs volume curve
- **Fitted parameters**: B₀, V₀, B'₀
- **Summary file**: Parameter values and bulk modulus

```
job_eos_DZ_ES0.010_SN0.15_*/
├── eos_plot.png
├── eos_summary.txt
└── eos_data.json
```

### Analysis Output

```
job_full_basis_analysis_*/
├── bulk_modulus_convergence.png  # B₀ vs parameters
├── optimal_parameters.txt        # Best parameters per basis
└── comparison_table.csv          # All combinations compared
```

**optimal_parameters.txt** example:
```
Optimal Parameters by Basis Set:

DZ:
  - PAO.EnergyShift: 0.015 Ry
  - PAO.SplitNorm: 0.20
  - Bulk modulus: 97.3 GPa
  - Converged: Yes

DZP:
  - PAO.EnergyShift: 0.010 Ry
  - PAO.SplitNorm: 0.15
  - Bulk modulus: 98.1 GPa
  - Converged: Yes
```

---

## With Custodian Error Handling

### Why Use Custodian?

When testing many parameter combinations, some may have SCF convergence issues:
- Aggressive PAO.EnergyShift values (< 0.005 Ry)
- Extreme PAO.SplitNorm values (< 0.10 or > 0.30)
- Large parameter sweeps where occasional failures would waste compute time

Custodian provides:
- Automatic SCF convergence rescue (5-level strategies)
- Mixer weight adjustment
- high error recovery rate

### Example with Custodian

```python
flow = EOSFullBasisConvergenceFlowMaker(
    dry_run=False,
    # Basis parameters
    basis_sizes=["DZ", "DZP"],
    energy_shifts=[0.01, 0.015],
    split_norms=[0.15, 0.20],
    # EOS settings
    linear_strain=(-0.04, 0.04),
    number_of_frames=5,
    a2s_kpts=[4, 4, 4],
    # Custodian error handling (propagates automatically)
    use_custodian=True,  # Enable automatic error recovery
    custodian_max_errors=10,  # Allow up to 10 recovery attempts
)
```

**Custodian Propagation**:
- Automatically propagates to `initial_relax_maker` and `eos_relax_maker`
- All child jobs inherit `use_custodian=True`
- No need to configure each maker separately

---

## Best Practices

✅ **Start small**: Test 2×2×2 combinations first (8 EOS calculations)
✅ **Use dry-run**: Verify structure before full run
✅ **Enable custodian**: Especially for aggressive parameter sweeps
✅ **Converge k-points**: Use converged k-point mesh from convergence studies
✅ **Check trends**: Bulk modulus should vary smoothly with parameters
✅ **Compare with basis-only**: Run EOSBasisConvergenceFlowMaker first for baseline

❌ **Don't test too many**: Start with 8-12 combinations, expand if needed
❌ **Don't use unconverged k-points**: Always use converged k-point mesh
❌ **Don't ignore failures**: If custodian can't fix, parameters may be unphysical
❌ **Don't skip initial relaxation**: Use relaxed structure for accurate results

---

## Analysis and Interpretation

### Finding Optimal Parameters

```python
import pandas as pd

# Load comparison table
df = pd.read_csv("job_full_basis_analysis_*/comparison_table.csv")

# Find optimal parameters for each basis
for basis in ["DZ", "DZP", "TZP"]:
    basis_df = df[df['basis'] == basis]

    # Find combination with B₀ closest to target (e.g., 98 GPa for Si)
    target_b0 = 98.0  # GPa
    basis_df['error'] = abs(basis_df['bulk_modulus'] - target_b0)
    optimal = basis_df.loc[basis_df['error'].idxmin()]

    print(f"\nOptimal {basis}:")
    print(f"  ES: {optimal['energy_shift']:.3f} Ry")
    print(f"  SN: {optimal['split_norm']:.2f}")
    print(f"  B₀: {optimal['bulk_modulus']:.1f} GPa")
```

### Convergence Criteria

**Energy Shift Convergence**:
- Bulk modulus change < 2 GPa between successive ES values
- Lattice parameter change < 0.01 Å

**Split Norm Convergence**:
- Bulk modulus change < 1 GPa between successive SN values
- Less sensitive than energy shift

### Validation

```python
# Compare with experimental data
exp_b0 = 98  # GPa for Si
calc_b0 = optimal['bulk_modulus']

agreement = abs(calc_b0 - exp_b0) / exp_b0 * 100
print(f"Agreement with experiment: {100-agreement:.1f}%")
# GGA typically accurate within ~5-10% for bulk modulus
```

---

## Troubleshooting

**Problem**: Many SCF convergence failures

**Solution**:
1. Enable custodian: `use_custodian=True`
2. Reduce parameter range (avoid ES < 0.005 Ry)
3. Check structure is relaxed: Use `initial_relax=True`

---

**Problem**: Bulk modulus varies erratically

**Solution**:
1. Increase volume points: `number_of_frames=9`
2. Use tighter convergence: `DM.Tolerance="1e-6"`
3. Check k-point convergence

---

**Problem**: All combinations give similar results

**Solution**:
- This may be correct! Some materials are insensitive to PAO parameters
- Check wider ES range: `[0.005, 0.01, 0.02, 0.03]`
- Verify with forces/stress convergence

---

**Problem**: Too expensive (taking days)

**Solution**:
1. Reduce combinations: Test 2×2×2 = 8 instead of 3×4×4 = 48
2. Reduce volumes: `number_of_frames=5` instead of 9
3. Coarser k-points for testing: `[4, 4, 4]`
4. Submit to HPC cluster

---

## Tutorial Files

This directory contains four tutorials demonstrating different approaches:

### Tutorial 01: Standard Full Optimization

**`EOSFullBasisConvergenceFlowMaker_01_full.py`**

Basic usage testing all parameter combinations:
```python
from atomate2.siesta.flows.eos import EOSFullBasisConvergenceFlowMaker

maker = EOSFullBasisConvergenceFlowMaker(
    basis_sizes=["DZ", "DZP"],
    energy_shifts=[0.01, 0.015],
    split_norms=[0.15, 0.20],
    linear_strain=(-0.04, 0.04),
    number_of_frames=5,
)
workflow = maker.make(structure)
```

**When to use**: Standard parameter optimization without error handling.

---

### Tutorial 02: With Custodian Error Handling

**`EOSFullBasisConvergenceFlowMaker_02_with_custodian.py`**

Enable automatic SCF convergence recovery:
```python
maker = EOSFullBasisConvergenceFlowMaker(
    basis_sizes=["DZ", "DZP"],
    energy_shifts=[0.01, 0.015],
    split_norms=[0.15, 0.20],
    use_custodian=True,  # Enable automatic error recovery
    custodian_max_errors=10,
)
```

**When to use**:
- Testing aggressive parameter values
- Production runs on HPC clusters
- Large parameter sweeps where failures would waste compute time

---

### Tutorial 03: Customizing with Powerups

**`EOSFullBasisConvergenceFlowMaker_03_powerups.py`**

Use powerups to add parameters after workflow creation:
```python
from atomate2.siesta.flows.eos import EOSFullBasisConvergenceFlowMaker
from atomate2.siesta.powerups import update_user_siesta_settings

# Create workflow with custodian
maker = EOSFullBasisConvergenceFlowMaker(
    basis_sizes=["DZ", "DZP"],
    energy_shifts=[0.01, 0.015],
    split_norms=[0.15, 0.20],
    use_custodian=True,
    custodian_max_errors=10,
)
workflow = maker.make(structure)

# Apply powerups to customize ALL jobs
workflow = update_user_siesta_settings(
    workflow,
    {
        "Mesh.Cutoff": "350 Ry",
        "SCF.Mixer.Weight": 0.05,
        "SCF.DM.Tolerance": 1.0e-5,
    },
)
```

**When to use**:
- Adding Mesh.Cutoff or SCF settings
- Fine-tuning parameters for specific runs
- Modifying existing workflows without recreating makers

---

### Tutorial 04: Using Tier Presets

**`EOSFullBasisConvergenceFlowMaker_04_presets.py`**

Apply production-quality parameter sets via tier presets:
```python
from atomate2.siesta.flows.eos import EOSFullBasisConvergenceFlowMaker
from atomate2.siesta.powerups import update_user_siesta_settings
from atomate2.siesta.sets.tiers import get_tier_preset

# Get preset configuration
preset = get_tier_preset("bulk_semiconductor")

# Create workflow with custodian
maker = EOSFullBasisConvergenceFlowMaker(
    basis_sizes=["DZ", "DZP"],
    energy_shifts=[0.01, 0.015],
    split_norms=[0.15, 0.20],
    use_custodian=True,
    custodian_max_errors=10,
)
workflow = maker.make(structure)

# Apply preset parameters with custom overrides
preset_params = preset["recommended_params"].copy()
preset_params["Mesh.Cutoff"] = "400 Ry"  # Custom override
workflow = update_user_siesta_settings(workflow, preset_params)
```

**When to use**:
- Production calculations requiring validated parameter sets
- Material-specific settings (semiconductors, metals, magnetic systems)
- Reproducible, documented workflows

**Available presets**:
- `relax_standard`: Standard relaxation
- `relax_high_accuracy`: High-accuracy relaxation
- `bulk_semiconductor`: Optimized for bulk semiconductors
- `bulk_metal`: Optimized for bulk metals
- `magnetic_*`: Optimized for magnetic systems

---

## Comparison: Four Methods

| Tutorial | Method | Custodian | Best For |
|----------|--------|-----------|----------|
| **01** | Standard | No | Quick parameter optimization |
| **02** | Custodian | Yes | Robust runs with error recovery |
| **03** | Powerups + Custodian | Yes | Adding Mesh.Cutoff, SCF settings |
| **04** | Presets + Custodian | Yes | Production with validated parameters |

### Recommendation

1. **Start with Tutorial 01** to understand the workflow
2. **Use Tutorial 02** for production runs (always enable custodian)
3. **Use Tutorial 03** for adding custom parameters
4. **Use Tutorial 04** for publication-quality calculations

---

## Next Steps

After completing full basis parameter optimization:
- [04-mechanical/01-ElasticFlowMaker](../../04-mechanical/01-ElasticFlowMaker/) - Use optimized parameters for elastic constants
- [06-vibrational-properties](../../06-vibrational-properties/) - Phonon calculations with optimized basis
- [03-surfaces-and-adsorption](../../03-surfaces-and-adsorption/) - Surface energy with production parameters

---

## Related Tutorials

- [01-SiestaEosFlowMaker](../01-SiestaEosFlowMaker/) - Basic EOS workflow
- [02-EOSBasisConvergenceFlowMaker](../02-EOSBasisConvergenceFlowMakere/) - Basis-only testing
- [EOS Overview](../README.md) - All EOS tutorials

---

**📚 [Back to EOS Workflows](../README.md)** | **📖 [All Tutorials](../../../README.md)**
