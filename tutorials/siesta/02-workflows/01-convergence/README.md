# 02: Convergence Studies

**Focus**: Systematic parameter optimization for production-quality calculations

**Difficulty**: Intermediate

**Prerequisites**:
- Completed [01-basics](../../01-basics/) tutorials
- Understanding of DFT convergence concepts
- Familiarity with k-points, mesh cutoff, and basis sets

---

## Tutorials in This Category

### 1. [KpointsConvergenceFlowMaker](KpointsConvergenceFlowMaker/)
**Description**: K-point mesh convergence testing for Brillouin zone sampling
**Difficulty**: Beginner
**Time**: ~5 min (dry-run), ~10-20 min (full calculation)
**Key Concepts**: K-point sampling, material dependence (metals vs insulators), anisotropic meshes
**📖 [Full Tutorial Guide →](KpointsConvergenceFlowMaker/README.md)**

### 2. [MeshCutoffConvergenceFlowMaker](MeshCutoffConvergenceFlowMaker/)
**Description**: Mesh cutoff convergence testing for real-space grid optimization
**Difficulty**: Beginner
**Time**: ~5 min (dry-run), ~10-15 min (full calculation)
**Key Concepts**: Real-space grids, pseudopotential dependence, monotonic convergence
**📖 [Full Tutorial Guide →](MeshCutoffConvergenceFlowMaker/README.md)**

### 3. [MeshKpointConvergenceFlowMaker](MeshKpointConvergenceFlowMaker/)
**Description**: Combined two-stage convergence (mesh cutoff → k-points) with early stopping
**Difficulty**: Intermediate
**Time**: ~10 min (dry-run), ~30-60 min (full calculation)
**Key Concepts**: Two-stage workflow, early stopping, multi-property criteria, many time savings
**📖 [Full Tutorial Guide →](MeshKpointConvergenceFlowMaker/README.md)**

### 4. [BasisParametersConvergenceFlowMaker](BasisParametersConvergenceFlowMaker/)
**Description**: PAO basis parameter convergence (EnergyShift + SplitNorm) with timing analysis
**Difficulty**: Intermediate
**Time**: ~5 min (dry-run), ~20 min (simple grid), ~1-2 hours (full grid)
**Key Concepts**: PAO.EnergyShift, PAO.SplitNorm, basis size trade-offs, custodian integration
**📖 [Full Tutorial Guide →](BasisParametersConvergenceFlowMaker/README.md)**

---

## Learning Path

**Important**: Convergence studies are essential before production calculations!

### Recommended Order

1. **Start here**: [MeshCutoffConvergenceFlowMaker](MeshCutoffConvergenceFlowMaker/) - Real-space grid convergence (MUST do first!)
2. **Next**: [KpointsConvergenceFlowMaker](KpointsConvergenceFlowMaker/) - K-point sampling (use converged mesh cutoff)
3. **Verify together**: [MeshKpointConvergenceFlowMaker](MeshKpointConvergenceFlowMaker/) - Two-stage combined convergence with early stopping
4. **Fine-tune**: [BasisParametersConvergenceFlowMaker](BasisParametersConvergenceFlowMaker/) - PAO basis parameters (optional but recommended)

---

## Why Convergence Testing?

Convergence testing ensures your results are:
- ✅ **Reliable**: Independent of numerical parameters
- ✅ **Reproducible**: Consistent across different setups
- ✅ **Efficient**: Balanced accuracy vs computational cost
- ✅ **Publishable**: Meeting scientific standards

**Rule of thumb**:
- Energy convergence: < 1 meV/atom
- Force convergence: < 0.01 eV/Å

---

## Quick Start

All convergence tutorials generate automatic plots and summaries:

```python
# Run in dry-run mode to preview test points
RUN_MODE = "dry_run"

# After verifying, run full convergence study
# RUN_MODE = "local"

# Results:
# - convergence_plot.png (automatic)
# - convergence_summary.txt (automatic)
# - convergence_results.json (for further analysis)
```

---

## Detailed Tutorial Guides

Each convergence workflow has its own comprehensive README with complete documentation:

### 📘 [KpointsConvergenceFlowMaker Guide](KpointsConvergenceFlowMaker/README.md)
**Complete guide to k-point mesh convergence** (250+ lines)
- K-point sampling theory and material dependence
- Workflow structure with quick start examples
- Common k-point ranges for different materials (metals, insulators, 2D)
- Anisotropic k-meshes for layered materials
- Troubleshooting and best practices

### 📗 [MeshCutoffConvergenceFlowMaker Guide](MeshCutoffConvergenceFlowMaker/README.md)
**Complete guide to mesh cutoff convergence** (250+ lines)
- Real-space grid fineness optimization
- Mesh cutoff ranges by material type
- Pseudopotential-specific guidelines
- Interpreting energy vs cutoff plots
- Common issues and solutions

### 📙 [MeshKpointConvergenceFlowMaker Guide](MeshKpointConvergenceFlowMaker/README.md)
**Complete guide to combined two-stage convergence** (300+ lines)
- Two-stage workflow (mesh cutoff → k-points)
- Early stopping feature (saves many computational time)
- Multi-property convergence criteria (energy, Fermi, forces, stress, bandgap)
- Stage-by-stage analysis with 12 automatic plots
- Advanced configurations for different material types

### 📕 [BasisParametersConvergenceFlowMaker Guide](BasisParametersConvergenceFlowMaker/README.md)
**Complete guide to PAO basis parameter convergence** (400+ lines)
- PAO.EnergyShift and PAO.SplitNorm optimization
- Basis size trade-offs and material-specific guidelines
- Si vs MgO comparison (why custodian is needed for ionic systems)
- Timing analysis and performance optimization
- Custodian integration for automatic error recovery

---

## Quick Reference Summary

### K-Points
- **Metals**: 6×6×6 to 12×12×12 (dense k-meshes, Fermi surface sampling critical)
- **Insulators/Semiconductors**: 4×4×4 to 8×8×8 (sparser meshes acceptable)
- **2D Materials**: Dense in-plane, sparse out-of-plane (e.g., 8×8×1)

### Mesh Cutoff
- **Typical range**: 200-400 Ry
- **Soft elements (C, Si, O, N)**: 200-300 Ry
- **Transition metals**: 300-400 Ry
- **Hard pseudopotentials**: 400-500+ Ry

### Basis Parameters (PAO)
- **PAO.EnergyShift**: 0.005-0.02 Ry (lower = larger basis, more accurate)
- **PAO.SplitNorm**: 0.10-0.30 (lower = more splitting, larger basis)
- **Typical**: EnergyShift=0.01 Ry, SplitNorm=0.15 (standard quality)

---

## Common Issues

### Issue 1: "Convergence tests take too long"
**Solution**:
- Start with dry-run mode to verify test points
- Use coarser initial sampling
- Run on HPC cluster (tutorial 04-infrastructure/02-job-submission)

### Issue 2: "Results don't converge"
**Solution**:
- Check if system is metallic (needs denser k-points)
- Try different basis set (DZP → TZP)
- Increase mesh cutoff range

### Issue 3: "How many test points?"
**Solution**:
- K-points: 5-7 test points (e.g., 2, 4, 6, 8, 10, 12, 14)
- Mesh cutoff: 6-8 test points (100-400 Ry)
- Basis parameters: 4-6 test points each

### Issue 4: "Unknown FDF parameter: fdf_arguments"
**Error**: `ValueError: Unknown FDF parameter(s): fdf_arguments`

**Cause**: Using deprecated `fdf_arguments` wrapper syntax

**Solution**:
```python
# ❌ OLD (doesn't work):
user_params = {
    "fdf_arguments": {
        "BandLines": [...]
    }
}

# ✅ NEW (correct):
user_params = {
    "%block BandLines": [...]  # Directly in user_params!
}
```

**Note**: Block parameters should be specified **directly** in `user_params`, NOT nested in `fdf_arguments`. See [FDF Block Parameters](#fdf-block-parameters-advanced) section below.

---

## Best Practices

1. **Always start with dry-run**: Verify test points before running
2. **One parameter at a time**: Don't vary k-points AND cutoff simultaneously
3. **Use reference structure**: Well-relaxed geometry
4. **Save results**: Keep convergence plots for your paper
5. **Document choices**: Note converged parameters in your workflow

---

## FDF Block Parameters (Advanced)

When you need to specify FDF block parameters in convergence workflows, use the `"%block ParamName"` syntax **directly** in `user_params`.

**IMPORTANT**: DO NOT wrap block parameters in `fdf_arguments` - this is deprecated!

### Correct Usage

```python
# ✅ CORRECT: Block parameters directly in user_params
from atomate2.siesta.flows.convergence import MeshCutoffConvergenceFlowMaker

maker = MeshCutoffConvergenceFlowMaker(
    cutoff_range=[100, 150, 200, 250, 300],
    user_params={
        "a2s_kpts": [6, 6, 6],

        # Custom band path for electronic structure
        "%block BandLines": [
            "1  0.0 0.0 0.0  L  # Gamma",
            "20 0.5 0.0 0.5  X",
            "20 0.5 0.25 0.75 W",
            "1  0.5 0.5 0.5  # L",
        ],
    },
)
```

### Incorrect Usage (Deprecated)

```python
# ❌ WRONG: Don't nest in fdf_arguments!
maker = MeshCutoffConvergenceFlowMaker(
    cutoff_range=[100, 150, 200, 250, 300],
    user_params={
        "fdf_arguments": {  # <-- This doesn't work!
            "BandLines": [...]
        }
    },
)
```

### Common Block Parameters for Convergence

- `"%block BandLines"` - K-path for band structure convergence
- `"%block kgrid.MonkhorstPack.Offset"` - Custom k-point offsets
- `"%block Geometry.Constraints"` - Fix atoms during convergence tests

For comprehensive examples, see [Advanced Features: FDF Block Inputs](../../03-advanced-features/01-parameter-systems/04-fdf-blocks/).

---

## Next Category

After completing convergence studies, proceed to:
- **[03-advanced-workflows](../../02-workflows/)** - Apply converged parameters to production workflows
- **[04-infrastructure](../../03-advanced-features/03-infrastructure/)** - Set up for large-scale calculations

---

*Back to [Main Tutorial Index](../README.md)*
