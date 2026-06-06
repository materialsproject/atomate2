# 01-Basics: Fundamental Makers and Single-Job Calculations

**Focus**: Core SIESTA Makers for single-calculation workflows

**Difficulty**: Beginner

**Prerequisites**:
- Python 3.9+
- SIESTA installed and configured
- Basic understanding of DFT calculations
- `~/.atomate2siesta.yaml` configuration file

---

## Tutorials in This Category

### [01-RelaxMaker](01-RelaxMaker/)
**Description**: Structural relaxation with fixed and variable cell options
**Difficulty**: Beginner
**Time**: ~2 min (dry-run), ~10-15 min (full calculation)
**Key Concepts**: RelaxMaker, fixed_cell_relaxation(), variable_cell_relaxation(), jobflow basics
**Files**: 8 tutorial scripts covering basic to advanced relaxation workflows

### [02-BandStructureMaker](02-BandStructureMaker/)
**Description**: Electronic band structure calculations with automatic k-path generation
**Difficulty**: Beginner
**Time**: ~2 min (dry-run), ~15-20 min (full calculation)
**Key Concepts**: BandStructureMaker, high-symmetry k-paths, electronic structure
**Files**: Multiple examples for different material types

### [03-LuaMaker](03-LuaMaker/)
**Description**: Lua scripting for advanced SIESTA features (FLOS optimization, geometry constraints)
**Difficulty**: Intermediate
**Time**: ~2 min (dry-run), ~15-20 min (full calculation)
**Key Concepts**: Lua scripting, FLOS library, geometry constraints, MD.TypeOfRun
**Files**: Basic Lua integration examples

### [04-RelaxMaker-StaticMaker](04-RelaxMaker-StaticMaker/)
**Description**: Multi-step workflows combining relaxation and static calculations
**Difficulty**: Beginner
**Time**: ~2 min (dry-run), ~20-30 min (full calculation)
**Key Concepts**: jobflow Flow, job chaining, relax → static workflows
**Files**: Sequential workflow examples with job dependencies

### [05-DOSMaker](05-DOSMaker/)
**Description**: Density of states (DOS) calculations
**Difficulty**: Beginner
**Time**: ~2 min (dry-run), ~10-15 min (full calculation)
**Key Concepts**: DOSMaker, electronic density of states, Fermi energy
**Files**: Basic DOS calculation examples

### [06-PDOSMaker](06-PDOSMaker/)
**Description**: Projected density of states (PDOS) calculations
**Difficulty**: Beginner
**Time**: ~2 min (dry-run), ~10-15 min (full calculation)
**Key Concepts**: PDOSMaker, orbital-projected DOS, atomic decomposition
**Files**: PDOS calculation with orbital analysis

### [07-PhonopyMaker](07-PhonopyMaker/)
**Description**: Phonopy-based phonon calculations with finite differences
**Difficulty**: Intermediate
**Time**: ~5 min (dry-run), ~1-2 hours (full calculation)
**Key Concepts**: PhonopyMaker, supercell generation, force constants, automatic plotting
**Files**: Basic phonopy integration examples

### [08-SiestaPhononMaker](08-SiestaPhononMaker/)
**Description**: Native SIESTA phonon calculations using force constant approach
**Difficulty**: Intermediate
**Time**: ~5 min (dry-run), ~30-60 min (full calculation)
**Key Concepts**: SiestaPhononMaker, SIESTA's built-in phonon features, MD.TypeOfRun
**Files**: SIESTA-native phonon examples

---

## Learning Path

We recommend following this sequence:

### Core Sequence (Essential)
1. **Start here**: [01-RelaxMaker](01-RelaxMaker/) - Learn the absolute basics of structural relaxation
2. **Multi-step**: [04-RelaxMaker-StaticMaker](04-RelaxMaker-StaticMaker/) - Combine multiple jobs into workflows
3. **Electronic**: [02-BandStructureMaker](02-BandStructureMaker/) - Electronic band structure analysis

### Electronic Structure Track
4. **DOS**: [05-DOSMaker](05-DOSMaker/) - Total density of states
5. **PDOS**: [06-PDOSMaker](06-PDOSMaker/) - Orbital-projected density of states

### Vibrational Properties Track
6. **Phonopy**: [07-PhonopyMaker](07-PhonopyMaker/) - Phonon calculations with phonopy
7. **Native**: [08-SiestaPhononMaker](08-SiestaPhononMaker/) - SIESTA's built-in phonon features

### Advanced Control (Optional)
8. **Lua**: [03-LuaMaker](03-LuaMaker/) - Advanced optimization with Lua scripting

---

## Quick Start Pattern

All tutorials in this category follow a consistent pattern:

```python
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker  # or other Maker
from jobflow import run_locally

# 1. Load structure
structure = Structure.from_file("../00-structures/Si.cif")

# 2. Create maker
maker = RelaxMaker.fixed_cell_relaxation()

# 3. Generate job
job = maker.make(structure)

# 4. Run (choose one)
# DRY-RUN (preview only)
results = run_locally(job, create_folders=True)

# Or LOCAL execution
# from atomate2.siesta.utils import run_and_report
# run_and_report(job)

# Or SUBMIT to cluster
# from jobflow_remote import submit_flow
# submit_flow(job, project="my_project")
```

---

## Execution Modes

All tutorials support three execution modes:

```python
# Mode 1: DRY-RUN (preview only, no calculations)
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)

# Mode 2: LOCAL (run calculations locally)
maker = RelaxMaker.fixed_cell_relaxation()
run_and_report(job)

# Mode 3: SUBMIT (submit to HPC cluster)
from jobflow_remote import submit_flow
submit_flow(job, project="production")
```

**Recommendation**: Always start with `dry_run=True` to verify your setup!

---

## Common Issues

### Issue 1: "SIESTA command not found"
**Solution**: Configure `SIESTA_CMD` in `~/.atomate2siesta.yaml`:
```yaml
SIESTA_CMD: "mpirun -np 4 siesta < siesta.fdf > siesta.out"
```

### Issue 2: "Pseudopotential files not found"
**Solution**: Install pseudopotentials and configure path:
```bash
atomate2siesta-pseudos install ONCVPSP-PBE-SR-PDv0.4-Standard
```

Then configure in `~/.atomate2siesta.yaml`:
```yaml
SIESTA_PP_PATH: "~/.atomate2siesta/pseudos/ONCVPSP-PBE-SR-PDv0.4-Standard"
```

### Issue 3: "Structure file not found"
**Solution**: Check that `00-structures/` directory exists at tutorials root:
```bash
ls ../00-structures/
```

### Issue 4: "Module not found: atomate2.siesta"
**Solution**: Install atomate2siesta in development mode:
```bash
cd /path/to/atomate2siesta
pip install -e .
```

### Issue 5: "Unknown FDF parameter: fdf_arguments"
**Error**: `ValueError: Unknown FDF parameter(s): fdf_arguments`

**Cause**: Using deprecated `fdf_arguments` wrapper syntax

**Solution**:
```python
# ❌ OLD (doesn't work):
user_params = {
    "fdf_arguments": {
        "DM.InitSpin": [...],
        "Geometry.Constraints": [...]
    }
}

# ✅ NEW (correct):
user_params = {
    "%block DM.InitSpin": [...],          # Directly in user_params!
    "%block Geometry.Constraints": [...],
}
```

**Note**: Block parameters should be specified **directly** in `user_params`, NOT nested in `fdf_arguments`. See [FDF Block Parameters](#fdf-block-parameters-advanced) section above.

---

## Maker Reference

| Maker | Class Method | Purpose |
|-------|-------------|---------|
| RelaxMaker | `fixed_cell_relaxation()` | Relax atomic positions only |
| RelaxMaker | `variable_cell_relaxation()` | Relax atoms + cell |
| BandStructureMaker | `band_structure()` | Electronic band structure |
| DOSMaker | `dos()` | Density of states |
| PDOSMaker | `pdos()` | Projected DOS |
| PhonopyMaker | `phonon()` | Phonon with phonopy |
| SiestaPhononMaker | `phonon()` | Native SIESTA phonon |
| LuaMaker | `lua_optimization()` | Lua-scripted optimization |

---

## Parameter Customization

### Using Tier Presets (Recommended)
```python
from atomate2.siesta.sets.tiers import apply_tier_preset

maker = RelaxMaker.fixed_cell_relaxation()
maker = apply_tier_preset(maker, "relax_standard")
```

### Direct Parameter Specification
```python
maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "PAO.BasisSize": "DZP",
        "a2s_kpts": [6, 6, 6],
        "Mesh.Cutoff": "300 Ry",
    }
)
```

### Override Preset Parameters
```python
maker = RelaxMaker.fixed_cell_relaxation()
maker = apply_tier_preset(
    maker,
    "relax_standard",
    override_params={
        "a2s_kpts": [8, 8, 8],  # Override preset's k-points
        "Spin": "polarized",  # Add spin polarization
    }
)
```

### FDF Block Parameters (Advanced)

For FDF block parameters (like `DM.InitSpin`, `Geometry.Constraints`, etc.), use the `"%block ParamName"` syntax **directly** in `user_params`:

```python
# ✅ CORRECT: Block parameters directly in user_params
maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "Spin": "polarized",
        "%block DM.InitSpin": [
            "1  +2.0",  # Atom 1: +2.0 μB
            "2  -2.0",  # Atom 2: -2.0 μB
        ],
    }
)

# ❌ WRONG: Don't nest in fdf_arguments!
maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "fdf_arguments": {  # <-- This doesn't work!
            "DM.InitSpin": [...]
        }
    }
)
```

**Note**: The `fdf_arguments` wrapper is deprecated. See individual tutorial READMEs for comprehensive examples.

See **[03-advanced-features/01-tier-system](../03-advanced-features/01-tier-system/)** for detailed tier preset documentation.

---

## Next Steps

After completing the basics, proceed to:

### Immediate Next Steps
- **[02-workflows/01-convergence](../02-workflows/01-convergence/)** - Learn systematic parameter optimization
- **[03-advanced-features/01-tier-system](../03-advanced-features/01-tier-system/)** - Material-specific parameter presets

### By Research Interest
- **Electronic properties**: [02-workflows/02-equation-of-states](../02-workflows/02-equation-of-states/)
- **Vibrational properties**: [02-workflows/06-vibrational-properties](../02-workflows/06-vibrational-properties/)
- **Surface chemistry**: [02-workflows/03-surfaces-and-adsorption](../02-workflows/03-surfaces-and-adsorption/)
- **Mechanical properties**: [02-workflows/04-mechanical](../02-workflows/04-mechanical/)

---

## Tutorial Metrics

- **Total tutorials**: 8 fundamental Makers
- **Total example files**: 30+ Python scripts
- **Coverage**: All basic single-job Makers
- **Difficulty**: Beginner (7 tutorials) + Intermediate (1 tutorial)

---

*Back to [Main Tutorial Index](../README.md)*
