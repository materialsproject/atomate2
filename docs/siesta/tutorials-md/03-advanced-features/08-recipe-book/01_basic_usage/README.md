# Recipe Book: Basic Usage Examples

This directory contains 7 standalone examples demonstrating the basic usage of the Recipe Book system. Each example is self-contained and can be run independently.

## Overview

The Recipe Book provides high-level workflows for complete material characterization with significant code reduction. Instead of writing 50+ lines of boilerplate, you can generate complete workflows with a single function call.

## Examples

### 1. Structure Analysis (`01_structure_analysis.py`)
**Start here!** Always analyze your structure first to see what parameters the Recipe Book will use automatically.

```bash
python 01_structure_analysis.py
```

**Key features:**
- Material type detection (metal, semiconductor, insulator)
- Recommended parameter presets
- Estimated computation time
- Automatic k-point and cutoff suggestions

---

### 2. Complete Material Study (`02_complete_material_study.py`)
The ultimate one-liner for comprehensive material characterization.

```bash
python 02_complete_material_study.py
```

**Calculates:**
- Electronic structure (bands, DOS)
- Mechanical properties (elastic constants, bulk modulus)
- Thermal properties (phonons, QHA, thermal expansion)

**Execution modes:**
- `run_locally`: Full calculation (several hours)
- `dry_run`: Preview inputs only (instant)

---

### 3. Quick Characterization (`03_quick_characterization.py`)
Fast preliminary study (~1-2 hours) for quick screening.

```bash
python 03_quick_characterization.py
```

**Includes:**
- Structure relaxation
- Band structure
- Essential properties

**Use case:** Quick material screening before expensive calculations

---

### 4. Selective Properties (`04_selective_properties.py`)
Calculate only specific property categories to save time.

```bash
python 04_selective_properties.py
```

**Available properties:**
- `electronic`: Bands, DOS, effective masses
- `mechanical`: Elastic constants, bulk modulus
- `thermal`: Phonons, QHA, thermal expansion

**Example:**
```python
flow = RecipeBook.complete_material_study(
    structure,
    properties=["electronic", "mechanical"]  # Skip thermal
)
```

---

### 5. With Convergence Testing (`05_with_convergence_testing.py`)
Test computational parameters before running expensive calculations.

```bash
python 05_with_convergence_testing.py
```

**Tests:**
- K-point convergence
- Mesh cutoff convergence
- Basis set convergence

**Benefit:** Ensures optimal accuracy/cost trade-off (adds ~30% time)

---

### 6. Dry-Run Mode (`06_dry_run_mode.py`)
Generate all input files WITHOUT running calculations (99.9% time savings!).

```bash
python 06_dry_run_mode.py
```

**Perfect for:**
- Checking parameters before expensive runs
- Debugging workflow setup
- Learning SIESTA input file structure

**Output:** All `.fdf`, `.psml`, and structure files in `dry_run_output/`

---

### 7. Different Materials (`07_different_materials.py`)
Demonstrates automatic material type detection and parameter adjustment.

```bash
python 07_different_materials.py
```

**Examples:**
- **Silicon** (semiconductor): Standard k-points, no smearing
- **Aluminum** (metal): Higher k-point density, Fermi-Dirac smearing

**Key point:** Recipe Book automatically adjusts parameters based on material type!

---

## Execution Options

All examples (except #1 and #6) support two execution modes:

### Run Locally
```python
from jobflow import run_locally

flow = RecipeBook.complete_material_study(structure)
results = run_locally(flow, create_folders=True)
```

**Pros:** Get real results
**Cons:** Takes hours to complete
**Use when:** You need actual calculation results

### Dry-Run Mode
```python
flow = RecipeBook.complete_material_study(structure, dry_run=True)
results = run_locally(flow, create_folders=True)
```

**Pros:** Instant feedback (seconds instead of hours)
**Cons:** No actual calculations performed
**Use when:** Testing, debugging, or previewing inputs

---

## Quick Start

```bash
# 1. Always start with structure analysis
python 01_structure_analysis.py

# 2. Try dry-run mode to preview inputs
python 06_dry_run_mode.py

# 3. Run a quick characterization
python 03_quick_characterization.py
```

## Tips

1. **Always analyze first**: `RecipeBook.print_analysis()` shows what parameters will be used
2. **Use dry-run mode**: Perfect for checking setup before expensive calculations
3. **Start small**: Try `quick_characterization()` before `complete_material_study()`
4. **Selective properties**: Only calculate what you need to save time
5. **Convergence testing**: Adds 30% time but ensures optimal parameters

## Next Steps

- **02_material_specific_workflows/**: Specialized workflows for different material types
- **03_property_calculations/**: Individual property calculations
- **04_advanced_features/**: Custom parameters, convergence testing, error handling

## Common Workflows

### Fast screening
```python
flow = RecipeBook.quick_characterization(structure)
```

### Electronic properties only
```python
flow = RecipeBook.complete_material_study(structure, properties=["electronic"])
```

### With convergence tests
```python
flow = RecipeBook.complete_material_study(
    structure,
    properties=["electronic"],
    test_convergence=True
)
```

### Preview inputs
```python
flow = RecipeBook.complete_material_study(structure, dry_run=True)
```

---

**Need help?** Check `tutorials/08-recipe-book/README.md` for complete documentation.
