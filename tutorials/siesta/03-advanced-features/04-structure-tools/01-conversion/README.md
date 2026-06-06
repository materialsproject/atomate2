# Structure File Conversion Tutorial

## Overview

This tutorial demonstrates structure format conversion in atomate2siesta using pymatgen, sisl, and custom powerups.

## Tutorial Scripts

| Script | Description | Key Features |
|--------|-------------|--------------|
| `01_basic_conversion.py` | Pymatgen conversion basics | CIF, XSF, JSON, POSCAR |
| `02_siesta_formats.py` | Reading SIESTA FDF files | siesta_to_pymatgen powerup |
| `03_dry_run_formats.py` | Dry-run format generation | All output formats |
| `04_xsf_cif_molecule.py` | Molecule format conversion | XSF molecular format |

**Run the tutorials:**

```bash
cd tutorials/03-advanced-features/04-structure-tools/01-conversion
python 01_basic_conversion.py
python 02_siesta_formats.py
python 03_dry_run_formats.py
python 04_xsf_cif_molecule.py
```

---

## Quick Reference

### Reading Structures with Pymatgen

```python
from pymatgen.core import Structure, Molecule

# Auto-detect format for crystals
structure = Structure.from_file("structure.cif")
structure = Structure.from_file("structure.xsf")
structure = Structure.from_file("POSCAR")
structure = Structure.from_file("structure.json")

# Read molecules
molecule = Molecule.from_file("molecule.xsf")
molecule = Molecule.from_file("molecule.xyz")
```

### Writing Structures with Pymatgen

```python
# Write various formats
structure.to(filename="output.cif", fmt="cif")
structure.to(filename="output.xsf", fmt="xsf")
structure.to(filename="POSCAR", fmt="poscar")
structure.to(filename="output.json", fmt="json")
structure.to(filename="output.yaml", fmt="yaml")
```

### Supported Formats

| Format | Extension | Use Case | Notes |
|--------|-----------|----------|-------|
| CIF | .cif | Standard crystallographic | Most common format |
| XSF | .xsf | XCrySDen visualization | Good for molecules too |
| POSCAR | POSCAR | VASP compatibility | No extension |
| JSON | .json | Python-friendly | Includes metadata |
| YAML | .yaml | Human-readable | Config-style format |

---

## Reading SIESTA Files

### Using siesta_to_pymatgen Powerup

The `siesta_to_pymatgen()` powerup reads SIESTA FDF and XV files:

```python
from atomate2.siesta.powerups import siesta_to_pymatgen

# Read from SIESTA FDF file (input structure)
structure = siesta_to_pymatgen("siesta.fdf", use_xv=False)

# Read from SIESTA XV file (relaxed structure)
structure = siesta_to_pymatgen("siesta.XV", use_xv=True)  # default
```

**Important distinctions:**

| File | When Available | Contains |
|------|----------------|----------|
| `siesta.fdf` | Always (input file) | Initial structure from input |
| `siesta.XV` | After SIESTA runs | **Relaxed structure** (output) |

**Dry-run mode:** Only generates `siesta.fdf` (input files), not `siesta.XV` (output files require actual SIESTA execution).

### Dry-Run Output Location

When using `dry_run=True`, files are in nested directories:

```
job_*/dry_run_output/*/siesta.fdf    # Input file (initial structure)
```

When using `dry_run=False`, files are in job root:

```
job_*/siesta.fdf    # Input file
job_*/siesta.XV     # Output file (relaxed structure)
```

### Example: Reading from Dry-Run

```python
from glob import glob
from atomate2.siesta.powerups import siesta_to_pymatgen

# Find FDF file from dry_run
fdf_files = glob("job_*/dry_run_output/*/siesta.fdf")
if fdf_files:
    structure = siesta_to_pymatgen(fdf_files[0], use_xv=False)
```

### Example: Reading Relaxed Structure

```python
# Read relaxed structure from actual SIESTA run
structure_relaxed = siesta_to_pymatgen("job_*/siesta.XV")
```

---

## Writing SIESTA Input Files

### Using SiestaInputGenerator

```python
from atomate2.siesta.sets.base import SiestaInputGenerator
from pymatgen.core import Structure

structure = Structure.from_file("structure.cif")
input_gen = SiestaInputGenerator()
input_set = input_gen.get_input_set(structure)
input_set.write_input("output_dir/")
# Creates: siesta.fdf, structure.fdf (in structure.fdf format, not XV)
```

### Using Dry-Run Mode (Recommended)

```python
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import run_locally

maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
job = maker.make(structure)
run_locally(job, create_folders=True)
# Creates complete input files in: job_*/dry_run_output/*/
```

---

## CLI Conversion Tool

The `atomate2siesta-structure convert` command provides automatic format detection:

```bash
# CIF to SIESTA FDF (automatic detection)
atomate2siesta-structure convert structure.cif --write-fdf

# XSF to CIF (automatic detection)
atomate2siesta-structure convert structure.xsf --write-cif

# SIESTA FDF to multiple formats
atomate2siesta-structure convert siesta.fdf --write-cif --write-xsf --write-json

# XV file to CIF (relaxed structure)
atomate2siesta-structure convert siesta.XV --write-cif

# Multiple output formats at once
atomate2siesta-structure convert structure.cif --write-fdf --write-xsf --write-json
```

**Supported input formats:** `.fdf`, `.xv`/`.XV`, `.cif`, `.xsf` (automatic detection)

**Supported output formats:**
- CIF (`.cif`) - Standard crystallographic
- XSF (`.xsf`) - Visualization format
- JSON (`.json`) - Python-friendly
- FDF (SIESTA input) - For SIESTA calculations
- Pickle formats (sisl/ASE/pymatgen)

---

## Dry-Run Format Options

Control output structure format in dry-run mode:

```python
from atomate2.siesta.jobs.core import RelaxMaker

# Generate different formats
maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=True,
    dry_run_format="xsf"  # Options: "cif", "xsf", "json", "POSCAR"
)
job = maker.make(structure)
run_locally(job)
```

**Available formats:**
- `"cif"` - CIF format (default)
- `"xsf"` - XSF format
- `"json"` - JSON format
- `"POSCAR"` - VASP format

---

## Common Workflows

### Workflow 1: CIF → SIESTA → Relaxed Structure → XSF

```python
# 1. Start with CIF
structure = Structure.from_file("structure.cif")

# 2. Run SIESTA relaxation
maker = RelaxMaker.fixed_cell_relaxation()
job = maker.make(structure)
results = run_locally(job, create_folders=True)

# 3. Read relaxed structure
structure_relaxed = siesta_to_pymatgen("job_*/siesta.XV")

# 4. Convert to XSF for visualization
structure_relaxed.to(filename="relaxed.xsf", fmt="xsf")
```

### Workflow 2: Preview Input Files Only

```python
# Generate input files without running SIESTA
structure = Structure.from_file("structure.cif")
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
job = maker.make(structure)
run_locally(job, create_folders=True)

# Read generated FDF
fdf_files = glob("job_*/dry_run_output/*/siesta.fdf")
structure_check = siesta_to_pymatgen(fdf_files[0], use_xv=False)
```

### Workflow 3: CLI Batch Conversion

```bash
# Convert all CIF files to FDF
for file in *.cif; do
    atomate2siesta-structure convert "$file" --write-fdf
done

# Convert SIESTA output to visualization format
atomate2siesta-structure convert job_*/siesta.XV --write-xsf --write-cif
```

---

## Best Practices

1. **Always prefer pymatgen** for standard formats (CIF, XSF, JSON)
2. **Use siesta_to_pymatgen** only for SIESTA-specific files (FDF, XV)
3. **Use dry-run mode** to preview input files before expensive calculations
4. **Read XV files for relaxed structures**, not FDF files (which have initial structure)
5. **Use glob patterns** to handle jobflow directory names with timestamps

---

## Common Issues

### Issue: "No such file or directory: job_*/siesta.XV"

**Cause:** XV files are only created by actual SIESTA execution, not dry-run mode.

**Solution:** Remove `dry_run=True` to run actual calculation:
```python
maker = RelaxMaker.fixed_cell_relaxation(dry_run=False)  # Will generate XV
```

### Issue: FDF file not found in expected location

**Cause:** Dry-run creates files in nested `dry_run_output/*/` directories.

**Solution:** Use glob to find files:
```python
from glob import glob
fdf_files = glob("job_*/dry_run_output/*/siesta.fdf")
```

### Issue: Structure coordinates differ from expected

**Cause:** Reading FDF file (input) instead of XV file (output).

**Solution:** Always use XV for relaxed structures:
```python
structure = siesta_to_pymatgen("job_*/siesta.XV")  # Relaxed
# NOT: siesta_to_pymatgen("job_*/siesta.fdf")      # Initial
```

---

## See Also

- **Pymatgen docs**: https://pymatgen.org/
- **sisl docs**: https://sisl.readthedocs.io/
- **Dry-run tutorial**: `03-infrastructure/05-dry-run-preview/`
- **Structure manipulation**: `04-structure-tools/02-manipulation/`
- **CLI structure tools**: `atomate2siesta-structure --help`

---

**Summary:** Use pymatgen for standard conversions, siesta_to_pymatgen for SIESTA files, and dry-run mode to preview calculations!
