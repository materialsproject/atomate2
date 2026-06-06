# BandStructureFlowMaker Tutorials

This directory contains tutorials for the `BandStructureFlowMaker`, a complete workflow
for computing electronic band structures with SIESTA.

## Overview

`BandStructureFlowMaker` provides an automated workflow for band structure calculations:

1. **Structure Relaxation** (optional): Optimize geometry before band calculation
2. **SCF Calculation**: Compute ground-state electronic structure
3. **Band Structure**: Calculate eigenvalues along high-symmetry k-path
4. **Analysis**: Extract band gap, VBM, CBM, and generate plots

## Tutorials

| File | Description | Runtime |
|------|-------------|---------|
| `01_basic.py` | Complete workflow with relaxation | ~30 min |
| `02_skip_relaxation.py` | Skip relaxation for pre-relaxed structures | ~15 min |
| `03_custom_parameters.py` | High-accuracy TZP settings | ~45-60 min |
| `04_with_tier_preset.py` | Using tier presets for parameters | varies |
| `05_dry_run.py` | Preview workflow without running SIESTA | ~2 sec |

## Quick Start

```python
from atomate2.siesta.flows.bands import BandStructureFlowMaker
from pymatgen.core import Structure
from jobflow import run_locally

structure = Structure.from_file("Si.cif")
maker = BandStructureFlowMaker()
flow = maker.make(structure)
results = run_locally(flow, create_folders=True)
```

## Key Features

### Automatic K-Path Generation
The workflow automatically determines the high-symmetry k-path based on crystal symmetry
using pymatgen/seekpath conventions. No manual k-point specification needed.

### Band Gap Analysis
Automatically extracts:
- Band gap (eV)
- VBM (Valence Band Maximum)
- CBM (Conduction Band Minimum)
- Gap type (direct/indirect/metallic)

### Output Files
- `band_structure_summary.txt`: Text summary of electronic properties
- `band_structure.png`: Publication-quality band structure plot

## Customization Options

### Skip Relaxation
For pre-relaxed structures:
```python
maker = BandStructureFlowMaker(relax_maker=None)
```

### Custom Energy Range
For wider/narrower plot range:
```python
maker = BandStructureFlowMaker(energy_range=(-10, 10))  # eV from Fermi
```

### Disable Plotting
To skip plot generation:
```python
maker = BandStructureFlowMaker(plot_bands=False)
```

### Custom Makers
For fine-grained control:
```python
from atomate2.siesta.jobs.core import StaticMaker, BandStructureMaker

scf_maker = StaticMaker(user_params={"PAO.BasisSize": "TZP"})
bands_maker = BandStructureMaker.bandstructure_calculation()

maker = BandStructureFlowMaker(
    scf_maker=scf_maker,
    bands_maker=bands_maker,
)
```

## Tips

1. **Start with dry run**: Use `dry_run=True` to validate parameters before production
2. **Match basis sets**: Use the same basis for SCF and bands for consistency
3. **Dense SCF k-grid**: Use denser k-points in SCF for accurate Fermi level
4. **Pre-relax for speed**: Skip relaxation if structure is already optimized

## See Also

- `tutorials/01-basics/02-BandStructureMaker/`: Basic band structure job (no workflow)
- `tutorials/01-basics/05-DOSMaker/`: Density of states calculations
- `atomate2siesta-info workflows BandStructureFlowMaker --full`: Full documentation
