# Band Structure Workflows

This section covers electronic band structure calculations using SIESTA workflows.

## Available Tutorials

### 01-BandStructureFlowMaker

Complete workflow for band structure calculations including:
- Optional structure relaxation
- Self-consistent field (SCF) calculation
- Band structure along high-symmetry k-path
- Automatic band gap analysis
- Publication-quality plotting

**Tutorials:**
- `01_basic.py` - Complete workflow with relaxation
- `02_skip_relaxation.py` - Skip relaxation for pre-relaxed structures
- `03_custom_parameters.py` - High-accuracy settings (TZP, 400 Ry)
- `04_with_tier_preset.py` - Using tier presets
- `05_dry_run.py` - Preview workflow without running SIESTA

## Quick Example

```python
from atomate2.siesta.flows.bands import BandStructureFlowMaker
from pymatgen.core import Structure
from jobflow import run_locally

structure = Structure.from_file("Si.cif")
maker = BandStructureFlowMaker()
flow = maker.make(structure)
results = run_locally(flow, create_folders=True)
```

## Key Outputs

- **band_structure_summary.txt**: Band gap, VBM, CBM information
- **band_structure.png**: Band dispersion plot

## See Also

- `01-basics/02-BandStructureMaker/`: Single band structure job (no workflow)
- `01-basics/05-DOSMaker/`: Density of states calculations
