# Tutorial 06: Tier Presets and Customization

Learn how to use, modify, and create tier presets in atomate2siesta.

## Directory Structure

```
06-tier-presets-customization/
├── 01-apply-preset/          # How to apply different presets
│   ├── 01_relax_standard.py
│   ├── 02_surface_metal.py
│   ├── 03_adsorbate_screening.py
│   ├── 04_2d_metal.py
│   └── 05_high_accuracy.py
├── 02-modify-preset/          # How to customize preset parameters
│   ├── 01_modify_kpts.py
│   ├── 02_modify_multiple.py
│   ├── 03_add_spin.py
│   ├── 04_add_vdw.py
│   ├── 05_add_dftu.py
│   ├── 06_tighten_convergence.py
│   └── 07_add_constraints.py
└── 03-create-preset/          # How to create custom presets
    └── 03_create_custom_preset.py
```

## Part 1: Apply Tier Presets

**Directory**: `01-apply-preset/`

Learn how to discover and apply material-specific presets.

### Examples:

1. **`01_relax_standard.py`** - Standard relaxation (intermediate tier)
2. **`02_surface_metal.py`** - Metallic surfaces with MP smearing
3. **`03_adsorbate_screening.py`** - Fast screening (basic tier)
4. **`04_2d_metal.py`** - 2D materials with dense in-plane kpts
5. **`05_high_accuracy.py`** - High-quality results with TZP basis

### Usage:
```bash
cd 01-apply-preset
python 01_relax_standard.py
```

All scripts use `dry_run=True` for instant preview without running SIESTA.

### CLI Discovery:
```bash
atomate2siesta-presets list                      # See all material-specific presets
atomate2siesta-presets show relax_standard       # Show preset details
atomate2siesta-presets category surface          # List surface presets
```

---

## Part 2: Modify Preset Parameters

**Directory**: `02-modify-preset/`

Learn how to customize presets by modifying parameters and adding new ones.

### Examples:

1. **`01_modify_kpts.py`** - Modify single parameter (k-points)
2. **`02_modify_multiple.py`** - Modify multiple parameters at once
3. **`03_add_spin.py`** - Add spin polarization
4. **`04_add_vdw.py`** - Add van der Waals corrections
5. **`05_add_dftu.py`** - Add DFT+U for correlated systems
6. **`06_tighten_convergence.py`** - Tighten convergence criteria
7. **`07_add_constraints.py`** - Add geometry constraints

### Usage:
```bash
cd 02-modify-preset
python 03_add_spin.py
```

### Key Pattern:
```python
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
maker = apply_tier_preset(
    maker,
    "relax_standard",
    override_params={
        "kpts": [6, 6, 6],        # Modify preset param
        "Spin": "polarized",       # Add new param
    },
)
```

---

## Part 3: Create Custom Presets

**Directory**: `03-create-preset/`

Learn the structure and workflow for creating your own presets.

### File:

**`03_create_custom_preset.py`** - Documentation and examples

This file shows:
- Preset dictionary structure
- MOF screening preset example
- Catalysis preset with vdW example
- High-throughput screening preset
- Naming conventions for automatic CLI detection
- Testing workflow

### Usage:
```bash
cd 03-create-preset
python 03_create_custom_preset.py  # Shows documentation
```

---

## Quick Reference

### Applying a Preset
```python
from atomate2.siesta.sets.tiers import apply_tier_preset

maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
maker = apply_tier_preset(maker, "relax_standard")
job = maker.make(structure)
```

### Modifying Parameters
```python
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
maker = apply_tier_preset(
    maker,
    "relax_standard",
    override_params={"kpts": [6, 6, 6]},
)
```

### Adding New Parameters
```python
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
maker = apply_tier_preset(
    maker,
    "relax_standard",
    override_params={
        "Spin": "polarized",
        "vdw": "DRSLL",
    },
)
```

---

## Output Files

All examples generate files in:
```
job_*/dry_run_output/*/siesta.fdf
```

Check these files to verify preset parameters.

---

## Available Presets (26 Total)

Use CLI to explore:
```bash
atomate2siesta-presets list              # All presets
atomate2siesta-presets category 2d       # 2D material presets
atomate2siesta-presets category surface  # Surface presets
atomate2siesta-presets defaults          # Tier-level defaults
```

### Categories:
- **2d** (8): 2D materials (graphene, TMDs, etc.)
- **surface** (3): Surfaces and adsorption
- **structural** (3): Bulk relaxation
- **phonon** (2): Vibrational properties
- **magnetic** (2): Spin-polarized systems
- **electronic** (3): Band structure, DOS
- **optical** (1): Optical properties
- **molecular** (1): Gas-phase molecules
- **performance** (3): Large systems, HPC

---

## Next Steps

1. Run examples in order: 01 → 02 → 03
2. Inspect generated `siesta.fdf` files
3. Try modifying parameters for your materials
4. Create custom presets following Part 3 guide

---

## Documentation

- **Tier System**: `docs/source/tier-system.rst`
- **CLI Tools**: `docs/source/cli-tools.rst`
- **Parameters**: `docs/source/siesta-inputs.rst`

---

**Last Updated**: November 2025 (v1.0.0)
