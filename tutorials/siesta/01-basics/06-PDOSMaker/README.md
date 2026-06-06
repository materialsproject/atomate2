# Tutorial: Projected Density of States (PDOS) with PDOSMaker

**Category**: 01-basics/06-PDOSMaker
**Difficulty**: Beginner
**Time**: ~2 min (dry-run), ~15-20 min (full calculation)

---

## Overview

This tutorial demonstrates how to calculate the projected density of states (PDOS) using `PDOSMaker`. While total DOS shows overall electronic state density, PDOS decomposes it by atomic sites and orbital angular momentum, revealing which atoms and orbitals contribute to electronic states at each energy.

**What PDOS Tells You**:
- **Orbital character**: Which orbitals (s, p, d, f) contribute at each energy
- **Atomic contributions**: Which atoms dominate electronic states
- **Bonding analysis**: Orbital overlap and hybridization
- **Magnetic moments**: Spin-resolved PDOS for magnetic systems

---

## What You'll Learn

- Using `PDOSMaker` for projected DOS calculations
- Atom and orbital-resolved electronic structure
- PDOS analysis and interpretation
- Customizing PDOS parameters for specific atoms/orbitals
- Spin-resolved PDOS for magnetic systems

---

## Prerequisites

- **Required tutorials**: [05-DOSMaker](../05-DOSMaker/)
- **Required knowledge**: Understanding of atomic orbitals and DOS
- **SIESTA configuration**: `~/.atomate2siesta.yaml` set up correctly
- **Structure files**: Located in [00-structures](../../00-structures/)

---

## Key Concepts

### Total DOS vs PDOS

**Total DOS**: $g(E) = \sum_{n,k} \delta(E - E_{n,k})$
- Shows overall density of states
- Single curve
- Good for band gaps and Fermi level

**Projected DOS**: $g_{\alpha,l}(E) = \sum_{n,k} |c_{\alpha,l}^{n,k}|^2 \delta(E - E_{n,k})$
- Decomposes DOS by atom $\alpha$ and orbital $l$
- Multiple curves (one per atom/orbital type)
- Reveals electronic character

**Relationship**: $\sum_{\alpha,l} g_{\alpha,l}(E) = g(E)$

### PDOS Decomposition Levels

SIESTA provides PDOS at multiple levels:

1. **By Atom**: Individual atomic contributions
   ```
   Atom 1 (Si): Total PDOS
   Atom 2 (Si): Total PDOS
   Atom 3 (O):  Total PDOS
   ```

2. **By Orbital (l quantum number)**:
   - s-orbitals (l=0)
   - p-orbitals (l=1)
   - d-orbitals (l=2)
   - f-orbitals (l=3)

3. **By ml quantum number** (optional):
   - px, py, pz (for p-orbitals)
   - dxy, dyz, dzx, dx²-y², dz² (for d-orbitals)

4. **Spin-resolved** (for magnetic systems):
   - Spin-up channel
   - Spin-down channel

### Key PDOS Parameters

```python
user_params = {
    "a2s_kpts": [8, 8, 8],          # Dense mesh required
    "Spin": "polarized",            # For magnetic systems

    # PDOS block parameters (directly in user_params, NOT in fdf_arguments!)
    "%block ProjectedDensityOfStates": [
        "-20.0  10.0  0.1  500  eV"  # Emin Emax dE npts units
    ]
}
```

---

## Tutorial Files

This directory contains 2 examples:

1. **`PDOSMaker_pdos_all_atoms.py`** - PDOS for all atoms in structure
2. **`PDOSMaker_custom_dos_parameters.py`** - Custom energy range and resolution

---

## Quick Start

### Example 1: Basic PDOS Calculation

```python
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import PDOSMaker
from jobflow import run_locally

# Load structure (e.g., SiO2, MgO with multiple atom types)
structure = Structure.from_file("../../00-structures/MgO.cif")

# Create maker and run
maker = PDOSMaker(dry_run=True)
job = maker.make(structure)
results = run_locally(job, create_folders=True)
```

### Example 2: PDOS with Custom Parameters

```python
maker = PDOSMaker(
    user_params={
        "a2s_kpts": [12, 12, 12],   # Dense k-points
        "PAO.BasisSize": "DZP",
        "%block ProjectedDensityOfStates": [
            "-15.0  15.0  0.05  600  eV"  # Custom energy range
        ]
    },
    dry_run=True
)
```

### Example 3: Spin-Resolved PDOS

```python
from atomate2.siesta.sets.utils import get_default_initial_magnetic_moments

# Set up magnetic system
magmoms = get_default_initial_magnetic_moments(structure)
structure.add_site_property("magmom", magmoms)

maker = PDOSMaker(
    user_params={
        "Spin": "polarized",
        "a2s_magnetic_ordering": "ferromagnetic",
        "a2s_kpts": [10, 10, 10],
    },
    dry_run=True
)
```

### Example 4: PDOS After Relaxation

```python
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import Flow

# Relax → PDOS workflow
relax_maker = RelaxMaker.fixed_cell_relaxation()
pdos_maker = PDOSMaker()

relax_job = relax_maker.make(structure)
pdos_job = pdos_maker.make(structure, prev_dir=relax_job.output.dir_name)

workflow = Flow([relax_job, pdos_job])
results = run_locally(workflow, create_folders=True)
```

---

## Run Modes

### 1. Dry-Run Mode

```bash
python PDOSMaker_pdos_all_atoms.py  # With dry_run=True
```

**Check PDOS settings**:
```bash
grep -A5 "ProjectedDensityOfStates" preview_output/job_*/siesta.fdf
```

### 2. Local Execution

```bash
# Edit: Set dry_run=False
python PDOSMaker_pdos_all_atoms.py
```

**Output**:
```
job_*/
├── siesta.fdf
├── siesta.out
├── siesta.PDOS.xml      # PDOS in XML format (preferred)
├── siesta.DOS           # Total DOS
└── [SIESTA output files]
```

**Time**: ~15-20 minutes (PDOS requires denser k-points)

---

## Expected Output

### PDOS File Format

SIESTA outputs PDOS in XML format (`siesta.PDOS.xml`):

```xml
<pdos>
  <nspin>1</nspin>
  <norbitals>26</norbitals>
  <orbital_info>
    <orbital index="1" atom_index="1" species="Si" n="3" l="0" m="0" z="1" />
    <orbital index="2" atom_index="1" species="Si" n="3" l="1" m="-1" z="1" />
    ...
  </orbital_info>
  <data>
    <!-- Energy grid and PDOS values -->
  </data>
</pdos>
```

### Analyzing PDOS with Pymatgen

```python
from pymatgen.io.siesta import SiestaOutput
import matplotlib.pyplot as plt

# Read PDOS data
output = SiestaOutput("job_*/siesta.out")
pdos = output.get_pdos()

# Get specific atom PDOS
atom_pdos = pdos.get_site_dos(output.structure[0])  # First atom

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot s, p, d orbitals separately
energies = pdos.energies - pdos.efermi  # Relative to Fermi
ax.plot(energies, atom_pdos.get_dos("s"), label="s-orbital", linewidth=2)
ax.plot(energies, atom_pdos.get_dos("p"), label="p-orbital", linewidth=2)
ax.plot(energies, atom_pdos.get_dos("d"), label="d-orbital", linewidth=2)

ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Fermi level')
ax.set_xlabel('Energy - E$_F$ (eV)', fontsize=12)
ax.set_ylabel('PDOS (states/eV)', fontsize=12)
ax.set_title(f'PDOS for Atom 1 ({output.structure[0].specie})', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pdos_atom1.png', dpi=300)
plt.show()
```

### Comparing Multiple Atoms

```python
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, site in enumerate(output.structure[:4]):  # First 4 atoms
    ax = axes[i]
    site_pdos = pdos.get_site_dos(site)

    energies = pdos.energies - pdos.efermi
    ax.plot(energies, site_pdos.get_dos("s"), label="s", linewidth=2)
    ax.plot(energies, site_pdos.get_dos("p"), label="p", linewidth=2)
    ax.plot(energies, site_pdos.get_dos("d"), label="d", linewidth=2)

    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('E - E$_F$ (eV)')
    ax.set_ylabel('PDOS (states/eV)')
    ax.set_title(f'Atom {i+1}: {site.specie}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pdos_comparison.png', dpi=300)
plt.show()
```

### Spin-Resolved PDOS

```python
if output.is_spin_polarized:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    site_pdos = pdos.get_site_dos(output.structure[0])
    energies = pdos.energies - pdos.efermi

    # Spin up
    ax1.plot(energies, site_pdos.get_dos("s", spin="up"), label="s ↑")
    ax1.plot(energies, site_pdos.get_dos("p", spin="up"), label="p ↑")
    ax1.plot(energies, site_pdos.get_dos("d", spin="up"), label="d ↑")
    ax1.set_title("Spin Up")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Spin down
    ax2.plot(energies, site_pdos.get_dos("s", spin="down"), label="s ↓")
    ax2.plot(energies, site_pdos.get_dos("p", spin="down"), label="p ↓")
    ax2.plot(energies, site_pdos.get_dos("d", spin="down"), label="d ↓")
    ax2.set_title("Spin Down")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pdos_spin_resolved.png', dpi=300)
```

---

## Common Issues

### Issue 1: "PDOS files not generated"

**Symptoms**: Only total DOS file exists, no PDOS files

**Causes & Solutions**:

1. **Missing PDOS block**:
   ```python
   # Add to user_params
   "fdf_arguments": {
       "ProjectedDensityOfStates": ["-10.0  10.0  0.1  200  eV"]
   }
   ```

2. **Wrong SIESTA version**: PDOS.xml requires SIESTA 4.1+
   ```bash
   siesta --version
   ```

### Issue 2: "PDOS too noisy or has strange features"

**Cause**: Insufficient k-point sampling

**Solution**: PDOS requires denser k-points than total DOS
```python
user_params = {
    "kpts": [12, 12, 12]  # At least 10×10×10 for PDOS
}
```

### Issue 3: "Cannot distinguish orbital contributions"

**Cause**: Need to specify projections explicitly

**Solution**: Use `%block PDOS.kgrid_Monkhorst_Pack` for better control
```python
"fdf_arguments": {
    "PDOS.kgrid_Monkhorst_Pack": [
        "12  0  0  0.0",
        " 0 12  0  0.0",
        " 0  0 12  0.0"
    ]
}
```

### Issue 4: "Pymatgen cannot read PDOS"

**Cause**: Old SIESTA output format

**Solution**:
1. Use SIESTA 4.1+ which outputs PDOS.xml
2. Or parse manually from older format files

### Issue 5: "Unknown FDF parameter: fdf_arguments"

**Error**: `ValueError: Unknown FDF parameter(s): fdf_arguments`

**Cause**: Using deprecated `fdf_arguments` wrapper syntax

**Solution**:
```python
# ❌ OLD (doesn't work):
user_params = {
    "fdf_arguments": {
        "ProjectedDensityOfStates": [...],
        "PDOS.kgrid_Monkhorst_Pack": [...]
    }
}

# ✅ NEW (correct):
user_params = {
    "%block ProjectedDensityOfStates": [...],  # Directly in user_params!
    "%block PDOS.kgrid.MonkhorstPack": [...],
}
```

**Note**: Block parameters should be specified **directly** in `user_params`, NOT nested in `fdf_arguments`. See [FDF Block Parameters](#fdf-block-parameters-advanced) section below.

---

## PDOS Analysis Examples

### 1. Identify Band Character

```python
def identify_band_character(pdos, energy_window=(-2, 0)):
    """Determine which orbitals dominate near Fermi level."""
    energies = pdos.energies - pdos.efermi
    mask = (energies > energy_window[0]) & (energies < energy_window[1])

    dos_s = pdos.get_dos("s")[mask].sum()
    dos_p = pdos.get_dos("p")[mask].sum()
    dos_d = pdos.get_dos("d")[mask].sum()

    total = dos_s + dos_p + dos_d
    print(f"Near Fermi level ({energy_window[0]} to {energy_window[1]} eV):")
    print(f"  s-character: {dos_s/total*100:.1f}%")
    print(f"  p-character: {dos_p/total*100:.1f}%")
    print(f"  d-character: {dos_d/total*100:.1f}%")
```

### 2. Compare Cation vs Anion

```python
def compare_cation_anion(structure, pdos):
    """Compare PDOS of cation and anion sites."""
    # Assume first half are cations, second half anions
    n_atoms = len(structure)

    cation_pdos = pdos.get_site_dos_list(structure[:n_atoms//2])
    anion_pdos = pdos.get_site_dos_list(structure[n_atoms//2:])

    # Sum PDOS
    cation_total = sum([d.get_densities() for d in cation_pdos])
    anion_total = sum([d.get_densities() for d in anion_pdos])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    energies = pdos.energies - pdos.efermi
    ax.plot(energies, cation_total, label="Cation", linewidth=2)
    ax.plot(energies, anion_total, label="Anion", linewidth=2)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_xlabel('E - E$_F$ (eV)')
    ax.set_ylabel('PDOS (states/eV)')
    plt.savefig('cation_anion_pdos.png', dpi=300)
```

### 3. Orbital Overlap Analysis

```python
def calculate_overlap(pdos1, pdos2, energy_window=(-5, 5)):
    """Calculate PDOS overlap between two sites."""
    energies = pdos.energies - pdos.efermi
    mask = (energies > energy_window[0]) & (energies < energy_window[1])

    dos1 = pdos1.get_densities()[mask]
    dos2 = pdos2.get_densities()[mask]

    # Overlap integral
    overlap = np.trapz(dos1 * dos2, energies[mask])
    norm1 = np.trapz(dos1**2, energies[mask])
    norm2 = np.trapz(dos2**2, energies[mask])

    overlap_normalized = overlap / np.sqrt(norm1 * norm2)
    return overlap_normalized
```

---

## PDOS Parameter Reference

### ProjectedDensityOfStates Block

```python
user_params = {
    "%block ProjectedDensityOfStates": [
        "Emin  Emax  dE  npts  units"
    ]
}
```

**Parameters**:
- `Emin`: Minimum energy (eV or Ry)
- `Emax`: Maximum energy
- `dE`: Energy spacing (optional if npts given)
- `npts`: Number of energy points
- `units`: "eV" or "Ry"

**Example**:
```python
user_params = {
    "%block ProjectedDensityOfStates": [
        "-20.0  10.0  0.1  300  eV"  # -20 to +10 eV, 300 points
    ]
}
```

### PDOS K-Grid

For more control:
```python
user_params = {
    "%block PDOS.kgrid.MonkhorstPack": [
        "12  0  0  0.0",
        " 0 12  0  0.0",
        " 0  0 12  0.0"
    ]
}
```

---

## FDF Block Parameters (Advanced)

When you need to specify FDF block parameters (like custom PDOS energy ranges or k-grids), use the `"%block ParamName"` syntax **directly** in `user_params`.

**IMPORTANT**: DO NOT wrap block parameters in `fdf_arguments` - this is deprecated!

### Correct Usage

```python
# ✅ CORRECT: Block parameters directly in user_params
from atomate2.siesta.jobs.core import PDOSMaker

maker = PDOSMaker(
    user_params={
        "a2s_kpts": [12, 12, 12],
        "PAO.BasisSize": "DZP",
        "Spin": "polarized",

        # ProjectedDensityOfStates block (custom PDOS parameters)
        "%block ProjectedDensityOfStates": [
            "EF -20.0 10.0 0.05 600 eV",  # Energy: -20 to +10 eV, 600 points
        ],

        # Custom PDOS k-point grid
        "%block PDOS.kgrid.MonkhorstPack": [
            "12  0  0  0.0",
            " 0 12  0  0.0",
            " 0  0 12  0.0",
        ],
    },
    dry_run=True
)
```

### Incorrect Usage (Deprecated)

```python
# ❌ WRONG: Don't nest in fdf_arguments!
maker = PDOSMaker(
    user_params={
        "fdf_arguments": {  # <-- This doesn't work!
            "ProjectedDensityOfStates": [...],
            "PDOS.kgrid_Monkhorst_Pack": [...]
        }
    }
)
```

### Common Block Parameters for PDOS

- `"%block ProjectedDensityOfStates"` - Custom PDOS energy range and resolution
- `"%block PDOS.kgrid.MonkhorstPack"` - K-point grid for PDOS sampling
- `"%block DOS.kgrid.MonkhorstPack"` - K-point grid for total DOS
- `"%block DM.InitSpin"` - Initial magnetic moments for spin-polarized calculations

For comprehensive examples, see:
- [02-fdf-block-inputs](../../03-advanced-features/02-fdf-block-inputs/)
- [16-magnetic-calculations](../../03-advanced-features/16-magnetic-calculations/)

---

## Alternative: Using CLI Tool

```bash
# Interactive mode
atomate2siesta-maker --interactive
# Select: "pdos" → Choose structure → Done!

# Command-line mode
atomate2siesta-maker pdos MgO.cif
```

---

## Tips for Success

✅ **Use dense k-points**: 12×12×12 minimum for good PDOS
✅ **Check all orbitals**: Don't forget d and f if present
✅ **Compare with total DOS**: PDOS components should sum to total
✅ **Use spin polarization**: For magnetic systems, always include spin
✅ **Relax first**: Use optimized geometry for accurate PDOS
✅ **Block parameters**: Use `"%block ParamName"` directly in `user_params` - NO `fdf_arguments` wrapper!

---

## Next Steps

After completing PDOS tutorials:

1. **Optical properties**: [03-advanced-features/07-optical-properties](../../03-advanced-features/07-optical-properties/)
2. **COOP/COHP analysis**: [03-advanced-features/15-hamiltonian-overlap](../../03-advanced-features/15-hamiltonian-overlap/)
3. **Magnetic systems**: [03-advanced-features/16-magnetic-calculations](../../03-advanced-features/16-magnetic-calculations/)
4. **DFT+U**: [03-advanced-features/08-dftu](../../03-advanced-features/08-dftu/) - For correlated d/f electrons

---

## References

- **SIESTA Manual**: PDOS chapter
- **Mulliken Analysis**: Understanding orbital projections
- **Pymatgen PDOS**: [Documentation](https://pymatgen.org/pymatgen.electronic_structure.dos.html#pymatgen.electronic_structure.dos.CompleteDos)

---

*Back to [01-basics](../README.md) | [Main Tutorial Index](../../README.md)*
