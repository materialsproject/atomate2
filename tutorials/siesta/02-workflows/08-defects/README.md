# Tutorial: Defect Calculations

**Category**: 02-workflows/08-defects
**Difficulty**: Advanced
**Time**: Varies (dry-run ~5 min, full calculations ~hours to days)

---

## Overview

This tutorial category demonstrates defect formation energy calculations using SIESTA with pymatgen-analysis-defects integration. Learn how to compute formation energies for vacancies, substitutions, and interstitials with proper finite-size corrections.

**What Defect Calculations Tell You**:
- **Formation energy (Ef)**: Energy cost to create a defect (eV)
- **Charge transition levels**: Defect ionization energies
- **Defect concentrations**: Thermodynamic equilibrium populations
- **Doping limits**: Maximum achievable carrier concentrations
- **Compensation mechanisms**: Competing defect reactions

---

## What You'll Learn

- Using `DefectFlowMaker` for automated defect workflows
- Vacancy, substitution, and interstitial calculations
- Charged defect calculations with proper corrections
- Finite-size corrections (FNV, Freysoldt, Kumagai)
- Chemical potential handling for non-stoichiometric conditions
- Defect concentration vs temperature/Fermi level

---

## Prerequisites

- **Required**: [01-basics/01-RelaxMaker](../../01-basics/01-RelaxMaker/) completed
- **Required**: Understanding of defect chemistry and thermodynamics
- **Recommended**: [01-convergence](../01-convergence/) - Converged parameters essential!
- **Recommended**: Familiarity with pymatgen-analysis-defects package
- **Structure files**: Located in [00-structures](../../00-structures/)

---

## Key Concepts

### Defect Formation Energy

The formation energy of defect D in charge state q at Fermi level EF:

$
E_f[D^q] = E_{tot}[D^q] - E_{host} - \sum_i n_i \mu_i + q(E_F + E_VBM) + E_{corr}
$

Where:
- **E_tot[D^q]**: Total energy of defective supercell
- **E_host**: Total energy of pristine supercell
- **n_i**: Number of atoms added (positive) or removed (negative)
- **μ_i**: Chemical potential of species i
- **E_F**: Fermi level (0 = VBM, E_gap = CBM)
- **E_VBM**: Valence band maximum energy
- **E_corr**: Finite-size correction

### Defect Types

**Vacancy**: Remove atom from lattice site
- Example: V_O in MgO (oxygen vacancy)
- Formation reaction: O_O → V_O + ½O2(g)

**Substitution**: Replace host atom with dopant
- Example: Al_Mg in MgO (Al on Mg site)
- Formation reaction: Mg_Mg + Al → Al_Mg + Mg

**Interstitial**: Add atom to empty site
- Example: Li_i in MgO (interstitial Li)
- Formation reaction: Li → Li_i

### Charge States

Defects can be neutral or charged:
- **Vacancy**: Often positive (missing electrons)
  - V_O in oxides: 0, +1, +2
- **Substitutional dopants**: Determined by valence difference
  - Al_Mg in MgO: -1 (Al³⁺ on Mg²⁺ site)
- **Interstitials**: Often positive
  - Li_i: 0, +1

### Supercell Size Requirements

**Minimum size**: ~10 Å separation between periodic defect images
- Typical: 3×3×3 to 5×5×5 supercells
- Larger for charged defects (need to test convergence)

**Why**: Avoid spurious defect-defect interactions

### Finite-Size Corrections

**Problem**: Periodic boundary conditions → artificial interactions

**Correction schemes**:
1. **FNV** (Freysoldt-Neugebauer-Van de Walle): Electrostatic + potential alignment
2. **Freysoldt**: Similar, different implementation
3. **Kumagai**: More sophisticated for anisotropic systems

**When needed**: Essential for charged defects, minor for neutral

---

## Tutorial Subdirectories

### [01-DefectFlowMaker](01-DefectFlowMaker/)
**Description**: Complete automated defect workflow
**Tutorial Files**:
- `DefectFlowMaker_01_basic_vacancy.py` - Simple vacancy (neutral)
- `DefectFlowMaker_02_charged_vacancy.py` - Charged vacancy with corrections
- `DefectFlowMaker_03_all_vacancies.py` - All vacancy types in compound
- `DefectFlowMaker_04_substitution_dopant.py` - Substitutional doping
- `DefectFlowMaker_05_interstitial.py` - Interstitial defects
- `CorrectionComparisonFlowMaker.py` - Compare correction schemes

**Features**:
- Automatic defect generation from structure
- Bulk reference calculation
- Defect supercell relaxation
- Finite-size corrections
- Formation energy calculation
- Charge transition level analysis

---

## Quick Start

### Example 1: Simple Vacancy

```python
from atomate2.siesta.flows.defects import DefectFlowMaker
from pymatgen.core import Structure
from jobflow import run_locally

# Load pristine structure
structure = Structure.from_file("MgO.cif")

# Create vacancy defect workflow
flow = DefectFlowMaker(
    defect_type="vacancy",
    site_index=0,  # Remove atom at site 0
    charge_states=[0],  # Neutral vacancy
    supercell_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],  # 3×3×3
)

# Run
workflow = flow.make(structure)
results = run_locally(workflow, create_folders=True)
```

### Example 2: Charged Vacancy with Corrections

```python
flow = DefectFlowMaker(
    defect_type="vacancy",
    site_index=0,  # Oxygen site in MgO
    charge_states=[0, +1, +2],  # Test multiple charge states
    supercell_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
    correction_scheme="freysoldt",  # Enable finite-size corrections
)

workflow = flow.make(structure)
results = run_locally(workflow, create_folders=True)
```

### Example 3: Substitutional Dopant

```python
from pymatgen.core import Element

flow = DefectFlowMaker(
    defect_type="substitution",
    site_index=0,  # Mg site
    substitution_species=Element("Al"),  # Replace Mg with Al
    charge_states=[-1, 0],  # Al³⁺ on Mg²⁺ → -1 charge
    supercell_matrix=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],  # 4×4×4 (larger for charged)
    correction_scheme="kumagai",
)

workflow = flow.make(structure)
```

### Example 4: All Vacancies in Compound

```python
# Automatically generate all unique vacancy types
flow = DefectFlowMaker(
    defect_type="all_vacancies",  # V_Mg and V_O
    charge_states={
        "Mg": [0, +1, +2],
        "O": [0, +1, +2],
    },
    supercell_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
)

workflow = flow.make(structure)
# Will create 6 defect calculations (2 species × 3 charge states)
```

---

## Configuration Options

### Charge States

```python
# Single neutral defect
charge_states=[0]

# Multiple charge states
charge_states=[0, +1, +2]  # Test 0, +1, +2

# Negative charges (electron acceptors)
charge_states=[-2, -1, 0]

# Wide range (comprehensive study)
charge_states=[-2, -1, 0, +1, +2]
```

### Supercell Size

```python
# Small (testing only)
supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]]  # 2×2×2

# Standard (neutral defects)
supercell_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]]  # 3×3×3

# Large (charged defects)
supercell_matrix=[[4, 0, 0], [0, 4, 0], [0, 0, 4]]  # 4×4×4

# Very large (publication quality)
supercell_matrix=[[5, 0, 0], [0, 5, 0], [0, 0, 5]]  # 5×5×5
```

### Correction Schemes

```python
# No correction (neutral defects)
correction_scheme=None

# Freysoldt (most common)
correction_scheme="freysoldt"

# FNV (alternative)
correction_scheme="fnv"

# Kumagai (best for anisotropic)
correction_scheme="kumagai"
```

---

## Output

### Defect Formation Energy

```
job_defect_analysis_*/
├── formation_energies.json    # Ef vs charge state and Fermi level
├── charge_transition_levels.json  # ε(q1/q2) values
├── defect_concentrations.json  # c(T, EF)
├── formation_energy_plot.png  # Ef vs EF diagram
└── defect_summary.txt         # Text summary
```

### Analyzing Results

```python
import json
import matplotlib.pyplot as plt

# Read formation energies
with open("job_defect_analysis_*/formation_energies.json") as f:
    ef_data = json.load(f)

# Plot formation energy vs Fermi level
fermi_levels = ef_data['fermi_levels']  # 0 to Eg
ef_vs_fermi = ef_data['formation_energies']  # Dictionary by charge state

for q, ef_values in ef_vs_fermi.items():
    plt.plot(fermi_levels, ef_values, label=f"q={q}")

plt.xlabel("Fermi level (eV)")
plt.ylabel("Formation energy (eV)")
plt.title("V_O formation energy vs Fermi level")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("ef_vs_fermi.png", dpi=300)
```

### Charge Transition Levels

```python
# Read transition levels
with open("job_defect_analysis_*/charge_transition_levels.json") as f:
    ctl_data = json.load(f)

print("Charge Transition Levels:")
for transition, energy in ctl_data.items():
    print(f"  ε({transition}): {energy:.2f} eV")

# Example output:
# ε(0/+1): 1.25 eV
# ε(+1/+2): 2.87 eV
```

**Interpretation**: Fermi level where defect changes charge state

---

## Best Practices

✅ **Large supercells**: ≥3×3×3 for neutral, ≥4×4×4 for charged defects
✅ **Converged parameters**: Use converged k-points, basis, cutoff
✅ **Test supercell convergence**: Compare 3×3×3 vs 4×4×4 vs 5×5×5
✅ **Use corrections for charged defects**: freysoldt or kumagai scheme
✅ **Tight relaxation**: MD.MaxForceTol < 0.02 eV/Å for defect supercells
✅ **Same parameters**: Bulk and defect calculations must use identical settings
✅ **Chemical potentials**: Consider O-rich vs O-poor limits

❌ **Don't use small supercells**: < 2×2×2 gives spurious interactions
❌ **Don't mix parameters**: Bulk and defect must match exactly
❌ **Don't skip corrections**: Essential for charged defects
❌ **Don't ignore band gap errors**: GGA underestimates gaps → affects Ef
❌ **Don't forget chemical potential limits**: μ must respect stability

---

## Chemical Potentials

Formation energy depends on chemical environment:

**Example: Oxygen vacancy in MgO**

O-rich limit (μ_O = ½E[O2]):
```python
mu_O = 0.5 * E_O2  # DFT energy of O2 molecule
Ef[V_O] = E_defect - E_bulk + mu_O + q*EF
```

O-poor limit (μ_O determined by stability):
```python
# MgO must be stable: μ_Mg + μ_O = E[MgO]
mu_O = E_MgO - mu_Mg_max
# where mu_Mg_max = E[Mg bulk]
```

**Range**: μ_O(poor) ≤ μ_O ≤ μ_O(rich)

---

## Troubleshooting

**Problem**: Formation energies change significantly with supercell size

**Solution**:
1. Increase supercell: 3×3×3 → 4×4×4 → 5×5×5
2. Check defect-defect separation > 10 Å
3. Enable finite-size corrections for charged defects
4. Test correction scheme convergence

---

**Problem**: Charged defect energies unrealistic

**Solution**:
1. **Enable corrections**: `correction_scheme="freysoldt"`
2. Check band gap: GGA underestimates → affects charge states
3. Verify VBM/CBM alignment between bulk and defect
4. Use scissor correction for band gap (advanced)

---

**Problem**: Different correction schemes give different results

**Solution**:
- Normal variation: ~0.1-0.3 eV difference
- Large difference (> 0.5 eV) → supercell too small
- Test all three schemes, report range
- Kumagai most robust for anisotropic systems

---

**Problem**: Defect relaxation doesn't converge

**Solution**:
1. Loosen tolerance: `MD.MaxForceTol="0.05 eV/Ang"`
2. Enable custodian: Automatic error recovery
3. Check for structural reconstruction (may need different initial guess)
4. Increase MD.NumCGSteps

---

## Defect Concentrations

From formation energies, compute equilibrium concentrations:

**Dilute limit**:

$
c_D = N_{\text{sites}} \exp\left(-\frac{E_f}{k_B T}\right)
$

```python
import numpy as np

Ef = 2.5  # eV (formation energy)
T = 1000  # K
kB = 8.617e-5  # eV/K
N_sites = 1.0  # sites per unit cell (normalized)

c = N_sites * np.exp(-Ef / (kB * T))
print(f"Defect concentration at {T} K: {c:.2e}")
```

**Example**: Ef = 2.5 eV at 1000 K → c ≈ 10⁻⁶ (1 ppm)

---

## Next Steps

After completing defect calculations:

1. **Compare with experiments**: Literature defect formation energies
2. **Doping study**: Systematic substitutional dopants
3. **Defect migration**: [05-barriers](../05-barriers/) - NEB for vacancy hopping
4. **Optical transitions**: Defect-related absorption/emission (future tutorials)

---

## Related Tutorials

- [05-barriers](../05-barriers/) - Defect migration barriers (NEB)
- [01-convergence](../01-convergence/) - Parameter convergence (critical!)
- [03-advanced-features/05-charged-calculations](../../03-advanced-features/02-physics-features/05-charge/) - Charged species

---

## References

- **Defect formation energies**: Freysoldt et al., Rev. Mod. Phys. 86, 253 (2014)
- **Finite-size corrections**: Freysoldt et al., Phys. Rev. Lett. 102, 016402 (2009)
- **Kumagai correction**: Kumagai & Oba, Phys. Rev. B 89, 195205 (2014)
- **Charge transition levels**: Lany & Zunger, Phys. Rev. B 78, 235104 (2008)
- **pymatgen-analysis-defects**: https://github.com/materialsproject/pymatgen-analysis-defects

---

*Back to [02-workflows](../README.md) | [Main Tutorial Index](../../README.md)*
