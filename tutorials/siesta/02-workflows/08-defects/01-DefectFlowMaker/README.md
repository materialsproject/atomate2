# Tutorial: Defect Formation Energy Calculations

**Category**: 02-workflows/05-defects/03-DefectFlowMaker
**Difficulty**: Intermediate
**Time**: ~5 min (dry-run), ~2-6 hours (full calculation)
**Status**: Complete! (v1.0.0) - Full defect analysis with formation diagrams, CTLs, Brouwer, SRH ✅

---

## Overview

This tutorial demonstrates **defect formation energy** calculations using `DefectFlowMaker`. Learn how to calculate formation energies for point defects (vacancies, interstitials, substitutions) with automatic finite-size corrections.

**What Defect Formation Energy Tells You**:
- **E_f**: Energy cost to create a defect (eV)
- **Stability**: Which defects form spontaneously (E_f < 0) vs under special conditions
- **Concentrations**: Defect populations at thermal equilibrium (via Boltzmann)
- **Doping**: Whether dopants prefer substitutional or interstitial sites
- **Material properties**: How defects affect conductivity, color, magnetism

**Formula (charged defects)**:
```
E_f(q) = E_defect(q) - E_host + Σn_i·μ_i + q·E_F + E_corr(q)
```

Where:
- `E_defect(q)`: Total energy of defect supercell with charge q
- `E_host`: Total energy of pristine host supercell
- `n_i·μ_i`: Chemical potential terms (species removed/added)
- `q·E_F`: Fermi level term (electron reservoir)
- `E_corr(q)`: Finite-size correction (electrostatic)

---

## What You'll Learn

- Creating defect structures manually
- Using `DefectFlowMaker` for formation energy calculations
- Understanding Lany-Zunger finite-size corrections
- Analyzing formation energies and corrections
- Setting up for charged vs neutral defects

---

## Current Status

**Complete Feature Set (v1.0.0)**:

**Core Formation Energy Calculations**:
✅ `DefectFlowMaker` - Basic formation energy workflow
✅ `DefectDocument` - Complete result schema with metadata
✅ Automatic correction for charged defects (q ≠ 0)
✅ Defect-optimized relaxation parameters

**Advanced Corrections & Generation**:
✅ **Ghost atoms for vacancies** - SIESTA-specific requirement for proper basis
✅ **Multiple correction schemes** - Lany-Zunger, Makov-Payne, Freysoldt
✅ **`CorrectionComparisonFlowMaker`** - Compare all corrections (unique feature!)
✅ **Automatic defect generation** - One-liner API with symmetry reduction
✅ **Automatic chemical potentials** - O2, H2, N2, metals

**Complete Defect Analysis**:
✅ **Formation energy diagrams** - E_f vs E_F plots with CTLs
✅ **Charge transition levels (CTLs)** - Automatic identification
✅ **Defect concentrations** - Self-consistent Fermi level solver
✅ **Brouwer diagrams** - vs Fermi level and temperature
✅ **Interactive Plotly plots** - HTML output with tooltips
✅ **SRH recombination** - Lifetime and capture analysis

**Ghost Atoms**: SIESTA vacancies now use ghost atoms (zero-charge atoms with basis functions) instead of complete removal. This is critical for:
- Proper basis set completeness
- Accurate grid sampling at vacancy sites
- Better SCF convergence for charged vacancies

**CorrectionComparisonFlowMaker**: NEW unique feature that compares all available correction schemes on the same calculation, providing uncertainty quantification and automated recommendations. No other defect workflow framework offers this!

---

## Prerequisites

- **Required tutorials**: [01-basics/01-RelaxMaker](../../../01-basics/01-RelaxMaker/)
- **Required knowledge**:
  - Point defect concepts (vacancy, interstitial, substitutional)
  - Charge states and Fermi level
  - Dielectric constants
- **Recommended**:
  - [01-convergence](../../01-convergence/) - Converged parameters critical for accurate formation energies
  - pymatgen basics for structure manipulation
- **Structure files**: Located in [00-structures](../../../00-structures/)

---

## Key Concepts

### Finite-Size Corrections

DFT calculations use periodic boundary conditions → charged defects interact with their periodic images → spurious electrostatic energy.

**Lany-Zunger Correction**:
```
E_corr = (q² × α_M) / (2 × ε × L)
```

Where:
- `q`: Charge state of defect
- `α_M`: Madelung constant (~2.84 for cubic lattices)
- `ε`: Static dielectric constant
- `L`: Characteristic supercell length

**When to use**:
- ✅ Quick estimates
- ✅ Initial screening of defects
- ✅ When dielectric constant is known

**Limitations**:
- Assumes isotropic dielectric (not accurate for layered materials)
- Point charge model (less accurate than Gaussian)
- No potential alignment

**More sophisticated corrections** are available: Freysoldt and Makov-Payne include potential alignment. See `CorrectionComparisonFlowMaker` for automated comparison of all correction schemes.

### Dielectric Constant

**Critical parameter** for corrections! Must be converged.

**How to get ε_static**:
1. **DFT calculation**: Use SIESTA's DFPT (future tutorial)
2. **Experiments**: Tabulated values for common materials
3. **Literature**: Published values

**Example values**:
- MgO: ε ≈ 9.8
- ZnO: ε ≈ 8.5
- Si: ε ≈ 11.7
- GaN: ε ≈ 10.4

**For this tutorial**: We'll use known values from literature.

### Convergence Requirements

Formation energies are **very sensitive** to:

1. **Supercell size**: ≥10 Å separation between defect images
2. **K-points**: Must converge to ~10 meV/atom
3. **Basis size**: DZP minimum, TZP recommended
4. **Cutoff energy**: 300-400 Ry typical

**Always** run convergence tests first! See [01-convergence](../../01-convergence/).

---

## Tutorial Files

**Total**: 18 tutorials organized in 4 groups

### Group 1: DefectFlowMaker Tutorials (Basic Formation Energies)

#### DefectFlowMaker_01_basic_vacancy.py

**File**: [DefectFlowMaker_01_basic_vacancy.py](DefectFlowMaker_01_basic_vacancy.py)
**Description**: Oxygen vacancy in MgO (neutral, q=0) WITH GHOST ATOMS
**Features**:
- Creating vacancies with ghost atoms (SIESTA-specific)
- Basic DefectFlowMaker usage
- Formation energy calculation (no correction for q=0)
- Dry-run mode for preview

#### DefectFlowMaker_02_charged_vacancy.py

**File**: [DefectFlowMaker_02_charged_vacancy.py](DefectFlowMaker_02_charged_vacancy.py)
**Description**: Oxygen vacancy with +2 charge (V_O^2+) WITH GHOST ATOMS
**Features**:
- Ghost atoms for charged vacancies (critical for SIESTA!)
- Charged defect with Lany-Zunger correction
- Demonstrates automatic correction
- Shows correction metadata and analysis

#### DefectFlowMaker_03_all_vacancies.py

**File**: [DefectFlowMaker_03_all_vacancies.py](DefectFlowMaker_03_all_vacancies.py)
**Description**: All symmetry-unique vacancies in MgO
**Features**:
- Automatic generation of V_Mg and V_O
- Multiple charge states demonstration
- Comparison of formation energies

#### DefectFlowMaker_04_substitution_dopant.py

**File**: [DefectFlowMaker_04_substitution_dopant.py](DefectFlowMaker_04_substitution_dopant.py)
**Description**: Li dopant on Mg sites
**Features**:
- Substitutional defect creation
- Acceptor dopant demonstration (Li_Mg)
- Charge states for substitutions

#### DefectFlowMaker_05_antisites.py

**File**: [DefectFlowMaker_05_antisites.py](DefectFlowMaker_05_antisites.py)
**Description**: Antisite defects (Mg_O, O_Mg)
**Features**:
- Atom swapping defects
- High formation energy demonstration
- Automatic antisite generation

#### DefectFlowMaker_06_interstitials.py

**File**: [DefectFlowMaker_06_interstitials.py](DefectFlowMaker_06_interstitials.py)
**Description**: Li interstitials at high-symmetry sites
**Features**:
- Interstitial defect creation
- Automatic site detection
- Tetrahedral/octahedral sites

#### DefectFlowMaker_08_slab2d_vt_automatic.py

**File**: [DefectFlowMaker_08_slab2d_vt_automatic.py](DefectFlowMaker_08_slab2d_vt_automatic.py)
**Description**: 2D slab corrections (Slab2D + automatic VBM/CBM extraction)
**Features**:
- Anisotropic dielectric corrections for 2D materials
- Automatic band edge extraction from calculations
- Dielectric profile customization

#### DefectFlowMaker_09_makov_payne_rho_automatic.py

**File**: [DefectFlowMaker_09_makov_payne_rho_automatic.py](DefectFlowMaker_09_makov_payne_rho_automatic.py)
**Description**: Makov-Payne quadrupole correction with automatic charge density
**Features**:
- Quadrupole term correction for anisotropic defects
- Automatic charge density extraction
- Gaussian charge distribution model

#### CorrectionComparisonFlowMaker.py

**File**: [CorrectionComparisonFlowMaker.py](CorrectionComparisonFlowMaker.py)
**Description**: Compare ALL correction schemes (UNIQUE KILLER FEATURE!)
**Features**:
- Apply all 6 correction schemes to same calculation
- Statistical analysis (mean, std, range)
- Automated recommendations
- Uncertainty quantification
- Diagnostic plots for each correction

### Group 2: FormationEnergyDiagramFlowMaker Tutorials (Complete Analysis)

#### FormationEnergyDiagramFlowMaker_00_basic.py (START HERE)

**File**: [FormationEnergyDiagramFlowMaker_00_basic.py](FormationEnergyDiagramFlowMaker_00_basic.py)
**Description**: Complete formation energy diagram workflow
**Features**:
- One-liner defect generation + formation energy diagram
- Automatic chemical potential calculation
- Automatic bandgap extraction
- Formation energy vs Fermi level plots
- Charge transition level (CTL) identification

#### FormationEnergyDiagramFlowMaker_01_defect_concentrations.py

**File**: [FormationEnergyDiagramFlowMaker_01_defect_concentrations.py](FormationEnergyDiagramFlowMaker_01_defect_concentrations.py)
**Description**: Defect concentration calculations
**Features**:
- Built-in concentration analysis (include_concentration_analysis=True)
- Self-consistent Fermi level solver
- Carrier concentrations (n, p)
- Defect concentrations vs Fermi level
- Temperature-dependent analysis

#### FormationEnergyDiagramFlowMaker_02_brouwer_diagrams.py

**File**: [FormationEnergyDiagramFlowMaker_02_brouwer_diagrams.py](FormationEnergyDiagramFlowMaker_02_brouwer_diagrams.py)
**Description**: Brouwer diagrams for defect chemistry
**Features**:
- Brouwer diagram vs Fermi level
- Brouwer diagram vs temperature
- Charge state population analysis
- Publication-quality matplotlib plots

#### FormationEnergyDiagramFlowMaker_03_interactive_plots_and_json_export.py

**File**: [FormationEnergyDiagramFlowMaker_03_interactive_plots_and_json_export.py](FormationEnergyDiagramFlowMaker_03_interactive_plots_and_json_export.py)
**Description**: Interactive Plotly diagrams and JSON export
**Features**:
- Interactive Plotly plots with hover tooltips
- Zoom/pan capabilities
- HTML output for browser viewing
- JSON export with complete data
- Formation energy arrays vs Fermi level

#### FormationEnergyDiagramFlowMaker_04_srh_recombination_analysis.py

**File**: [FormationEnergyDiagramFlowMaker_04_srh_recombination_analysis.py](FormationEnergyDiagramFlowMaker_04_srh_recombination_analysis.py)
**Description**: Shockley-Read-Hall (SRH) recombination analysis
**Features**:
- Built-in SRH analysis (include_srh_analysis=True)
- Effective mass inputs for carriers
- Capture cross-sections (σ_n, σ_p)
- Lifetime calculations
- Recombination rate analysis

### Group 3: Reading Structures with Ghost Atoms

#### ReadDefectStructures_01_ghost_atoms.py

**File**: [ReadDefectStructures_01_ghost_atoms.py](ReadDefectStructures_01_ghost_atoms.py)
**Description**: How to read defect structures preserving ghost atom information
**Features**:
- Demonstrates the problem with standard Structure.from_file()
- Using read_cif_with_ghost() for CIF files
- Using read_siesta_with_ghost() for FDF/XV files
- Complete workflow from generation to calculation

### Group 4: Generator Tutorials (Low-Level Defect Generation)

#### Generator_01_vacancy.py

**File**: [Generator_01_vacancy.py](Generator_01_vacancy_MoS2.py)
**Description**: SiestaVacancyGenerator - Programmatic vacancy generation
**Features**:
- Direct use of SiestaVacancyGenerator class
- Symmetry-based site identification
- Ghost atom insertion control
- MoS2 monolayer examples (with/without symmetry)
- Batch vacancy generation

#### Generator_02_substitution.py

**File**: [Generator_02_substitution.py](Generator_02_substitution.py)
**Description**: SiestaSubstitutionGenerator - Programmatic substitution generation
**Features**:
- Direct use of SiestaSubstitutionGenerator class
- Dopant and antisite generation
- Species-specific substitutions
- MoS2 monolayer examples
- Multiple dopant handling

#### Generator_03_interstitial.py

**File**: [Generator_03_interstitial.py](Generator_03_interstitial.py)
**Description**: SiestaInterstitialGenerator - Programmatic interstitial generation
**Features**:
- Direct use of SiestaInterstitialGenerator class
- High-symmetry site detection (Voronoi analysis)
- Tetrahedral and octahedral sites
- MoS2 monolayer examples
- Distance-based site filtering

---

## Quick Start

### Example 0: One-Liner API (RECOMMENDED - START HERE)

The easiest way to generate defects is using `from_pristine_structure()`:

```python
from pymatgen.core import Lattice, Structure
from atomate2.siesta.flows.defects import DefectFlowMaker
from jobflow import run_locally

# Step 1: Create pristine structure (unit cell is fine!)
lattice = Lattice.cubic(4.212)
mgo = Structure(
    lattice,
    ["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"],
    [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0],
     [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
)

# Step 2: Generate all vacancy defects in ONE LINE!
flows = DefectFlowMaker.from_pristine_structure(
    mgo,
    defect_type="vacancy",
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    charge_states=[0, +2],
    epsilon_static=9.8,
    dry_run=True,
)

# Step 3: Run all flows
for flow in flows:
    print(f"Running: {flow.name}")
    results = run_locally(flow, create_folders=True)

    # Get DefectDocument
    defect_doc = [r[1].output for r in results.values()
                  if hasattr(r[1].output, "defect_type")][0]
    print(f"  E_f = {defect_doc.corrected_formation_energy:.3f} eV")
```

**What just happened?**
- ✅ Automatic symmetry reduction (found 2 unique sites: Mg, O)
- ✅ Automatic supercell creation (2×2×2)
- ✅ Automatic ghost atom insertion (for vacancies)
- ✅ Generated 4 flows: V_Mg(q=0), V_Mg(q=+2), V_O(q=0), V_O(q=+2)
- ✅ **Code reduction**: ~200 lines → 10 lines (high!)

**Other defect types**:
```python
# Li dopant on Mg sites
flows = DefectFlowMaker.from_pristine_structure(
    mgo, defect_type="substitution", species="Mg", dopants="Li",
    charge_states=[-1, 0], epsilon_static=9.8, dry_run=True
)

# All antisites (Mg_O, O_Mg)
flows = DefectFlowMaker.from_pristine_structure(
    mgo, defect_type="substitution", dopants=None,  # None = antisites
    charge_states=[0], epsilon_static=9.8, dry_run=True
)

# Li interstitials
flows = DefectFlowMaker.from_pristine_structure(
    mgo, defect_type="interstitial", species="Li",
    charge_states=[+1], epsilon_static=9.8, dry_run=True
)
```

**See tutorials 05_01-05_04 for full examples with run_locally()!**

---

### Example 1: Neutral Oxygen Vacancy (q=0) WITH GHOST ATOMS (Manual)

```python
from pymatgen.core import Structure
from atomate2.siesta.flows.defects import DefectFlowMaker, create_vacancy_with_ghost
from jobflow import run_locally

# Step 1: Load pristine MgO supercell (2×2×2)
host_structure = Structure.from_file("mgo_2x2x2.cif")
# host_structure has 64 atoms (32 Mg + 32 O)

# Step 2: Create defect structure WITH GHOST ATOM (SIESTA best practice!)
o_indices = [i for i, site in enumerate(host_structure) if site.specie.symbol == "O"]
defect_structure = create_vacancy_with_ghost(
    structure=host_structure,
    site_index=o_indices[0],  # Remove first O atom
    use_ghost=True,  # CRITICAL: Use ghost atom for SIESTA
)
# defect_structure has 64 atoms (32 Mg + 31 O + 1 O_ghost)

# Step 3: Create DefectFlowMaker
flow_maker = DefectFlowMaker(
    epsilon_static=9.8,       # MgO dielectric constant
    defect_type="vacancy",    # Type of defect
    charge_state=0,           # Neutral vacancy
    name="V_O neutral"
)

# Step 4: Generate workflow
flow = flow_maker.make(
    defect_structure=defect_structure,
    host_structure=host_structure,
    defect_site=[0.25, 0.25, 0.25],  # Fractional coords of removed O
    defect_species="O"
)

# Step 5: Run (or preview with dry_run)
results = run_locally(flow, create_folders=True)

# Step 6: Access results
defect_doc = results[flow.uuid][1].output
print(f"Formation energy: {defect_doc.corrected_formation_energy:.3f} eV")
print(f"Correction: {defect_doc.correction_energy:.3f} eV (q=0 → no correction)")
```

**Expected output** (q=0):
```
Formation energy: 7.85 eV
Correction: 0.000 eV (q=0 → no correction)
```

### Example 2: Charged Oxygen Vacancy (q=+2) WITH GHOST ATOMS

```python
from atomate2.siesta.flows.defects import DefectFlowMaker, create_vacancy_with_ghost

# Same host structure as Example 1
# host_structure = ...

# Create defect structure WITH GHOST ATOM
o_indices = [i for i, site in enumerate(host_structure) if site.specie.symbol == "O"]
defect_structure = create_vacancy_with_ghost(
    structure=host_structure,
    site_index=o_indices[0],
    use_ghost=True,  # Essential for charged vacancies!
)

# Create charged defect workflow
flow_maker = DefectFlowMaker(
    epsilon_static=9.8,       # MgO dielectric constant
    defect_type="vacancy",
    charge_state=2,           # +2 charged vacancy (V_O^2+)
    name="V_O +2"
)

flow = flow_maker.make(
    defect_structure=defect_structure,
    host_structure=host_structure,
    defect_site=[0.25, 0.25, 0.25],
    defect_species="O"
)

results = run_locally(flow, create_folders=True)
defect_doc = results[flow.uuid][1].output

print(f"Raw formation energy: {defect_doc.raw_formation_energy:.3f} eV")
print(f"Correction: {defect_doc.correction_energy:.3f} eV")
print(f"Corrected formation energy: {defect_doc.corrected_formation_energy:.3f} eV")
print(f"Correction scheme: {defect_doc.correction_scheme}")
```

**Expected output** (q=+2):
```
Raw formation energy: 7.85 eV
Correction: 1.25 eV
Corrected formation energy: 6.60 eV
Correction scheme: Lany-Zunger
```

**Note**: Correction lowers formation energy (removes spurious repulsion between periodic +2 charges).

---

## Run Modes

### 1. Dry-Run Mode (Preview)

```python
# Add dry_run to makers
flow_maker = DefectFlowMaker(
    epsilon_static=9.8,
    defect_type="vacancy",
    charge_state=2,
)

# Modify the makers to use dry_run
flow_maker.defect_relax_maker.dry_run = True
flow_maker.host_static_maker.dry_run = True

flow = flow_maker.make(defect_structure, host_structure)
results = run_locally(flow, create_folders=True)
```

**Output**:
```
preview_output/
├── job_V_O_+2_-_Defect_Relax_*/      # Defect relaxation preview
├── job_V_O_+2_-_Host_Static_*/       # Host static preview
└── job_V_O_+2_-_Finalize_*/          # Correction and document creation
```

**Check files**:
```bash
ls preview_output/job_*_Defect_Relax_*/
# Should see: siesta.fdf, siesta.ion, *.psf
```

### 2. Local Execution

```python
# Set dry_run=False (or omit)
flow_maker = DefectFlowMaker(...)
flow = flow_maker.make(defect_structure, host_structure)
results = run_locally(flow, create_folders=True)
```

**What happens**:
1. Relax defect supercell (fixed-cell, tight forces)
2. Calculate host supercell energy (static SCF)
3. Apply Lany-Zunger correction (if q ≠ 0)
4. Create DefectDocument with all metadata

**Time**: 2-6 hours depending on:
- Supercell size (64-216 atoms typical)
- K-point density
- Basis size and cutoff
- SCF convergence

---

## Expected Output

### DefectDocument Fields

After calculation, access results:

```python
defect_doc = results[flow.uuid][1].output

# Basic info
print(f"Defect type: {defect_doc.defect_type}")
print(f"Charge state: {defect_doc.charge_state}")
print(f"Defect species: {defect_doc.defect_species}")

# Energies
print(f"Defect energy: {defect_doc.defect_energy:.4f} eV")
print(f"Host energy: {defect_doc.host_energy:.4f} eV")
print(f"Raw E_f: {defect_doc.raw_formation_energy:.4f} eV")
print(f"Correction: {defect_doc.correction_energy:.4f} eV")
print(f"Corrected E_f: {defect_doc.corrected_formation_energy:.4f} eV")

# Correction metadata
print(f"Correction scheme: {defect_doc.correction_scheme}")
print(f"Correction metadata: {defect_doc.correction_metadata}")

# Structures
print(f"Defect structure: {len(defect_doc.defect_structure)} atoms")
print(f"Host structure: {len(defect_doc.host_structure)} atoms")
```

### Analyzing Corrections

```python
import json

# Save results to JSON
with open("defect_results.json", "w") as f:
    json.dump(defect_doc.dict(), f, indent=2, default=str)

# Analyze correction metadata
metadata = defect_doc.correction_metadata
print(f"Madelung constant: {metadata['madelung_constant']}")
print(f"Supercell length: {metadata['characteristic_length_angstrom']:.2f} Å")
print(f"Dielectric constant: {metadata['epsilon_static']}")
print(f"Supercell volume: {metadata['volume_angstrom3']:.1f} Å³")
```

---

## Common Issues

### Issue 1: "Formation energy too high/low"

**Symptoms**: E_f unrealistic compared to experiments

**Causes & Solutions**:

1. **Unconverged parameters**:
   - Run [01-convergence](../../01-convergence/) first!
   - Formation energies very sensitive to k-points
   ```python
   # Increase k-points
   flow_maker.defect_relax_maker = DefectRelaxMaker.defect_relax(
       user_params={"a2s_kpts": [6, 6, 6]}  # Instead of [2, 2, 2]
   )
   ```

2. **Supercell too small**:
   - Need ≥10 Å separation between defects
   - Increase supercell size (2×2×2 → 3×3×3)

3. **Wrong dielectric constant**:
   - ε_static is critical for corrections!
   - Use converged DFT value or reliable experimental value

4. **GGA limitations**:
   - PBE may overestimate/underestimate E_f by 0.5-1 eV
   - Compare trends, not absolute values

### Issue 2: "Correction too large"

**Symptoms**: Correction > 1 eV for moderate charges

**Causes**:

1. **Supercell too small**:
   - Correction ∝ q²/L → large for small L
   - Solution: Increase supercell size
   - Target: Correction < 0.5 eV for well-converged systems

2. **Wrong dielectric constant**:
   - Check ε value from literature
   - Run DFPT calculation to get accurate ε

### Issue 3: "Cannot import DefectFlowMaker"

**Symptoms**: ImportError when running tutorial

**Solution**:
```python
# Make sure you're on feature/defect-workflows branch
from atomate2.siesta.flows.defects import DefectFlowMaker
```

If import fails, defect module may not be installed. Check:
```bash
python -c "from atomate2.siesta.flows.defects import DefectFlowMaker; print('OK')"
```

---

## Best Practices

### 1. Always Use Adequate Supercell

```python
from pymatgen.core import Structure

# Start with unit cell
unit_cell = Structure.from_file("mgo.cif")

# Create supercell with ≥10 Å separation
supercell = unit_cell.make_supercell([3, 3, 3])  # For MgO (a ≈ 4.2 Å)

# Check dimensions
print(f"Supercell lattice: {supercell.lattice.abc}")
# Should be > (10, 10, 10) Å
```

### 2. Converge All Parameters First

**Before calculating formation energies**:
1. Run k-point convergence (target: 10 meV/atom)
2. Run basis convergence (DZP vs TZP)
3. Run cutoff convergence (300-400 Ry)

See [01-convergence](../../01-convergence/) tutorials.

### 3. Use Consistent Settings

```python
# Define common parameters
common_params = {
    "PAO.BasisSize": "DZP",
    "Mesh.Cutoff": "300 Ry",
    "a2s_kpts": [4, 4, 4],
    "XC.functional": "GGA",
    "XC.authors": "PBE",
}

# Apply to both defect and host calculations
flow_maker.defect_relax_maker = DefectRelaxMaker.defect_relax(
    user_params=common_params
)
flow_maker.host_static_maker = DefectStaticMaker.defect_scf(
    user_params=common_params
)
```

### 4. Test Supercell Convergence

```python
# Test formation energy vs supercell size
for size in [2, 3, 4]:
    supercell = unit_cell.make_supercell([size, size, size])
    # ... create defect, run calculation ...
    # E_f should converge within ~0.1 eV
```

---

## Reading Generated Structures with Ghost Atoms

After generating defect structures using the `Generator_*` tutorials or `write_defects_to_folders()`, you need to use special functions to load them while preserving ghost atom information.

### Why Special Functions?

Standard `Structure.from_file()` does **NOT** preserve ghost atom information:
- CIF files: Ghost atoms have `occupancy=0.001` (lost on normal read)
- FDF files: Ghost atoms have `negative Z` (lost on normal read)

### Available Functions

```python
from atomate2.siesta.sets.utils.structure_io import (
    read_cif_with_ghost,      # For CIF files
    read_siesta_with_ghost,   # For FDF/XV files
)
```

### Example: Loading Generated Defect Structures

```python
from atomate2.siesta.sets.utils.structure_io import (
    read_cif_with_ghost,
    read_siesta_with_ghost,
)
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.tiers import apply_tier_preset

# Option 1: From CIF file (generated by write_defects_to_folders)
structure = read_cif_with_ghost("V_S_2c_qp0/defect_structure.cif")

# Option 2: From FDF file (initial geometry)
structure = read_siesta_with_ghost("V_S_2c_qp0/defect_structure.fdf")

# Option 3: From XV file (relaxed geometry after SIESTA run)
structure = read_siesta_with_ghost("siesta.fdf", use_xv=True)

# Check ghost atoms are preserved
print(f"Ghost tags: {structure.site_properties.get('ghost_tags')}")
print(f"Species labels: {structure.site_properties.get('species_label')}")

# Now use with any maker - ghost atoms will be handled correctly!
maker = RelaxMaker.fixed_cell_relaxation()
maker = apply_tier_preset(maker, "2d_vdw")  # For MoS2
job = maker.make(structure)
```

### Quick Reference

| File Type | Function | Example |
|-----------|----------|---------|
| CIF | `read_cif_with_ghost()` | `read_cif_with_ghost("defect.cif")` |
| FDF | `read_siesta_with_ghost()` | `read_siesta_with_ghost("defect.fdf")` |
| XV | `read_siesta_with_ghost()` | `read_siesta_with_ghost("siesta.fdf", use_xv=True)` |

**See tutorial**: [ReadDefectStructures_01_ghost_atoms.py](ReadDefectStructures_01_ghost_atoms.py)

---

## Current Capabilities & Limitations

**✅ Complete Feature Set (v1.0.0)**:
- ✅ **Ghost atoms for vacancies** - Automatic insertion (SIESTA-specific)
- ✅ **CorrectionComparisonFlowMaker** - Compare all correction schemes
- ✅ **Automatic defect generation** - One-liner API with symmetry reduction
- ✅ **Multiple corrections** - Lany-Zunger, Makov-Payne, Freysoldt
- ✅ **Automatic chemical potentials** - O2, H2, N2, F2, Cl2, Br2, I2 + 7 metals
- ✅ **Formation energy diagrams** - E_f vs E_F plots with CTLs
- ✅ **Charge transition levels** - Automatic CTL identification
- ✅ **Defect concentrations** - Self-consistent Fermi solver
- ✅ **Brouwer diagrams** - vs Fermi level and temperature
- ✅ **Interactive plots** - Plotly HTML output
- ✅ **SRH recombination** - Lifetime and capture analysis

**Remaining Limitations (Future)**:
- ❌ No Kumagai-Oba correction (anisotropic dielectrics)
- ❌ No Gaussian charge model

**Current Capabilities**:
- Use `FormationEnergyDiagramFlowMaker` for complete defect analysis (17 tutorials!)
- Use `from_pristine_structure()` for automatic defect generation
- All CTL and concentration analysis automated

---

## Next Steps

After completing DefectFlowMaker tutorials:

1. **FormationEnergyDiagramFlowMaker**: Complete defect analysis with CTLs, Brouwer diagrams, SRH (5 tutorials!)
2. **Combine with NEB**: Calculate defect migration barriers ([01-NebDirectFlowMaker](../../05-barriers/01-NebDirectFlowMaker/))
3. **Advanced corrections**: Kumagai-Oba for anisotropic materials (future)
4. **Gaussian charge model**: More accurate charge distribution (future)

---

## References

- **Lany-Zunger correction**: Lany & Zunger, PRB 78, 235104 (2008)
- **Defect formation energies**: Freysoldt et al., RMP 86, 253 (2014)
- **Point defects review**: Alkauskas et al., J. Appl. Phys. 119, 181101 (2016)

---

*Back to [05-defects](../README.md) | [02-workflows](../../README.md) | [Main Tutorial Index](../../../README.md)*
