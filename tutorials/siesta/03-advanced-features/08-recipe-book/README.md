# Recipe Book: One-Line Workflows

The **Recipe Book** is the fastest, easiest way to set up atomate2siesta workflows. It provides high-level "recipes" that turn complex multi-step calculations into simple one-liners.

**📁 Tutorial Structure** (Streamlined Dec 2025):
- `01_basic_usage/` - 7 essential examples (12-27 lines each)
- `03_electronic_recipes/` - Band structure & DOS (37 lines)
- `04_mechanical_recipes/` - Elastic & EOS (43 lines)
- `05_thermal_recipes/` - Phonons & QHA (43 lines)
- `06_catalysis_recipes/` - Surfaces & adsorption (48 lines)
- `07_convergence_recipes/` - K-points & basis (42 lines)
- `08_combined_recipes/` - High-throughput (52 lines)

**Total**: 14 files, 433 lines (minimal context, maximum clarity)

## 🎯 Key Benefits

- **⚡ 10x faster**: Reduce 50+ lines of setup code to 1 line
- **🤖 Smart defaults**: Automatic parameter detection based on your structure
- **📚 Best practices**: Encodes expert knowledge and proven workflows
- **🎨 Flexible**: Easy to customize while keeping simplicity
- **📊 Comprehensive**: Covers all major material properties

## 📖 Table of Contents

1. [Quick Start](#quick-start)
2. [Structure Analysis](#structure-analysis)
3. [Complete Characterization](#complete-characterization)
4. [Electronic Properties](#electronic-properties)
5. [Mechanical Properties](#mechanical-properties)
6. [Thermal Properties](#thermal-properties)
7. [Surface & Catalysis](#surface--catalysis)
8. [Convergence Testing](#convergence-testing)
9. [Application-Specific Recipes](#application-specific-recipes)
10. [Customization](#customization)

---

## Quick Start

### The Ultimate One-Liner

```python
from atomate2.siesta.recipes import RecipeBook
from pymatgen.core import Structure
from jobflow import run_locally

# Load your structure
structure = Structure.from_file("POSCAR")

# Complete material characterization in ONE LINE!
flow = RecipeBook.complete_material_study(structure)

# Run it
results = run_locally(flow, create_folders=True)
```

That's it! This automatically:
- ✅ Analyzes your material type (metal/semiconductor/insulator)
- ✅ Selects optimal SIESTA parameters
- ✅ Calculates electronic properties (bands, DOS)
- ✅ Calculates mechanical properties (elastic constants, bulk modulus)
- ✅ Calculates thermal properties (phonons, QHA, thermal expansion)
- ✅ Generates publication-quality plots

---

## Structure Analysis

Before running calculations, you can analyze your structure to see what the Recipe Book recommends:

```python
from atomate2.siesta.recipes import RecipeBook
from pymatgen.core import Structure

structure = Structure.from_file("POSCAR")

# Print comprehensive analysis
RecipeBook.print_analysis(structure)
```

**Output**:
```
======================================================================
Material Analysis: Si2
======================================================================

📊 Basic Properties:
  - Formula: Si2
  - Atoms: 2
  - Volume: 40.05 ų
  - Density: 2.33 g/cm³

🔬 Electronic Properties:
  - Type: Insulator/Semiconductor
  - Magnetic elements: No
  - Heavy elements: No
  - Max Z: 14

🔮 Structural Properties:
  - Space group: 227
  - Crystal system: cubic
  - Layered: No

⚙️ Recommended SIESTA Settings:
  - K-points: [8, 8, 8]
  - Mesh cutoff: 300 Ry
  - Basis size: DZP
  - Tier: basic
  - Preset: relax_standard

💰 Computational Estimates:
  - Est. time: 1.5 hours
  - Est. memory: 2.1 GB
  - Recommended cores: 4

======================================================================
```

---

## Complete Characterization

### Full Material Study

```python
# Calculate ALL properties
flow = RecipeBook.complete_material_study(structure)
```

### Quick Characterization

For preliminary studies (~1-2 hours):

```python
# Fast essential properties only
flow = RecipeBook.quick_characterization(structure)
```

### Selective Properties

Choose which property categories to calculate:

```python
# Only electronic and mechanical
flow = RecipeBook.complete_material_study(
    structure,
    properties=["electronic", "mechanical"]
)

# With convergence testing
flow = RecipeBook.complete_material_study(
    structure,
    properties=["electronic", "thermal"],
    test_convergence=True  # Ensures optimal parameters
)
```

---

## Electronic Properties

### Complete Electronic Structure

```python
from atomate2.siesta.recipes import electronic_properties

# Relaxation + bands + DOS
flow = electronic_properties(structure)
```

### Band Structure Only

```python
from atomate2.siesta.recipes import band_structure_workflow

# Quick band structure
flow = band_structure_workflow(structure, relax_first=True)
```

### Density of States

```python
from atomate2.siesta.recipes import dos_workflow

# High-density k-point mesh for DOS
flow = dos_workflow(structure, dos_kpts_density=2000)
```

### Optical Properties

```python
from atomate2.siesta.recipes import optical_properties

# Dielectric function, absorption, reflectivity
flow = optical_properties(structure, energy_range=(0, 15))  # 0-15 eV
```

### Material-Specific Electronic Calculations

```python
from atomate2.siesta.recipes import metal_properties, semiconductor_properties

# Optimized for metals (MP occupation, electronic temperature)
flow = metal_properties(al_structure)

# Optimized for semiconductors (accurate band gap)
flow = semiconductor_properties(si_structure)
```

---

## Mechanical Properties

### Complete Mechanical Characterization

```python
from atomate2.siesta.recipes import mechanical_properties

# EOS + elastic constants + mechanical properties
flow = mechanical_properties(structure)
```

### Elastic Constants Only

```python
from atomate2.siesta.recipes import elastic_constants_workflow

# Full elastic tensor
flow = elastic_constants_workflow(structure)
```

### Equation of State (EOS) and Bulk Modulus

```python
from atomate2.siesta.recipes import eos_workflow

# EOS fit gives you BOTH E(V) curve AND bulk modulus
flow = eos_workflow(structure, number_of_frames=9)
# Output: bulk_modulus, equilibrium_volume, E0, EOS_fit
```

### Pressure-Dependent EOS

```python
from atomate2.siesta.recipes import pressure_eos_workflow

# EOS from 0-100 GPa
flow = pressure_eos_workflow(
    structure,
    pressure_range=(0, 100),  # GPa
    number_of_frames=11
)
```

### Hardness Estimation

```python
from atomate2.siesta.recipes import hardness_estimation

# Empirical hardness from elastic constants
flow = hardness_estimation(structure)
```

---

## Thermal Properties

### Complete Thermal Characterization

```python
from atomate2.siesta.recipes import thermal_properties

# Phonons + Grüneisen + QHA + thermal expansion
flow = thermal_properties(structure)
```

### Phonons Only

```python
from atomate2.siesta.recipes import phonon_workflow

# Phonon dispersion + DOS
flow = phonon_workflow(
    structure,
    supercell_matrix=(2, 2, 2)  # Auto-detected if not specified
)
```

### Grüneisen Parameters

```python
from atomate2.siesta.recipes import gruneisen_workflow

# Mode-dependent Grüneisen parameters
flow = gruneisen_workflow(structure, volume_change=0.01)  # ±1%
```

### Quasi-Harmonic Approximation

```python
from atomate2.siesta.recipes import qha_workflow

# Temperature-dependent thermodynamics
flow = qha_workflow(
    structure,
    temperature_range=(0, 1500, 20)  # 0-1500 K, 20 K steps
)
```

### Thermal Expansion

```python
from atomate2.siesta.recipes import thermal_expansion_workflow

# Combines Grüneisen + QHA for α(T)
flow = thermal_expansion_workflow(structure)
```

### High-Temperature Properties

```python
from atomate2.siesta.recipes import high_temperature_properties

# Extended temperature range
flow = high_temperature_properties(
    ceramic_structure,
    max_temperature=3000  # Up to 3000 K
)
```

### Vibrational Stability Check

```python
from atomate2.siesta.recipes import vibrational_stability_check

# Check for imaginary modes
flow = vibrational_stability_check(structure)
```

---

## Surface & Catalysis

### Surface Energy Calculation

```python
from atomate2.siesta.recipes import surface_energy_workflow

# Multiple surface facets
flow = surface_energy_workflow(
    bulk_structure,
    miller_indices=[(1,0,0), (1,1,0), (1,1,1)],  # Auto-detected if None
    slab_layers=5,
    vacuum=15.0  # Angstroms
)
```

### Adsorption Site Scanning

```python
from atomate2.siesta.recipes import adsorption_scanning_workflow
from pymatgen.core import Molecule

# Create adsorbate
co_molecule = Molecule(["C", "O"], [[0,0,0], [0,0,1.15]])

# Grid-based adsorption scan
flow = adsorption_scanning_workflow(
    slab_structure,
    adsorbate=co_molecule,
    grid_density=(7, 7),  # 7×7 grid
    height_above_surface=2.0
)
```

### Complete Catalysis Study

```python
from atomate2.siesta.recipes import catalysis_study

# Surface energy + adsorption for multiple molecules
h2 = Molecule(["H", "H"], [[0,0,0], [0,0,0.74]])
o2 = Molecule(["O", "O"], [[0,0,0], [0,0,1.21]])

flow = catalysis_study(
    pt_structure,
    adsorbates=[h2, o2],
    miller_indices=[(1,1,1)]
)
```

---

## Convergence Testing

### Complete Convergence Suite

```python
from atomate2.siesta.recipes import convergence_suite

# Test k-points + cutoff + basis
flow = convergence_suite(
    structure,
    property="energy",  # or "forces", "stress"
    tolerance=0.001  # eV/atom
)
```

### K-Points Convergence

```python
from atomate2.siesta.recipes import kpoints_convergence

# K-point mesh convergence
flow = kpoints_convergence(
    structure,
    tolerance=0.0005,
    kpts_range=[2, 4, 6, 8, 10, 12]  # Auto-detected if None
)
```

### Mesh Cutoff Convergence

```python
from atomate2.siesta.recipes import mesh_cutoff_convergence

# Real-space mesh cutoff
flow = mesh_cutoff_convergence(
    structure,
    cutoff_range=[200, 250, 300, 350, 400]  # Ry, auto-detected if None
)
```

### Basis Convergence

```python
from atomate2.siesta.recipes import basis_convergence

# Test basis size, energy shift, split norm
flow = basis_convergence(
    structure,
    test_basis_size=True,
    test_energy_shift=True,
    test_split_norm=True
)
```

### Quick vs Complete Convergence

```python
from atomate2.siesta.recipes import quick_convergence_check, complete_convergence

# Fast preliminary check (~30 min)
flow = quick_convergence_check(structure)

# Exhaustive for publication (~3-4 hours)
flow = complete_convergence(structure, tolerance=0.0001)
```

---

## Application-Specific Recipes

### Battery Cathode Screening

```python
from atomate2.siesta.recipes import battery_cathode_screening

# Optimized for Li-ion cathodes
flow = battery_cathode_screening(licoo2_structure)
```

### Thermoelectric Analysis

```python
from atomate2.siesta.recipes import thermoelectric_analysis

# Electronic + thermal + mechanical
flow = thermoelectric_analysis(pbs_structure)
```

### High-Temperature Ceramics

```python
from atomate2.siesta.recipes import high_temperature_ceramic

# Mechanical + thermal stability
flow = high_temperature_ceramic(
    alumina_structure,
    max_temperature=2500
)
```

### Magnetic Materials

```python
from atomate2.siesta.recipes import magnetic_material_study

# Spin-polarized calculations
flow = magnetic_material_study(fe_structure)
```

### Semiconductor Devices

```python
from atomate2.siesta.recipes import semiconductor_device_study

# Accurate band gap + optical
flow = semiconductor_device_study(si_structure)
```

### Phase Transitions

```python
from atomate2.siesta.recipes import structural_phase_transition

# Phonon stability + pressure dependence
flow = structural_phase_transition(structure)
```

---

## Customization

### Override Auto Parameters

```python
# Specify custom parameters
flow = RecipeBook.electronic_properties(
    structure,
    auto_params=True,  # Still use auto-detection
    user_params={
        "kpts": [12, 12, 12],  # Override k-points
        "Mesh.Cutoff": "400 Ry",  # Override cutoff
        "PAO.BasisSize": "TZP"  # Override basis
    }
)
```

### Choose Tier and Preset

```python
# Explicitly set tier/preset
flow = RecipeBook.mechanical_properties(
    structure,
    tier="advanced",
    preset="phonon_high_accuracy"
)
```

### Dry-Run Mode

Preview workflows without running calculations:

```python
# Generate all input files, but don't run
flow = RecipeBook.complete_material_study(
    structure,
    dry_run=True  # Just generate inputs
)
run_locally(flow, create_folders=True)
# Check dry_run_output/ directory for all input files
```

### Selective Workflow Steps

```python
# Customize what's included
flow = RecipeBook.mechanical_properties(
    structure,
    include_eos=True,
    include_elastic=False  # Skip elastic constants
)

flow = RecipeBook.thermal_properties(
    structure,
    include_phonons=True,
    include_gruneisen=False,  # Skip Grüneisen
    include_qha=True
)
```

---

## Comparison: Before vs After

### Before Recipe Book (50+ lines)

```python
from atomate2.siesta.jobs.core import RelaxMaker, BandStructureMaker
from atomate2.siesta.flows.elastic import ElasticMaker
from atomate2.siesta.jobs.phonon import PhonopyMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import Flow

# Manual structure analysis...
is_metal = check_if_metal(structure)
kpts = calculate_kpoints(structure)
cutoff = determine_cutoff(structure)
# ... more analysis ...

# Create makers with parameters
relax_maker = RelaxMaker(
    user_params={
        "kpts": kpts,
        "Mesh.Cutoff": cutoff,
        "PAO.BasisSize": "DZP",
        "OccupationFunction": "MP" if is_metal else "FD",
        # ... many more parameters ...
    }
)
relax_maker = apply_tier_preset(relax_maker, "relax_standard")

band_maker = BandStructureMaker(user_params={...})
elastic_maker = ElasticMaker(bulk_relax_maker=..., elastic_relax_maker=...)
phonon_maker = PhonopyMaker(static_maker=..., supercell_matrix=...)

# Create jobs
relax_job = relax_maker.make(structure)
band_job = band_maker.make(relax_job.output.structure, prev_dir=relax_job.output.dir_name)
elastic_job = elastic_maker.make(relax_job.output.structure)
phonon_job = phonon_maker.make(relax_job.output.structure)

# Assemble flow
flow = Flow([relax_job, band_job, elastic_job, phonon_job], ...)
```

### After Recipe Book (1 line!)

```python
from atomate2.siesta.recipes import RecipeBook

# That's it!
flow = RecipeBook.complete_material_study(structure)
```

**Result**: **50x less code**, automatic parameter selection, best practices built-in!

---

## Examples in This Directory

- `01_basic_usage.py` - Simple examples of main recipes
- `02_electronic_properties.py` - Electronic structure workflows
- `03_mechanical_properties.py` - Mechanical property calculations
- `04_thermal_properties.py` - Phonons and thermal expansion
- `05_convergence_testing.py` - Parameter convergence
- `06_application_recipes.py` - Domain-specific workflows
- `07_customization.py` - Advanced customization

---

## Tips & Best Practices

1. **Start with analysis**: Always run `RecipeBook.print_analysis()` first to see recommendations

2. **Use dry-run mode**: Preview workflows before running expensive calculations:
   ```python
   flow = RecipeBook.complete_material_study(structure, dry_run=True)
   ```

3. **Test convergence for important calculations**:
   ```python
   flow = RecipeBook.complete_material_study(structure, test_convergence=True)
   ```

4. **Choose appropriate recipes**: Use application-specific recipes when available
   ```python
   RecipeBook.battery_cathode_screening()  # Better than generic complete_material_study()
   ```

5. **Customize when needed**: Auto params are good, but you can always override:
   ```python
   flow = RecipeBook.electronic_properties(
       structure,
       user_params={"kpts": [16, 16, 16]}  # Denser for accurate DOS
   )
   ```

---

## Troubleshooting

### "ImportError: cannot import name 'RecipeBook'"

Make sure you have the latest atomate2siesta:
```bash
pip install -e . --upgrade
```

### "Structure analysis failed"

Some complex structures may need manual parameters:
```python
flow = RecipeBook.complete_material_study(
    structure,
    auto_params=False,  # Disable auto-detection
    user_params={...}  # Provide parameters manually
)
```

### "Calculation taking too long"

Use quick recipes for preliminary studies:
```python
RecipeBook.quick_characterization(structure)  # Faster
RecipeBook.quick_convergence_check(structure)  # For convergence
```

---

## Next Steps

- Check individual example scripts in this directory
- Read the main atomate2siesta documentation
- Explore the recipe source code: `src/atomate2/siesta/recipes/`

Happy calculating! 🚀
