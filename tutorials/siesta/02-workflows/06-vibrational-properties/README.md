# 05: Vibrational Properties

**Focus**: Phonons, Grüneisen parameters, and thermal properties

**Difficulty**: Advanced

**Prerequisites**:
- Completed [02-convergence](../../02-workflows/01-convergence/) (converged parameters essential!)
- Understanding of phonons and lattice dynamics
- Familiarity with phonopy package
- Well-relaxed structures (forces < 0.01 eV/Å)

---

## Tutorials in This Category

### [01-phonons](01-phonons/)
**Description**: Phonon calculations using phonopy integration with automatic plotting
**Difficulty**: Advanced
**Time**: ~10 min (dry-run), ~2-6 hours (full calculation)
**Key Concepts**: Phonopy, supercells, force constants, phonon band structure, DOS, thermal properties

### [02-gruneisen-parameters](02-gruneisen-parameters/)
**Description**: Grüneisen parameters from volume-dependent phonon calculations
**Difficulty**: Advanced
**Time**: ~15 min (dry-run), ~6-12 hours (full calculation)
**Key Concepts**: Grüneisen parameters, mode-by-mode analysis, thermal expansion, anharmonicity

### [03-qha-thermodynamics](03-qha-thermodynamics/)
**Description**: Quasi-harmonic approximation (QHA) for temperature-dependent thermodynamics
**Difficulty**: Advanced
**Time**: ~20 min (dry-run), ~12-24 hours (full calculation)
**Key Concepts**: QHA, thermal expansion, heat capacity, Gibbs free energy vs T and P

---

## Learning Path

These tutorials build on each other:

1. **Foundation**: [01-phonons](01-phonons/) - Master basic phonon calculations
2. **Anharmonicity**: [02-gruneisen-parameters](02-gruneisen-parameters/) - Add volume dependence
3. **Thermodynamics**: [03-qha-thermodynamics](03-qha-thermodynamics/) - Complete T,P-dependent properties

**Important**: Each tutorial requires the previous one as foundation!

---

## Why Vibrational Properties?

Vibrational calculations enable:
- ✅ **Thermodynamic stability**: Free energy vs temperature
- ✅ **Thermal expansion**: α(T) from QHA
- ✅ **Heat capacity**: Cv(T) and Cp(T)
- ✅ **Phase transitions**: Detect structural instabilities
- ✅ **Spectroscopy**: Compare with IR/Raman experiments
- ✅ **Thermal conductivity**: Starting point for transport properties

---

## Computational Requirements

### Phonon Calculations
- **Supercell size**: 2×2×2 minimum (3×3×3 for accuracy)
- **Number of calculations**: 10-100+ (depends on symmetry)
- **Time per calculation**: Similar to relaxation
- **Total time**: Hours to days

**Example**: Si 2×2×2 supercell
- 64 atoms → ~6 irreducible displacements
- ~30 min per displacement = 3 hours total

### Grüneisen Parameters
- **Multiply phonon cost by 3-5** (different volumes)
- **Example**: Si phonons × 5 volumes = 15 hours

### QHA Calculations
- **Multiply Grüneisen cost by 1.5** (additional analysis)
- **Example**: Si QHA = ~20-25 hours total

**Recommendation**: Use HPC cluster (see [04-infrastructure/02-job-submission](../../03-advanced-features/03-infrastructure/02-jobflow-remote/))

---

## Critical Requirements

### 1. Well-Relaxed Structure ⚠️
```python
# Forces MUST be < 0.01 eV/Å
# Otherwise phonon frequencies will be inaccurate
```

### 2. Converged Parameters ⚠️
```python
# Use converged k-points, mesh cutoff, basis
# From category 02-convergence tutorials
```

### 3. Symmetry Considerations
```python
# Higher symmetry = fewer displacements
# Cubic: ~6-10 displacements
# Low symmetry: 50-100+ displacements
```

### 4. Supercell Size
```python
# Minimum: 2×2×2 (quick test)
# Recommended: 3×3×3 (publication quality)
# Large systems: 2×2×2 may suffice
```

---

## Quick Start

### Basic Phonon Calculation

```python
from atomate2.siesta.jobs.phonon import PhonopyMaker

# Create phonon workflow
phonon = PhonopyMaker(
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    dry_run=True  # Preview first!
)

workflow = phonon.make(structure)
results = run_locally(workflow)

# Generated:
# - phonon_band_structure.png
# - phonon_dos.png
# - thermal_properties.png
# - force_constants.yaml
```

### Grüneisen Parameters

```python
from atomate2.siesta.flows.phonon import SiestaGruneisenFlowMaker

# Grüneisen workflow
# Note: dry_run=True will generate input files for all volume points
# but will fail at analysis step (requires real force data)
gruneisen = SiestaGruneisenFlowMaker(
    perc_vol=0.02  # ±2% volume change
)

workflow = gruneisen.make(structure)

# Generated (after full calculation):
# - gruneisen_vs_frequency.png
# - gruneisen_distribution.png
# - thermal_expansion.png
# - gruneisen_summary.txt
```

### QHA Thermodynamics

```python
from atomate2.siesta.flows.phonon import SiestaQhaFlowMaker

# Full QHA workflow
# Note: dry_run=True will generate input files for all volume points
# but will fail at analysis step (requires real force and energy data)
qha = SiestaQhaFlowMaker(
    number_of_frames=5  # Number of volumes to calculate
)

workflow = qha.make(structure)

# Generated:
# - qha_thermal_expansion.png
# - qha_heat_capacity.png
# - qha_gibbs_energy.png
# - qha_summary.txt
```

---

## Common Issues

### Issue 1: "Imaginary frequencies (negative)"
**Cause**: Structure not fully relaxed OR unstable structure
**Solution**:
```python
# Re-relax with tighter criteria
relax = RelaxMaker.variable_cell_relaxation()
# Check forces < 0.01 eV/Å before phonon calculation
```

### Issue 2: "Too many displacement calculations"
**Cause**: Low symmetry structure
**Solution**:
- Use symmetry-reduced displacements (automatic in phonopy)
- Start with smaller supercell (2×2×2)
- Consider if high-symmetry approximation is acceptable

### Issue 3: "Phonon calculation takes forever"
**Cause**: Large supercell or many irreducible displacements
**Solution**:
```python
# Submit to HPC cluster
from jobflow_remote import submit_flow
submit_flow(workflow)

# Or use job arrays for parallel execution
```

### Issue 4: "Thermal properties look wrong"
**Cause**: Insufficient supercell size or unconverged parameters
**Solution**:
- Increase supercell size (2×2×2 → 3×3×3)
- Verify k-point convergence
- Check mesh cutoff convergence

### Issue 5: "QHA fails at high temperature"
**Cause**: Anharmonic effects too large for QHA
**Solution**:
- QHA valid up to ~0.5-0.7 × melting temperature
- Consider molecular dynamics for higher T

---

## Best Practices

### Before Running Phonons

1. **Relax thoroughly**:
   ```python
   # Use tight convergence criteria
   maker = RelaxMaker.variable_cell_relaxation()
   # Check forces < 0.01 eV/Å
   ```

2. **Test supercell size**:
   ```python
   # Start small (2×2×2), check convergence
   # Increase to 3×3×3 for publication
   ```

3. **Preview displacements**:
   ```python
   # Use dry_run to see number of calculations
   phonon = PhononMaker(dry_run=True)
   # Adjust supercell if too many displacements
   ```

### During Calculation

1. **Monitor progress**: Check calculation outputs periodically
2. **Check forces**: Verify forces are consistent across displacements
3. **Save intermediates**: Keep all displacement calculation outputs

### After Calculation

1. **Check phonon plots**: Look for imaginary frequencies
2. **Validate thermal properties**: Compare with experiments if available
3. **Document parameters**: Record supercell size, convergence parameters
4. **Archive results**: Save force_constants.yaml and analysis files

---

## Expected Results

### Phonon Calculation
- Phonon band structure plot
- Phonon density of states
- Thermal properties (Cv, S, F vs T)
- `force_constants.yaml` (for further analysis)

### Grüneisen Parameters
- Mode-dependent Grüneisen parameters
- Distribution histogram
- Thermal expansion coefficient α(T)
- Material classification (soft/hard modes)

### QHA Thermodynamics
- Thermal expansion α(T)
- Heat capacity Cv(T) and Cp(T)
- Gibbs free energy G(T, P)
- Equilibrium volume vs T
- Bulk modulus vs T

---

## Validation

### Compare with Experiments

1. **Phonon frequencies**: IR/Raman spectroscopy
2. **Thermal expansion**: Dilatometry measurements
3. **Heat capacity**: Calorimetry (Cv, Cp)
4. **Elastic properties**: Ultrasonic measurements

### Typical Accuracy

- **Phonon frequencies**: ±10-50 cm⁻¹ (GGA)
- **Thermal expansion**: ±10-20%
- **Heat capacity**: ±5-10% (low T), ±20% (high T)

---

## Next Steps

After completing vibrational properties:
- **[06-surfaces-and-adsorption](../../02-workflows/03-surfaces-and-adsorption/)** - Surface phonons and adsorbate vibrations
- **[07-advanced-features](../../03-advanced-features/)** - Advanced analysis and customization

---

*Back to [Main Tutorial Index](../README.md)*
