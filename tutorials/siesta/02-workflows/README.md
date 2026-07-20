# 02-Workflows: Multi-Step FlowMaker Workflows

**Focus**: Production-ready multi-step workflows for materials properties

**Difficulty**: Intermediate to Advanced

**Prerequisites**:
- Completed [01-basics](../01-basics/)
- Understanding of convergence testing
- Familiarity with jobflow Flow concepts
- Knowledge of materials properties (EOS, elasticity, phonons)

---

## Tutorials in This Category

This category contains 7 workflow subcategories covering all production-level calculations:

### [01-convergence](01-convergence/)
**Description**: Systematic convergence testing for k-points, mesh cutoff, and basis parameters
**Workflows**: MeshCutoffConvergenceFlowMaker, KpointsConvergenceFlowMaker, MeshKpointConvergenceFlowMaker, BasisConvergenceFlowMaker
**Difficulty**: Intermediate
**Time**: ~5 min (dry-run), ~1-3 hours (full calculation)
**Key Concepts**: Convergence criteria, multi-property tracking (energy, forces, stress, Fermi, bandgap), automatic plotting
**Files**: 6 tutorials covering mesh cutoff, k-points, combined mesh+kpoints, basis parameters

### [02-equation-of-states](02-equation-of-states/)
**Description**: Equation of State (EOS) calculations for bulk modulus and equilibrium volume
**Workflows**: EOSFlowMaker
**Difficulty**: Intermediate
**Time**: ~5 min (dry-run), ~30-60 min (full calculation)
**Key Concepts**: Volume scaling, Birch-Murnaghan fitting, bulk modulus, pressure-volume curves
**Files**: Multiple examples with different volume ranges and fitting methods

### [03-surfaces-and-adsorption](03-surfaces-and-adsorption/)
**Description**: Surface energy calculations and adsorption site scanning
**Workflows**: SurfaceEnergyFlowMaker, AdsorptionScanFlowMaker
**Difficulty**: Advanced
**Time**: ~5 min (dry-run), ~2-6 hours (full calculation)
**Key Concepts**: Surface generation, slab models, adsorption site scanning, height scanning, 3D potential energy surfaces, automatic plotting
**Files**: 12 tutorials covering surface basics, multi-surface comparison, adsorption scanning with grid/height modes

### [04-mechanical](04-mechanical/)
**Description**: Elastic constants and mechanical properties
**Workflows**: ElasticFlowMaker
**Difficulty**: Advanced
**Time**: ~5 min (dry-run), ~1-2 hours (full calculation)
**Key Concepts**: Elastic tensor, strain perturbations, bulk/shear modulus, Young's modulus, Poisson's ratio
**Files**: Elastic constant calculation examples

### [05-barriers](05-barriers/)
**Description**: Transition state search using Nudged Elastic Band (NEB) method
**Workflows**: NebDirectFlowMaker, AseNebFlowMaker
**Difficulty**: Advanced
**Time**: ~5 min (dry-run), ~3-12 hours (full calculation)
**Key Concepts**: Reaction barriers, transition states, climbing image, force projections, minimum energy pathways
**Files**: NEB tutorials with direct and ASE implementations

### [06-vibrational-properties](06-vibrational-properties/)
**Description**: Phonon calculations, Grüneisen parameters, and quasi-harmonic approximation (QHA)
**Workflows**: SiestaPhononFlowMaker, SiestaGruneisenFlowMaker, SiestaQhaFlowMaker
**Difficulty**: Intermediate to Advanced
**Time**: ~5 min (dry-run), ~1-6 hours (full calculation)
**Key Concepts**: Phonon band structure, phonon DOS, thermal properties, thermal expansion, Grüneisen parameters, automatic plotting
**Files**: 5 comprehensive tutorials (01-phonons, 02-gruneisen, 03-qha) with tier preset examples

### [07-bands](07-bands/)
**Description**: Electronic band structure calculations with automatic k-path generation
**Workflows**: BandStructureFlowMaker
**Difficulty**: Intermediate
**Time**: ~5 min (dry-run), ~30-60 min (full calculation)
**Key Concepts**: Electronic band structure, high-symmetry k-paths, band gaps, Fermi surfaces
**Files**: Band structure calculation tutorials

---

## Learning Path

Choose tutorials based on your research needs:

### Essential for All Users
1. **[01-convergence](01-convergence/)** - MUST DO FIRST for production calculations
   - Learn systematic parameter optimization
   - Multi-property convergence testing
   - Establish baseline parameters for your system

### Structural Properties Track
2. **[02-equation-of-states](02-equation-of-states/)** - Bulk modulus and equilibrium volume
3. **[04-mechanical](04-mechanical/)** - Complete elastic tensor and mechanical properties

### Surface Chemistry Track
2. **[03-surfaces-and-adsorption](03-surfaces-and-adsorption/)** - Surface energy and adsorption sites
   - Surface slab generation and termination discovery
   - Grid-based adsorption site scanning
   - Height-scanned 3D potential energy surfaces

### Vibrational Properties Track
2. **[06-vibrational-properties](06-vibrational-properties/)** - Phonons, Grüneisen, QHA
   - Phonon band structure and DOS
   - Thermal expansion coefficients
   - Quasi-harmonic approximation for thermodynamics

### Electronic Structure Track
2. **[07-bands](07-bands/)** - Electronic band structure
   - High-symmetry k-path generation
   - Band gap calculations
   - Density of states

### Reaction Kinetics Track
2. **[05-barriers](05-barriers/)** - Transition state search with NEB
   - Nudged Elastic Band calculations
   - Reaction barriers and pathways
   - Climbing image optimization

---

## Workflow Characteristics

All workflows in this category:
- ✅ **Multi-step**: Automatically manage job dependencies
- ✅ **Self-contained**: No manual intervention required
- ✅ **Production-ready**: Suitable for publication-quality results
- ✅ **Automatic analysis**: Generate plots and summary files
- ✅ **Database-compatible**: Store results in MongoDB
- ✅ **Checkpoint/restart**: Resume failed workflows (where applicable)
- ✅ **Tier preset compatible**: Use material-specific parameter sets

---

## Quick Start Pattern

All FlowMakers follow a consistent pattern:

```python
from pymatgen.core import Structure
from atomate2.siesta.flows.convergence import MeshCutoffConvergenceFlowMaker
from jobflow import run_locally

# 1. Load structure
structure = Structure.from_file("../00-structures/Si.cif")

# 2. Create workflow maker
flow_maker = MeshCutoffConvergenceFlowMaker(
    meshes=[100, 150, 200, 250, 300, 350, 400],  # Ry
    dry_run=True,  # Preview mode
)

# 3. Generate workflow
workflow = flow_maker.make(structure)

# 4. Run workflow
results = run_locally(workflow, create_folders=True)
```

---

## Execution Modes

All workflows support three execution modes:

### 1. Dry-Run Mode (Preview)
```python
flow_maker = MeshCutoffConvergenceFlowMaker(
    meshes=[100, 200, 300],
    dry_run=True,  # Preview only
)
workflow = flow_maker.make(structure)
results = run_locally(workflow, create_folders=True)
```

### 2. Local Execution
```python
flow_maker = MeshCutoffConvergenceFlowMaker(
    meshes=[100, 200, 300],
)
workflow = flow_maker.make(structure)
results = run_locally(workflow, create_folders=True)
```

### 3. Remote HPC Submission
```python
from jobflow_remote import submit_flow

flow_maker = MeshCutoffConvergenceFlowMaker(
    meshes=[100, 200, 300],
)
workflow = flow_maker.make(structure)
submit_flow(workflow, project="production", worker="my_cluster")
```

---

## Using Tier Presets with Workflows

All FlowMakers accept tier presets or custom parameters:

```python
from atomate2.siesta.sets.tiers import apply_tier_preset

# Method 1: Apply tier preset to flow_maker's relax_maker
flow_maker = MeshCutoffConvergenceFlowMaker(
    meshes=[100, 200, 300],
    relax_maker=apply_tier_preset(
        RelaxMaker.fixed_cell_relaxation(),
        "relax_standard"
    )
)

# Method 2: Use tier preset recipes (see 03-advanced-features/08-recipe-book)
from atomate2.siesta.recipes import RecipeBook
flow = RecipeBook.convergence_study(structure, tier="intermediate")
```

---

## Common Workflow Parameters

### Convergence Workflows
- **Mesh cutoff range**: 100-400 Ry (7-9 points)
- **K-points range**: [2,2,2] to [12,12,12] (4-6 points)
- **Convergence criteria**: Energy (0.01 eV), Forces (0.05 eV/Å), Stress (0.1 GPa)
- **Multi-property tracking**: Energy, Fermi energy, band gap, forces, stress

### EOS Workflows
- **Volume range**: ±6% (scales: 0.94-1.06)
- **Number of points**: 7-11 (more for accurate fitting)
- **Fitting method**: Birch-Murnaghan (3rd or 4th order)

### Surface Workflows
- **Slab thickness**: 10-20 Å (depends on material)
- **Vacuum spacing**: 15-20 Å
- **Adsorption grid**: 4×4 or 6×6 points
- **Height scanning**: 2.0-3.5 Å (0.1-0.2 Å steps)

### Phonon Workflows
- **Supercell size**: 2×2×2 or larger (min_length ≥ 15 Å)
- **Displacement**: 0.01 Å (default)
- **Force constants**: Symmetry-reduced
- **Automatic plotting**: Band structure, DOS, thermal properties

### Elastic Workflows
- **Strain magnitude**: 0.01 (1% strain)
- **Number of strains**: 24 (cubic) or more (lower symmetry)
- **Symmetry**: Automatic detection and reduction

---

## Expected Output Files

### Convergence Workflows
- `mesh_cutoff_convergence.png` - Convergence plot (6 properties)
- `convergence_summary.json` - Converged parameters
- `parameter_evolution.log` - Full parameter tracking

### EOS Workflow
- `eos_plot.png` - E vs V curve with fitted EOS
- `eos_summary.txt` - Bulk modulus, equilibrium volume
- Fitted parameters (B₀, V₀, B'₀)

### Surface Workflow
- `multi_surface_comparison.png` - Energy comparison plot
- `multi_surface_summary.txt` - Surface energies table
- Individual surface structures (CIF files)

### Adsorption Workflow
- `adsorption_energy_map.png` - Energy heatmap (2D or height-resolved)
- `adsorption_sites.png` - Discrete site energies
- `optimal_height_map.png` - Best height at each (x,y) position
- `best_structure.cif` - Lowest energy configuration
- `adsorption_summary.txt` - Best site and energy

### Phonon Workflow
- `phonon_bands.png` - Phonon dispersion curves
- `phonon_dos.png` - Phonon density of states
- `thermal_properties.png` - Cv, entropy, free energy vs T
- `force_constants.hdf5` - Force constants matrix

### Grüneisen Workflow
- 6 plots: band structure, DOS, thermal properties, Grüneisen parameters, thermal expansion, heat capacity
- `gruneisen_summary.txt` - Thermal expansion coefficients

---

## Common Issues

### Issue 1: "Convergence test takes too long"
**Solution**:
- Use dry-run to verify number of jobs
- Start with fewer test points (5 instead of 9)
- Submit to HPC cluster (see [03-advanced-features/03-infrastructure](../03-advanced-features/03-infrastructure/))
- Use checkpoint/restart for failed jobs

### Issue 2: "EOS fitting fails"
**Solution**:
- Increase volume range (±8% or ±10%)
- Add more data points (9 or 11 volumes)
- Check for structural instabilities
- Ensure relaxations converged properly

### Issue 3: "Surface energy is negative or unrealistic"
**Solution**:
- Increase slab thickness (>10 Å)
- Increase vacuum spacing (>15 Å)
- Check surface is charge-neutral
- Use converged bulk parameters

### Issue 4: "Phonon has imaginary modes at Γ"
**Solution**:
- Ensure structure is fully relaxed and stable
- Increase supercell size (min_length ≥ 15 Å)
- Check force convergence threshold
- Use tighter SCF convergence

### Issue 5: "Elastic constants don't satisfy stability criteria"
**Solution**:
- Ensure structure is at equilibrium (relaxed)
- Use well-converged parameters
- Check for numerical noise (reduce strain magnitude)
- Verify crystal symmetry is correct

---

## Best Practices

1. **Always converge first**: Complete [01-convergence](01-convergence/) before other workflows
2. **Start with dry-run**: Preview structure transformations and job counts
3. **Use tier presets**: Material-specific presets for consistent parameters
4. **Enable database**: Store results for later analysis (see [03-advanced-features/03-infrastructure](../03-advanced-features/03-infrastructure/))
5. **Document parameters**: Keep workflow parameters for reproducibility
6. **Validate results**: Compare with literature/experiments
7. **Checkpoint long jobs**: Enable restart for multi-hour calculations

---

## Workflow Scaling

| Workflow | Typical Jobs | Est. Time (local) | Est. Time (HPC) |
|----------|--------------|-------------------|-----------------|
| Mesh cutoff conv. | 7-9 | 1-2 hours | 10-20 min |
| K-points conv. | 5-7 | 1-3 hours | 15-30 min |
| Combined conv. | 10-15 | 3-6 hours | 30-60 min |
| EOS | 7-11 | 30-60 min | 10-15 min |
| Elastic | 24+ | 1-2 hours | 20-40 min |
| Phonon | 20-50 | 2-6 hours | 30-90 min |
| Grüneisen | 60-150 | 6-12 hours | 1-3 hours |
| Adsorption scan | 16-36 | 2-4 hours | 30-60 min |

*Times assume Si-like system with DZP basis. Scale accordingly for larger systems.*

---

## Next Steps

After completing workflows, proceed to:

### Advanced Workflow Features
- **[03-advanced-features/04-structure-tools/03-powerups](../03-advanced-features/04-structure-tools/03-powerups/)** - Customize workflows dynamically
- **[03-advanced-features/08-recipe-book](../03-advanced-features/08-recipe-book/)** - High-level workflow system (recipe book)
- **[03-advanced-features/03-infrastructure](../03-advanced-features/03-infrastructure/)** - Database, HPC, error handling

### Specialized Calculations
- **[03-advanced-features/02-physics-features/03-magnetic](../03-advanced-features/02-physics-features/03-magnetic/)** - Spin-polarized systems
- **[03-advanced-features/02-physics-features/03-magnetic](../03-advanced-features/02-physics-features/03-magnetic/)** - DFT+U for correlated electrons
- **[03-advanced-features/02-physics-features/02-optical](../03-advanced-features/02-physics-features/02-optical/)** - Optical absorption

---

## Tutorial Metrics

- **Total workflow categories**: 7 main categories
- **Total FlowMakers**: 13+ production workflows
- **Total tutorial files**: 50+ Python scripts
- **Coverage**: All major materials properties
- **Difficulty**: Intermediate (convergence, EOS, phonons, bands) + Advanced (surfaces, barriers/NEB, elastic, mechanical)

---

*Back to [Main Tutorial Index](../README.md)*
