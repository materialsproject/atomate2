# Quickstart Guide: Your First Calculation in 5 Minutes

Welcome to **atomate2siesta**! This guide will walk you through your first DFT calculation in just a few minutes.

## Prerequisites

- Python 3.9+ installed
- SIESTA installed and accessible
- 5 minutes of your time

## Step 1: Installation (1 minute)

```bash
# Install atomate2 with the SIESTA extra
pip install atomate2[siesta]
```

## Step 2: Configuration (1 minute)

Create a configuration file at `~/.atomate2siesta.yaml`:

```yaml
# Minimal configuration
SIESTA_CMD: "siesta < siesta.fdf > siesta.out"
SIESTA_PP_PATH: "/path/to/your/pseudopotentials"
```

**Note**: Replace `/path/to/your/pseudopotentials` with your actual pseudopotential directory.

### Quick Config Check

Verify your setup:

```bash
python -c "from atomate2.siesta import SETTINGS; print(SETTINGS)"
```

You should see your configuration printed.

## Step 3: Your First Calculation (3 minutes)

Create a file called `my_first_calculation.py`:

```python
"""
Your First Calculation: Silicon Relaxation
===========================================

This script performs a simple structure relaxation of crystalline silicon.
"""

from pymatgen.core import Structure, Lattice
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import run_locally

# Step 1: Create a silicon structure
lattice = Lattice.cubic(5.43)  # Silicon lattice constant in Angstroms
structure = Structure(
    lattice=lattice,
    species=["Si", "Si"],
    coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
)

print("Structure created:")
print(structure)
print()

# Step 2: Create a relaxation job
maker = RelaxMaker.fixed_cell_relaxation()
job = maker.make(structure)

print("Job created:")
print(f"  Name: {job.name}")
print(f"  Function: {job.function.__name__}")
print()

# Step 3: Run the calculation locally
print("Running calculation...")
print("This will take 1-2 minutes depending on your system.")
print()

result = run_locally(job, create_folders=True)

# Step 4: Check the results
print("Calculation complete!")
print()
print("Results:")
print(f"  Final energy: {result[job.uuid][1].output.energy:.4f} eV")
print(f"  Job directory: {result[job.uuid][1].output.dir_name}")
print()
print("Success! You've completed your first atomate2siesta calculation.")
```

### Run It!

```bash
python my_first_calculation.py
```

**Expected output**:
```
Structure created:
Full Formula (Si2)
Reduced Formula: Si
abc   :   5.430000   5.430000   5.430000
...

Job created:
  Name: relax
  Function: run_siesta
...

Running calculation...
This will take 1-2 minutes depending on your system.

Calculation complete!

Results:
  Final energy: -215.7234 eV
  Job directory: job_2025-11-05-...

Success! You've completed your first atomate2siesta calculation.
```

## What Just Happened?

1. **Structure Creation**: You created a crystalline silicon structure using pymatgen
2. **Job Setup**: `RelaxMaker` configured a SIESTA relaxation calculation
3. **Execution**: `run_locally()` executed SIESTA and collected results
4. **Results**: The final energy and relaxed structure were saved

## Next Steps

### Learn More About Relaxation
- 📚 **Tutorial 01-basics/01-RelaxMaker**: Detailed relaxation guide
- 📚 **Tutorial 03-advanced-features/01-parameter-systems/01-tier-system**: Customize SIESTA parameters with tiers

### Try Other Calculations
```python
from atomate2.siesta.jobs.core import StaticMaker, BandStructureMaker

# Static energy calculation
static_job = StaticMaker().make(structure)

# Band structure calculation
bands_job = BandStructureMaker().make(structure)
```

### Use the Recipe Book (High-Level API)
```python
from atomate2.siesta.recipes import RecipeBook

# Complete material study in one line!
flow = RecipeBook.complete_material_study(structure)
```

### Explore Workflows
- 📚 **Tutorial 02-workflows/01-convergence**: k-points and basis convergence
- 📚 **Tutorial 02-workflows/02-equation-of-states**: EOS calculations
- 📚 **Tutorial 02-workflows/06-vibrational-properties**: Phonons, Grüneisen, QHA
- 📚 **Tutorial 02-workflows/03-surfaces-and-adsorption**: Surface energy and adsorption

### Production Setup
- 📚 **Tutorial 03-advanced-features/03-infrastructure/01-database-storage**: Save results to MongoDB
- 📚 **Tutorial 03-advanced-features/03-infrastructure/02-jobflow-remote**: Submit to HPC clusters
- 📚 **Tutorial 03-advanced-features/03-infrastructure/03-error-handling**: Automatic error recovery

## Common Issues

### "SIESTA command not found"
**Solution**: Update your `SIESTA_CMD` in `~/.atomate2siesta.yaml` with the full path:
```yaml
SIESTA_CMD: "/usr/local/bin/siesta < siesta.fdf > siesta.out"
```

### "Pseudopotentials not found"
**Solution**: Download pseudopotentials and update `SIESTA_PP_PATH`:
```bash
# Using the CLI tool
atomate2siesta-pseudos available
atomate2siesta-pseudos install psf
```

### "SCF not converged"
**Solution**: See the [Troubleshooting Guide](https://materialsproject.github.io/atomate2/siesta/troubleshooting.html) for SCF convergence tips.

## Quick Reference

### Basic Pattern
```python
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import run_locally

# 1. Create structure
structure = Structure(...)

# 2. Create maker
maker = RelaxMaker()

# 3. Make job
job = maker.make(structure)

# 4. Run
result = run_locally(job, create_folders=True)
```

### Common Makers
```python
from atomate2.siesta.jobs.core import (
    RelaxMaker,        # Structure relaxation
    StaticMaker,       # Single-point energy
    BandStructureMaker,# Band structure
)

from atomate2.siesta.flows.phonon import SiestaPhononFlowMaker  # Phonons
from atomate2.siesta.flows.eos import EOSMaker        # Equation of state
```

### Customize Parameters
```python
maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "PAO.BasisSize": "DZP",
        "kpts": [8, 8, 8],
        "Mesh.Cutoff": "300 Ry",
    }
)
```

## Help & Support

- 📖 **Full Documentation**: `docs/source/index.rst`
- 📚 **All Tutorials**: `tutorials/README.md`
- 🐛 **Issues**: https://github.com/materialsproject/atomate2/issues
- 💬 **Discussions**: GitHub Discussions

## Congratulations!

You've successfully run your first atomate2siesta calculation! 🎉

Now explore the tutorials to learn about advanced workflows, convergence testing, and production deployment.

Happy computing! 🚀
