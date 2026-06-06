# Atomate2SIESTA Tutorials

Comprehensive tutorials for learning atomate2siesta workflows, from basic calculations to advanced features and real-world case studies.

## 🚀 New to atomate2siesta?

**Start here**: [**QUICKSTART Guide**](QUICKSTART.md) - Complete your first calculation in 5 minutes!

## 🆘 Need Help?

- **[Troubleshooting Guide](../docs/source/troubleshooting.rst)** - Solutions to common issues
- **Cheat Sheets**: [Workflows](../docs/cheatsheets/COMMON_WORKFLOWS.md) | [CLI](../docs/cheatsheets/CLI_COMMANDS.md) | [Parameters](../docs/cheatsheets/PARAMETER_REFERENCE.md)

## Tutorial Organization

The tutorials are organized into four main categories plus case studies:

### 00. Structure Files

**[00-structures/](00-structures/)** - Example structure files (CIF, POSCAR, etc.) used across tutorials

Common structures:
- Si (diamond), GaAs (zincblende)
- MgO (rocksalt), CuO (monoclinic)
- Graphene, BN monolayer (2D materials)
- Cu (FCC), Fe (BCC) (metals)

### 01. Basics (8 tutorials)

**[01-basics/](01-basics/)** - Fundamental Makers and single-job calculations

1. **[01-RelaxMaker](01-basics/01-RelaxMaker/)** - Structural relaxation (fixed/variable cell)
2. **[02-BandStructureMaker](01-basics/02-BandStructureMaker/)** - Electronic band structure
3. **[03-LuaMaker](01-basics/03-LuaMaker/)** - Lua scripting for advanced SIESTA features
4. **[04-RelaxMaker-StaticMaker](01-basics/04-RelaxMaker-StaticMaker/)** - Multi-step workflows
5. **[05-DOSMaker](01-basics/05-DOSMaker/)** - Density of states calculations
6. **[06-PDOSMaker](01-basics/06-PDOSMaker/)** - Projected density of states
7. **[07-PhonopyMaker](01-basics/07-PhonopyMaker/)** - Phonopy-based phonon calculations
8. **[08-SiestaPhononMaker](01-basics/08-SiestaPhononMaker/)** - Native SIESTA phonon calculations

**Learning path**: Start with 01-RelaxMaker → 02-BandStructureMaker → 04-RelaxMaker-StaticMaker

### 02. Workflows (7 categories)

**[02-workflows/](02-workflows/)** - Multi-step FlowMaker workflows for production calculations

#### **[01-convergence/](02-workflows/01-convergence/)**
- K-points convergence
- Mesh cutoff convergence
- Combined mesh+kpoints convergence
- Basis parameter convergence

#### **[02-equation-of-states/](02-workflows/02-equation-of-states/)**
- EOS calculations
- Bulk modulus determination
- Pressure-volume curves

#### **[03-surfaces-and-adsorption/](02-workflows/03-surfaces-and-adsorption/)**
- Surface energy calculations
- Adsorption site scanning
- Height-scanned adsorption (3D potential energy surfaces)

#### **[04-mechanical/](02-workflows/04-mechanical/)**
- Elastic constants
- Bulk modulus
- Mechanical properties

#### **[05-barriers/](02-workflows/05-barriers/)**
- NEB (Nudged Elastic Band) calculations
- Transition state search
- Energy barrier determination

#### **[06-vibrational-properties/](02-workflows/06-vibrational-properties/)**
- Phonon calculations with automatic plotting
- Grüneisen parameters
- Quasi-harmonic approximation (QHA)
- Thermal expansion

#### **[07-bands/](02-workflows/07-bands/)**
- Electronic band structure workflows
- Automatic k-path generation
- Band structure + DOS combined analysis

**Learning path**:
1. Convergence (essential for all calculations)
2. EOS or mechanical properties
3. Surfaces/vibrational based on research needs

### 03. Advanced Features (20 topics)

**[03-advanced-features/](03-advanced-features/)** - Specialized calculations and advanced functionality

1. **[01-tier-system](03-advanced-features/01-parameter-systems/01-tier-system/)** - Tier-based parameter presets (basic → expert)
2. **[02-fdf-block-inputs](03-advanced-features/01-parameter-systems/04-fdf-blocks/)** - FDF block parameters (electric fields, constraints, PDOS, custom k-paths)
3. **[03-infrastructure](03-advanced-features/03-infrastructure/)** - Database, HPC submission, dry-run mode
4. **[03-powerups](03-advanced-features/04-structure-tools/03-powerups/)** - Workflow customization and parameter modification
5. **[04-structure-conversion](03-advanced-features/04-structure-tools/01-conversion/)** - XV, FDF, XSF, CIF file conversion
6. **[05-dos-calculations](03-advanced-features/02-physics-features/01-dos/)** - Advanced DOS analysis
7. **[06-phonon-inputs](03-advanced-features/02-physics-features/04-phonon/)** - Phonon calculation customization
8. **[06-tier-presets-customization](03-advanced-features/01-parameter-systems/02-presets/)** - Material-specific presets (material-specific presets in 10 categories)
9. **[07-basis-set-customization](03-advanced-features/01-parameter-systems/03-basis/)** - Custom basis sets, species variants, PAO.Basis
10. **[07-optical-properties](03-advanced-features/02-physics-features/02-optical/)** - Optical calculations
11. **[08-recipe-book](03-advanced-features/08-recipe-book/)** - High-level workflow system (significant code reduction)
12. **[09-charge-dipole-efield](03-advanced-features/02-physics-features/05-charge/)** - Electric fields and dipole corrections
13. **[09-structure-manipulation](03-advanced-features/04-structure-tools/02-manipulation/)** - Structure tools (16 subcommands)
14. **[10-grid-output](03-advanced-features/05-output-viz/01-grid/)** - Grid-based output (charge density, potentials)
15. **[11-denchar-visualization](03-advanced-features/05-output-viz/02-denchar/)** - Denchar visualization tools
16. **[12-parallel-computation](03-advanced-features/06-performance/01-parallel/)** - Parallel execution settings
17. **[13-netcdf-output](03-advanced-features/05-output-viz/03-netcdf/)** - NetCDF output format
18. **[14-efficiency-options](03-advanced-features/06-performance/02-efficiency/)** - Performance optimization
19. **[15-hamiltonian-overlap](03-advanced-features/07-advanced-siesta/01-hamiltonian/)** - Hamiltonian and overlap matrix settings
20. **[16-magnetic-calculations](03-advanced-features/02-physics-features/03-magnetic/)** - Spin-polarized calculations, DM.InitSpin, magnetic moments

**Highlights**:
- **Recipe Book (08)**: One-line workflows, fully documented
- **Tier Presets (06)**: 26 material-specific presets with automatic CLI discovery
- **Magnetic Calculations (16)**: Automatic magnetic moment detection, FM/AFM/custom orderings
- **Basis Customization (07)**: Species variants, per-atom basis, PAO.Basis builder

### 05. Troubleshooting

**[05-troubleshooting/](05-troubleshooting/)** - Debugging and optimization guides

- **[common_errors/](05-troubleshooting/common_errors/)** - SCF convergence issues, frequent error messages and solutions
- **[debugging_workflows/](05-troubleshooting/debugging_workflows/)** - Workflow debugging strategies, job tracing
- **[performance_optimization/](05-troubleshooting/performance_optimization/)** - Parallel efficiency, computational cost reduction

## Quick Start Pattern

All tutorials follow this consistent pattern:

```python
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker  # or FlowMaker
from jobflow import run_locally

# 1. Create or load structure
structure = Structure.from_file("structure.cif")

# 2. Create maker
maker = RelaxMaker()

# 3. Make the job
job = maker.make(structure)

# 4. Run locally
run_locally(job, create_folders=True)
```

## Recommended Learning Paths

### 🌱 **For Absolute Beginners**
1. **QUICKSTART.md** - First calculation in 5 minutes
2. **01-basics/01-RelaxMaker** - Understand basic workflow
3. **01-basics/02-BandStructureMaker** - Electronic properties
4. **01-basics/04-RelaxMaker-StaticMaker** - Multi-step workflows

### 📊 **For Convergence Testing**
1. **02-workflows/01-convergence/** - K-points and mesh cutoff
2. **03-advanced-features/01-parameter-systems/03-basis/** - Basis optimization
3. **03-advanced-features/01-parameter-systems/01-tier-system/** - Tier presets for quick setup

### 🚀 **For Production Calculations**
1. Complete convergence studies (02-workflows/01-convergence)
2. Apply tier presets (03-advanced-features/01-parameter-systems/02-presets)
3. Set up infrastructure (03-advanced-features/03-infrastructure)
4. Use recipe book for one-line workflows (03-advanced-features/08-recipe-book)

### 🔬 **For Vibrational Properties**
1. **02-workflows/06-vibrational-properties/01-phonons/** - Basic phonon calculations
2. **02-workflows/06-vibrational-properties/02-gruneisen/** - Thermal expansion
3. **02-workflows/06-vibrational-properties/03-qha/** - Quasi-harmonic approximation

### 🧲 **For Magnetic Systems**
1. **03-advanced-features/02-physics-features/03-magnetic/** - Spin polarization, magnetic moments, FM/AFM orderings
2. **03-advanced-features/01-parameter-systems/04-fdf-blocks/** - DFT+U parameters for correlated electrons

### 🔧 **For HPC Users**
1. **03-advanced-features/03-infrastructure/** - Database, jobflow-remote
2. **03-advanced-features/06-performance/01-parallel/** - Parallel settings
3. **03-advanced-features/06-performance/02-efficiency/** - Performance tuning

### 📚 **For Advanced Users**
1. **03-advanced-features/08-recipe-book/** - High-level workflow system
2. **03-advanced-features/04-structure-tools/03-powerups/** - Workflow customization
3. **03-advanced-features/01-parameter-systems/03-basis/** - Species variants, PAO.Basis

## Requirements

### Installation
```bash
pip install atomate2[siesta]
```

### Configuration
Create `~/.atomate2siesta.yaml`:
```yaml
SIESTA_CMD: "mpirun -np 4 siesta < siesta.fdf > siesta.out"
SIESTA_PP_PATH: "/path/to/pseudopotentials"
```

### Pseudopotentials
Install from CLI:
```bash
# Install all 6 available sets
atomate2siesta-pseudos install --all

# Or specific set
atomate2siesta-pseudos install ONCVPSP-PBE-SR-PDv0.4-Standard
```

Available sets:
- ONCVPSP-PBE-SR-PDv0.4-Standard (recommended for most systems)
- ONCVPSP-PBEsol-PDv0.4
- PseudoDojo-PBE-SR-v0.4-standard
- PseudoDojo-LDA-SR-v0.4-standard
- PseudoDojo-PBEsol-SR-v0.4-standard
- GGA-PBE (legacy ABINIT format)

## Tutorial Features

### 📖 Documentation
- **Detailed READMEs**: Step-by-step instructions in each tutorial
- **Inline comments**: Extensively commented code
- **Learning objectives**: Clear goals for each tutorial

### 📊 Automatic Outputs
- **Convergence plots**: Publication-quality PNG files
- **Result summaries**: JSON and text formats
- **Timing analysis**: Computational cost tracking
- **Phonon plots**: Automatic band structure, DOS, thermal properties

### 🎯 Multiple Difficulty Levels
- **Simple examples**: Basic functionality (01_basic.py)
- **Detailed examples**: Advanced features (02_detailed.py)
- **Custom examples**: Real-world scenarios (03_custom.py)

### 🔧 CLI Integration
Many tutorials demonstrate CLI tools:
- **atomate2siesta-maker** - Interactive workflow generation
- **atomate2siesta-presets** - Tier preset exploration
- **atomate2siesta-recipe** - Recipe book browser
- **atomate2siesta-structure** - Structure manipulation (16 subcommands)
- **atomate2siesta-pseudos** - Pseudopotential management

## Finding Tutorials

### By Topic
- **Relaxation**: 01-basics/01-RelaxMaker
- **Band structure**: 01-basics/02-BandStructureMaker
- **Phonons**: 01-basics/07-PhonopyMaker, 02-workflows/06-vibrational-properties
- **Surfaces**: 02-workflows/03-surfaces-and-adsorption
- **Convergence**: 02-workflows/01-convergence
- **Magnetic**: 03-advanced-features/02-physics-features/03-magnetic
- **DFT+U**: 03-advanced-features/02-physics-features/03-magnetic
- **Recipes**: 03-advanced-features/08-recipe-book

### By Workflow Type
- **Single job**: 01-basics/
- **Multi-step**: 01-basics/04-RelaxMaker-StaticMaker, 02-workflows/
- **High-level**: 03-advanced-features/08-recipe-book

### By Material Type
- **Bulk crystals**: Most tutorials use Si, MgO
- **2D materials**: 03-advanced-features/01-parameter-systems/01-tier-system (2D presets)
- **Surfaces**: 02-workflows/03-surfaces-and-adsorption
- **Magnetic**: 03-advanced-features/02-physics-features/03-magnetic

## Getting Help

### Documentation
- **Main docs**: `../docs/source/index.rst`
- **CLI reference**: `../docs/source/cli-tools.rst`
- **API docs**: `../docs/source/modules.rst`
- **Troubleshooting**: `../docs/source/troubleshooting.rst`

### Interactive Tools
```bash
# Browse tutorials
atomate2siesta-tutorials

# List workflows
atomate2siesta-info workflows

# Interactive workflow generation
atomate2siesta-maker --interactive

# Explore tier presets
atomate2siesta-presets list
```

### Community
- **Issues**: https://github.com/materialsproject/atomate2/issues
- **Discussions**: GitHub Discussions

## Contributing

### Adding New Tutorials

1. **Choose location** based on tutorial type:
   - Basic Maker → `01-basics/`
   - FlowMaker workflow → `02-workflows/`
   - Advanced feature → `03-advanced-features/`

2. **Create directory** with descriptive name:
   ```bash
   mkdir -p 03-advanced-features/17-my-new-feature
   ```

3. **Add files**:
   - `README.md` - Overview, objectives, instructions
   - `01_basic.py` - Simple example
   - `02_detailed.py` - Advanced features (optional)
   - Example structures (or use `00-structures/`)

4. **Update this README**: Add entry in appropriate section

### Tutorial Template Structure

```
my-tutorial/
├── README.md           # Overview and instructions
├── 01_basic.py         # Simple example
├── 02_detailed.py      # Advanced example (optional)
├── example.cif         # Example structure (optional)
└── expected_output/    # Reference outputs (optional)
    ├── results.json
    └── plot.png
```

## Tutorial Metrics

- **Total tutorials**: 30+ main topics
- **Organization**: 3 main categories (basics, workflows, advanced)
- **FlowMakers covered**: 13 workflows (all production-ready)
- **Makers covered**: 8+ basic Makers
- **Advanced features**: 16 specialized topics
- **Recipe book**: one-line workflows, comprehensive tutorials
- **CLI tools**: 13 command-line utilities

## Recent Additions (v1.0.0 - December 2025)

- ✅ Consolidated adsorption output files (single directory)
- ✅ Height-scanned adsorption (3D potential energy surfaces)
- ✅ Optimal height map visualization
- ✅ Sign-only DM.InitSpin mode for magnetic calculations
- ✅ Phonon tutorials reorganized to `02-workflows/06-vibrational-properties/`
- ✅ Tutorial browser CLI: `atomate2siesta-tutorials`
- ✅ MyST parser integration for documentation

## Quick Reference

### Essential Commands
```bash
# Browse tutorials
atomate2siesta-tutorials list
atomate2siesta-tutorials show 01-basics/01-RelaxMaker

# Generate workflow (interactive)
atomate2siesta-maker --interactive

# List all workflows
atomate2siesta-info workflows

# Explore tier presets
atomate2siesta-presets list
atomate2siesta-presets show relax_standard

# Recipe book
atomate2siesta-recipe list
atomate2siesta-recipe show complete_material_study
```

### Common Patterns
```python
# Tier presets
from atomate2.siesta.sets.tiers import apply_tier_preset
maker = apply_tier_preset(RelaxMaker(), "relax_standard")

# Recipe book
from atomate2.siesta.recipes import RecipeBook
flow = RecipeBook.complete_material_study(structure)

# Powerups
from atomate2.siesta.powerups import update_user_siesta_settings
job = update_user_siesta_settings(job, {"kpts": [6,6,6]})
```

---

**Happy Learning!** 🎓

For questions, issues, or contributions, visit:
https://github.com/materialsproject/atomate2
