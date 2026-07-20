# 03-Advanced Features: Specialized Calculations and Advanced Functionality

**Focus**: Advanced SIESTA features, parameter customization, and infrastructure

**Difficulty**: Intermediate to Expert

**Prerequisites**:
- Completed [01-basics](../01-basics/) and [02-workflows](../02-workflows/)
- Understanding of SIESTA FDF format
- Familiarity with materials properties and advanced DFT concepts

---

## Overview

This category contains **8 organized groups** covering specialized calculations, parameter systems, and infrastructure. Previously scattered across 20 directories, tutorials are now logically grouped by topic for easier navigation.

**Total**: ~120 Python tutorial files organized in 8 thematic groups

---

## Tutorial Groups

### 01. [Parameter Systems](01-parameter-systems/)
**Topics**: Tier system, material presets, basis sets, FDF blocks
**Directories**: 4 (tier-system, presets, basis, fdf-blocks)
**Files**: ~32 tutorials
**Difficulty**: Beginner to Advanced

Control calculation parameters through hierarchical tier system, material-specific presets, custom basis sets, and advanced FDF block inputs.

**Key Features**:
- 5-tier hierarchy (basic → intermediate → advanced → expert)
- 27 material-specific presets across 9 categories
- Custom basis sets with species variants and per-atom control
- FDF block parameters for advanced features

**CLI Tools**: `atomate2siesta-presets list`

---

### 02. [Physics Features](02-physics-features/)
**Topics**: DOS, optical, magnetic, phonon, charged systems
**Directories**: 5 (dos, optical, magnetic, phonon, charge)
**Files**: ~17 tutorials
**Difficulty**: Intermediate to Advanced

Advanced physics calculations including density of states, optical properties, magnetism, phonon parameters, and charged/dipole systems.

**Key Features**:
- Automatic magnetic moment initialization (DM.InitSpin)
- Optical property calculations (dielectric, absorption)
- Advanced DOS customization
- Phonon force constant control
- Electric field and dipole corrections

---

### 03. [Infrastructure](03-infrastructure/)
**Topics**: Database, HPC submission, error handling, restart, dry-run
**Directories**: 5 (database, jobflow-remote, error-handling, restart, dry-run)
**Files**: 24 tutorials
**Difficulty**: Intermediate to Advanced

Complete infrastructure setup for production workflows including MongoDB integration, HPC job submission with jobflow-remote, custodian error handling, checkpoint/restart systems, and dry-run preview mode.

**Key Features**:
- MongoDB database storage
- HPC job submission via jobflow-remote
- Automatic error recovery with custodian
- Checkpoint/restart for long calculations
- Dry-run mode for parameter preview

---

### 04. [Structure Tools](04-structure-tools/)
**Topics**: Format conversion, manipulation, powerups
**Directories**: 3 (conversion, manipulation, powerups)
**Files**: ~11 tutorials
**Difficulty**: Beginner to Intermediate

Structure file format conversion, manipulation via CLI tools, and dynamic workflow customization with powerups.

**Key Features**:
- Convert between XV, FDF, XSF, CIF, VASP POSCAR
- 16 structure manipulation subcommands in 4 tiers
- Dynamic parameter modification with powerups
- `update_user_siesta_settings()` workflow transformation

**CLI Tools**: `atomate2siesta-structure <command>`

---

### 05. [Output & Visualization](05-output-viz/)
**Topics**: Grid output, denchar visualization, NetCDF
**Directories**: 3 (grid, denchar, netcdf)
**Files**: ~6 tutorials
**Difficulty**: Intermediate

Output file configuration for charge density grids, wavefunction visualization with denchar, and NetCDF format for analysis.

**Key Features**:
- SaveRho, SaveElectrostaticPotential, SaveTotalPotential
- Denchar visualization tools
- NetCDF compressed output format
- Grid-based output for post-processing

---

### 06. [Performance](06-performance/)
**Topics**: Parallel computation, efficiency options
**Directories**: 2 (parallel, efficiency)
**Files**: ~5 tutorials
**Difficulty**: Intermediate to Advanced

Performance optimization through MPI parallelization, memory management, and convergence acceleration.

**Key Features**:
- K-point and orbital parallelization
- DiagUseElpa for efficient diagonalization
- SolutionMethod optimization
- Memory and I/O tuning

---

### 07. [Advanced SIESTA](07-advanced-siesta/)
**Topics**: Hamiltonian/overlap matrices
**Directories**: 1 (hamiltonian)
**Files**: 3 tutorials
**Difficulty**: Advanced

Advanced SIESTA features for Hamiltonian and overlap matrix output, COOP/COHP analysis.

**Key Features**:
- SaveHS for Hamiltonian/overlap matrices
- LocalDensityOfStates analysis
- COOP/COHP chemical bonding analysis

---

### 08. [Recipe Book](08-recipe-book/)
**Topics**: High-level one-line workflows
**Directories**: 9 (by recipe category)
**Files**: comprehensive tutorials
**Difficulty**: Beginner to Advanced

**⭐ Production-ready high-level workflow system with one-line recipes achieving significant code reduction.**

Complete material studies, electronic properties, mechanical properties, thermal analysis, catalysis, convergence testing, and more—all in single-line function calls.

**Key Features**:
- recipes across 6 categories
- One-line workflows: `RecipeBook.complete_material_study(structure)`
- fully documented with comprehensive tutorials
- significant code reduction vs manual workflows

**CLI Tools**: `atomate2siesta-recipe list` and `atomate2siesta-recipe show <name>`

---

## Learning Paths

### Beginner Path
1. Start with **01-Parameter-Systems/01-tier-system** (basic tier usage)
2. Explore **01-Parameter-Systems/02-presets** (material presets)
3. Try **08-Recipe-Book** (one-line workflows)
4. Setup **03-Infrastructure/05-dry-run** (preview mode)

### Intermediate Path
1. Master **01-Parameter-Systems/03-basis** (custom basis sets)
2. Learn **02-Physics-Features** (DOS, optical, magnetic)
3. Configure **03-Infrastructure** (database, HPC, errors)
4. Use **04-Structure-Tools/03-powerups** (dynamic customization)

### Advanced Path
1. Explore **02-Physics-Features/03-magnetic** (complex spin states)
2. Master **03-Infrastructure/02-jobflow-remote** (HPC workflows)
3. Optimize with **06-Performance** (parallel computation)
4. Analyze with **07-Advanced-SIESTA** (Hamiltonian matrices)

---

## Quick Reference

### Most Used
- **Tier presets**: `01-parameter-systems/02-presets/`
- **Recipe book**: `08-recipe-book/`
- **HPC setup**: `03-infrastructure/02-jobflow-remote/`
- **Magnetic calculations**: `02-physics-features/03-magnetic/`

### By Feature
- **Parameter control**: Group 01 (Parameter Systems)
- **Physics calculations**: Group 02 (Physics Features)
- **Production setup**: Group 03 (Infrastructure)
- **Quick workflows**: Group 08 (Recipe Book)

### By Difficulty
- **Beginner**: Tier system, presets, recipe book, format conversion
- **Intermediate**: DOS, optical, powerups, grid output, parallel
- **Advanced**: Magnetic, basis customization, performance tuning, Hamiltonian

---

## CLI Tools

```bash
# Parameter presets
atomate2siesta-presets list
atomate2siesta-presets show relax_standard

# Recipe book
atomate2siesta-recipe list
atomate2siesta-recipe show complete_material_study

# Structure tools
atomate2siesta-structure info structure.cif
atomate2siesta-structure convert structure.fdf structure.cif

# Database
atomate2siesta-database test
atomate2siesta-database list

# Tutorials browser
atomate2siesta-tutorials list 03-advanced-features
atomate2siesta-tutorials search "magnetic"
```

---

## Highlights

### Recipe Book (Group 08)
- **One-line workflows** across 6 categories
- **significant code reduction** vs manual implementation
- fully documented with comprehensive tutorials
- extensive tutorial documentation

### Infrastructure (Group 03)
- **24 comprehensive tutorials** for production setup
- **MongoDB integration** with automatic storage
- **HPC job submission** via jobflow-remote
- **Error recovery** with custodian (high success rate)
- **Checkpoint/restart** for long calculations

### Parameter Systems (Group 01)
- **5-tier hierarchy** with automatic module activation
- **27 material presets** across 9 categories
- **Custom basis sets** with 4-phase species variant system
- **Automatic preset discovery** via CLI

### Physics Features (Group 02)
- **Automatic magnetic initialization** for 3d metals (Cr, Mn, Fe, Co, Ni, Cu)
- **Optical property** calculations (dielectric, absorption, refraction)
- **Advanced DOS** customization with broadening
- **Electric field** and dipole corrections

---

## Directory Structure

```
03-advanced-features/
├── 01-parameter-systems/          # Tiers, presets, basis, FDF (4 dirs, ~32 files)
├── 02-physics-features/           # DOS, optical, magnetic, etc (5 dirs, ~17 files)
├── 03-infrastructure/             # Database, HPC, errors (5 dirs, 24 files)
├── 04-structure-tools/            # Conversion, manipulation (3 dirs, ~11 files)
├── 05-output-viz/                 # Grid, denchar, NetCDF (3 dirs, ~6 files)
├── 06-performance/                # Parallel, efficiency (2 dirs, ~5 files)
├── 07-advanced-siesta/            # Hamiltonian/overlap (1 dir, 3 files)
└── 08-recipe-book/                # One-line workflows (9 dirs, 23 files)
```

**Total**: 8 groups, 32 subdirectories, ~120 tutorial files

---

## Related Documentation

- **[02-workflows](../02-workflows/)**: Multi-step calculation workflows
- **[01-basics](../01-basics/)**: Getting started tutorials
- **Main tutorial index**: [tutorials/README.md](../README.md)
- **CLI reference**: Run `atomate2siesta-<tool> --help` for any CLI tool

---

*Reorganized: 2026-01-13 (20 directories → 8 logical groups)*
