# Structure Manipulation Tutorials

This directory contains comprehensive tutorials for the `atomate2siesta-structure` command-line tool, which provides 16 subcommands organized into 4 tiers for crystal structure manipulation and analysis.

## Overview

The structure manipulation commands enable researchers to:
- **Validate and convert** structures between formats
- **Create and modify** crystal structures programmatically
- **Analyze and compare** structures quantitatively
- **Prepare structures** for DFT calculations

All tutorials use practical examples with real crystal structures and demonstrate command-line usage with expected outputs.

## Tutorial Organization

### Tutorial 1: Structure Comparison Workflow
**File**: `01_structure_comparison.py` (287 lines)

Learn how to compare structures quantitatively using the `compare` command.

**Topics Covered**:
- Before/after optimization comparison
- Format conversion verification
- Standardization result validation
- Supercell generation verification
- Tolerance-based comparison
- Composition comparison

**Example Usage**:
```bash
# Compare two structures
atomate2siesta-structure compare si_original.cif si_optimized.cif

# With custom tolerance
atomate2siesta-structure compare si_original.cif si_perturbed.cif --tolerance 0.1

# Detailed site analysis
atomate2siesta-structure compare si_original.cif si_perturbed.cif --verbose
```

**Key Commands**:
- `compare` - Quantitative structure comparison (RMSD, lattice, composition)

**Run Tutorial**:
```bash
python 01_structure_comparison.py
```

---

### Tutorial 2: Cell Standardization
**File**: `02_cell_standardization.py` (389 lines)

Master cell standardization for DFT efficiency and database comparison.

**Topics Covered**:
- Conventional → Primitive (DFT efficiency: 8→2 atoms = 4× speedup)
- Primitive → Conventional (visualization)
- International standard settings
- Symmetry analysis workflow
- Custom symmetry precision
- Workflow integration diagrams
- Before/after comparison
- Multi-format output

**Example Usage**:
```bash
# Convert to primitive cell (faster DFT)
atomate2siesta-structure standardize si_conventional.cif --primitive

# Convert to conventional cell (easier visualization)
atomate2siesta-structure standardize si_primitive.cif --conventional

# International standard setting
atomate2siesta-structure standardize tio2.cif --international

# Custom symmetry tolerance
atomate2siesta-structure standardize si_distorted.cif --primitive --symprec 0.1
```

**Key Commands**:
- `standardize` - Cell standardization (primitive/conventional/international)

**Computational Savings**:
```
Atoms Reduction | DFT Speedup (approx)
----------------|---------------------
8 → 2           | 4× faster
4 → 2           | 2× faster
6 → 3           | 2× faster
```

**Run Tutorial**:
```bash
python 02_cell_standardization.py
```

---

### Tutorial 3: Surface Preparation Pipeline
**File**: `03_surface_preparation.py` (570+ lines)

Complete workflow for preparing surface slabs for DFT calculations.

**Topics Covered**:
- Basic surface slab generation
- Multiple surface orientations (100, 110, 111)
- Vacuum thickness optimization
- Symmetric vs asymmetric slabs
- Complete surface preparation pipeline
- Surface termination analysis
- Layer-by-layer convergence
- Integration with DFT workflows
- Advanced surface features

**Example Usage**:
```bash
# Generate Cu(111) surface
atomate2siesta-structure slab cu_bulk.cif \
    --miller-indices 1,1,1 \
    --min-slab-size 10.0 \
    --min-vacuum-size 15.0 \
    --symmetric

# Complete pipeline: bulk → slab → primitive → orthogonalize
atomate2siesta-structure slab mgo_bulk.cif --miller-indices 1,0,0 --min-slab-size 12.0
atomate2siesta-structure standardize slab_mgo_bulk.cif --primitive
atomate2siesta-structure optimize-cell primitive_slab_mgo_bulk.cif --orthogonalize
```

**Key Commands**:
- `slab` - Surface slab generation
- `standardize` - Reduce to primitive surface cell
- `optimize-cell` - Orthogonalize for better k-points

**Surface Guidelines**:
```
Parameter        | Typical Values
-----------------|----------------------------------
Slab thickness   | 5-7 layers (10-14 Å)
Vacuum size      | 15-20 Å
k-point density  | 2× bulk density in slab plane
```

**Run Tutorial**:
```bash
python 03_surface_preparation.py
```

---

### Tutorial 4: Complete Structure Manipulation Workflow
**File**: `04_complete_workflow.py` (650+ lines)

End-to-end workflows combining multiple commands for real research scenarios.

**Topics Covered**:

**Workflow 1: Materials Discovery Pipeline**
- Generate derivative structures (perovskite screening)
- Batch validation and standardization
- Systematic comparison with reference
- DFT workflow generation for top candidates

**Workflow 2: Surface Adsorption Study**
- Bulk → Surface → Adsorbate → Attach → Optimize → DFT
- CO adsorption on Cu(111) for catalysis
- Multiple adsorption sites (top/bridge/hollow)
- Complete bash script with error handling

**Workflow 3: Structure Database Curation**
- Download → Validate → Standardize → Compare → Archive
- Transition metal oxide database
- Python script with quality assurance
- Automated report generation

**Workflow 4: Multi-System Comparison**
- Compare SiO2 polymorphs (quartz, cristobalite, etc.)
- Pairwise comparison matrix
- Density trend analysis
- Summary report generation

**Example Workflows**:
```bash
# Materials discovery (bash)
for element in Sr Ba Pb; do
    atomate2siesta-structure substitute CaTiO3.cif Ca $element
    atomate2siesta-structure validate substituted_CaTiO3.cif
    atomate2siesta-structure standardize substituted_CaTiO3.cif --primitive
done

# Surface adsorption (complete pipeline)
atomate2siesta-structure slab cu_bulk.cif --miller-indices 1,1,1
atomate2siesta-structure standardize slab_cu_bulk.cif --primitive
atomate2siesta-structure optimize-cell primitive_slab_cu_bulk.cif --orthogonalize
atomate2siesta-structure molecule --formula CO
atomate2siesta-structure attach optimized_surface.cif co_molecule.cif --position top
atomate2siesta-maker relax attached_structure.cif --preset surface_standard
```

**Key Integration Patterns**:
- Command chaining with bash scripts
- Error handling and validation checkpoints
- Python wrappers for complex logic
- Automated DFT workflow generation

**Run Tutorial**:
```bash
python 04_complete_workflow.py
```

---

## Command Reference

### Tier 1: Basic Information and Conversion (4 commands)
```bash
atomate2siesta-structure info <file>                 # Display structure properties
atomate2siesta-structure convert <file> --format cif # Convert between formats
atomate2siesta-structure validate <file>             # Validate structure integrity
atomate2siesta-structure molecule --formula H2O      # Create molecular structures
```

### Tier 2: Structure Generation and Modification (4 commands)
```bash
atomate2siesta-structure supercell <file> --matrix 2 2 2       # Create supercells
atomate2siesta-structure slab <file> --miller-indices 1,1,1    # Generate surface slabs
atomate2siesta-structure attach <slab> <mol> --position top    # Attach molecules
atomate2siesta-structure perturb <file> --distance 0.1         # Perturb atomic positions
```

### Tier 3: Composition and Organization (5 commands)
```bash
atomate2siesta-structure remove-species <file> H               # Remove elements
atomate2siesta-structure substitute <file> Ca Sr               # Substitute elements
atomate2siesta-structure merge <file1> <file2>                 # Merge structures
atomate2siesta-structure sort <file> --by electronegativity    # Sort atoms
atomate2siesta-structure symmetry <file>                       # Analyze symmetry
```

### Tier 4: Analysis and Optimization (3 commands)
```bash
atomate2siesta-structure compare <file1> <file2>               # Compare structures
atomate2siesta-structure standardize <file> --primitive        # Cell standardization
atomate2siesta-structure optimize-cell <file> --orthogonalize  # Cell optimization
```

---

## Quick Start

### 1. Run All Tutorials
```bash
cd tutorials/09-structure-manipulation/
for tutorial in *.py; do
    echo "Running: $tutorial"
    python $tutorial
done
```

### 2. Interactive Learning
Start with Tutorial 1 and work sequentially through all tutorials. Each tutorial builds on concepts from previous ones.

### 3. Copy-Paste Examples
All command examples in tutorials can be copy-pasted directly into your terminal. Structure files are created programmatically by each tutorial.

---

## Prerequisites

### Required Packages
```bash
pip install atomate2[siesta] pymatgen
```

### Verify Installation
```bash
atomate2siesta-structure --help
```

You should see the help message with all 16 subcommands listed.

---

## Typical Workflows

### DFT Calculation Preparation
```bash
# 1. Standardize to primitive (faster DFT)
atomate2siesta-structure standardize input.cif --primitive

# 2. Verify structure
atomate2siesta-structure validate primitive_input.cif

# 3. Generate DFT workflow
atomate2siesta-maker relax primitive_input.cif --preset relax_standard
```

### Surface Energy Calculation
```bash
# 1. Create surface slab
atomate2siesta-structure slab bulk.cif --miller-indices 1,1,1

# 2. Standardize and optimize
atomate2siesta-structure standardize slab_bulk.cif --primitive
atomate2siesta-structure optimize-cell primitive_slab_bulk.cif --orthogonalize

# 3. Generate workflow
atomate2siesta-maker relax optimized_slab.cif --preset surface_standard
```

### Format Conversion Pipeline
```bash
# 1. Convert to desired format
atomate2siesta-structure convert input.cif --format xsf

# 2. Verify conversion preserved structure
atomate2siesta-structure compare input.cif structure.xsf

# 3. Check no errors introduced
atomate2siesta-structure validate structure.xsf
```

### Materials Screening
```bash
# Batch process multiple structures
for file in materials/*.cif; do
    # Standardize
    atomate2siesta-structure standardize $file --primitive

    # Validate
    if atomate2siesta-structure validate primitive_$file; then
        # Generate workflow
        atomate2siesta-maker relax primitive_$file --preset relax_standard
    fi
done
```

---

## Best Practices

### 1. Always Validate
```bash
# After any structure manipulation
atomate2siesta-structure validate output.cif
```

### 2. Compare Before/After
```bash
# Verify transformations preserved essential properties
atomate2siesta-structure compare input.cif output.cif
```

### 3. Use Primitive Cells for DFT
```bash
# Reduces computation time
atomate2siesta-structure standardize structure.cif --primitive
```

### 4. Standardize Before Database Submission
```bash
# Ensures consistent format
atomate2siesta-structure standardize structure.cif --conventional
```

### 5. Check Orthogonality for Surfaces
```bash
# Better k-point sampling
atomate2siesta-structure optimize-cell slab.cif --orthogonalize
```

---

## Troubleshooting

### Common Issues

**Issue**: "Structure file not found"
```bash
# Solution: Check file path is absolute or relative to current directory
atomate2siesta-structure info ./structures/si.cif
```

**Issue**: "Invalid structure"
```bash
# Solution: Run validate to see specific errors
atomate2siesta-structure validate structure.cif --verbose
```

**Issue**: "Standardization changed atom count unexpectedly"
```bash
# Solution: Compare with original, may be reducing to primitive
atomate2siesta-structure compare original.cif primitive_original.cif
```

**Issue**: "RMSD too high after transformation"
```bash
# Solution: Use larger tolerance for relaxed structures
atomate2siesta-structure compare struct1.cif struct2.cif --tolerance 0.1
```

---

## Additional Resources

### Full Documentation
```bash
# List all commands
atomate2siesta-structure --help

# Command-specific help
atomate2siesta-structure compare --help
atomate2siesta-structure standardize --help
atomate2siesta-structure slab --help
```

### Online Documentation
- **CLI Reference**: `docs/source/cli-tools.rst`
- **Project Docs**: `docs/source/index.rst`
- **Main README**: `../README.md`

### Related Tutorials
- **01-basics**: Basic relaxation and band structure
- **03-advanced-workflows**: EOS, elastic constants, NEB
- **06-surfaces-and-adsorption**: Surface energy workflows
- **08-recipe-book**: High-level workflow system

---

## Tutorial Statistics

| Tutorial | Lines | Topics | Commands | Examples |
|----------|-------|--------|----------|----------|
| 01_structure_comparison.py | 287 | 6 | 1 | 6 |
| 02_cell_standardization.py | 389 | 8 | 1 | 8 |
| 03_surface_preparation.py | 570+ | 9 | 3 | 9 |
| 04_complete_workflow.py | 650+ | 4 workflows | 16 | 4 |
| **Total** | **~1,900** | **27** | **16** | **27** |

---

## Contributing

If you find errors or have suggestions for improving these tutorials:

1. Open an issue: https://github.com/materialsproject/atomate2/issues
2. Submit a pull request with improvements
3. Share your own workflow examples

---

## License

These tutorials are part of the atomate2siesta project and are distributed under the same license.

---

## Contact

For questions about these tutorials or the `atomate2siesta-structure` command:
- GitHub: https://github.com/materialsproject/atomate2
- Documentation: See `docs/` directory

---

**Happy structure manipulation!** 🔬⚛️
