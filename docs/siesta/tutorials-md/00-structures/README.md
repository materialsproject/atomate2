# Tutorial 00: Structure Files

This directory contains structure files (CIF format) used across all tutorials.

## Available Structures

### Si_mp-149_primitive.cif
Silicon primitive cell (2 atoms)
- **Formula**: Si₂
- **Space Group**: Fd-3m (227)
- **Lattice**: FCC, a = 5.43 Å
- **Used in**: Most tutorials (relaxation, band structure, convergence tests, EOS, elastic constants)

### Si_mp-149_conventional_standard.cif
Silicon conventional cell (8 atoms)
- **Formula**: Si₈
- **Space Group**: Fd-3m (227)
- **Lattice**: Cubic, a = 5.43 Å
- **Used in**: Selected tutorials demonstrating larger systems

### G.cif
Graphene structure
- **Formula**: C₂
- **Space Group**: P6/mmm (191)
- **Lattice**: Hexagonal
- **Used in**: NEB calculations tutorial

## Usage

All tutorial scripts reference structures from this directory using relative paths:

```python
from pymatgen.core import Structure

structure = Structure.from_file("../00-structures/Si_mp-149_primitive.cif")
```

## Adding New Structures

To add structures for your own tutorials:

1. Place CIF file in this directory
2. Update tutorial scripts with path: `"../00-structures/your_structure.cif"`
3. Document the structure in this README

## Structure Sources

- Silicon structures: Materials Project (mp-149)
- Graphene: Standard graphene monolayer

## Notes

- All structures are in Crystallographic Information File (CIF) format
- Structures are standardized using pymatgen conventions
- Primitive cells are preferred for computational efficiency
- Conventional cells are used when demonstrating symmetry or larger systems
