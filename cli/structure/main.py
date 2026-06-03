#!/usr/bin/env python
"""
Main CLI group for structure manipulation tools.

This module provides the main command group for all structure-related operations.
"""

import click
from atomate2.siesta.cli.structure.convert import main as convert_command
from atomate2.siesta.cli.structure.info import main as info_command
from atomate2.siesta.cli.structure.scale import scale as scale_command
from atomate2.siesta.cli.structure.supercell import supercell as supercell_command
from atomate2.siesta.cli.structure.rotate import rotate as rotate_command
from atomate2.siesta.cli.structure.translate import translate as translate_command
from atomate2.siesta.cli.structure.slab import slab as slab_command
from atomate2.siesta.cli.structure.vacuum import vacuum as vacuum_command
from atomate2.siesta.cli.structure.stack import stack as stack_command
from atomate2.siesta.cli.structure.substitute import substitute as substitute_command
from atomate2.siesta.cli.structure.remove import remove as remove_command
from atomate2.siesta.cli.structure.add import add as add_command
from atomate2.siesta.cli.structure.perturb import perturb as perturb_command
from atomate2.siesta.cli.structure.compare import compare as compare_command
from atomate2.siesta.cli.structure.standardize import standardize as standardize_command
from atomate2.siesta.cli.structure.optimize_cell import (
    optimize_cell as optimize_cell_command,
)


@click.group()
def cli():
    """Structure manipulation and conversion tools for SIESTA.

    Convert between different structure formats (sisl, ASE, pymatgen),
    remove ghost atoms, export to various file formats, and perform
    crystallographic operations (scale, rotate, translate, supercell).

    Available Commands:
        # Tier 1: Basic Operations
        convert    - Convert structure formats (SIESTA ↔ CIF/POSCAR/XSF)
        info       - Display structure information and analysis
        scale      - Scale lattice parameters (EOS, pressure, strain)
        supercell  - Generate supercells (phonons, defects, surfaces)
        rotate     - Rotate structures (alignment, reorientation)
        translate  - Translate atomic positions (centering, interfaces)

        # Tier 2: Surface & 2D Operations
        slab       - Generate surface slabs (adsorption, catalysis)
        vacuum     - Add vacuum spacing (2D materials, surfaces)
        stack      - Stack layers (heterostructures, multilayers)

        # Tier 3: Advanced Atomic Manipulation
        substitute - Substitute atoms (doping, alloying)
        remove     - Remove atoms (vacancies, cleanup)
        add        - Add atoms/molecules (adsorbates, dopants)
        perturb    - Random perturbations (MD, transition states)

        # Tier 4: Analysis & Optimization
        compare      - Compare two structures (RMSD, lattice differences)
        standardize  - Convert to conventional/primitive cells
        optimize-cell - Cell optimization (Niggli, orthogonalization)

    Examples:
        atomate2siesta-structure convert siesta.fdf --write-xsf --write-cif
        atomate2siesta-structure info structure.cif
        atomate2siesta-structure scale structure.cif --factor 1.05
        atomate2siesta-structure supercell structure.cif --matrix 2 2 2
        atomate2siesta-structure rotate structure.cif --axis z --angle 45
        atomate2siesta-structure translate structure.cif --center
    """
    pass


# Add subcommands
cli.add_command(convert_command, name="convert")
cli.add_command(info_command, name="info")
cli.add_command(scale_command, name="scale")
cli.add_command(supercell_command, name="supercell")
cli.add_command(rotate_command, name="rotate")
cli.add_command(translate_command, name="translate")
cli.add_command(slab_command, name="slab")
cli.add_command(vacuum_command, name="vacuum")
cli.add_command(stack_command, name="stack")
cli.add_command(substitute_command, name="substitute")
cli.add_command(remove_command, name="remove")
cli.add_command(add_command, name="add")
cli.add_command(perturb_command, name="perturb")
cli.add_command(compare_command, name="compare")
cli.add_command(standardize_command, name="standardize")
cli.add_command(optimize_cell_command, name="optimize-cell")


if __name__ == "__main__":
    cli()
