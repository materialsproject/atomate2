"""Standardize crystal structures to conventional or primitive cells.

This module provides the `standardize` command for converting structures to
standard conventional, primitive, or international settings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from rich.console import Console
from rich.table import Table

console = Console()


def load_structure(file_path: str) -> Structure:
    """Load structure from file."""
    return Structure.from_file(file_path)


def save_structure(structure: Structure, filename: str, fmt: str) -> None:
    """Save structure to file."""
    from ase import Atoms
    from ase.io import write as ase_write

    if fmt == "cif":
        from atomate2.siesta.sets.utils.structure_io import write_cif_with_ghost

        write_cif_with_ghost(structure, filename)
    elif fmt == "poscar":
        structure.to(filename=filename, fmt="poscar")
    elif fmt == "xsf":
        atoms = Atoms(
            symbols=[str(site.specie) for site in structure],
            positions=structure.cart_coords,
            cell=structure.lattice.matrix,
            pbc=True,
        )
        ase_write(filename, atoms, format="xsf")
    elif fmt == "json":
        structure.to(filename=filename, fmt="json")
    elif fmt in ["fdf", "XV"]:
        import sisl

        geom = sisl.Geometry.new(structure)
        if fmt == "fdf":
            geom.write(filename)
        else:  # XV
            geom.write(
                filename.replace(".xv", ".XV") if ".xv" in filename else filename
            )
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def show_structure_comparison(original: Structure, modified: Structure) -> None:
    """Show before/after comparison table."""
    table = Table(title="Structure Comparison", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Before", style="yellow")
    table.add_column("After", style="green")
    table.add_column("Change", style="magenta")

    # Sites
    table.add_row(
        "Sites",
        str(original.num_sites),
        str(modified.num_sites),
        f"{modified.num_sites - original.num_sites:+d}",
    )

    # Volume
    table.add_row(
        "Volume (Ų)",
        f"{original.volume:.2f}",
        f"{modified.volume:.2f}",
        f"{modified.volume - original.volume:+.2f}",
    )

    # Lattice parameters
    for param in ["a", "b", "c"]:
        orig_val = getattr(original.lattice, param)
        mod_val = getattr(modified.lattice, param)
        table.add_row(
            f"{param} (Å)",
            f"{orig_val:.4f}",
            f"{mod_val:.4f}",
            f"{mod_val - orig_val:+.4f}",
        )

    console.print(table)


@click.command()
@click.argument("structure_file", type=click.Path(exists=True))
@click.option(
    "--conventional",
    is_flag=True,
    help="Convert to conventional cell",
)
@click.option(
    "--primitive",
    is_flag=True,
    help="Convert to primitive cell",
)
@click.option(
    "--international",
    is_flag=True,
    help="Use international standard setting",
)
@click.option(
    "--symprec",
    type=float,
    default=0.01,
    help="Symmetry precision (Å, default: 0.01)",
)
@click.option(
    "--angle-tolerance",
    type=float,
    default=5.0,
    help="Angle tolerance (degrees, default: 5.0)",
)
@click.option(
    "-o",
    "--output",
    type=str,
    help="Output filename (default: <mode>_<input>)",
)
@click.option(
    "--format",
    type=click.Choice(["cif", "poscar", "xsf", "json", "fdf", "XV"]),
    default="cif",
    help="Output format (default: cif)",
)
@click.option(
    "--show-before-after",
    is_flag=True,
    help="Show before/after comparison table",
)
def standardize(
    structure_file: str,
    conventional: bool,
    primitive: bool,
    international: bool,
    symprec: float,
    angle_tolerance: float,
    output: str | None,
    format: str,  # noqa: A002 Click option name mirrors the CLI --format flag
    show_before_after: bool,
) -> None:
    """Standardize crystal structure to conventional or primitive cell.

    Converts structures to standard cells using spglib:
    - Conventional cell (standard crystallographic cell)
    - Primitive cell (smallest repeating unit)
    - International standard setting

    Examples
    --------
        # Convert to conventional cell
        atomate2siesta-structure standardize structure.cif --conventional

        # Convert to primitive cell
        atomate2siesta-structure standardize structure.cif --primitive

        # International standard setting
        atomate2siesta-structure standardize structure.cif --international

        # Custom symmetry precision
        atomate2siesta-structure standardize structure.cif --primitive --symprec 0.1
    """
    try:
        # Validate options
        mode_count = sum([conventional, primitive, international])
        if mode_count == 0:
            console.print(
                "[bold red]Error:[/bold red] Must specify one of: "
                "--conventional, --primitive, or --international"
            )
            sys.exit(1)
        if mode_count > 1:
            console.print(
                "[bold red]Error:[/bold red] Can only specify one standardization mode"
            )
            sys.exit(1)

        # Load structure
        console.print("\n[bold cyan]Loading structure...[/bold cyan]")
        structure = load_structure(structure_file)
        original_structure = structure.copy()

        console.print(f"  Formula: {structure.formula}")
        console.print(f"  Sites: {structure.num_sites}")
        console.print(f"  Volume: {structure.volume:.2f} ų\n")

        # Analyze symmetry
        console.print("[bold cyan]Analyzing symmetry...[/bold cyan]")
        sga = SpacegroupAnalyzer(
            structure, symprec=symprec, angle_tolerance=angle_tolerance
        )

        console.print(f"  Space group: {sga.get_space_group_symbol()}")
        console.print(f"  Space group number: {sga.get_space_group_number()}")
        console.print(f"  Crystal system: {sga.get_crystal_system()}")
        console.print(f"  Point group: {sga.get_point_group_symbol()}\n")

        # Perform standardization
        console.print("[bold cyan]Standardizing structure...[/bold cyan]")

        if conventional:
            standardized = sga.get_conventional_standard_structure()
            mode_name = "conventional"
            console.print("  Mode: Conventional cell")
        elif primitive:
            standardized = sga.get_primitive_standard_structure()
            mode_name = "primitive"
            console.print("  Mode: Primitive cell")
        else:  # international
            standardized = sga.get_refined_structure()
            mode_name = "international"
            console.print("  Mode: International standard setting")

        console.print(f"  New sites: {standardized.num_sites}")
        console.print(f"  New volume: {standardized.volume:.2f} ų\n")

        # Show comparison
        if show_before_after:
            show_structure_comparison(original_structure, standardized)

        # Determine output filename
        if output is None:
            input_path = Path(structure_file)
            base_name = input_path.stem
            ext = format if format != "XV" else "xv"
            output = f"{mode_name}_{base_name}.{ext}"

        # Save structure
        console.print(
            f"[bold cyan]Saving standardized structure to {output}...[/bold cyan]"
        )
        save_structure(standardized, output, format)

        console.print(
            f"\n[bold green]✓ Successfully standardized to "
            f"{mode_name} cell![/bold green]"
        )
        console.print(f"  Output: {output}\n")

        # Show summary
        _show_summary(original_structure, standardized, mode_name)

    except Exception as e:  # noqa: BLE001 friendly CLI error reporting
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _show_summary(
    original: Structure,
    standardized: Structure,
    mode: str,  # noqa: ARG001 kept for signature symmetry
) -> None:
    """Show summary of standardization."""
    table = Table(title="Standardization Summary", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Original", style="yellow")
    table.add_column("Standardized", style="green")
    table.add_column("Change", style="magenta")

    # Number of sites
    site_multiplier = standardized.num_sites / original.num_sites
    table.add_row(
        "Sites",
        str(original.num_sites),
        str(standardized.num_sites),
        f"×{site_multiplier:.2f}",  # noqa: RUF001
    )

    # Volume
    vol_multiplier = standardized.volume / original.volume
    table.add_row(
        "Volume (Ų)",
        f"{original.volume:.2f}",
        f"{standardized.volume:.2f}",
        f"×{vol_multiplier:.2f}",  # noqa: RUF001
    )

    # Lattice parameters
    for param in ["a", "b", "c"]:
        orig_val = getattr(original.lattice, param)
        std_val = getattr(standardized.lattice, param)
        change = std_val - orig_val
        table.add_row(
            f"{param} (Å)",
            f"{orig_val:.4f}",
            f"{std_val:.4f}",
            f"{change:+.4f}",
        )

    # Angles
    for param in ["alpha", "beta", "gamma"]:
        orig_val = getattr(original.lattice, param)
        std_val = getattr(standardized.lattice, param)
        change = std_val - orig_val
        table.add_row(
            f"{param} (°)",
            f"{orig_val:.2f}",
            f"{std_val:.2f}",
            f"{change:+.2f}",
        )

    console.print(table)
    console.print()


if __name__ == "__main__":
    standardize()
