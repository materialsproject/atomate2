#!/usr/bin/env python3
# ruff: noqa: EXE001
"""CLI for translating atomic positions.

This module provides the `translate` subcommand for atomate2siesta-structure.
"""

from __future__ import annotations

import click
from pymatgen.core import Structure
from rich.console import Console
from rich.table import Table

console = Console()


@click.command()
@click.argument("structure_file", type=click.Path(exists=True))
@click.option(
    "--vector",
    type=float,
    nargs=3,
    help="Translation vector in Cartesian coordinates (x, y, z in Å)",
)
@click.option(
    "--fractional",
    type=float,
    nargs=3,
    help="Translation vector in fractional coordinates",
)
@click.option(
    "--element",
    type=str,
    help="Translate only this element (e.g., 'Cu', 'O')",
)
@click.option(
    "--center",
    is_flag=True,
    help="Center structure in unit cell (geometric center → [0.5, 0.5, 0.5])",
)
@click.option(
    "--wrap",
    is_flag=True,
    default=True,
    help="Wrap atoms back into cell (default: True)",
)
@click.option(
    "--no-wrap",
    is_flag=True,
    help="Don't wrap atoms back into cell",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file (default: translated_<input>)",
)
@click.option(
    "--format",
    type=click.Choice(["cif", "poscar", "xsf", "json"]),
    default="cif",
    help="Output format (default: cif)",
)
@click.option(
    "--show-before-after",
    is_flag=True,
    help="Show first few atomic positions before/after translation",
)
def translate(
    structure_file: str,
    vector: tuple[float, float, float] | None,
    fractional: tuple[float, float, float] | None,
    element: str | None,
    center: bool,
    wrap: bool,
    no_wrap: bool,
    output: str | None,
    format: str,  # noqa: A002 Click option name mirrors the CLI --format flag
    show_before_after: bool,
) -> None:
    """Translate atomic positions for alignment, centering, or interface building.

    Supports Cartesian and fractional translations, element-selective shifts,
    and automatic centering. Useful for preparing surfaces, interfaces, and
    heterostructures.

    Examples
    --------
        # Translate all atoms by 2.5 Å in z
        atomate2siesta-structure translate Si.cif --vector 0 0 2.5

        # Fractional translation
        atomate2siesta-structure translate Si.cif --fractional 0 0 0.1

        # Center structure in cell
        atomate2siesta-structure translate Si.cif --center

        # Shift only Cu atoms
        atomate2siesta-structure translate CuO.cif --element Cu --vector 0 0 0.5

        # No wrapping (allow atoms outside cell)
        atomate2siesta-structure translate Si.cif --vector 0 0 5 --no-wrap

        # Show position changes
        atomate2siesta-structure translate Si.cif --fractional 0 0 0.1 \
--show-before-after
    """
    # Handle wrap/no-wrap options
    if no_wrap:
        wrap = False

    # Validate options
    methods = [
        vector is not None,
        fractional is not None,
        center,
    ]
    if sum(methods) == 0:
        console.print(
            "[red]Error: Must specify one translation method:\n"
            "  --vector, --fractional, or --center[/red]"
        )
        raise click.Abort
    if sum(methods) > 1:
        console.print("[red]Error: Only specify one translation method[/red]")
        raise click.Abort

    try:
        # Load structure
        structure = Structure.from_file(structure_file)
        console.print(
            f"\n[cyan]Loaded structure: {structure.composition.reduced_formula}[/cyan]"
        )
        console.print(f"  Formula: {structure.composition.formula}")
        console.print(f"  Sites: {structure.num_sites}")

        if element:
            n_element = sum(1 for site in structure if site.specie.symbol == element)
            console.print(f"  {element} sites: {n_element}")
            if n_element == 0:
                console.print(
                    f"[red]Error: No {element} atoms found in structure[/red]"
                )
                raise click.Abort  # noqa: TRY301 intentional early exit within try

        # Store original positions for comparison
        if show_before_after:
            orig_frac_coords = [site.frac_coords for site in structure]

        # Perform translation
        translated_structure = structure.copy()
        translation_description = ""

        if center:
            # Center structure
            import numpy as np

            # Calculate geometric center
            frac_coords = np.array([site.frac_coords for site in translated_structure])
            center_frac = np.mean(frac_coords, axis=0)

            # Translation to center (0.5, 0.5, 0.5)
            shift = np.array([0.5, 0.5, 0.5]) - center_frac

            if element:
                # Center only selected element
                indices = [
                    i
                    for i, site in enumerate(translated_structure)
                    if site.specie.symbol == element
                ]
                translated_structure.translate_sites(
                    indices=indices, vector=shift, frac_coords=True
                )
                translation_description = f"Centered {element} atoms in unit cell"
            else:
                # Center all atoms
                translated_structure.translate_sites(
                    indices=range(len(translated_structure)),
                    vector=shift,
                    frac_coords=True,
                )
                translation_description = "Centered structure in unit cell"

            display_vector = (
                f"Δ(frac) = [{shift[0]:.4f}, {shift[1]:.4f}, {shift[2]:.4f}]"
            )

        elif vector is not None:
            # Cartesian translation
            if element:
                indices = [
                    i
                    for i, site in enumerate(translated_structure)
                    if site.specie.symbol == element
                ]
                translated_structure.translate_sites(
                    indices=indices, vector=vector, frac_coords=False
                )
                translation_description = (
                    f"Translated {element} atoms: "
                    f"[{vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}] Å"
                )
            else:
                translated_structure.translate_sites(
                    indices=range(len(translated_structure)),
                    vector=vector,
                    frac_coords=False,
                )
                translation_description = (
                    f"Cartesian translation: "
                    f"[{vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}] Å"
                )

            display_vector = (
                f"Δ(cart) = [{vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}] Å"
            )

        elif fractional is not None:
            # Fractional translation
            if element:
                indices = [
                    i
                    for i, site in enumerate(translated_structure)
                    if site.specie.symbol == element
                ]
                translated_structure.translate_sites(
                    indices=indices, vector=fractional, frac_coords=True
                )
                translation_description = (
                    f"Translated {element} atoms: "
                    f"[{fractional[0]:.4f}, {fractional[1]:.4f}, "
                    f"{fractional[2]:.4f}] (frac)"
                )
            else:
                translated_structure.translate_sites(
                    indices=range(len(translated_structure)),
                    vector=fractional,
                    frac_coords=True,
                )
                translation_description = (
                    f"Fractional translation: "
                    f"[{fractional[0]:.4f}, {fractional[1]:.4f}, "
                    f"{fractional[2]:.4f}]"
                )

            display_vector = (
                f"Δ(frac) = [{fractional[0]:.4f}, "
                f"{fractional[1]:.4f}, {fractional[2]:.4f}]"
            )

        # Apply wrapping if requested
        if wrap:
            import numpy as np

            # Wrap all fractional coordinates back to [0, 1)
            for i, site in enumerate(translated_structure):
                frac_coords = site.frac_coords
                wrapped_coords = frac_coords - np.floor(frac_coords)
                translated_structure.replace(
                    i,
                    site.specie,
                    wrapped_coords,
                    properties=site.properties,
                )

        # Display translation information
        console.print(f"\n[yellow]{translation_description}[/yellow]")
        console.print(f"  {display_vector}")
        if wrap:
            console.print("  Wrapping: Applied (atoms kept in [0, 1))")
        else:
            console.print("  Wrapping: Disabled (atoms may be outside cell)")

        # Show before/after positions
        if show_before_after:
            n_show = min(5, len(structure))
            table = Table(
                title=f"Position Changes (first {n_show} atoms)",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Index", style="cyan")
            table.add_column("Element", style="green")
            table.add_column("Before (frac)", style="yellow")
            table.add_column("After (frac)", style="red")

            for i in range(n_show):
                orig = orig_frac_coords[i]
                new = translated_structure[i].frac_coords
                table.add_row(
                    str(i),
                    str(structure[i].specie),
                    f"[{orig[0]:.4f}, {orig[1]:.4f}, {orig[2]:.4f}]",
                    f"[{new[0]:.4f}, {new[1]:.4f}, {new[2]:.4f}]",
                )

            console.print("\n")
            console.print(table)

        # Structure comparison
        console.print("\n[cyan]Structure unchanged:[/cyan]")
        console.print(f"  Sites: {translated_structure.num_sites}")
        console.print(
            f"  Volume: {translated_structure.volume:.3f} Å³ (same as original)"
        )

        # Save structure
        if output is None:
            from pathlib import Path

            input_path = Path(structure_file)
            output = f"translated_{input_path.name}"

        if format == "cif":
            from atomate2.siesta.sets.utils.structure_io import write_cif_with_ghost

            write_cif_with_ghost(translated_structure, output)
        elif format == "poscar":
            translated_structure.to(filename=output, fmt="poscar")
        elif format == "xsf":
            from pymatgen.io.xcrysden import XSF

            xsf = XSF(translated_structure)
            with open(output, "w") as f:
                f.write(xsf.to_str())
        elif format == "json":
            translated_structure.to(filename=output, fmt="json")

        console.print(f"\n[green]✓ Translated structure saved to: {output}[/green]")

        # Usage tips
        if not show_before_after:
            console.print(
                "\n[dim]Tip: Use --show-before-after to see position changes[/dim]"
            )
        if wrap and (vector or fractional):
            console.print(
                "[dim]Tip: Use --no-wrap to allow atoms outside cell boundaries[/dim]"
            )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort from e


if __name__ == "__main__":
    translate()
