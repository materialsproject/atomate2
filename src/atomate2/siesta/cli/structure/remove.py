#!/usr/bin/env python3
"""CLI for removing atoms from structures.

This module provides the `remove` subcommand for atomate2siesta-structure.
"""

from __future__ import annotations

import numpy as np
import click
from rich.console import Console
from rich.table import Table

from pymatgen.core import Element, Structure

console = Console()


@click.command()
@click.argument("structure_file", type=click.Path(exists=True))
@click.option(
    "--element",
    type=str,
    help="Remove all atoms of this element",
)
@click.option(
    "--sites",
    type=str,
    help="Site indices to remove (comma-separated, 0-based)",
)
@click.option(
    "--near",
    type=str,
    help="Remove atoms near this position (x,y,z in Cartesian Å or fractional)",
)
@click.option(
    "--radius",
    type=float,
    help="Radius for --near option (in Å)",
)
@click.option(
    "--fractional",
    is_flag=True,
    help="Treat --near coordinates as fractional",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file (default: removed_<input>)",
)
@click.option(
    "--format",
    type=click.Choice(["cif", "poscar", "xsf", "json", "fdf", "XV"]),
    default="cif",
    help="Output format (default: cif)",
)
@click.option(
    "--show-removed",
    is_flag=True,
    help="Show which atoms were removed",
)
def remove(
    structure_file,
    element,
    sites,
    near,
    radius,
    fractional,
    output,
    format,
    show_removed,
):
    """Remove atoms from a structure (vacancies, cleanup, surface modifications).

    Remove atoms by element, site index, or proximity to a point. Useful for
    creating vacancies, removing adsorbates, or cleaning up structures.

    Examples:

        # Remove all hydrogen atoms
        atomate2siesta-structure remove structure.cif --element H

        # Remove specific sites
        atomate2siesta-structure remove structure.cif --sites 0,5,10

        # Remove atoms within 2.5 Å of a point
        atomate2siesta-structure remove structure.cif --near 0,0,5.0 --radius 2.5

        # Remove atoms near fractional coordinate
        atomate2siesta-structure remove structure.cif --near 0.5,0.5,0.5 --radius 2.0 --fractional

        # Show what was removed
        atomate2siesta-structure remove structure.cif --element O --show-removed
    """
    try:
        # Validate options
        if not any([element, sites, near]):
            console.print(
                "[red]Error: Must specify at least one of --element, --sites, or --near[/red]"
            )
            raise click.Abort()

        if near and not radius:
            console.print("[red]Error: --near requires --radius[/red]")
            raise click.Abort()

        # Load structure
        structure = Structure.from_file(structure_file)
        console.print(
            f"\n[cyan]Loaded structure: {structure.composition.reduced_formula}[/cyan]"
        )
        console.print(f"  Formula: {structure.composition.formula}")
        console.print(f"  Sites: {structure.num_sites}")

        # Determine sites to remove
        sites_to_remove = set()

        # By element
        if element:
            try:
                elem = Element(element)
                element_indices = [
                    i
                    for i, site in enumerate(structure)
                    if site.specie.symbol == elem.symbol
                ]
                sites_to_remove.update(element_indices)
                console.print(
                    f"[yellow]Found {len(element_indices)} {elem.symbol} atoms to remove[/yellow]"
                )
            except Exception as e:
                console.print(f"[red]Error: Invalid element symbol: {e}[/red]")
                raise click.Abort()

        # By site indices
        if sites:
            try:
                site_indices = [int(x) for x in sites.split(",")]
                # Validate
                for idx in site_indices:
                    if idx < 0 or idx >= structure.num_sites:
                        console.print(
                            f"[red]Error: Site index {idx} out of range (0-{structure.num_sites-1})[/red]"
                        )
                        raise click.Abort()
                sites_to_remove.update(site_indices)
                console.print(
                    f"[yellow]Adding {len(site_indices)} sites by index[/yellow]"
                )
            except ValueError:
                console.print("[red]Error: Invalid site indices format[/red]")
                raise click.Abort()

        # By proximity
        if near:
            try:
                coords = [float(x) for x in near.split(",")]
                if len(coords) != 3:
                    console.print("[red]Error: --near must be x,y,z (3 values)[/red]")
                    raise click.Abort()

                if fractional:
                    # Convert to Cartesian
                    center = structure.lattice.get_cartesian_coords(coords)
                else:
                    center = np.array(coords)

                # Find atoms within radius
                near_indices = []
                for i, site in enumerate(structure):
                    dist = np.linalg.norm(site.coords - center)
                    if dist <= radius:
                        near_indices.append(i)

                sites_to_remove.update(near_indices)
                coord_type = "fractional" if fractional else "Cartesian"
                console.print(
                    f"[yellow]Found {len(near_indices)} atoms within {radius:.2f} Å of {coord_type} {coords}[/yellow]"
                )

            except ValueError:
                console.print("[red]Error: Invalid coordinates format[/red]")
                raise click.Abort()

        if not sites_to_remove:
            console.print("[yellow]Warning: No atoms match removal criteria[/yellow]")
            raise click.Abort()

        console.print(
            f"\n[yellow]Total atoms to remove: {len(sites_to_remove)}[/yellow]"
        )

        # Show what will be removed
        if show_removed:
            _display_removed_sites(structure, sorted(sites_to_remove))

        # Remove sites
        sites_to_keep = [
            i for i in range(structure.num_sites) if i not in sites_to_remove
        ]
        new_structure = Structure.from_sites([structure[i] for i in sites_to_keep])

        # Display comparison
        _display_removal_info(structure, new_structure, len(sites_to_remove))

        # Determine output filename
        if output is None:
            from pathlib import Path

            input_path = Path(structure_file)
            if element:
                output = f"removed_{element}_{input_path.stem}.{format}"
            else:
                output = (
                    f"removed_{len(sites_to_remove)}atoms_{input_path.stem}.{format}"
                )

        # Save structure
        _save_structure(new_structure, output, format)
        console.print(
            f"\n[green]✓ Structure with removed atoms saved to: {output}[/green]"
        )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()


def _save_structure(structure, filename, format):
    """Save structure to file."""
    if format == "cif":
        from atomate2.siesta.sets.utils.structure_io import write_cif_with_ghost

        write_cif_with_ghost(structure, filename)
    elif format == "poscar":
        structure.to(filename=filename, fmt="poscar")
    elif format == "xsf":
        from pymatgen.io.xcrysden import XSF

        xsf = XSF(structure)
        with open(filename, "w") as f:
            f.write(xsf.to_str())
    elif format == "json":
        structure.to(filename=filename, fmt="json")
    elif format == "fdf":
        import sisl

        geom = sisl.get_sile(structure).read_geometry()
        with sisl.get_sile(filename, "w") as fdf:
            fdf.write_geometry(geom)
    elif format == "XV":
        import sisl

        geom = sisl.get_sile(structure).read_geometry()
        geom.write(filename)


def _display_removal_info(original, new_structure, n_removed):
    """Display removal information."""
    console.print("\n[yellow]Removal Summary:[/yellow]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Original", style="green")
    table.add_column("After Removal", style="yellow")

    table.add_row(
        "Formula", original.composition.formula, new_structure.composition.formula
    )
    table.add_row("Sites", str(original.num_sites), str(new_structure.num_sites))
    table.add_row(
        "Volume (Å³)",
        f"{original.volume:.3f}",
        f"{new_structure.volume:.3f}",
    )
    table.add_row("—", "—", "—")
    table.add_row("Atoms removed", "—", str(n_removed))
    table.add_row(
        "Remaining",
        "—",
        f"{new_structure.num_sites} ({100*new_structure.num_sites/original.num_sites:.1f}%)",
    )

    console.print(table)


def _display_removed_sites(structure, removed_indices):
    """Display removed sites."""
    console.print("\n[cyan]Atoms to be Removed:[/cyan]")

    if not removed_indices:
        console.print("  None")
        return

    # Show up to 20 sites
    n_show = min(20, len(removed_indices))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Site Index", style="cyan")
    table.add_column("Element", style="green")
    table.add_column("Position (Å)", style="yellow")

    for i in range(n_show):
        idx = removed_indices[i]
        site = structure[idx]
        pos = f"({site.coords[0]:.3f}, {site.coords[1]:.3f}, {site.coords[2]:.3f})"
        table.add_row(str(idx), site.specie.symbol, pos)

    if len(removed_indices) > n_show:
        table.add_row(
            "...",
            "...",
            f"... and {len(removed_indices) - n_show} more",
        )

    console.print(table)


if __name__ == "__main__":
    remove()
