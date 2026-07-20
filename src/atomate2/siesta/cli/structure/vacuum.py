#!/usr/bin/env python3
"""CLI for adding/removing vacuum in structures.

This module provides the `vacuum` subcommand for atomate2siesta-structure.
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
    "--thickness",
    type=float,
    required=True,
    help="Vacuum thickness in Å",
)
@click.option(
    "--direction",
    type=click.Choice(["a", "b", "c"]),
    default="c",
    help="Direction to add vacuum (default: c)",
)
@click.option(
    "--center",
    is_flag=True,
    help="Center structure in the vacuum (recommended for slabs)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file (default: vacuum_<input>)",
)
@click.option(
    "--format",
    type=click.Choice(["cif", "poscar", "xsf", "json", "fdf", "XV"]),
    default="cif",
    help="Output format (default: cif)",
)
@click.option(
    "--show-layers",
    is_flag=True,
    help="Show atomic layer positions before/after",
)
def vacuum(
    structure_file,
    thickness,
    direction,
    center,
    output,
    format,
    show_layers,
):
    """Add or adjust vacuum in structures (slabs, 2D materials, molecules).

    Useful for preparing surface slabs, 2D materials, and isolated molecules
    by adding vacuum space in the specified direction.

    Examples
    --------
        # Add 15 Å vacuum in c direction
        atomate2siesta-structure vacuum slab.cif --thickness 15

        # Add vacuum and center slab
        atomate2siesta-structure vacuum slab.cif --thickness 20 --center

        # Add vacuum in b direction (for layered structures)
        atomate2siesta-structure vacuum structure.cif --thickness 10 --direction b

        # Show layer positions
        atomate2siesta-structure vacuum slab.cif --thickness 15 --center --show-layers
    """
    try:
        # Load structure
        structure = Structure.from_file(structure_file)
        console.print(
            f"\n[cyan]Loaded structure: {structure.composition.reduced_formula}[/cyan]"
        )
        console.print(f"  Formula: {structure.composition.formula}")
        console.print(f"  Sites: {structure.num_sites}")

        # Get direction index
        dir_map = {"a": 0, "b": 1, "c": 2}
        dir_idx = dir_map[direction]
        dir_label = ["a", "b", "c"][dir_idx]

        # Calculate current cell length for vacuum addition
        current_cell_length = structure.lattice.abc[dir_idx]

        # Calculate new cell length (add vacuum to current cell)
        new_cell_length = current_cell_length + thickness

        console.print(
            f"\n[yellow]Adding {thickness:.3f} Å vacuum in {dir_label} direction[/yellow]"
        )

        if show_layers:
            _display_layer_positions(structure, dir_idx, "Before")

        # Create new structure with vacuum
        # We need to preserve cartesian coordinates while changing lattice
        from pymatgen.core import Lattice

        # Get original cartesian coordinates
        cart_coords = structure.cart_coords.copy()
        species = [site.species for site in structure]

        # Get lattice matrix and modify the specified direction
        lattice_matrix = structure.lattice.matrix.copy()
        scale_factor = new_cell_length / structure.lattice.abc[dir_idx]
        lattice_matrix[dir_idx] *= scale_factor

        # Create new lattice
        new_lattice = Lattice(lattice_matrix)

        # Create new structure with preserved cartesian coordinates
        vacuum_structure = Structure(
            new_lattice,
            species,
            cart_coords,
            coords_are_cartesian=True,
            site_properties=structure.site_properties,
        )

        # If centering, shift atoms to center
        if center:
            # Calculate shift needed to center in the new cell
            dir_coords_new = vacuum_structure.cart_coords[:, dir_idx]
            current_min = dir_coords_new.min()
            current_max = dir_coords_new.max()
            current_center = (current_min + current_max) / 2

            new_cell_length_actual = new_lattice.abc[dir_idx]
            target_center = new_cell_length_actual / 2

            shift = target_center - current_center

            # Create shift vector
            shift_vector = [0.0, 0.0, 0.0]
            shift_vector[dir_idx] = shift

            # Translate all sites
            vacuum_structure.translate_sites(
                indices=range(len(vacuum_structure)),
                vector=shift_vector,
                frac_coords=False,
            )

        if show_layers:
            _display_layer_positions(vacuum_structure, dir_idx, "After")

        # Display comparison
        _display_vacuum_info(
            structure, vacuum_structure, dir_idx, dir_label, thickness, center
        )

        # Save structure
        if output is None:
            from pathlib import Path

            input_path = Path(structure_file)
            output = f"vacuum_{input_path.name}"

        if format == "cif":
            from atomate2.siesta.sets.utils.structure_io import write_cif_with_ghost

            write_cif_with_ghost(vacuum_structure, output)
        elif format == "poscar":
            vacuum_structure.to(filename=output, fmt="poscar")
        elif format == "xsf":
            from pymatgen.io.xcrysden import XSF

            xsf = XSF(vacuum_structure)
            with open(output, "w") as f:
                f.write(xsf.to_str())
        elif format == "json":
            vacuum_structure.to(filename=output, fmt="json")
        elif format == "fdf":
            # Convert to sisl geometry and write FDF
            import sisl

            geom = sisl.get_sile(vacuum_structure).read_geometry()
            with sisl.get_sile(output, "w") as fdf:
                fdf.write_geometry(geom)
        elif format == "XV":
            # Convert to sisl geometry and write XV
            import sisl

            geom = sisl.get_sile(vacuum_structure).read_geometry()
            geom.write(output)

        console.print(f"\n[green]✓ Structure with vacuum saved to: {output}[/green]")

        # Usage tips
        if not center:
            console.print(
                "\n[dim]Tip: Use --center to center the slab in the vacuum space[/dim]"
            )
        if not show_layers:
            console.print(
                "[dim]Tip: Use --show-layers to see atomic positions before/after[/dim]"
            )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()


def _display_vacuum_info(
    orig_struct, vacuum_struct, dir_idx, dir_label, vacuum_thickness, centered
):
    """Display comparison of structures before/after vacuum addition."""
    console.print("\n[yellow]Structure Comparison:[/yellow]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Original", style="green")
    table.add_column("With Vacuum", style="yellow")

    # Lattice parameters
    orig_abc = orig_struct.lattice.abc
    vac_abc = vacuum_struct.lattice.abc

    for i, label in enumerate(["a (Å)", "b (Å)", "c (Å)"]):
        orig = orig_abc[i]
        vac = vac_abc[i]
        if i == dir_idx:
            table.add_row(label, f"{orig:.4f}", f"{vac:.4f} (+{vac - orig:.3f})")
        else:
            table.add_row(label, f"{orig:.4f}", f"{vac:.4f}")

    # Volume
    table.add_row("—", "—", "—")
    table.add_row(
        "Volume (Å³)",
        f"{orig_struct.volume:.3f}",
        f"{vacuum_struct.volume:.3f}",
    )

    # Calculate actual slab thickness
    orig_coords = orig_struct.cart_coords[:, dir_idx]
    vac_coords = vacuum_struct.cart_coords[:, dir_idx]

    orig_thickness = orig_coords.max() - orig_coords.min()
    vac_thickness = vac_coords.max() - vac_coords.min()

    table.add_row("—", "—", "—")
    table.add_row("Slab thickness", f"{orig_thickness:.3f} Å", f"{vac_thickness:.3f} Å")

    # Vacuum space
    orig_vacuum = orig_abc[dir_idx] - orig_thickness
    vac_vacuum = vac_abc[dir_idx] - vac_thickness

    table.add_row(
        "Vacuum space",
        f"{orig_vacuum:.3f} Å",
        f"{vac_vacuum:.3f} Å (+{vac_vacuum - orig_vacuum:.3f})",
    )

    console.print(table)

    if centered:
        console.print(f"\n[cyan]Structure centered in {dir_label} direction[/cyan]")


def _display_layer_positions(structure, dir_idx, label):
    """Display atomic layer positions."""
    console.print(f"\n[cyan]Layer Positions ({label}):[/cyan]")

    coords = structure.cart_coords
    dir_coords = coords[:, dir_idx]

    # Find unique layers (tolerance 0.5 Å)
    unique_z = []
    for z in sorted(dir_coords):
        if not unique_z or abs(z - unique_z[-1]) > 0.5:
            unique_z.append(z)

    console.print(f"  Found {len(unique_z)} layers")

    # Show first 5 and last 5
    n_show = min(5, len(unique_z))

    for i in range(n_show):
        z = unique_z[i]
        n_atoms = sum(1 for zz in dir_coords if abs(zz - z) < 0.5)
        console.print(f"    Layer {i}: z = {z:.3f} Å ({n_atoms} atoms)")

    if len(unique_z) > 2 * n_show:
        console.print("    ...")

    if len(unique_z) > n_show:
        for i in range(max(n_show, len(unique_z) - n_show), len(unique_z)):
            z = unique_z[i]
            n_atoms = sum(1 for zz in dir_coords if abs(zz - z) < 0.5)
            console.print(f"    Layer {i}: z = {z:.3f} Å ({n_atoms} atoms)")

    # Show range
    console.print(
        f"  Range: {dir_coords.min():.3f} to {dir_coords.max():.3f} Å "
        f"(thickness: {dir_coords.max() - dir_coords.min():.3f} Å)"
    )


if __name__ == "__main__":
    vacuum()
