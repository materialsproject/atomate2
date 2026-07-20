#!/usr/bin/env python3
"""CLI for stacking layers to create heterostructures and multilayers.

This module provides the `stack` subcommand for atomate2siesta-structure.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from pymatgen.core import Structure

console = Console()


@click.command()
@click.argument("structure1", type=click.Path(exists=True))
@click.argument("structure2", type=click.Path(exists=True), required=False)
@click.option(
    "--direction",
    type=click.Choice(["a", "b", "c"]),
    default="c",
    help="Stacking direction (default: c)",
)
@click.option(
    "--spacing",
    type=float,
    default=3.0,
    help="Spacing between layers in Å (default: 3.0)",
)
@click.option(
    "--repetitions",
    type=str,
    help="Repeat pattern as comma-separated integers (e.g., '2,3' for 2x structure1 + 3x structure2)",
)
@click.option(
    "--center",
    is_flag=True,
    default=True,
    help="Center the stack in the cell (default: True)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file (default: stacked_<input1>_<input2>)",
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
    help="Show layer information after stacking",
)
def stack(
    structure1,
    structure2,
    direction,
    spacing,
    repetitions,
    center,
    output,
    format,
    show_layers,
):
    """Stack layers to create heterostructures and multilayers.

    Create vertical stacks of 2D materials, slabs, or molecules with controlled
    spacing. Supports both heterostructures (two different materials) and
    multilayers (repeated structure).

    Examples:

        # Simple bilayer (same material)
        atomate2siesta-structure stack graphene.cif --spacing 3.35

        # Heterostructure (two materials)
        atomate2siesta-structure stack MoS2.cif WS2.cif --spacing 3.0

        # Multilayer with repetition pattern
        atomate2siesta-structure stack layer.cif --repetitions 5

        # Complex heterostructure pattern (2 MoS2 + 3 WS2)
        atomate2siesta-structure stack MoS2.cif WS2.cif --repetitions 2,3

        # Stack along different direction
        atomate2siesta-structure stack slab1.cif slab2.cif --direction a
    """
    try:
        # Load first structure
        struct1 = Structure.from_file(structure1)
        console.print(
            f"\n[cyan]Loaded first structure: {struct1.composition.reduced_formula}[/cyan]"
        )
        console.print(f"  Formula: {struct1.composition.formula}")
        console.print(f"  Sites: {struct1.num_sites}")

        # Determine stacking mode
        if structure2:
            # Heterostructure mode
            struct2 = Structure.from_file(structure2)
            console.print(
                f"\n[cyan]Loaded second structure: {struct2.composition.reduced_formula}[/cyan]"
            )
            console.print(f"  Formula: {struct2.composition.formula}")
            console.print(f"  Sites: {struct2.num_sites}")
            mode = "heterostructure"
        else:
            # Multilayer mode (same structure repeated)
            struct2 = None
            mode = "multilayer"

        # Parse repetitions
        if repetitions:
            reps = [int(x) for x in repetitions.split(",")]
            if mode == "multilayer" and len(reps) > 1:
                console.print(
                    "[yellow]Warning: Only one repetition value needed for multilayer. Using first value.[/yellow]"
                )
                reps = [reps[0]]
            elif mode == "heterostructure" and len(reps) == 1:
                # If only one value given for heterostructure, assume same for both
                reps = [reps[0], reps[0]]
            elif mode == "heterostructure" and len(reps) != 2:
                console.print(
                    "[red]Error: Heterostructure requires 1 or 2 repetition values[/red]"
                )
                raise click.Abort()
        else:
            # Default repetitions
            if mode == "multilayer":
                reps = [2]  # Default bilayer
            else:
                reps = [1, 1]  # One layer of each

        # Get direction index
        dir_map = {"a": 0, "b": 1, "c": 2}
        dir_idx = dir_map[direction]
        dir_label = ["a", "b", "c"][dir_idx]

        console.print(f"\n[yellow]Stacking Mode: {mode.title()}[/yellow]")
        if mode == "multilayer":
            console.print(f"  Repeating structure {reps[0]} times")
        else:
            console.print(f"  Pattern: {reps[0]}x structure1 + {reps[1]}x structure2")
        console.print(f"  Direction: {dir_label}")
        console.print(f"  Spacing: {spacing:.3f} Å")

        # Create stacked structure
        if mode == "multilayer":
            stacked = _stack_multilayer(struct1, reps[0], dir_idx, spacing, center)
        else:
            stacked = _stack_heterostructure(
                struct1, struct2, reps[0], reps[1], dir_idx, spacing, center
            )

        # Display information
        _display_stack_info(struct1, struct2, stacked, dir_idx, dir_label, mode, reps)

        if show_layers:
            _display_layer_info(stacked, dir_idx)

        # Determine output filename
        if output is None:
            from pathlib import Path

            path1 = Path(structure1)
            if structure2:
                path2 = Path(structure2)
                output = f"stacked_{path1.stem}_{path2.stem}.{format}"
            else:
                output = f"stacked_{reps[0]}x_{path1.stem}.{format}"

        # Save structure
        _save_structure(stacked, output, format)
        console.print(f"\n[green]✓ Stacked structure saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()


def _stack_multilayer(structure, n_layers, dir_idx, spacing, center):
    """Stack multiple copies of the same structure."""
    if n_layers < 2:
        console.print("[yellow]Warning: Using at least 2 layers[/yellow]")
        n_layers = 2

    # Get structure thickness in stacking direction
    coords = structure.cart_coords
    dir_coords = coords[:, dir_idx]
    thickness = dir_coords.max() - dir_coords.min()

    # Calculate total cell length needed
    total_thickness = n_layers * thickness + (n_layers - 1) * spacing
    current_cell_length = structure.lattice.abc[dir_idx]

    # Determine new cell length (add extra vacuum if centering)
    if center:
        new_cell_length = total_thickness + 2 * spacing  # Add vacuum on both sides
    else:
        new_cell_length = total_thickness

    # Create new lattice
    from pymatgen.core import Lattice, Structure

    lattice_matrix = structure.lattice.matrix.copy()
    scale_factor = new_cell_length / current_cell_length
    lattice_matrix[dir_idx] *= scale_factor
    new_lattice = Lattice(lattice_matrix)

    # Collect all atoms with shifted positions
    all_species = []
    all_coords = []

    # Calculate base offset (if centering)
    if center:
        base_offset = spacing
    else:
        base_offset = 0.0

    # Stack layers
    for i in range(n_layers):
        layer_offset = base_offset + i * (thickness + spacing)

        for site in structure:
            all_species.append(site.species)
            cart_coord = site.coords.copy()
            cart_coord[dir_idx] += layer_offset
            all_coords.append(cart_coord)

    # Create stacked structure
    stacked = Structure(
        new_lattice,
        all_species,
        all_coords,
        coords_are_cartesian=True,
    )

    return stacked


def _stack_heterostructure(struct1, struct2, n1, n2, dir_idx, spacing, center):
    """Stack two different structures with repetition pattern."""
    # Get thicknesses
    coords1 = struct1.cart_coords
    coords2 = struct2.cart_coords
    thickness1 = coords1[:, dir_idx].max() - coords1[:, dir_idx].min()
    thickness2 = coords2[:, dir_idx].max() - coords2[:, dir_idx].min()

    # Check lattice compatibility in non-stacking directions
    abc1 = list(struct1.lattice.abc)
    abc2 = list(struct2.lattice.abc)
    angles1 = struct1.lattice.angles
    angles2 = struct2.lattice.angles

    # Remove stacking direction for comparison
    abc1_plane = [abc1[i] for i in range(3) if i != dir_idx]
    abc2_plane = [abc2[i] for i in range(3) if i != dir_idx]
    angles1_plane = [angles1[i] for i in range(3) if i != dir_idx]
    angles2_plane = [angles2[i] for i in range(3) if i != dir_idx]

    # Check if lattices are compatible (within 5%)
    lattice_mismatch = False
    for a1, a2 in zip(abc1_plane, abc2_plane):
        if abs(a1 - a2) / max(a1, a2) > 0.05:
            lattice_mismatch = True
    for ang1, ang2 in zip(angles1_plane, angles2_plane):
        if abs(ang1 - ang2) > 5.0:
            lattice_mismatch = True

    if lattice_mismatch:
        console.print(
            "[yellow]Warning: In-plane lattice parameters differ by >5%. "
            "Consider using supercell matching first.[/yellow]"
        )

    # Use first structure's lattice as base
    total_thickness = n1 * thickness1 + n2 * thickness2 + (n1 + n2 - 1) * spacing

    if center:
        new_cell_length = total_thickness + 2 * spacing
        base_offset = spacing
    else:
        new_cell_length = total_thickness
        base_offset = 0.0

    # Create new lattice
    from pymatgen.core import Lattice, Structure

    lattice_matrix = struct1.lattice.matrix.copy()
    scale_factor = new_cell_length / struct1.lattice.abc[dir_idx]
    lattice_matrix[dir_idx] *= scale_factor
    new_lattice = Lattice(lattice_matrix)

    # Collect all atoms
    all_species = []
    all_coords = []
    current_offset = base_offset

    # Add layers from struct1
    for i in range(n1):
        for site in struct1:
            all_species.append(site.species)
            cart_coord = site.coords.copy()
            cart_coord[dir_idx] += current_offset
            all_coords.append(cart_coord)
        current_offset += thickness1 + spacing

    # Subtract last spacing, add layers from struct2
    current_offset -= spacing

    for i in range(n2):
        for site in struct2:
            all_species.append(site.species)
            cart_coord = site.coords.copy()
            cart_coord[dir_idx] += current_offset
            all_coords.append(cart_coord)
        current_offset += thickness2 + spacing

    # Create stacked structure
    stacked = Structure(
        new_lattice,
        all_species,
        all_coords,
        coords_are_cartesian=True,
    )

    return stacked


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
        # Convert to sisl geometry and write FDF
        import sisl

        geom = sisl.get_sile(structure).read_geometry()
        with sisl.get_sile(filename, "w") as fdf:
            fdf.write_geometry(geom)
    elif format == "XV":
        # Convert to sisl geometry and write XV
        import sisl

        geom = sisl.get_sile(structure).read_geometry()
        geom.write(filename)


def _display_stack_info(struct1, struct2, stacked, dir_idx, dir_label, mode, reps):
    """Display information about the stacked structure."""
    console.print("\n[yellow]Stacking Summary:[/yellow]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    if mode == "multilayer":
        table.add_column("Original", style="green")
        table.add_column("Stacked", style="yellow")
    else:
        table.add_column("Structure 1", style="green")
        table.add_column("Structure 2", style="blue")
        table.add_column("Stacked", style="yellow")

    # Formulas
    if mode == "multilayer":
        table.add_row(
            "Formula", struct1.composition.formula, stacked.composition.formula
        )
        table.add_row("Sites", str(struct1.num_sites), str(stacked.num_sites))
    else:
        table.add_row(
            "Formula",
            struct1.composition.formula,
            struct2.composition.formula,
            stacked.composition.formula,
        )
        table.add_row(
            "Sites",
            str(struct1.num_sites),
            str(struct2.num_sites),
            str(stacked.num_sites),
        )

    # Lattice parameters
    table.add_row("—", "—", "—", "—" if mode == "heterostructure" else "")

    abc1 = struct1.lattice.abc
    abc_stack = stacked.lattice.abc

    for i, label in enumerate(["a (Å)", "b (Å)", "c (Å)"]):
        if mode == "multilayer":
            if i == dir_idx:
                table.add_row(label, f"{abc1[i]:.4f}", f"{abc_stack[i]:.4f}")
            else:
                table.add_row(label, f"{abc1[i]:.4f}", f"{abc_stack[i]:.4f}")
        else:
            abc2 = struct2.lattice.abc
            if i == dir_idx:
                table.add_row(
                    label, f"{abc1[i]:.4f}", f"{abc2[i]:.4f}", f"{abc_stack[i]:.4f}"
                )
            else:
                table.add_row(
                    label, f"{abc1[i]:.4f}", f"{abc2[i]:.4f}", f"{abc_stack[i]:.4f}"
                )

    console.print(table)

    # Layer count
    if mode == "multilayer":
        console.print(f"\n[cyan]Total layers: {reps[0]}[/cyan]")
    else:
        console.print(
            f"\n[cyan]Total layers: {reps[0]} (structure 1) + {reps[1]} (structure 2) = {reps[0] + reps[1]}[/cyan]"
        )


def _display_layer_info(structure, dir_idx):
    """Display layer-by-layer information."""
    console.print("\n[cyan]Layer Information:[/cyan]")

    coords = structure.cart_coords
    z_coords = coords[:, dir_idx]

    # Find unique layers (tolerance 0.5 Å)
    unique_z = []
    for z in sorted(z_coords):
        if not unique_z or abs(z - unique_z[-1]) > 0.5:
            unique_z.append(z)

    console.print(f"  Detected {len(unique_z)} distinct layers")

    # Display first 5 and last 5 layers
    n_show = min(5, len(unique_z))
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Layer", style="cyan")
    table.add_column("Position (Å)", style="green")
    table.add_column("Atoms", style="yellow")
    table.add_column("Species", style="blue")

    for i in range(n_show):
        z = unique_z[i]
        layer_indices = [j for j, zz in enumerate(z_coords) if abs(zz - z) < 0.5]
        layer_sites = [structure[j] for j in layer_indices]
        species = ", ".join(sorted(set(str(s.specie) for s in layer_sites)))

        table.add_row(str(i), f"{z:.3f}", str(len(layer_indices)), species)

    if len(unique_z) > 2 * n_show:
        table.add_row("...", "...", "...", "...")

    if len(unique_z) > n_show:
        for i in range(max(n_show, len(unique_z) - n_show), len(unique_z)):
            z = unique_z[i]
            layer_indices = [j for j, zz in enumerate(z_coords) if abs(zz - z) < 0.5]
            layer_sites = [structure[j] for j in layer_indices]
            species = ", ".join(sorted(set(str(s.specie) for s in layer_sites)))

            table.add_row(str(i), f"{z:.3f}", str(len(layer_indices)), species)

    console.print(table)

    # Calculate interlayer spacings
    if len(unique_z) > 1:
        spacings = [unique_z[i + 1] - unique_z[i] for i in range(len(unique_z) - 1)]
        avg_spacing = sum(spacings) / len(spacings)
        min_spacing = min(spacings)
        max_spacing = max(spacings)

        console.print("\n[cyan]Interlayer Spacing:[/cyan]")
        console.print(f"  Average: {avg_spacing:.3f} Å")
        console.print(f"  Range: {min_spacing:.3f} - {max_spacing:.3f} Å")


if __name__ == "__main__":
    stack()
