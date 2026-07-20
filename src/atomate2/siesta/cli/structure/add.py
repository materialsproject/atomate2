#!/usr/bin/env python3
"""CLI for adding atoms/molecules to structures.

This module provides the `add` subcommand for atomate2siesta-structure.
"""

from __future__ import annotations

import click
import numpy as np
from pymatgen.core import Element, Molecule, Structure
from rich.console import Console
from rich.table import Table

console = Console()

# Simple molecule library
MOLECULE_LIBRARY = {
    "H2": {
        "species": ["H", "H"],
        "coords": [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
    },
    "O2": {
        "species": ["O", "O"],
        "coords": [[0.0, 0.0, 0.0], [1.21, 0.0, 0.0]],
    },
    "H2O": {
        "species": ["O", "H", "H"],
        "coords": [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    },
    "CO": {
        "species": ["C", "O"],
        "coords": [[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]],
    },
    "CO2": {
        "species": ["O", "C", "O"],
        "coords": [[-1.16, 0.0, 0.0], [0.0, 0.0, 0.0], [1.16, 0.0, 0.0]],
    },
    "NH3": {
        "species": ["N", "H", "H", "H"],
        "coords": [
            [0.0, 0.0, 0.0],
            [1.01, 0.0, 0.0],
            [-0.34, 0.95, 0.0],
            [-0.34, -0.48, 0.82],
        ],
    },
    "CH4": {
        "species": ["C", "H", "H", "H", "H"],
        "coords": [
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [-0.36, 1.03, 0.0],
            [-0.36, -0.51, 0.89],
            [-0.36, -0.51, -0.89],
        ],
    },
}


@click.command()
@click.argument("structure_file", type=click.Path(exists=True))
@click.option(
    "--element",
    type=str,
    help="Add single atom of this element",
)
@click.option(
    "--molecule",
    type=str,
    help=f"Add molecule from library: {', '.join(MOLECULE_LIBRARY.keys())}",
)
@click.option(
    "--molecule-file",
    type=click.Path(exists=True),
    help="Add molecule from XYZ file",
)
@click.option(
    "--position",
    type=str,
    help="Position to add atom/molecule (x,y,z in Cartesian Å or fractional)",
)
@click.option(
    "--on-top",
    is_flag=True,
    help="Place molecule on top of structure (z_max + distance)",
)
@click.option(
    "--on-bottom",
    is_flag=True,
    help="Place molecule on bottom of structure (z_min - distance)",
)
@click.option(
    "--distance",
    type=float,
    default=2.5,
    help="Distance from surface for --on-top/--on-bottom (default: 2.5 Å)",
)
@click.option(
    "--fractional",
    is_flag=True,
    help="Treat --position as fractional coordinates",
)
@click.option(
    "--rotate",
    type=str,
    help="Rotate molecule: euler angles as 'alpha,beta,gamma' in degrees",
)
@click.option(
    "--align-to",
    type=click.Choice(["x", "y", "z"]),
    help="Align molecule's principal axis to this direction",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file (default: added_<input>)",
)
@click.option(
    "--format",
    type=click.Choice(["cif", "poscar", "xsf", "json", "fdf", "XV"]),
    default="cif",
    help="Output format (default: cif)",
)
@click.option(
    "--show-added",
    is_flag=True,
    help="Show which atoms were added",
)
def add(
    structure_file,
    element,
    molecule,
    molecule_file,
    position,
    on_top,
    on_bottom,
    distance,
    fractional,
    rotate,
    align_to,
    output,
    format,
    show_added,
):
    """Add atoms or molecules to a structure (adsorbates, dopants, interstitials).

    Add single atoms or molecules at specified positions. Supports a library of
    common molecules or loading from XYZ files. Can orient molecules and place
    them automatically on top/bottom of surfaces.

    Examples
    --------
        # Add oxygen atom at Cartesian position
        atomate2siesta-structure add surface.cif --element O --position 0,0,5.0

        # Add H2O molecule from library
        atomate2siesta-structure add surface.cif --molecule H2O --position 0,0,2.5

        # Add molecule on top of slab (2.5 Å above)
        atomate2siesta-structure add slab.cif --molecule CO --on-top

        # Add on bottom with custom distance
        atomate2siesta-structure add slab.cif --molecule H2O --on-bottom --distance 3.0

        # Add molecule at fractional coordinate
        atomate2siesta-structure add surface.cif --molecule CO --position 0.5,0.5,0.9 --fractional

        # Add with rotation (Euler angles)
        atomate2siesta-structure add surface.cif --molecule H2O --on-top --rotate 0,45,90

        # Align molecule to z-axis
        atomate2siesta-structure add surface.cif --molecule CO2 --on-top --align-to z

        # Add molecule from XYZ file
        atomate2siesta-structure add surface.cif --molecule-file benzene.xyz --position 0,0,3.0
    """
    try:
        # Validate options
        if not any([element, molecule, molecule_file]):
            console.print(
                "[red]Error: Must specify one of --element, --molecule, or --molecule-file[/red]"
            )
            raise click.Abort()

        if sum([bool(element), bool(molecule), bool(molecule_file)]) > 1:
            console.print(
                "[red]Error: Can only use one of --element, --molecule, or --molecule-file[/red]"
            )
            raise click.Abort()

        if sum([bool(position), on_top, on_bottom]) != 1:
            console.print(
                "[red]Error: Must specify exactly one of --position, --on-top, or --on-bottom[/red]"
            )
            raise click.Abort()

        if on_top and on_bottom:
            console.print("[red]Error: Cannot use both --on-top and --on-bottom[/red]")
            raise click.Abort()

        # Load structure
        structure = Structure.from_file(structure_file)
        console.print(
            f"\n[cyan]Loaded structure: {structure.composition.reduced_formula}[/cyan]"
        )
        console.print(f"  Formula: {structure.composition.formula}")
        console.print(f"  Sites: {structure.num_sites}")

        # Determine what to add
        new_species = []
        new_coords = []

        if element:
            # Single atom
            try:
                elem = Element(element)
                new_species = [elem.symbol]
                new_coords = [[0.0, 0.0, 0.0]]  # Will be positioned later
                add_type = f"atom ({elem.symbol})"
            except Exception as e:
                console.print(f"[red]Error: Invalid element symbol: {e}[/red]")
                raise click.Abort()

        elif molecule:
            # Molecule from library
            if molecule not in MOLECULE_LIBRARY:
                console.print(f"[red]Error: Molecule '{molecule}' not in library[/red]")
                console.print(
                    f"Available molecules: {', '.join(MOLECULE_LIBRARY.keys())}"
                )
                raise click.Abort()

            mol_data = MOLECULE_LIBRARY[molecule]
            new_species = mol_data["species"]
            new_coords = np.array(mol_data["coords"])
            add_type = f"molecule ({molecule}, {len(new_species)} atoms)"

        elif molecule_file:
            # Molecule from file
            try:
                mol = Molecule.from_file(molecule_file)
                new_species = [str(site.specie) for site in mol]
                new_coords = np.array([site.coords for site in mol])
                add_type = f"molecule from file ({len(new_species)} atoms)"
            except Exception as e:
                console.print(f"[red]Error loading molecule file: {e}[/red]")
                raise click.Abort()

        # Apply rotation/alignment if molecule
        if len(new_species) > 1:
            new_coords = _apply_orientation(new_coords, rotate, align_to, new_species)

        # Determine position
        if on_top or on_bottom:
            # Automatic positioning on top/bottom
            z_coords = structure.cart_coords[:, 2]
            if on_top:
                z_position = z_coords.max() + distance
                placement = f"on top (z_max + {distance:.2f} Å)"
            else:  # on_bottom
                z_position = z_coords.min() - distance
                placement = f"on bottom (z_min - {distance:.2f} Å)"

            # Center in x-y plane
            cart_coords = structure.cart_coords
            x_center = (cart_coords[:, 0].max() + cart_coords[:, 0].min()) / 2
            y_center = (cart_coords[:, 1].max() + cart_coords[:, 1].min()) / 2

            cart_position = np.array([x_center, y_center, z_position])
            console.print(f"\n[yellow]Placing {placement}[/yellow]")
            console.print(
                f"  Position: ({cart_position[0]:.3f}, {cart_position[1]:.3f}, {cart_position[2]:.3f}) Å"
            )

        else:
            # User-specified position
            try:
                coords = [float(x) for x in position.split(",")]
                if len(coords) != 3:
                    console.print(
                        "[red]Error: --position must be x,y,z (3 values)[/red]"
                    )
                    raise click.Abort()
            except ValueError:
                console.print("[red]Error: Invalid position format[/red]")
                raise click.Abort()

            if fractional:
                cart_position = structure.lattice.get_cartesian_coords(coords)
                console.print(
                    f"\n[yellow]Adding at fractional position: {coords}[/yellow]"
                )
                console.print(
                    f"  Cartesian: ({cart_position[0]:.3f}, {cart_position[1]:.3f}, {cart_position[2]:.3f}) Å"
                )
            else:
                cart_position = np.array(coords)
                console.print(
                    f"\n[yellow]Adding at Cartesian position: ({coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}) Å[/yellow]"
                )

        console.print(f"[yellow]Adding {add_type}[/yellow]")

        # Center molecule at specified position
        mol_center = np.array(new_coords).mean(axis=0)
        final_coords = [cart_position + (c - mol_center) for c in new_coords]

        # Create new structure with added atoms
        new_structure = structure.copy()
        for species, coord in zip(new_species, final_coords):
            new_structure.append(species, coord, coords_are_cartesian=True)

        # Display what was added
        if show_added:
            _display_added_atoms(new_species, final_coords)

        # Display comparison
        _display_addition_info(structure, new_structure, len(new_species), add_type)

        # Determine output filename
        if output is None:
            from pathlib import Path

            input_path = Path(structure_file)
            if element:
                output = f"added_{element}_{input_path.stem}.{format}"
            elif molecule:
                output = f"added_{molecule}_{input_path.stem}.{format}"
            else:
                output = f"added_molecule_{input_path.stem}.{format}"

        # Save structure
        _save_structure(new_structure, output, format)
        console.print(
            f"\n[green]✓ Structure with added atoms saved to: {output}[/green]"
        )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()


def _apply_orientation(coords, rotate, align_to, species):
    """Apply rotation/alignment to molecule coordinates."""
    coords = np.array(coords)

    # Align to axis if requested
    if align_to:
        # Calculate principal axis (line from first to last atom)
        if len(coords) > 1:
            principal_axis = coords[-1] - coords[0]
            principal_axis = principal_axis / np.linalg.norm(principal_axis)

            # Target axis
            target = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[align_to]

            # Rotation axis and angle
            rotation_axis = np.cross(principal_axis, target)
            if np.linalg.norm(rotation_axis) > 1e-6:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.clip(np.dot(principal_axis, target), -1.0, 1.0))

                # Apply rotation
                coords = _rotate_coords(coords, rotation_axis, angle)
                console.print(
                    f"[dim]Aligned molecule to {align_to}-axis ({np.degrees(angle):.1f}° rotation)[/dim]"
                )

    # Apply Euler rotation if requested
    if rotate:
        try:
            angles = [float(x) for x in rotate.split(",")]
            if len(angles) != 3:
                console.print(
                    "[red]Error: --rotate must be alpha,beta,gamma (3 angles)[/red]"
                )
                raise click.Abort()

            alpha, beta, gamma = np.radians(angles)

            # Rotation matrices (ZYZ Euler convention)
            Rz1 = np.array(
                [
                    [np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha), np.cos(alpha), 0],
                    [0, 0, 1],
                ]
            )
            Ry = np.array(
                [
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)],
                ]
            )
            Rz2 = np.array(
                [
                    [np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1],
                ]
            )

            R = Rz2 @ Ry @ Rz1
            center = coords.mean(axis=0)
            coords = (coords - center) @ R.T + center

            console.print(
                f"[dim]Rotated molecule (α={angles[0]:.1f}°, β={angles[1]:.1f}°, γ={angles[2]:.1f}°)[/dim]"
            )

        except ValueError:
            console.print("[red]Error: Invalid rotation angles format[/red]")
            raise click.Abort()

    return coords


def _rotate_coords(coords, axis, angle):
    """Rotate coordinates around axis by angle using Rodrigues' formula."""
    axis = np.array(axis) / np.linalg.norm(axis)
    center = coords.mean(axis=0)
    coords_centered = coords - center

    # Rodrigues' rotation formula
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    rotated = (
        coords_centered * cos_angle
        + np.cross(axis, coords_centered) * sin_angle
        + axis[:, np.newaxis].T * (axis @ coords_centered.T) * (1 - cos_angle)
    )

    return rotated + center


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


def _display_addition_info(original, new_structure, n_added, add_type):
    """Display addition information."""
    console.print("\n[yellow]Addition Summary:[/yellow]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Original", style="green")
    table.add_column("After Addition", style="yellow")

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
    table.add_row("Atoms added", "—", f"{n_added} ({add_type})")
    table.add_row("Total atoms", "—", str(new_structure.num_sites))

    console.print(table)


def _display_added_atoms(species, coords):
    """Display added atoms."""
    console.print("\n[cyan]Added Atoms:[/cyan]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Element", style="cyan")
    table.add_column("Position (Å)", style="green")

    for sp, coord in zip(species, coords):
        pos = f"({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f})"
        table.add_row(sp, pos)

    console.print(table)


if __name__ == "__main__":
    add()
