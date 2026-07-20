#!/usr/bin/env python3
"""CLI for rotating structures.

This module provides the `rotate` subcommand for atomate2siesta-structure.
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
    "--axis",
    type=click.Choice(["x", "y", "z"]),
    help="Rotation axis (use with --angle)",
)
@click.option(
    "--angle",
    type=float,
    help="Rotation angle in degrees (counterclockwise, right-hand rule)",
)
@click.option(
    "--align-to-x",
    type=str,
    help="Align crystallographic direction [h,k,l] to x-axis (e.g., '1,1,0')",
)
@click.option(
    "--align-to-y",
    type=str,
    help="Align crystallographic direction [h,k,l] to y-axis",
)
@click.option(
    "--align-to-z",
    type=str,
    help="Align crystallographic direction [h,k,l] to z-axis",
)
@click.option(
    "--euler",
    type=float,
    nargs=3,
    help="Euler angles (α, β, γ) in degrees (ZXZ convention)",
)
@click.option(
    "--rotate-cell",
    is_flag=True,
    default=True,
    help="Rotate both cell and atoms (default: True)",
)
@click.option(
    "--rotate-atoms-only",
    is_flag=True,
    help="Rotate only atomic positions, keep cell fixed",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file (default: rotated_<input>)",
)
@click.option(
    "--format",
    type=click.Choice(["cif", "poscar", "xsf", "json"]),
    default="cif",
    help="Output format (default: cif)",
)
@click.option(
    "--show-angles",
    is_flag=True,
    help="Show angles between lattice vectors before/after",
)
def rotate(
    structure_file,
    axis,
    angle,
    align_to_x,
    align_to_y,
    align_to_z,
    euler,
    rotate_cell,
    rotate_atoms_only,
    output,
    format,
    show_angles,
):
    """Rotate structures for alignment, surface cuts, or reorientation.

    Supports axis-angle rotation, Euler angles, and alignment of crystallographic
    directions to Cartesian axes. Can rotate entire cell or just atoms.

    Examples
    --------
        # Rotate 45° about z-axis
        atomate2siesta-structure rotate Si.cif --axis z --angle 45

        # Align [111] direction to z-axis
        atomate2siesta-structure rotate Si.cif --align-to-z 1,1,1

        # Euler angle rotation (ZXZ convention)
        atomate2siesta-structure rotate Si.cif --euler 30 45 60

        # Rotate atoms only, keep cell fixed
        atomate2siesta-structure rotate Si.cif --axis z --angle 30 --rotate-atoms-only

        # With angle display
        atomate2siesta-structure rotate Si.cif --axis x --angle 90 --show-angles
    """
    # Validate options
    methods = [
        (axis is not None and angle is not None),
        align_to_x is not None,
        align_to_y is not None,
        align_to_z is not None,
        euler is not None,
    ]
    if sum(methods) == 0:
        console.print(
            "[red]Error: Must specify one rotation method:\n"
            "  --axis/--angle, --align-to-x/y/z, or --euler[/red]"
        )
        raise click.Abort()
    if sum(methods) > 1:
        console.print("[red]Error: Only specify one rotation method[/red]")
        raise click.Abort()

    if rotate_atoms_only and not rotate_cell:
        console.print(
            "[yellow]Warning: Both --rotate-atoms-only and --rotate-cell=False specified. "
            "Using --rotate-atoms-only.[/yellow]"
        )

    try:
        # Load structure
        structure = Structure.from_file(structure_file)
        console.print(
            f"\n[cyan]Loaded structure: {structure.composition.reduced_formula}[/cyan]"
        )
        console.print(f"  Formula: {structure.composition.formula}")
        console.print(f"  Sites: {structure.num_sites}")
        console.print(f"  Volume: {structure.volume:.3f} Å³")

        # Store original lattice for comparison
        orig_params = structure.lattice.parameters

        # Perform rotation
        rotated_structure = structure.copy()
        rotation_description = ""

        if axis is not None and angle is not None:
            # Axis-angle rotation
            import numpy as np
            from pymatgen.core.operations import SymmOp
            from scipy.spatial.transform import Rotation as R

            axis_map = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
            rotation_axis = axis_map[axis]

            # Create rotation matrix
            rot = R.from_rotvec(np.radians(angle) * np.array(rotation_axis))
            rotation_matrix = rot.as_matrix()

            if rotate_atoms_only:
                # Rotate atoms only

                new_coords = []
                for site in rotated_structure:
                    cart_coords = site.coords
                    rotated_coords = rotation_matrix @ cart_coords
                    new_coords.append(rotated_coords)

                # Create new structure with rotated atoms
                rotated_structure = Structure(
                    lattice=rotated_structure.lattice,
                    species=[site.specie for site in rotated_structure],
                    coords=new_coords,
                    coords_are_cartesian=True,
                    site_properties=rotated_structure.site_properties,
                )
                rotation_description = (
                    f"Axis-angle rotation: {angle:.1f}° about {axis}-axis (atoms only)"
                )
            else:
                # Rotate entire cell
                symm_op = SymmOp.from_rotation_and_translation(rotation_matrix)
                rotated_structure.apply_operation(symm_op)
                rotation_description = (
                    f"Axis-angle rotation: {angle:.1f}° about {axis}-axis"
                )

        elif align_to_x or align_to_y or align_to_z:
            # Alignment rotation
            if align_to_x:
                hkl = [float(x) for x in align_to_x.split(",")]
                target_axis = [1, 0, 0]
                target_name = "x"
            elif align_to_y:
                hkl = [float(x) for x in align_to_y.split(",")]
                target_axis = [0, 1, 0]
                target_name = "y"
            else:  # align_to_z
                hkl = [float(x) for x in align_to_z.split(",")]
                target_axis = [0, 0, 1]
                target_name = "z"

            import numpy as np
            from scipy.spatial.transform import Rotation as R

            # Convert Miller indices to Cartesian
            lattice = rotated_structure.lattice
            direction_cart = lattice.get_cartesian_coords(hkl)
            direction_cart /= np.linalg.norm(direction_cart)

            # Calculate rotation to align direction with target axis
            target_axis = np.array(target_axis)
            v = np.cross(direction_cart, target_axis)
            s = np.linalg.norm(v)
            c = np.dot(direction_cart, target_axis)

            if s < 1e-6:  # Already aligned or opposite
                if c < 0:  # Opposite direction - rotate 180°
                    # Find perpendicular axis
                    if abs(target_axis[2]) < 0.9:
                        perp = np.cross(target_axis, [0, 0, 1])
                    else:
                        perp = np.cross(target_axis, [1, 0, 0])
                    perp /= np.linalg.norm(perp)
                    rot = R.from_rotvec(np.pi * perp)
                    rotation_matrix = rot.as_matrix()
                else:  # Already aligned
                    rotation_matrix = np.eye(3)
            else:
                # General rotation
                v_skew = np.array(
                    [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
                )
                rotation_matrix = (
                    np.eye(3) + v_skew + v_skew @ v_skew * ((1 - c) / s**2)
                )

            from pymatgen.core.operations import SymmOp

            symm_op = SymmOp.from_rotation_and_translation(rotation_matrix)
            rotated_structure.apply_operation(symm_op)
            rotation_description = f"Alignment: [{','.join(str(int(h)) for h in hkl)}] → {target_name}-axis"

        elif euler is not None:
            # Euler angle rotation (ZXZ convention)
            import numpy as np
            from pymatgen.core.operations import SymmOp
            from scipy.spatial.transform import Rotation as R

            alpha, beta, gamma = euler
            rot = R.from_euler("ZXZ", [alpha, beta, gamma], degrees=True)
            rotation_matrix = rot.as_matrix()

            if rotate_atoms_only:
                # Rotate atoms only

                new_coords = []
                for site in rotated_structure:
                    cart_coords = site.coords
                    rotated_coords = rotation_matrix @ cart_coords
                    new_coords.append(rotated_coords)

                rotated_structure = Structure(
                    lattice=rotated_structure.lattice,
                    species=[site.specie for site in rotated_structure],
                    coords=new_coords,
                    coords_are_cartesian=True,
                    site_properties=rotated_structure.site_properties,
                )
                rotation_description = f"Euler angles (ZXZ): α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}° (atoms only)"
            else:
                symm_op = SymmOp.from_rotation_and_translation(rotation_matrix)
                rotated_structure.apply_operation(symm_op)
                rotation_description = (
                    f"Euler angles (ZXZ): α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}°"
                )

        # Display rotation information
        console.print(f"\n[yellow]{rotation_description}[/yellow]")

        # Create comparison table
        table = Table(
            title="Lattice Parameters Comparison",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Parameter", style="cyan")
        table.add_column("Original", style="green")
        table.add_column("Rotated", style="yellow")
        table.add_column("Change", style="red")

        rot_params = rotated_structure.lattice.parameters

        for i, label in enumerate(["a (Å)", "b (Å)", "c (Å)"]):
            orig = orig_params[i]
            rot = rot_params[i]
            change = ((rot - orig) / orig * 100) if orig != 0 else 0
            table.add_row(
                label,
                f"{orig:.4f}",
                f"{rot:.4f}",
                f"{change:+.2f}%" if abs(change) > 0.01 else "—",
            )

        # Angles (only if changed significantly)
        if show_angles or not rotate_cell:
            for i, label in enumerate(["α (°)", "β (°)", "γ (°)"]):
                orig = orig_params[i + 3]
                rot = rot_params[i + 3]
                change = rot - orig
                table.add_row(
                    label,
                    f"{orig:.4f}",
                    f"{rot:.4f}",
                    f"{change:+.2f}°" if abs(change) > 0.01 else "—",
                )

        # Volume
        vol_change = (
            (rotated_structure.volume - structure.volume) / structure.volume * 100
        )
        table.add_row(
            "Volume (Å³)",
            f"{structure.volume:.3f}",
            f"{rotated_structure.volume:.3f}",
            f"{vol_change:+.2f}%" if abs(vol_change) > 0.01 else "—",
        )

        console.print(table)

        # Additional info for atom-only rotation
        if rotate_atoms_only:
            console.print(
                "\n[dim]Note: Atoms rotated, cell unchanged (fractional coordinates modified)[/dim]"
            )

        # Save structure
        if output is None:
            from pathlib import Path

            input_path = Path(structure_file)
            output = f"rotated_{input_path.name}"

        if format == "cif":
            from atomate2.siesta.sets.utils.structure_io import write_cif_with_ghost

            write_cif_with_ghost(rotated_structure, output)
        elif format == "poscar":
            rotated_structure.to(filename=output, fmt="poscar")
        elif format == "xsf":
            from pymatgen.io.xcrysden import XSF

            xsf = XSF(rotated_structure)
            xsf.to_file(output)
        elif format == "json":
            rotated_structure.to(filename=output, fmt="json")

        console.print(f"\n[green]✓ Rotated structure saved to: {output}[/green]")

        # Usage tips
        if axis and angle:
            console.print(
                "\n[dim]Tip: Use --show-angles to see angle changes between lattice vectors[/dim]"
            )
        if not show_angles and rotate_cell:
            console.print("[dim]Tip: Use --rotate-atoms-only to keep cell fixed[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()


if __name__ == "__main__":
    rotate()
