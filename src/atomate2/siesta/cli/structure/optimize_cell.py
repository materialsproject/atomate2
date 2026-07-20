"""Optimize cell shape for better periodic calculations.

This module provides the `optimize-cell` command for optimizing cell shapes using
Niggli reduction and orthogonalization techniques.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from pymatgen.core import Structure
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
    "--niggli",
    is_flag=True,
    help="Apply Niggli reduction (find most reduced cell)",
)
@click.option(
    "--orthogonalize",
    is_flag=True,
    help="Find most orthogonal supercell",
)
@click.option(
    "--max-atoms",
    type=int,
    default=1000,
    help="Maximum atoms for orthogonalization (default: 1000)",
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
def optimize_cell(
    structure_file: str,
    niggli: bool,
    orthogonalize: bool,
    max_atoms: int,
    output: str | None,
    format: str,  # noqa: A002 Click option name mirrors the CLI --format flag
    show_before_after: bool,
) -> None:
    """Optimize cell shape for periodic calculations.

    Optimizes cell using:
    - Niggli reduction: Find the most reduced lattice representation
    - Orthogonalization: Find most orthogonal supercell (better for DFT)

    Examples
    --------
        # Niggli reduction
        atomate2siesta-structure optimize-cell structure.cif --niggli

        # Find orthogonal cell
        atomate2siesta-structure optimize-cell structure.cif --orthogonalize

        # Orthogonalize with custom max atoms
        atomate2siesta-structure optimize-cell structure.cif --orthogonalize \
--max-atoms 500
    """
    try:
        # Validate options
        if not niggli and not orthogonalize:
            console.print(
                "[bold red]Error:[/bold red] Must specify --niggli or --orthogonalize"
            )
            sys.exit(1)
        if niggli and orthogonalize:
            console.print(
                "[bold red]Error:[/bold red] Can only specify one optimization mode"
            )
            sys.exit(1)

        # Load structure
        console.print("\n[bold cyan]Loading structure...[/bold cyan]")
        structure = load_structure(structure_file)
        original_structure = structure.copy()

        console.print(f"  Formula: {structure.formula}")
        console.print(f"  Sites: {structure.num_sites}")
        console.print(f"  Volume: {structure.volume:.2f} ų")
        console.print(
            f"  Original angles: α={structure.lattice.alpha:.2f}°, "  # noqa: RUF001
            f"β={structure.lattice.beta:.2f}°, γ={structure.lattice.gamma:.2f}°\n"  # noqa: RUF001
        )

        # Perform optimization
        console.print("[bold cyan]Optimizing cell...[/bold cyan]")

        if niggli:
            optimized = structure.copy()
            optimized.lattice = structure.lattice.get_niggli_reduced_lattice()
            mode_name = "niggli"
            console.print("  Mode: Niggli reduction")
            console.print("  Finding most reduced lattice representation...")
        else:  # orthogonalize
            mode_name = "orthogonal"
            console.print("  Mode: Orthogonalization")
            console.print(f"  Maximum atoms: {max_atoms}")
            console.print("  Finding most orthogonal supercell...")

            # Try to find orthogonal supercell
            try:
                # Use pymatgen's transformation to find orthogonal supercell

                # Simple approach: try a few common supercell matrices
                best_ortho = None
                best_score = float("inf")

                for scale in range(1, 6):  # Try up to 5x5x5
                    for a in range(1, scale + 1):
                        for b in range(1, scale + 1):
                            for c in range(1, scale + 1):
                                if a * b * c * structure.num_sites > max_atoms:
                                    continue

                                test_structure = structure.copy()
                                test_structure.make_supercell([a, b, c])

                                # Calculate orthogonality score (deviation from 90°)
                                angles = [
                                    test_structure.lattice.alpha,
                                    test_structure.lattice.beta,
                                    test_structure.lattice.gamma,
                                ]
                                score = sum((angle - 90) ** 2 for angle in angles)

                                if score < best_score:
                                    best_score = score
                                    best_ortho = test_structure

                if best_ortho is None:
                    console.print(
                        "[yellow]Warning: Could not find orthogonal cell "
                        f"within {max_atoms} atoms limit[/yellow]"
                    )
                    optimized = structure.copy()
                else:
                    optimized = best_ortho
                    console.print(f"  Found supercell: {optimized.num_sites} atoms")
                    console.print(f"  Orthogonality score: {best_score:.2f}")

            except Exception as e:  # noqa: BLE001 friendly fallback to original cell
                console.print(
                    f"[yellow]Warning: Orthogonalization failed: {e}[/yellow]"
                )
                console.print("  Returning original structure")
                optimized = structure.copy()

        console.print(f"  New sites: {optimized.num_sites}")
        console.print(f"  New volume: {optimized.volume:.2f} ų")
        console.print(
            f"  New angles: α={optimized.lattice.alpha:.2f}°, "  # noqa: RUF001
            f"β={optimized.lattice.beta:.2f}°, γ={optimized.lattice.gamma:.2f}°\n"  # noqa: RUF001
        )

        # Show comparison
        if show_before_after:
            show_structure_comparison(original_structure, optimized)

        # Determine output filename
        if output is None:
            input_path = Path(structure_file)
            base_name = input_path.stem
            ext = format if format != "XV" else "xv"
            output = f"{mode_name}_{base_name}.{ext}"

        # Save structure
        console.print(
            f"[bold cyan]Saving optimized structure to {output}...[/bold cyan]"
        )
        save_structure(optimized, output, format)

        console.print(
            f"\n[bold green]✓ Successfully optimized cell ({mode_name})![/bold green]"
        )
        console.print(f"  Output: {output}\n")

        # Show summary
        _show_summary(original_structure, optimized, mode_name)

    except Exception as e:  # noqa: BLE001 friendly CLI error reporting
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _show_summary(original: Structure, optimized: Structure, mode: str) -> None:
    """Show summary of optimization."""
    table = Table(title="Cell Optimization Summary", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Original", style="yellow")
    table.add_column("Optimized", style="green")
    table.add_column("Change", style="magenta")

    # Number of sites
    site_multiplier = optimized.num_sites / original.num_sites
    table.add_row(
        "Sites",
        str(original.num_sites),
        str(optimized.num_sites),
        f"×{site_multiplier:.2f}",  # noqa: RUF001
    )

    # Volume
    vol_multiplier = optimized.volume / original.volume
    table.add_row(
        "Volume (Ų)",
        f"{original.volume:.2f}",
        f"{optimized.volume:.2f}",
        f"×{vol_multiplier:.2f}",  # noqa: RUF001
    )

    # Lattice parameters
    for param in ["a", "b", "c"]:
        orig_val = getattr(original.lattice, param)
        opt_val = getattr(optimized.lattice, param)
        change = opt_val - orig_val
        table.add_row(
            f"{param} (Å)",
            f"{orig_val:.4f}",
            f"{opt_val:.4f}",
            f"{change:+.4f}",
        )

    # Angles
    for param in ["alpha", "beta", "gamma"]:
        orig_val = getattr(original.lattice, param)
        opt_val = getattr(optimized.lattice, param)
        change = opt_val - orig_val
        table.add_row(
            f"{param} (°)",
            f"{orig_val:.2f}",
            f"{opt_val:.2f}",
            f"{change:+.2f}",
        )

    # Orthogonality metric
    orig_ortho = sum(
        (angle - 90) ** 2
        for angle in [
            original.lattice.alpha,
            original.lattice.beta,
            original.lattice.gamma,
        ]
    )
    opt_ortho = sum(
        (angle - 90) ** 2
        for angle in [
            optimized.lattice.alpha,
            optimized.lattice.beta,
            optimized.lattice.gamma,
        ]
    )

    table.add_row(
        "Orthogonality",
        f"{orig_ortho:.2f}",
        f"{opt_ortho:.2f}",
        f"{opt_ortho - orig_ortho:+.2f}",
    )

    console.print(table)
    console.print()

    if mode == "orthogonal":
        console.print(
            "[dim]Note: Lower orthogonality score means more orthogonal cell[/dim]\n"
        )


if __name__ == "__main__":
    optimize_cell()
