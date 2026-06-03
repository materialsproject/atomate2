#!/usr/bin/env python3
"""CLI for generating supercells.

This module provides the `supercell` subcommand for atomate2siesta-structure.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from pymatgen.core import Structure

console = Console()


@click.command()
@click.argument("structure_file", type=click.Path(exists=True))
@click.option(
    "--matrix",
    type=int,
    nargs=3,
    help="Supercell matrix (3 integers: nx ny nz for diagonal supercell)",
)
@click.option(
    "--min-length",
    type=float,
    help="Minimum supercell length in Å (automatically determines supercell size)",
)
@click.option(
    "--min-atoms",
    type=int,
    help="Minimum number of atoms (automatically determines supercell size)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file (default: supercell_<input>)",
)
@click.option(
    "--format",
    type=click.Choice(["cif", "poscar", "xsf", "json"]),
    default="cif",
    help="Output format (default: cif)",
)
@click.option(
    "--preserve-magmom",
    is_flag=True,
    default=True,
    help="Preserve magnetic moments in supercell (default: True)",
)
@click.option(
    "--show-estimate",
    is_flag=True,
    help="Show memory/time estimates for phonon/NEB calculations",
)
def supercell(
    structure_file,
    matrix,
    min_length,
    min_atoms,
    output,
    format,
    preserve_magmom,
    show_estimate,
):
    """Generate supercells for phonon, defect, or surface calculations.

    Supports diagonal supercells (nx×ny×nz), minimum length specification,
    or minimum atom count. Automatically propagates site properties like
    magnetic moments.

    Examples:

        # 2×2×2 supercell
        atomate2siesta-structure supercell Si.cif --matrix 2 2 2

        # Non-cubic supercell
        atomate2siesta-structure supercell Si.cif --matrix 2 2 1

        # Automatic sizing for 10 Å minimum length
        atomate2siesta-structure supercell Si.cif --min-length 10.0

        # At least 50 atoms
        atomate2siesta-structure supercell Si.cif --min-atoms 50

        # With computation estimates
        atomate2siesta-structure supercell Si.cif --matrix 3 3 3 --show-estimate
    """
    # Validate options
    options_count = sum(
        [matrix is not None, min_length is not None, min_atoms is not None]
    )
    if options_count == 0:
        console.print(
            "[red]Error: Must specify one of --matrix, --min-length, or --min-atoms[/red]"
        )
        raise click.Abort()
    if options_count > 1:
        console.print(
            "[red]Error: Only specify one supercell method (--matrix, --min-length, or --min-atoms)[/red]"
        )
        raise click.Abort()

    try:
        # Load structure
        structure = Structure.from_file(structure_file)
        console.print(
            f"\n[cyan]Loaded structure: {structure.composition.reduced_formula}[/cyan]"
        )
        console.print(f"  Formula: {structure.composition.formula}")
        console.print(f"  Sites: {structure.num_sites}")
        console.print(f"  Volume: {structure.volume:.3f} Å³")

        # Display lattice parameters
        params = structure.lattice.parameters
        console.print(
            f"  Lattice: a={params[0]:.3f} Å, b={params[1]:.3f} Å, c={params[2]:.3f} Å"
        )

        # Check for magnetic moments
        has_magmom = "magmom" in structure.site_properties
        if has_magmom:
            magmoms = structure.site_properties["magmom"]
            n_magnetic = sum(1 for m in magmoms if abs(m) > 0.01)
            console.print(f"  Magnetic sites: {n_magnetic}/{structure.num_sites}")

        # Determine supercell matrix
        if matrix is not None:
            # User-specified matrix
            sc_matrix = [[matrix[0], 0, 0], [0, matrix[1], 0], [0, 0, matrix[2]]]
            method = f"User-specified: {matrix[0]}×{matrix[1]}×{matrix[2]}"

        elif min_length is not None:
            # Automatic sizing based on minimum length
            import numpy as np

            abc = structure.lattice.abc
            nx = int(np.ceil(min_length / abc[0]))
            ny = int(np.ceil(min_length / abc[1]))
            nz = int(np.ceil(min_length / abc[2]))
            sc_matrix = [[nx, 0, 0], [0, ny, 0], [0, 0, nz]]
            method = f"Auto from min_length={min_length:.1f} Å: {nx}×{ny}×{nz}"

        elif min_atoms is not None:
            # Automatic sizing based on minimum atoms
            import numpy as np

            # Estimate cubic supercell needed
            n_unit = structure.num_sites
            scale_factor = (min_atoms / n_unit) ** (1 / 3)
            n = int(np.ceil(scale_factor))

            # Start with cubic and adjust if needed
            sc_matrix = [[n, 0, 0], [0, n, 0], [0, 0, n]]
            if n**3 * n_unit < min_atoms:
                n += 1
                sc_matrix = [[n, 0, 0], [0, n, 0], [0, 0, n]]

            method = f"Auto from min_atoms={min_atoms}: {n}×{n}×{n}"

        # Generate supercell
        supercell_structure = structure.copy()
        supercell_structure.make_supercell(sc_matrix)

        # Display supercell information
        console.print(f"\n[yellow]{method}[/yellow]")

        # Create comparison table
        table = Table(
            title="Structure Comparison", show_header=True, header_style="bold magenta"
        )
        table.add_column("Property", style="cyan")
        table.add_column("Unit Cell", style="green")
        table.add_column("Supercell", style="yellow")
        table.add_column("Multiplier", style="red")

        # Number of atoms
        multiplier = supercell_structure.num_sites / structure.num_sites
        table.add_row(
            "Atoms",
            str(structure.num_sites),
            str(supercell_structure.num_sites),
            f"×{multiplier:.0f}",
        )

        # Lattice parameters
        sc_params = supercell_structure.lattice.parameters
        for i, label in enumerate(["a (Å)", "b (Å)", "c (Å)"]):
            orig = params[i]
            sc = sc_params[i]
            mult = sc / orig
            table.add_row(label, f"{orig:.4f}", f"{sc:.4f}", f"×{mult:.2f}")

        # Volume
        vol_mult = supercell_structure.volume / structure.volume
        table.add_row(
            "Volume (Å³)",
            f"{structure.volume:.3f}",
            f"{supercell_structure.volume:.3f}",
            f"×{vol_mult:.1f}",
        )

        # Magnetic sites
        if has_magmom and preserve_magmom:
            sc_magmoms = supercell_structure.site_properties.get("magmom", [])
            sc_magnetic = sum(1 for m in sc_magmoms if abs(m) > 0.01)
            mag_mult = sc_magnetic / n_magnetic if n_magnetic > 0 else 0
            table.add_row(
                "Magnetic sites",
                str(n_magnetic),
                str(sc_magnetic),
                f"×{mag_mult:.0f}",
            )

        console.print(table)

        # Show computational estimates
        if show_estimate:
            n_atoms = supercell_structure.num_sites

            # Phonon calculation estimates (using finite differences)
            n_displacements = n_atoms * 3 * 2  # 3 directions × 2 directions (±)
            phonon_time_estimate = (
                n_displacements * 5
            )  # ~5 min per displacement (rough)

            # Memory estimate (rough)
            # Typical DFT: ~100 MB per atom for wavefunctions
            memory_estimate = n_atoms * 0.1  # GB

            estimate_text = f"""
[bold]Computational Estimates:[/bold]

[cyan]Phonon Calculation (Finite Differences):[/cyan]
  • Displacements: {n_displacements} (3 directions × 2 per atom)
  • Estimated time: ~{phonon_time_estimate:.0f} minutes ({phonon_time_estimate/60:.1f} hours)
    (assuming ~5 min per force calculation)
  • Memory: ~{memory_estimate:.1f} GB (rough estimate)

[cyan]NEB Calculation:[/cyan]
  • Typical images: 5-7 intermediate structures
  • Time per image: ~10-30 min
  • Total for 5 images: ~{5*15:.0f} min ({5*15/60:.1f} hours)

[yellow]Note:[/yellow] These are rough estimates. Actual time depends on:
  • k-point sampling
  • Basis set size
  • SCF convergence criteria
  • System complexity
"""
            console.print(Panel(estimate_text, title="Estimates", border_style="blue"))

        # Save supercell
        if output is None:
            from pathlib import Path

            input_path = Path(structure_file)
            output = f"supercell_{input_path.name}"

        if format == "cif":
            from atomate2.siesta.sets.utils.structure_io import write_cif_with_ghost

            write_cif_with_ghost(supercell_structure, output)
        elif format == "poscar":
            supercell_structure.to(filename=output, fmt="poscar")
        elif format == "xsf":
            from pymatgen.io.xcrysden import XSF

            xsf = XSF(supercell_structure)
            xsf.to_file(output)
        elif format == "json":
            supercell_structure.to(filename=output, fmt="json")

        console.print(f"\n[green]✓ Supercell saved to: {output}[/green]")

        # Verification tip
        if has_magmom and preserve_magmom:
            console.print(
                "[dim]  Magnetic moments preserved ✓ (magmom site property propagated)[/dim]"
            )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()


if __name__ == "__main__":
    supercell()
