#!/usr/bin/env python3
# ruff: noqa: EXE001
"""CLI for scaling lattice parameters.

This module provides the `scale` subcommand for atomate2siesta-structure.
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
    "--factor",
    type=float,
    help="Uniform scaling factor (e.g., 1.05 for 5% expansion)",
)
@click.option(
    "--abc",
    type=float,
    nargs=3,
    help="Non-uniform scaling factors for a, b, c lattice parameters",
)
@click.option(
    "--volume",
    type=float,
    help="Target volume in Å³ (scales uniformly to reach this volume)",
)
@click.option(
    "--strain",
    type=float,
    help="Volumetric strain (e.g., 0.05 for 5% volume increase)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file (default: scaled_<input>)",
)
@click.option(
    "--format",
    type=click.Choice(["cif", "poscar", "xsf", "json"]),
    default="cif",
    help="Output format (default: cif)",
)
@click.option(
    "--series",
    is_flag=True,
    help="Generate a series of scaled structures (requires --min and --max)",
)
@click.option(
    "--min",
    "min_factor",
    type=float,
    help="Minimum scaling factor for series",
)
@click.option(
    "--max",
    "max_factor",
    type=float,
    help="Maximum scaling factor for series",
)
@click.option(
    "--steps",
    type=int,
    default=11,
    help="Number of steps in series (default: 11)",
)
def scale(
    structure_file: str,
    factor: float | None,
    abc: tuple[float, float, float] | None,
    volume: float | None,
    strain: float | None,
    output: str | None,
    format: str,  # noqa: A002 Click option name mirrors the CLI --format flag
    series: bool,
    min_factor: float | None,
    max_factor: float | None,
    steps: int,
) -> None:
    """Scale lattice parameters of a structure.

    Supports uniform scaling, anisotropic scaling, volume-based scaling,
    and strain-based scaling. Can generate series of structures for EOS studies.

    Examples
    --------
        # Uniform 5% expansion
        atomate2siesta-structure scale Si.cif --factor 1.05

        # Anisotropic scaling (expand a, keep b, compress c)
        atomate2siesta-structure scale Si.cif --abc 1.05 1.0 0.95

        # Scale to specific volume
        atomate2siesta-structure scale Si.cif --volume 50.0

        # Apply volumetric strain
        atomate2siesta-structure scale Si.cif --strain 0.05

        # Generate series for EOS
        atomate2siesta-structure scale Si.cif --series --min 0.95 --max 1.05 --steps 11
    """
    # Validate options
    options_count = sum(
        [
            factor is not None,
            abc is not None,
            volume is not None,
            strain is not None,
            series,
        ]
    )
    if options_count == 0:
        console.print(
            "[red]Error: Must specify one of --factor, --abc, --volume, "
            "--strain, or --series[/red]"
        )
        raise click.Abort
    if options_count > 1:
        console.print("[red]Error: Only specify one scaling method[/red]")
        raise click.Abort

    if series and (min_factor is None or max_factor is None):
        console.print("[red]Error: --series requires --min and --max[/red]")
        raise click.Abort

    try:
        # Load structure
        structure = Structure.from_file(structure_file)
        console.print(
            f"\n[cyan]Loaded structure: {structure.composition.reduced_formula}[/cyan]"
        )
        console.print(f"  Formula: {structure.composition.formula}")
        console.print(f"  Sites: {structure.num_sites}")
        console.print(f"  Volume: {structure.volume:.3f} Å³")

        if series:
            # Generate series of scaled structures
            import numpy as np

            factors = np.linspace(min_factor, max_factor, steps)
            console.print(
                f"\n[yellow]Generating series with {steps} structures[/yellow]"
            )
            console.print(f"  Range: {min_factor:.4f} to {max_factor:.4f}")

            # Create output filenames
            from pathlib import Path

            input_path = Path(structure_file)
            base_name = input_path.stem

            for i, scale_factor in enumerate(factors):
                scaled_structure = structure.copy()
                scaled_structure.scale_lattice(
                    scaled_structure.volume * scale_factor**3
                )

                # Generate filename
                output_file = f"{base_name}_scale_{scale_factor:.4f}.{format}"

                # Write structure
                if format == "cif":
                    scaled_structure.to(filename=output_file, fmt="cif")
                elif format == "poscar":
                    scaled_structure.to(filename=output_file, fmt="poscar")
                elif format == "xsf":
                    from pymatgen.io.xcrysden import XSF

                    xsf = XSF(scaled_structure)
                    xsf.to_file(output_file)
                elif format == "json":
                    scaled_structure.to(filename=output_file, fmt="json")

                if i == 0 or i == steps - 1 or (i + 1) % 5 == 0:
                    console.print(
                        f"  [{i + 1:2d}/{steps}] {output_file}: "
                        f"V = {scaled_structure.volume:.3f} Å³"
                    )

            console.print(f"\n[green]✓ Generated {steps} scaled structures[/green]")

        else:
            # Single structure scaling
            scaled_structure = structure.copy()

            if factor is not None:
                # Uniform scaling
                scaled_structure.scale_lattice(structure.volume * factor**3)
                scale_type = f"Uniform scaling: factor = {factor:.4f}"

            elif abc is not None:
                # Anisotropic scaling
                lattice = structure.lattice
                new_lattice = lattice.matrix.copy()
                new_lattice[0] *= abc[0]
                new_lattice[1] *= abc[1]
                new_lattice[2] *= abc[2]

                from pymatgen.core import Lattice

                scaled_structure.lattice = Lattice(new_lattice)
                scale_type = (
                    f"Anisotropic scaling: a={abc[0]:.4f}, "
                    f"b={abc[1]:.4f}, c={abc[2]:.4f}"
                )

            elif volume is not None:
                # Volume-based scaling
                scaled_structure.scale_lattice(volume)
                scale_factor = (volume / structure.volume) ** (1 / 3)
                scale_type = (
                    f"Volume scaling: target = {volume:.3f} Å³ "
                    f"(factor = {scale_factor:.4f})"
                )

            elif strain is not None:
                # Strain-based scaling
                scale_factor = (1 + strain) ** (1 / 3)
                scaled_structure.scale_lattice(structure.volume * (1 + strain))
                scale_type = (
                    f"Volumetric strain: ε = {strain:.4f} (factor = {scale_factor:.4f})"
                )

            # Display results
            console.print(f"\n[yellow]{scale_type}[/yellow]")

            # Create comparison table
            table = Table(
                title="Lattice Parameters Comparison",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Parameter", style="cyan")
            table.add_column("Original", style="green")
            table.add_column("Scaled", style="yellow")
            table.add_column("Change", style="red")

            orig_params = structure.lattice.parameters
            scaled_params = scaled_structure.lattice.parameters

            for i, label in enumerate(
                ["a (Å)", "b (Å)", "c (Å)", "α (°)", "β (°)", "γ (°)"]  # noqa: RUF001
            ):
                orig = orig_params[i]
                scaled = scaled_params[i]
                change = ((scaled - orig) / orig * 100) if orig != 0 else 0
                table.add_row(
                    label,
                    f"{orig:.4f}",
                    f"{scaled:.4f}",
                    f"{change:+.2f}%",
                )

            # Volume
            vol_change = (
                (scaled_structure.volume - structure.volume) / structure.volume * 100
            )
            table.add_row(
                "Volume (Å³)",
                f"{structure.volume:.3f}",
                f"{scaled_structure.volume:.3f}",
                f"{vol_change:+.2f}%",
            )

            console.print(table)

            # Save structure
            if output is None:
                from pathlib import Path

                input_path = Path(structure_file)
                output = f"scaled_{input_path.name}"

            if format == "cif":
                scaled_structure.to(filename=output, fmt="cif")
            elif format == "poscar":
                scaled_structure.to(filename=output, fmt="poscar")
            elif format == "xsf":
                from pymatgen.io.xcrysden import XSF

                xsf = XSF(scaled_structure)
                xsf.to_file(output)
            elif format == "json":
                scaled_structure.to(filename=output, fmt="json")

            console.print(f"\n[green]✓ Scaled structure saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort from e


if __name__ == "__main__":
    scale()
