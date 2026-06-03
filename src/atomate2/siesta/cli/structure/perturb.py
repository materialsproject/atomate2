#!/usr/bin/env python3
"""CLI for perturbing structures with random displacements.

This module provides the `perturb` subcommand for atomate2siesta-structure.
"""

from __future__ import annotations

import random

import numpy as np
import click
from rich.console import Console
from rich.table import Table

from pymatgen.core import Structure

console = Console()

# Boltzmann constant in eV/K
KB = 8.617333262e-5


@click.command()
@click.argument("structure_file", type=click.Path(exists=True))
@click.option(
    "--amplitude",
    type=float,
    help="Random displacement amplitude in Å (uniform distribution)",
)
@click.option(
    "--temperature",
    type=float,
    help="Temperature for thermal displacements in K (Maxwell-Boltzmann)",
)
@click.option(
    "--n-configs",
    type=int,
    default=1,
    help="Number of perturbed configurations to generate (default: 1)",
)
@click.option(
    "--element",
    type=str,
    help="Only perturb atoms of this element",
)
@click.option(
    "--seed",
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file prefix (default: perturbed_<input>)",
)
@click.option(
    "--format",
    type=click.Choice(["cif", "poscar", "xsf", "json", "fdf", "XV"]),
    default="cif",
    help="Output format (default: cif)",
)
@click.option(
    "--show-stats",
    is_flag=True,
    help="Show displacement statistics",
)
def perturb(
    structure_file,
    amplitude,
    temperature,
    n_configs,
    element,
    seed,
    output,
    format,
    show_stats,
):
    """Perturb atomic positions with random displacements.

    Generate perturbed structures for MD initialization, transition state search,
    or ensemble generation. Supports uniform random displacements or thermal
    displacements based on Maxwell-Boltzmann distribution.

    Examples:

        # Random displacements with 0.1 Å amplitude
        atomate2siesta-structure perturb structure.cif --amplitude 0.1

        # Generate 10 configurations
        atomate2siesta-structure perturb structure.cif --amplitude 0.1 --n-configs 10

        # Thermal displacements at 300 K
        atomate2siesta-structure perturb structure.cif --temperature 300

        # Perturb only hydrogen atoms
        atomate2siesta-structure perturb structure.cif --amplitude 0.2 --element H

        # With reproducible random seed
        atomate2siesta-structure perturb structure.cif --amplitude 0.1 --seed 42

        # Show displacement statistics
        atomate2siesta-structure perturb structure.cif --amplitude 0.1 --show-stats
    """
    try:
        # Validate options
        if not any([amplitude, temperature]):
            console.print(
                "[red]Error: Must specify either --amplitude or --temperature[/red]"
            )
            raise click.Abort()

        if amplitude and temperature:
            console.print(
                "[red]Error: Cannot use both --amplitude and --temperature[/red]"
            )
            raise click.Abort()

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            console.print(f"[dim]Using random seed: {seed}[/dim]")

        # Load structure
        structure = Structure.from_file(structure_file)
        console.print(
            f"\n[cyan]Loaded structure: {structure.composition.reduced_formula}[/cyan]"
        )
        console.print(f"  Formula: {structure.composition.formula}")
        console.print(f"  Sites: {structure.num_sites}")

        # Determine which atoms to perturb
        if element:
            from pymatgen.core import Element

            try:
                elem = Element(element)
                perturb_indices = [
                    i
                    for i, site in enumerate(structure)
                    if site.specie.symbol == elem.symbol
                ]
                if not perturb_indices:
                    console.print(
                        f"[yellow]Warning: No {elem.symbol} atoms found[/yellow]"
                    )
                    raise click.Abort()
                console.print(
                    f"\n[yellow]Perturbing only {elem.symbol} atoms ({len(perturb_indices)} sites)[/yellow]"
                )
            except Exception as e:
                console.print(f"[red]Error: Invalid element symbol: {e}[/red]")
                raise click.Abort()
        else:
            perturb_indices = list(range(structure.num_sites))
            console.print(
                f"\n[yellow]Perturbing all {len(perturb_indices)} atoms[/yellow]"
            )

        # Determine perturbation mode
        if amplitude:
            mode = "uniform"
            console.print(
                f"[yellow]Mode: Uniform random (amplitude = {amplitude:.3f} Å)[/yellow]"
            )
        else:
            mode = "thermal"
            console.print(
                f"[yellow]Mode: Thermal (temperature = {temperature:.1f} K)[/yellow]"
            )

        # Generate perturbed configurations
        configs = []
        all_displacements = []

        for config_num in range(n_configs):
            # Create copy of structure
            perturbed = structure.copy()

            # Generate displacements
            displacements = []

            for site_idx in perturb_indices:
                site = structure[site_idx]

                if mode == "uniform":
                    # Uniform random displacement in spherical coordinates
                    # Random direction (uniform on sphere)
                    theta = np.random.uniform(0, 2 * np.pi)
                    phi = np.arccos(np.random.uniform(-1, 1))

                    # Random magnitude (uniform in amplitude)
                    r = np.random.uniform(0, amplitude)

                    # Convert to Cartesian
                    dx = r * np.sin(phi) * np.cos(theta)
                    dy = r * np.sin(phi) * np.sin(theta)
                    dz = r * np.cos(phi)

                else:  # thermal
                    # Maxwell-Boltzmann distribution
                    # sigma = sqrt(kB * T / m) but we use atomic mass
                    mass = site.specie.atomic_mass  # amu

                    # Thermal velocity width (Å)
                    # Using simplified formula: sigma ~ sqrt(kB*T/m) with units
                    # kB*T in eV, m in amu, need conversion factor
                    # 1 eV = 1.602e-19 J, 1 amu = 1.66e-27 kg
                    # Result in Å/fs, but we use as displacement width
                    sigma = (
                        np.sqrt(KB * temperature / mass) * 0.1
                    )  # Approximate scaling

                    # Gaussian displacements in each direction
                    dx = np.random.normal(0, sigma)
                    dy = np.random.normal(0, sigma)
                    dz = np.random.normal(0, sigma)

                displacement = np.array([dx, dy, dz])
                displacements.append(np.linalg.norm(displacement))

                # Apply displacement
                new_coords = site.coords + displacement
                perturbed[site_idx] = (site.specie, new_coords)

            configs.append(perturbed)
            all_displacements.extend(displacements)

        # Display statistics
        _display_perturbation_info(structure, configs[0], displacements, mode)

        if show_stats:
            _display_displacement_stats(all_displacements, n_configs, mode)

        # Save configurations
        if n_configs == 1:
            # Single configuration
            if output is None:
                from pathlib import Path

                input_path = Path(structure_file)
                if mode == "uniform":
                    output = f"perturbed_amp{amplitude:.2f}_{input_path.stem}.{format}"
                else:
                    output = f"perturbed_T{temperature:.0f}K_{input_path.stem}.{format}"

            _save_structure(configs[0], output, format)
            console.print(f"\n[green]✓ Perturbed structure saved to: {output}[/green]")
        else:
            # Multiple configurations
            from pathlib import Path

            input_path = Path(structure_file)
            saved_files = []

            for i, config in enumerate(configs):
                if output:
                    output_file = f"{output}_{i+1}.{format}"
                else:
                    if mode == "uniform":
                        output_file = f"perturbed_amp{amplitude:.2f}_{input_path.stem}_config{i+1}.{format}"
                    else:
                        output_file = f"perturbed_T{temperature:.0f}K_{input_path.stem}_config{i+1}.{format}"

                _save_structure(config, output_file, format)
                saved_files.append(output_file)

            console.print(
                f"\n[green]✓ Generated {n_configs} perturbed configurations:[/green]"
            )
            for f in saved_files[:5]:  # Show first 5
                console.print(f"  {f}")
            if len(saved_files) > 5:
                console.print(f"  ... and {len(saved_files) - 5} more")

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


def _display_perturbation_info(original, perturbed, displacements, mode):
    """Display perturbation information."""
    console.print("\n[yellow]Perturbation Summary:[/yellow]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Mode", mode.capitalize())
    table.add_row("Atoms perturbed", str(len(displacements)))
    table.add_row(
        "Mean displacement",
        f"{np.mean(displacements):.4f} Å",
    )
    table.add_row(
        "Max displacement",
        f"{np.max(displacements):.4f} Å",
    )
    table.add_row(
        "RMS displacement",
        f"{np.sqrt(np.mean(np.array(displacements)**2)):.4f} Å",
    )

    console.print(table)


def _display_displacement_stats(all_displacements, n_configs, mode):
    """Display detailed displacement statistics."""
    console.print("\n[cyan]Displacement Statistics (all configurations):[/cyan]")

    displacements = np.array(all_displacements)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Statistic", style="cyan")
    table.add_column("Value (Å)", style="yellow")

    table.add_row("Configurations", str(n_configs))
    table.add_row("Total displacements", str(len(displacements)))
    table.add_row("Mean", f"{displacements.mean():.4f}")
    table.add_row("Std Dev", f"{displacements.std():.4f}")
    table.add_row("Min", f"{displacements.min():.4f}")
    table.add_row("Max", f"{displacements.max():.4f}")
    table.add_row("RMS", f"{np.sqrt((displacements**2).mean()):.4f}")

    # Percentiles
    table.add_row("—", "—")
    table.add_row("25th percentile", f"{np.percentile(displacements, 25):.4f}")
    table.add_row("Median (50th)", f"{np.percentile(displacements, 50):.4f}")
    table.add_row("75th percentile", f"{np.percentile(displacements, 75):.4f}")

    console.print(table)


if __name__ == "__main__":
    perturb()
