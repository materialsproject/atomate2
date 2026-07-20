"""CLI for atomic substitution in structures.

This module provides the `substitute` subcommand for atomate2siesta-structure.
"""

from __future__ import annotations

import random

import click
from pymatgen.core import Element, Structure
from rich.console import Console
from rich.table import Table

console = Console()


@click.command()
@click.argument("structure_file", type=click.Path(exists=True))
@click.option(
    "--replace",
    type=str,
    required=True,
    help="Element replacement as 'OLD:NEW' (e.g., 'Fe:Co')",
)
@click.option(
    "--fraction",
    type=float,
    help="Fraction of atoms to substitute (0.0-1.0, for random substitution)",
)
@click.option(
    "--sites",
    type=str,
    help="Specific site indices to substitute (comma-separated, 0-based)",
)
@click.option(
    "--n-configs",
    type=int,
    default=1,
    help="Number of random configurations to generate (default: 1)",
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
    help="Output file (default: substituted_<input>)",
)
@click.option(
    "--format",
    type=click.Choice(["cif", "poscar", "xsf", "json", "fdf", "XV"]),
    default="cif",
    help="Output format (default: cif)",
)
@click.option(
    "--show-changes",
    is_flag=True,
    help="Show which atoms were substituted",
)
def substitute(
    structure_file: str,
    replace: str,
    fraction: float | None,
    sites: str | None,
    n_configs: int,
    seed: int | None,
    output: str | None,
    format: str,  # noqa: A002 Click --format option name
    show_changes: bool,
) -> None:
    """Substitute atoms in a structure (doping, alloying, element replacement).

    Replace specific elements or sites with different elements. Supports complete
    substitution, random partial substitution, and site-specific replacement.

    Examples
    --------
        # Replace all Fe with Co
        atomate2siesta-structure substitute structure.cif --replace Fe:Co

        # Random 25% substitution of Fe with Co
        atomate2siesta-structure substitute structure.cif \
            --replace Fe:Co --fraction 0.25

        # Replace specific sites
        atomate2siesta-structure substitute structure.cif --replace Fe:Co --sites 0,2,4

        # Generate 10 random configurations
        atomate2siesta-structure substitute structure.cif \
            --replace Fe:Co --fraction 0.5 --n-configs 10

        # With reproducible random seed
        atomate2siesta-structure substitute structure.cif \
            --replace Fe:Co --fraction 0.3 --seed 42
    """
    try:
        # Parse replacement
        if ":" not in replace:
            console.print(
                "[red]Error: --replace must be in format 'OLD:NEW' "
                "(e.g., 'Fe:Co')[/red]"
            )
            raise click.Abort  # noqa: TRY301

        old_element_str, new_element_str = replace.split(":")

        try:
            old_element = Element(old_element_str)
            new_element = Element(new_element_str)
        except Exception as e:
            console.print(f"[red]Error: Invalid element symbol: {e}[/red]")
            raise click.Abort from e

        # Validate options
        if fraction is not None and sites is not None:
            console.print("[red]Error: Cannot use both --fraction and --sites[/red]")
            raise click.Abort  # noqa: TRY301

        if fraction is not None and (fraction < 0.0 or fraction > 1.0):
            console.print("[red]Error: --fraction must be between 0.0 and 1.0[/red]")
            raise click.Abort  # noqa: TRY301

        # Load structure
        structure = Structure.from_file(structure_file)
        console.print(
            f"\n[cyan]Loaded structure: {structure.composition.reduced_formula}[/cyan]"
        )
        console.print(f"  Formula: {structure.composition.formula}")
        console.print(f"  Sites: {structure.num_sites}")

        # Find sites with old element
        old_element_indices = [
            i
            for i, site in enumerate(structure)
            if site.specie.symbol == old_element.symbol
        ]

        if not old_element_indices:
            console.print(
                f"[yellow]Warning: No {old_element.symbol} atoms "
                f"found in structure[/yellow]"
            )
            raise click.Abort  # noqa: TRY301

        console.print(
            f"\n[yellow]Found {len(old_element_indices)} "
            f"{old_element.symbol} atoms[/yellow]"
        )

        # Determine which sites to substitute
        if sites is not None:
            # Site-specific substitution
            try:
                site_indices = [int(x) for x in sites.split(",")]
                # Validate site indices
                for idx in site_indices:
                    if idx < 0 or idx >= structure.num_sites:
                        console.print(
                            f"[red]Error: Site index {idx} out of range "
                            f"(0-{structure.num_sites - 1})[/red]"
                        )
                        raise click.Abort
                    if structure[idx].specie.symbol != old_element.symbol:
                        console.print(
                            f"[yellow]Warning: Site {idx} is "
                            f"{structure[idx].specie.symbol}, "
                            f"not {old_element.symbol}[/yellow]"
                        )
                substitution_mode = "site-specific"
            except ValueError:
                console.print("[red]Error: Invalid site indices format[/red]")
                raise click.Abort from None
        elif fraction is not None:
            # Random partial substitution
            substitution_mode = "random"
        else:
            # Complete substitution
            site_indices = old_element_indices
            substitution_mode = "complete"

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            console.print(f"[dim]Using random seed: {seed}[/dim]")

        # Generate configurations
        configs = []
        for _ in range(n_configs):
            # Create copy of structure
            new_structure = structure.copy()

            # Determine sites to substitute for this config
            if substitution_mode == "random":
                n_to_substitute = max(1, int(len(old_element_indices) * fraction))
                config_sites = random.sample(old_element_indices, n_to_substitute)
            elif substitution_mode == "site-specific":
                config_sites = [
                    idx for idx in site_indices if idx in old_element_indices
                ]
            else:  # complete
                config_sites = site_indices

            # Perform substitution
            for site_idx in config_sites:
                new_structure[site_idx] = new_element

            configs.append((new_structure, config_sites))

        # Display substitution information
        _display_substitution_info(
            structure,
            configs[0][0],
            old_element,
            new_element,
            configs[0][1],
            substitution_mode,
        )

        if show_changes:
            _display_changes(structure, configs[0][1], old_element, new_element)

        # Save configurations
        if n_configs == 1:
            # Single configuration
            if output is None:
                from pathlib import Path

                input_path = Path(structure_file)
                output = (
                    f"substituted_{old_element.symbol}_to_"
                    f"{new_element.symbol}_{input_path.stem}.{format}"
                )

            _save_structure(configs[0][0], output, format)
            console.print(
                f"\n[green]✓ Substituted structure saved to: {output}[/green]"
            )
        else:
            # Multiple configurations
            from pathlib import Path

            input_path = Path(structure_file)
            saved_files = []

            for i, (config_struct, _) in enumerate(configs):
                if output:
                    output_file = f"{output}_{i + 1}.{format}"
                else:
                    output_file = (
                        f"substituted_{old_element.symbol}_to_"
                        f"{new_element.symbol}_{input_path.stem}_config{i + 1}.{format}"
                    )

                _save_structure(config_struct, output_file, format)
                saved_files.append(output_file)

            console.print(f"\n[green]✓ Generated {n_configs} configurations:[/green]")
            for f in saved_files[:5]:  # Show first 5
                console.print(f"  {f}")
            if len(saved_files) > 5:
                console.print(f"  ... and {len(saved_files) - 5} more")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort from e


def _save_structure(structure: Structure, filename: str, fmt: str) -> None:
    """Save structure to file."""
    if fmt == "cif":
        from atomate2.siesta.sets.utils.structure_io import write_cif_with_ghost

        write_cif_with_ghost(structure, filename)
    elif fmt == "poscar":
        structure.to(filename=filename, fmt="poscar")
    elif fmt == "xsf":
        from pymatgen.io.xcrysden import XSF

        xsf = XSF(structure)
        with open(filename, "w") as f:
            f.write(xsf.to_str())
    elif fmt == "json":
        structure.to(filename=filename, fmt="json")
    elif fmt == "fdf":
        import sisl

        geom = sisl.get_sile(structure).read_geometry()
        with sisl.get_sile(filename, "w") as fdf:
            fdf.write_geometry(geom)
    elif fmt == "XV":
        import sisl

        geom = sisl.get_sile(structure).read_geometry()
        geom.write(filename)


def _display_substitution_info(
    original: Structure,
    substituted: Structure,
    old_elem: Element,
    new_elem: Element,
    substituted_sites: list,
    mode: str,
) -> None:
    """Display substitution information."""
    console.print("\n[yellow]Substitution Summary:[/yellow]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Original", style="green")
    table.add_column("Substituted", style="yellow")

    # Formulas
    table.add_row(
        "Formula", original.composition.formula, substituted.composition.formula
    )

    # Element counts
    orig_count = sum(1 for site in original if site.specie.symbol == old_elem.symbol)
    new_count = sum(1 for site in substituted if site.specie.symbol == new_elem.symbol)

    table.add_row(
        f"{old_elem.symbol} count",
        str(orig_count),
        str(orig_count - len(substituted_sites)),
    )
    table.add_row(
        f"{new_elem.symbol} count",
        "0" if new_count == len(substituted_sites) else "...",
        str(new_count),
    )

    # Sites modified
    table.add_row("—", "—", "—")
    table.add_row("Sites modified", "—", str(len(substituted_sites)))
    table.add_row(
        "Substitution mode",
        "—",
        mode.replace("-", " ").title(),
    )

    console.print(table)


def _display_changes(
    structure: Structure,
    substituted_sites: list,
    old_elem: Element,
    new_elem: Element,
) -> None:
    """Display which atoms were changed."""
    console.print("\n[cyan]Substituted Sites:[/cyan]")

    if len(substituted_sites) == 0:
        console.print("  No sites substituted")
        return

    # Show up to 10 sites
    n_show = min(10, len(substituted_sites))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Site Index", style="cyan")
    table.add_column("Position (Å)", style="green")
    table.add_column("Old → New", style="yellow")

    for i in range(n_show):
        site_idx = substituted_sites[i]
        site = structure[site_idx]
        pos = f"({site.coords[0]:.3f}, {site.coords[1]:.3f}, {site.coords[2]:.3f})"
        table.add_row(str(site_idx), pos, f"{old_elem.symbol} → {new_elem.symbol}")

    if len(substituted_sites) > n_show:
        table.add_row(
            "...",
            "...",
            f"... and {len(substituted_sites) - n_show} more",
        )

    console.print(table)


if __name__ == "__main__":
    substitute()
