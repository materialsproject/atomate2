"""Compare two structures quantitatively.

This module provides the `compare` command for comparing two crystal structures.
"""
# ruff: noqa: RUF002 docstring uses intentional Greek crystallographic symbols

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
from pymatgen.core import Structure
from rich.console import Console
from rich.table import Table

console = Console()


def load_structure(file_path: str) -> Structure:
    """Load structure from file."""
    return Structure.from_file(file_path)


@click.command()
@click.argument("structure1", type=click.Path(exists=True))
@click.argument("structure2", type=click.Path(exists=True))
@click.option(
    "--tolerance",
    type=float,
    default=0.01,
    help="Tolerance for structure matching (Å, default: 0.01)",
)
@click.option(
    "--compare-lattice/--no-compare-lattice",
    default=True,
    help="Compare lattice parameters (default: True)",
)
@click.option(
    "--compare-sites/--no-compare-sites",
    default=True,
    help="Compare atomic sites (default: True)",
)
@click.option(
    "--calculate-rmsd/--no-calculate-rmsd",
    default=True,
    help="Calculate RMSD between structures (default: True)",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed site-by-site comparison",
)
def compare(
    structure1: str,
    structure2: str,
    tolerance: float,
    compare_lattice: bool,
    compare_sites: bool,
    calculate_rmsd: bool,
    verbose: bool,
) -> None:
    """Compare two crystal structures.

    Compares two structures by analyzing:
    - Lattice parameters (a, b, c, α, β, γ, volume)
    - Atomic composition and formula
    - Site positions and RMSD
    - Symmetry properties

    Examples
    --------
        # Basic comparison
        atomate2siesta-structure compare struct1.cif struct2.cif

        # With custom tolerance
        atomate2siesta-structure compare struct1.cif struct2.cif --tolerance 0.1

        # Lattice only
        atomate2siesta-structure compare struct1.cif struct2.cif --no-compare-sites

        # Detailed site-by-site comparison
        atomate2siesta-structure compare struct1.cif struct2.cif --verbose
    """
    try:
        # Load structures
        console.print("\n[bold cyan]Loading structures...[/bold cyan]")
        s1 = load_structure(structure1)
        s2 = load_structure(structure2)

        file1 = Path(structure1).name
        file2 = Path(structure2).name

        console.print(f"  Structure 1: {file1}")
        console.print(f"  Structure 2: {file2}\n")

        # Check if structures are the same
        if s1 == s2:
            console.print("[bold green]✓ Structures are identical![/bold green]\n")
            return

        # Compare basic properties
        _compare_basic_properties(s1, s2, file1, file2)

        # Compare lattice
        if compare_lattice:
            _compare_lattice(s1, s2, tolerance)

        # Compare composition
        _compare_composition(s1, s2)

        # Compare sites
        if compare_sites:
            _compare_sites(s1, s2, tolerance, verbose)

        # Calculate RMSD
        if calculate_rmsd:
            _calculate_rmsd(s1, s2, tolerance)

    except Exception as e:  # noqa: BLE001 friendly CLI error reporting
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        sys.exit(1)


def _compare_basic_properties(
    s1: Structure, s2: Structure, file1: str, file2: str
) -> None:
    """Compare basic structure properties."""
    table = Table(title="Basic Properties Comparison", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column(file1, style="yellow")
    table.add_column(file2, style="green")
    table.add_column("Match", style="magenta")

    # Number of sites
    n_sites_match = "✓" if s1.num_sites == s2.num_sites else "✗"
    table.add_row(
        "Number of Sites",
        str(s1.num_sites),
        str(s2.num_sites),
        n_sites_match,
    )

    # Formula
    formula_match = "✓" if s1.formula == s2.formula else "✗"
    table.add_row(
        "Chemical Formula",
        s1.formula,
        s2.formula,
        formula_match,
    )

    # Reduced formula
    reduced_match = (
        "✓" if s1.composition.reduced_formula == s2.composition.reduced_formula else "✗"
    )
    table.add_row(
        "Reduced Formula",
        s1.composition.reduced_formula,
        s2.composition.reduced_formula,
        reduced_match,
    )

    console.print(table)
    console.print()


def _compare_lattice(s1: Structure, s2: Structure, tolerance: float) -> None:
    """Compare lattice parameters."""
    table = Table(title="Lattice Parameters Comparison", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Structure 1", style="yellow")
    table.add_column("Structure 2", style="green")
    table.add_column("Difference", style="magenta")
    table.add_column("Match", style="white")

    lattice1 = s1.lattice
    lattice2 = s2.lattice

    # Lattice lengths (a, b, c)
    for param, val1, val2 in [
        ("a (Å)", lattice1.a, lattice2.a),
        ("b (Å)", lattice1.b, lattice2.b),
        ("c (Å)", lattice1.c, lattice2.c),
    ]:
        diff = abs(val1 - val2)
        match = "✓" if diff < tolerance else "✗"
        table.add_row(
            param,
            f"{val1:.6f}",
            f"{val2:.6f}",
            f"{diff:.6f}",
            match,
        )

    # Lattice angles (α, β, γ)  # noqa: RUF003
    for param, val1, val2 in [
        ("α (°)", lattice1.alpha, lattice2.alpha),  # noqa: RUF001
        ("β (°)", lattice1.beta, lattice2.beta),
        ("γ (°)", lattice1.gamma, lattice2.gamma),  # noqa: RUF001
    ]:
        diff = abs(val1 - val2)
        match = "✓" if diff < tolerance else "✗"
        table.add_row(
            param,
            f"{val1:.6f}",
            f"{val2:.6f}",
            f"{diff:.6f}",
            match,
        )

    # Volume
    vol_diff = abs(lattice1.volume - lattice2.volume)
    vol_match = "✓" if vol_diff < tolerance else "✗"
    table.add_row(
        "Volume (Ų)",
        f"{lattice1.volume:.6f}",
        f"{lattice2.volume:.6f}",
        f"{vol_diff:.6f}",
        vol_match,
    )

    console.print(table)
    console.print()


def _compare_composition(s1: Structure, s2: Structure) -> None:
    """Compare atomic composition."""
    table = Table(title="Composition Comparison", show_header=True)
    table.add_column("Element", style="cyan")
    table.add_column("Structure 1", style="yellow")
    table.add_column("Structure 2", style="green")
    table.add_column("Match", style="magenta")

    # Get all elements
    all_elements = set(s1.composition.elements) | set(s2.composition.elements)

    for element in sorted(all_elements, key=lambda e: e.Z):
        count1 = s1.composition[element]
        count2 = s2.composition[element]
        match = "✓" if count1 == count2 else "✗"

        table.add_row(
            str(element),
            f"{count1:.2f}" if count1 > 0 else "0",
            f"{count2:.2f}" if count2 > 0 else "0",
            match,
        )

    console.print(table)
    console.print()


def _compare_sites(
    s1: Structure, s2: Structure, tolerance: float, verbose: bool
) -> None:
    """Compare atomic sites."""
    if s1.num_sites != s2.num_sites:
        console.print(
            "[yellow]Warning: Structures have different number of sites. "
            "Skipping site comparison.[/yellow]\n"
        )
        return

    # Try to match sites and store all matches
    matched_sites = 0
    unmatched_sites = []
    all_site_matches = []  # Store all site matches for detailed view

    frac_coords1 = s1.frac_coords
    frac_coords2 = s2.frac_coords
    cart_coords1 = s1.cart_coords
    cart_coords2 = s2.cart_coords

    for i, (site1, frac1, cart1) in enumerate(
        zip(s1, frac_coords1, cart_coords1, strict=False)
    ):
        # Find closest site in structure 2
        min_dist = float("inf")
        best_match = None
        best_match_idx = None

        for j, (site2, frac2, _cart2) in enumerate(
            zip(s2, frac_coords2, cart_coords2, strict=False)
        ):
            if site1.specie != site2.specie:
                continue

            # Calculate distance (considering periodic boundary)
            dist = np.linalg.norm(s1.lattice.get_cartesian_coords(frac1 - frac2))
            if dist < min_dist:
                min_dist = dist
                best_match = (j, dist)
                best_match_idx = j

        if best_match and best_match[1] < tolerance:
            matched_sites += 1
            all_site_matches.append(
                (i, site1, frac1, cart1, best_match_idx, best_match[1], True)
            )
        else:
            unmatched_sites.append((i, site1, best_match))
            all_site_matches.append(
                (
                    i,
                    site1,
                    frac1,
                    cart1,
                    best_match_idx if best_match else None,
                    best_match[1] if best_match else None,
                    False,
                )
            )

    # Summary table
    table = Table(title="Site Comparison Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Total Sites", str(s1.num_sites))
    table.add_row(
        "Matched Sites", f"{matched_sites} ({matched_sites / s1.num_sites * 100:.1f}%)"
    )
    table.add_row("Unmatched Sites", str(len(unmatched_sites)))
    table.add_row("Tolerance", f"{tolerance} Å")

    console.print(table)
    console.print()

    # Detailed site-by-site comparison if verbose
    if verbose:
        detail_table = Table(
            title="Detailed Site-by-Site Comparison", show_header=True, show_lines=True
        )
        detail_table.add_column("Site", style="cyan", justify="right")
        detail_table.add_column("Element", style="yellow", justify="center")
        detail_table.add_column("Struct 1 (frac)", style="white")
        detail_table.add_column("Struct 2 (frac)", style="white")
        detail_table.add_column("Distance (Å)", style="magenta", justify="right")
        detail_table.add_column("Match", style="green", justify="center")

        for i, site1, frac1, _cart1, match_idx, dist, is_match in all_site_matches:
            if match_idx is not None:
                frac2 = frac_coords2[match_idx]
                frac2_str = f"({frac2[0]:.4f}, {frac2[1]:.4f}, {frac2[2]:.4f})"
                dist_str = f"{dist:.4f}" if dist is not None else "N/A"
            else:
                frac2_str = "No match"
                dist_str = "N/A"

            frac1_str = f"({frac1[0]:.4f}, {frac1[1]:.4f}, {frac1[2]:.4f})"
            match_str = "✓" if is_match else "✗"

            # Color code the row based on match status
            if is_match:
                style = "green"
            elif dist is not None and dist < tolerance * 2:
                style = "yellow"
            else:
                style = "red"

            detail_table.add_row(
                str(i),
                str(site1.specie),
                frac1_str,
                frac2_str,
                dist_str,
                f"[{style}]{match_str}[/{style}]",
            )

        console.print(detail_table)
        console.print()

    # Show unmatched sites summary (always show if there are unmatched sites)
    elif unmatched_sites:
        console.print("[bold yellow]Unmatched Sites Summary:[/bold yellow]")
        for i, site, match in unmatched_sites[:10]:  # Show first 10
            if match:
                console.print(
                    f"  Site {i} ({site.specie}): closest match at "
                    f"{match[1]:.4f} Å (exceeds tolerance)"
                )
            else:
                console.print(f"  Site {i} ({site.specie}): no matching element found")
        if len(unmatched_sites) > 10:
            console.print(f"  ... and {len(unmatched_sites) - 10} more")
        console.print(
            "\n[dim]Use --verbose for detailed site-by-site comparison[/dim]\n"
        )


def _calculate_rmsd(s1: Structure, s2: Structure, tolerance: float) -> None:
    """Calculate RMSD between structures."""
    if s1.num_sites != s2.num_sites:
        console.print(
            "[yellow]Warning: Cannot calculate RMSD for structures "
            "with different number of sites.[/yellow]\n"
        )
        return

    if s1.composition.reduced_formula != s2.composition.reduced_formula:
        console.print(
            "[yellow]Warning: Structures have different compositions. "
            "RMSD may not be meaningful.[/yellow]\n"
        )

    try:
        # Get Cartesian coordinates
        coords1 = s1.cart_coords
        coords2 = s2.cart_coords

        # Try to match sites by element
        matched_coords1 = []
        matched_coords2 = []

        for site1 in s1:
            # Find matching site in structure 2
            for j, site2 in enumerate(s2):
                if site1.specie == site2.specie:
                    matched_coords1.append(coords1[s1.index(site1)])
                    matched_coords2.append(coords2[j])
                    break

        if len(matched_coords1) == 0:
            console.print(
                "[yellow]Warning: No matching sites found. "
                "Cannot calculate RMSD.[/yellow]\n"
            )
            return

        # Calculate RMSD
        coords1_array = np.array(matched_coords1)
        coords2_array = np.array(matched_coords2)

        # Center coordinates
        coords1_centered = coords1_array - coords1_array.mean(axis=0)
        coords2_centered = coords2_array - coords2_array.mean(axis=0)

        # Calculate RMSD
        rmsd = np.sqrt(
            np.mean(np.sum((coords1_centered - coords2_centered) ** 2, axis=1))
        )

        # Display results
        table = Table(title="RMSD Analysis", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("RMSD", f"{rmsd:.6f} Å")
        table.add_row("Matched Sites", str(len(matched_coords1)))
        table.add_row("Within Tolerance", "✓" if rmsd < tolerance else "✗")

        console.print(table)
        console.print()

        if rmsd < tolerance:
            console.print(
                f"[bold green]✓ Structures are similar "
                f"(RMSD < {tolerance} Å)[/bold green]\n"
            )
        else:
            console.print(
                f"[bold yellow]⚠ Structures differ "
                f"(RMSD = {rmsd:.6f} Å > {tolerance} Å)[/bold yellow]\n"
            )

    except Exception as e:  # noqa: BLE001 friendly warning, RMSD is best-effort
        console.print(f"[yellow]Warning: RMSD calculation failed: {e}[/yellow]\n")


if __name__ == "__main__":
    compare()
