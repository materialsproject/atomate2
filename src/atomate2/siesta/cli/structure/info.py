#!/usr/bin/env python
"""
Structure information CLI tool.

Displays comprehensive information about structure files including:
- Crystal system and space group
- Lattice parameters
- Atomic composition
- Magnetic properties
- Symmetry operations
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def load_structure(file_path: str) -> Structure:
    """Load structure from various file formats.

    Supports: CIF, XSF, POSCAR, XV, FDF

    Args:
        file_path: Path to structure file

    Returns:
        pymatgen Structure object
    """
    import sisl

    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".fdf", ".xv"}:
        # Use sisl to read FDF or XV files
        try:
            geom = sisl.get_sile(str(path)).read_geometry()
            # Convert to pymatgen via ASE (preserves properties)
            atoms = geom.to.ase()
            return Structure.from_ase_atoms(atoms)
        except Exception as e:
            raise ValueError(f"Failed to read {suffix.upper()} file: {e}") from e
    else:
        # Use pymatgen for CIF, XSF, POSCAR, etc.
        return Structure.from_file(str(path))


def analyze_magnetic_properties(structure: Structure) -> dict[str, Any]:
    """Analyze magnetic properties of structure.

    Args:
        structure: pymatgen Structure object

    Returns:
        Dictionary with magnetic analysis
    """
    from atomate2.siesta.sets.utils import get_default_initial_magnetic_moments

    # Check if structure has magmom property
    has_magmom = "magmom" in structure.site_properties

    if has_magmom:
        magmoms = structure.site_properties["magmom"]
    else:
        # Try automatic detection
        magmoms = get_default_initial_magnetic_moments(structure)
        # If detection returns None (no magnetic elements), create zeros
        if magmoms is None:
            magmoms = [0.0] * len(structure)

    # Analyze magnetic moments
    n_magnetic = sum(1 for m in magmoms if abs(m) > 1e-6)
    unique_moments = sorted(set(round(m, 3) for m in magmoms if abs(m) > 1e-6))

    # Determine magnetic ordering
    if n_magnetic == 0:
        ordering = "Non-magnetic"
    elif all(m > 0 for m in magmoms if abs(m) > 1e-6):
        ordering = "Ferromagnetic (FM)"
    elif all(m < 0 for m in magmoms if abs(m) > 1e-6):
        ordering = "Ferromagnetic (FM)"
    elif len(unique_moments) > 1 or any(m < 0 for m in magmoms):
        ordering = "Antiferromagnetic (AFM) / Complex"
    else:
        ordering = "Unknown"

    return {
        "has_magmom": has_magmom,
        "n_magnetic_sites": n_magnetic,
        "n_total_sites": len(structure),
        "unique_moments": unique_moments,
        "ordering": ordering,
        "magmoms": magmoms,
    }


@click.command()
@click.argument("structure_file", type=click.Path(exists=True))
@click.option(
    "--symprec",
    type=float,
    default=0.01,
    help="Symmetry precision for space group detection (default: 0.01 Å)",
)
@click.option(
    "--angle-tolerance",
    type=float,
    default=5.0,
    help="Angle tolerance for space group detection (default: 5 degrees)",
)
@click.option(
    "--magnetic/--no-magnetic",
    default=True,
    help="Analyze magnetic properties (default: True)",
)
@click.option(
    "--sites/--no-sites",
    default=False,
    help="Show atomic sites with coordinates and properties (default: False)",
)
@click.option(
    "--max-sites",
    type=int,
    default=50,
    help="Maximum number of sites to display (default: 50)",
)
def main(
    structure_file: str,
    symprec: float,
    angle_tolerance: float,
    magnetic: bool,
    sites: bool,
    max_sites: int,
):
    """Display comprehensive information about a structure file.

    Supports CIF, XSF, XV, FDF, POSCAR formats.

    Example:
        atomate2siesta-structure info structure.cif
        atomate2siesta-structure info siesta.fdf --symprec 0.001
        atomate2siesta-structure info structure.xsf --no-magnetic
    """
    console = Console()

    try:
        # Load structure
        with console.status(f"[bold blue]Loading {structure_file}..."):
            structure = load_structure(structure_file)

        # Analyze symmetry
        with console.status("[bold blue]Analyzing symmetry..."):
            sga = SpacegroupAnalyzer(
                structure, symprec=symprec, angle_tolerance=angle_tolerance
            )
            spacegroup = sga.get_space_group_symbol()
            spacegroup_number = sga.get_space_group_number()
            crystal_system = sga.get_crystal_system()
            point_group = sga.get_point_group_symbol()

        # Get conventional and primitive structures
        conventional = sga.get_conventional_standard_structure()
        try:
            primitive = sga.get_primitive_standard_structure()
        except Exception:
            primitive = structure

        # Create header panel
        header = (
            f"[bold cyan]Structure Information:[/bold cyan] {Path(structure_file).name}"
        )
        console.print(Panel(header, style="bold blue"))

        # Table 1: Crystal System
        table_crystal = Table(title="Crystal System & Symmetry", show_header=True)
        table_crystal.add_column("Property", style="cyan")
        table_crystal.add_column("Value", style="green")

        table_crystal.add_row("Formula", structure.composition.reduced_formula)
        table_crystal.add_row(
            "Full Formula", structure.composition.alphabetical_formula
        )
        table_crystal.add_row("Space Group", f"{spacegroup} (#{spacegroup_number})")
        table_crystal.add_row("Crystal System", crystal_system)
        table_crystal.add_row("Point Group", point_group)
        table_crystal.add_row("Number of Sites", str(len(structure)))
        table_crystal.add_row("Number of Sites (Primitive)", str(len(primitive)))
        table_crystal.add_row("Number of Sites (Conventional)", str(len(conventional)))

        console.print(table_crystal)
        console.print()

        # Table 2: Lattice Parameters
        table_lattice = Table(title="Lattice Parameters", show_header=True)
        table_lattice.add_column("Parameter", style="cyan")
        table_lattice.add_column("Value", style="green")

        lattice = structure.lattice
        table_lattice.add_row("a", f"{lattice.a:.6f} Å")
        table_lattice.add_row("b", f"{lattice.b:.6f} Å")
        table_lattice.add_row("c", f"{lattice.c:.6f} Å")
        table_lattice.add_row("α", f"{lattice.alpha:.4f}°")
        table_lattice.add_row("β", f"{lattice.beta:.4f}°")
        table_lattice.add_row("γ", f"{lattice.gamma:.4f}°")
        table_lattice.add_row("Volume", f"{lattice.volume:.6f} Ų")

        console.print(table_lattice)
        console.print()

        # Table 3: Atomic Composition
        table_composition = Table(title="Atomic Composition", show_header=True)
        table_composition.add_column("Element", style="cyan")
        table_composition.add_column("Count", style="green")
        table_composition.add_column("Atomic %", style="yellow")

        elem_dict = structure.composition.get_el_amt_dict()
        total_atoms = sum(elem_dict.values())

        for elem, count in sorted(elem_dict.items()):
            percentage = (count / total_atoms) * 100
            table_composition.add_row(elem, f"{int(count)}", f"{percentage:.2f}%")

        console.print(table_composition)
        console.print()

        # Table 4: Magnetic Properties (if requested)
        if magnetic:
            with console.status("[bold blue]Analyzing magnetic properties..."):
                mag_info = analyze_magnetic_properties(structure)

            table_magnetic = Table(title="Magnetic Properties", show_header=True)
            table_magnetic.add_column("Property", style="cyan")
            table_magnetic.add_column("Value", style="green")

            table_magnetic.add_row(
                "Has magmom Property",
                "Yes" if mag_info["has_magmom"] else "No (Auto-detected)",
            )
            table_magnetic.add_row(
                "Magnetic Sites",
                f"{mag_info['n_magnetic_sites']} / {mag_info['n_total_sites']}",
            )
            table_magnetic.add_row("Magnetic Ordering", mag_info["ordering"])

            if mag_info["unique_moments"]:
                moments_str = ", ".join(
                    f"{m:.3f} μB" for m in mag_info["unique_moments"]
                )
                table_magnetic.add_row("Unique Moments", moments_str)
            else:
                table_magnetic.add_row("Unique Moments", "None")

            console.print(table_magnetic)
            console.print()

            # Show per-atom magnetic moments if any exist
            if mag_info["n_magnetic_sites"] > 0:
                table_moments = Table(title="Magnetic Moments (μB)", show_header=True)
                table_moments.add_column("Site", style="cyan")
                table_moments.add_column("Element", style="yellow")
                table_moments.add_column("Moment", style="green")

                # Limit to 20 rows for display
                shown_count = 0
                for i, (site, moment) in enumerate(
                    zip(structure, mag_info["magmoms"], strict=False)
                ):
                    moment_val: float = float(moment)
                    if abs(moment_val) > 1e-6:  # Only show magnetic sites
                        if shown_count < 20:
                            table_moments.add_row(
                                str(i + 1),
                                site.specie.symbol,
                                f"{moment_val:+.3f}",
                            )
                            shown_count += 1

                if mag_info["n_magnetic_sites"] > 20:
                    console.print(
                        f"[yellow]Showing first 20 of {mag_info['n_magnetic_sites']} magnetic sites[/yellow]"
                    )

                console.print(table_moments)
                console.print()

        # Table 5: Atomic Sites (if requested)
        if sites:
            table_sites = Table(title="Atomic Sites", show_header=True)
            table_sites.add_column("idx", style="dim", justify="right")
            table_sites.add_column("Element", style="cyan")
            table_sites.add_column("x (frac)", style="green", justify="right")
            table_sites.add_column("y (frac)", style="green", justify="right")
            table_sites.add_column("z (frac)", style="green", justify="right")
            table_sites.add_column("x (Å)", style="yellow", justify="right")
            table_sites.add_column("y (Å)", style="yellow", justify="right")
            table_sites.add_column("z (Å)", style="yellow", justify="right")

            # Check for species_label property
            has_species_label = "species_label" in structure.site_properties
            if has_species_label:
                table_sites.add_column("Label", style="magenta")

            n_sites = len(structure)
            for i, site in enumerate(structure):
                if i >= max_sites:
                    break
                frac = site.frac_coords
                cart = site.coords
                row = [
                    str(i),  # 0-based index for Python manipulation
                    site.specie.symbol,
                    f"{frac[0]:.6f}",
                    f"{frac[1]:.6f}",
                    f"{frac[2]:.6f}",
                    f"{cart[0]:.4f}",
                    f"{cart[1]:.4f}",
                    f"{cart[2]:.4f}",
                ]
                if has_species_label:
                    label = structure.site_properties["species_label"][i]
                    row.append(str(label))
                table_sites.add_row(*row)

            if n_sites > max_sites:
                console.print(
                    f"[yellow]Showing first {max_sites} of {n_sites} sites. "
                    f"Use --max-sites to show more.[/yellow]"
                )

            console.print(table_sites)
            console.print()

        # Summary info
        console.print(
            f"[dim]Symmetry analysis parameters: symprec={symprec} Å, angle_tolerance={angle_tolerance}°[/dim]"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e!s}")
        raise click.Abort from e


if __name__ == "__main__":
    main()
