"""CLI for generating surface slabs.

This module provides the `slab` subcommand for atomate2siesta-structure.
"""

from __future__ import annotations

import click
from pymatgen.core import Structure
from pymatgen.core.surface import (
    SlabGenerator,
    get_symmetrically_distinct_miller_indices,
)
from rich.console import Console
from rich.table import Table

console = Console()


@click.command()
@click.argument("structure_file", type=click.Path(exists=True))
@click.option(
    "--miller",
    type=str,
    help="Miller indices as comma-separated h,k,l (e.g., '1,1,1' or '1,0,0')",
)
@click.option(
    "--layers",
    type=int,
    default=5,
    help="Minimum number of unit planes along surface normal (default: 5). "
    "Result may have more atomic layers depending on structure.",
)
@click.option(
    "--vacuum",
    type=float,
    default=15.0,
    help="Vacuum thickness in Å (default: 15.0)",
)
@click.option(
    "--min-slab-size",
    type=float,
    help="Minimum slab thickness in Å (alternative to --layers)",
)
@click.option(
    "--termination",
    type=int,
    help="Specific termination index (0-based, use --list-terminations to see options)",
)
@click.option(
    "--list-terminations",
    is_flag=True,
    help="List all possible terminations without generating slabs",
)
@click.option(
    "--all-terminations",
    is_flag=True,
    help="Generate all possible terminations",
)
@click.option(
    "--all-surfaces",
    is_flag=True,
    help="Generate all symmetry-unique surfaces up to --max-index",
)
@click.option(
    "--max-index",
    type=int,
    default=1,
    help="Maximum Miller index for --all-surfaces (default: 1)",
)
@click.option(
    "--symmetric/--no-symmetric",
    default=True,
    help="Generate symmetric slabs (default: --symmetric)",
)
@click.option(
    "--center-slab/--no-center-slab",
    default=True,
    help="Center slab in cell (default: --center-slab)",
)
@click.option(
    "--in-unit-planes",
    is_flag=True,
    help="Use unit planes for layer counting (auto-enabled when using --layers)",
)
@click.option(
    "--primitive",
    is_flag=True,
    help="Generate primitive slab (default: conventional)",
)
@click.option(
    "--orthogonal",
    is_flag=True,
    help="Make slab orthogonal (orthogonalize in-plane lattice vectors)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file (default: slab_<hkl>_<input>)",
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
    help="Show layer information and atomic positions",
)
@click.option(
    "--vdw-layers",
    type=int,
    help="Number of van der Waals layers for layered materials (e.g., MoS2 trilayers). "
    "Uses vdW gap detection to ensure complete layers with no extra atoms.",
)
def slab(
    structure_file: str,
    miller: str | None,
    layers: int,
    vacuum: float,
    min_slab_size: float | None,
    termination: int | None,
    list_terminations: bool,
    all_terminations: bool,
    all_surfaces: bool,
    max_index: int,
    symmetric: bool,
    center_slab: bool,
    in_unit_planes: bool,
    primitive: bool,
    orthogonal: bool,
    output: str | None,
    format: str,  # noqa: A002 Click --format option name
    show_layers: bool,
    vdw_layers: int | None,
) -> None:
    """Generate surface slabs for adsorption and catalysis studies.

    Supports Miller index specification, termination discovery, vacuum control,
    and automatic generation of all symmetry-unique surfaces.

    Examples
    --------
        # Basic (111) surface with 5 layers and 15 Å vacuum
        atomate2siesta-structure slab bulk.cif --miller 1,1,1

        # (100) surface with specific parameters
        atomate2siesta-structure slab bulk.cif --miller 1,0,0 --layers 7 --vacuum 20

        # List all possible terminations first
        atomate2siesta-structure slab bulk.cif --miller 1,1,1 --list-terminations

        # Generate specific termination
        atomate2siesta-structure slab bulk.cif --miller 1,1,1 --termination 1

        # Generate all terminations
        atomate2siesta-structure slab bulk.cif --miller 1,1,1 --all-terminations

        # Generate all low-index surfaces
        atomate2siesta-structure slab bulk.cif --all-surfaces --max-index 1

        # Minimum slab thickness instead of layer count
        atomate2siesta-structure slab bulk.cif --miller 1,1,1 --min-slab-size 10.0
    """
    # Validate options
    if vdw_layers is not None:
        # --vdw-layers doesn't need --miller (always uses basal plane)
        pass
    elif all_surfaces:
        if miller:
            console.print(
                "[yellow]Warning: --miller ignored when using --all-surfaces[/yellow]"
            )
    elif not miller and not all_surfaces:
        console.print(
            "[red]Error: Must specify either --miller, --all-surfaces, "
            "or --vdw-layers[/red]"
        )
        raise click.Abort

    if termination is not None and all_terminations:
        console.print(
            "[red]Error: Cannot use both --termination and --all-terminations[/red]"
        )
        raise click.Abort

    # When using --layers (not --min-slab-size), automatically enable in_unit_planes
    # so that layers are counted correctly instead of being treated as Angstroms
    auto_enabled_unit_planes = False
    if not min_slab_size and not in_unit_planes:
        in_unit_planes = True
        auto_enabled_unit_planes = True

    try:
        # Load structure
        structure = Structure.from_file(structure_file)
        console.print(
            f"\n[cyan]Loaded bulk structure: "
            f"{structure.composition.reduced_formula}[/cyan]"
        )
        console.print(f"  Formula: {structure.composition.formula}")
        console.print(f"  Sites: {structure.num_sites}")
        console.print(f"  Space Group: {structure.get_space_group_info()[0]}")

        # Store original vacuum in Angstroms for display
        vacuum_angstrom = vacuum

        # Handle --vdw-layers for layered materials (MoS2, graphene, etc.)
        if vdw_layers is not None:
            console.print(
                f"\n[yellow]Generating {vdw_layers} vdW layers "
                f"for layered material[/yellow]"
            )
            slab_struct = _generate_vdw_slab(structure, vdw_layers, vacuum)

            if slab_struct is None:
                console.print(
                    "[red]Error: Could not detect vdW layers. "
                    "Structure may not be a layered material.[/red]"
                )
                raise click.Abort  # noqa: TRY301

            # Display info
            coords = slab_struct.cart_coords
            slab_thickness = coords[:, 2].max() - coords[:, 2].min()
            actual_vacuum = slab_struct.lattice.c - slab_thickness

            console.print("\n[cyan]vdW Slab Generated:[/cyan]")
            console.print(f"  Formula: {slab_struct.composition.formula}")
            console.print(f"  Atoms: {slab_struct.num_sites}")
            console.print(f"  vdW layers: {vdw_layers}")
            console.print(f"  Slab thickness: {slab_thickness:.3f} Å")
            console.print(f"  Vacuum: {actual_vacuum:.3f} Å")

            # Determine output filename
            if output:
                output_file = output
            else:
                from pathlib import Path

                input_path = Path(structure_file)
                output_file = f"slab_vdw{vdw_layers}_{input_path.stem}.{format}"

            _save_structure(slab_struct, output_file, format)
            console.print(f"\n[green]✓ Slab saved to: {output_file}[/green]")
            return

        if all_surfaces:
            # Generate all symmetry-unique surfaces
            console.print(
                f"\n[yellow]Generating all surfaces up to Miller index "
                f"{max_index}[/yellow]"
            )

            indices = get_symmetrically_distinct_miller_indices(structure, max_index)
            console.print(f"  Found {len(indices)} symmetry-unique surfaces")

            generated_count = 0
            for hkl in indices:
                miller_h, miller_k, miller_l = hkl
                console.print(
                    f"\n[cyan]Surface ({miller_h},{miller_k},{miller_l}):[/cyan]"
                )

                # When in_unit_planes=True, convert vacuum from Angstroms to planes
                effective_vacuum = vacuum
                if in_unit_planes:
                    d_hkl = structure.lattice.d_hkl(hkl)
                    effective_vacuum = vacuum_angstrom / d_hkl
                    console.print(
                        f"  d_hkl = {d_hkl:.3f} Å, vacuum = "
                        f"{effective_vacuum:.1f} planes ({vacuum_angstrom:.1f} Å)"
                    )

                # Generate slab
                slabgen = SlabGenerator(
                    structure,
                    hkl,
                    min_slab_size=min_slab_size or layers,
                    min_vacuum_size=effective_vacuum,
                    in_unit_planes=in_unit_planes,
                    center_slab=center_slab,
                    primitive=primitive,
                )

                slabs = slabgen.get_slabs(
                    bonds=None, ftol=0.1, tol=0.1, symmetrize=symmetric
                )

                console.print(f"  Generated {len(slabs)} slab(s)")

                # Save first slab of each surface
                if slabs:
                    slab_struct = slabs[0]

                    # Orthogonalize if requested
                    if orthogonal:
                        slab_struct = slab_struct.get_orthogonal_c_slab()

                    from pathlib import Path

                    input_path = Path(structure_file)
                    output_file = (
                        f"slab_{miller_h}{miller_k}{miller_l}_"
                        f"{input_path.stem}.{format}"
                    )

                    _save_structure(slab_struct, output_file, format)
                    console.print(f"  Saved: {output_file}")
                    generated_count += 1

            console.print(
                f"\n[green]✓ Generated {generated_count} surface slabs[/green]"
            )
            return

        # Parse Miller indices
        hkl = tuple(int(x) for x in miller.split(","))
        if len(hkl) != 3:
            console.print("[red]Error: Miller indices must be h,k,l (3 values)[/red]")
            raise click.Abort  # noqa: TRY301

        console.print(
            f"\n[yellow]Generating ({hkl[0]},{hkl[1]},{hkl[2]}) surface[/yellow]"
        )

        # Calculate d_hkl spacing
        d_hkl = structure.lattice.d_hkl(hkl)

        # When in_unit_planes=True, convert vacuum from Angstroms to planes
        effective_vacuum = vacuum
        if in_unit_planes:
            effective_vacuum = vacuum_angstrom / d_hkl

        # Display generation parameters
        console.print("\n[cyan]Slab Generation Parameters:[/cyan]")
        console.print(f"  d_hkl spacing: {d_hkl:.3f} Å")
        if min_slab_size:
            console.print(f"  Minimum slab size: {min_slab_size:.2f} Å")
        else:
            expected_thickness = layers * d_hkl
            console.print(
                f"  Minimum unit planes: {layers} (~{expected_thickness:.1f} Å)"
            )
            if auto_enabled_unit_planes:
                console.print(
                    "  [dim](in_unit_planes auto-enabled for layer counting)[/dim]"
                )
        if in_unit_planes:
            console.print(
                f"  Vacuum: {effective_vacuum:.1f} planes (~{vacuum_angstrom:.1f} Å)"
            )
        else:
            console.print(f"  Vacuum thickness: {vacuum:.2f} Å")
        console.print(f"  In unit planes: {in_unit_planes}")
        console.print(f"  Symmetric: {symmetric}")
        console.print(f"  Center slab: {center_slab}")
        console.print(f"  Primitive: {primitive}")
        if orthogonal:
            console.print(f"  Orthogonalize: {orthogonal}")

        # Create SlabGenerator
        slabgen = SlabGenerator(
            structure,
            hkl,
            min_slab_size=min_slab_size or layers,
            min_vacuum_size=effective_vacuum,
            in_unit_planes=in_unit_planes,
            center_slab=center_slab,
            primitive=primitive,
        )

        # Get all slabs (different terminations)
        slabs = slabgen.get_slabs(bonds=None, ftol=0.1, tol=0.1, symmetrize=symmetric)

        console.print("\n[cyan]Results:[/cyan]")
        console.print(f"  Found {len(slabs)} possible termination(s)")

        if list_terminations:
            # Just list terminations
            _display_terminations(slabs, structure, hkl)
            return

        # Determine which slabs to generate
        if all_terminations:
            slabs_to_generate = list(enumerate(slabs))
        elif termination is not None:
            if termination >= len(slabs):
                console.print(
                    f"[red]Error: Termination {termination} not found. "
                    f"Available: 0-{len(slabs) - 1}[/red]"
                )
                raise click.Abort  # noqa: TRY301
            slabs_to_generate = [(termination, slabs[termination])]
        else:
            # Generate first termination by default
            slabs_to_generate = [(0, slabs[0])]

        # Generate and save slabs
        for idx, slab_struct in slabs_to_generate:
            # Orthogonalize if requested
            if orthogonal:
                slab_struct = slab_struct.get_orthogonal_c_slab()  # noqa: PLW2901

            # Display slab information
            _display_slab_info(slab_struct, structure, hkl, idx, show_layers)

            # Determine output filename
            if output and len(slabs_to_generate) == 1:
                output_file = output
            else:
                from pathlib import Path

                input_path = Path(structure_file)
                miller_h, miller_k, miller_l = hkl
                if all_terminations or len(slabs_to_generate) > 1:
                    output_file = (
                        f"slab_{miller_h}{miller_k}{miller_l}_term{idx}_"
                        f"{input_path.stem}.{format}"
                    )
                else:
                    output_file = (
                        f"slab_{miller_h}{miller_k}{miller_l}_"
                        f"{input_path.stem}.{format}"
                    )

            # Save structure
            _save_structure(slab_struct, output_file, format)
            console.print(f"\n[green]✓ Slab saved to: {output_file}[/green]")

        # Usage tips
        if len(slabs) > 1 and not all_terminations and termination is None:
            console.print(
                f"\n[dim]Tip: Found {len(slabs)} terminations. "
                f"Use --list-terminations to see all, "
                f"or --all-terminations to generate all[/dim]"
            )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort from e


def _save_structure(structure: Structure, filename: str, fmt: str) -> None:
    """Save structure to file."""
    if fmt == "cif":
        structure.to(filename=filename, fmt="cif")
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
        # Convert to sisl geometry and write FDF
        import sisl

        geom = sisl.get_sile(structure).read_geometry()
        with sisl.get_sile(filename, "w") as fdf:
            fdf.write_geometry(geom)
    elif fmt == "XV":
        # Convert to sisl geometry and write XV
        import sisl

        geom = sisl.get_sile(structure).read_geometry()
        geom.write(filename)


def _display_terminations(
    slabs: list,
    bulk_structure: Structure,  # noqa: ARG001 kept for signature parity
    hkl: tuple,
) -> None:
    """Display all possible terminations."""
    console.print(
        f"\n[cyan]Available Terminations for ({hkl[0]},{hkl[1]},{hkl[2]}):[/cyan]"
    )

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Index", style="cyan")
    table.add_column("Layers", style="green")
    table.add_column("Atoms", style="yellow")
    table.add_column("Thickness (Å)", style="blue")
    table.add_column("Surface Energy Site", style="red")

    for idx, slab in enumerate(slabs):
        # Calculate slab thickness (c-direction)
        coords = slab.cart_coords
        thickness = coords[:, 2].max() - coords[:, 2].min()

        # Get surface termination info
        surface_sites = _get_surface_sites(slab)
        top_species = ", ".join(sorted({str(s.specie) for s in surface_sites[:3]}))

        table.add_row(
            str(idx),
            str(len(slab.sites)),
            str(slab.num_sites),
            f"{thickness:.3f}",
            top_species,
        )

    console.print(table)
    console.print("\n[dim]Use --termination N to generate specific termination[/dim]")
    console.print("[dim]Use --all-terminations to generate all[/dim]")


def _display_slab_info(
    slab: Structure,
    bulk: Structure,
    hkl: tuple,  # noqa: ARG001 kept for signature parity
    term_idx: int,
    show_layers: bool,
) -> None:
    """Display detailed slab information."""
    console.print(f"\n[yellow]Slab Information (Termination {term_idx}):[/yellow]")

    # Calculate metrics
    coords = slab.cart_coords
    slab_thickness = coords[:, 2].max() - coords[:, 2].min()
    cell_c = slab.lattice.c
    vacuum_thickness = cell_c - slab_thickness

    # Count layers (group atoms by z-coordinate with tolerance 0.5 Å)
    z_coords = coords[:, 2]
    unique_z: list[float] = []
    for z in sorted(z_coords):
        if not unique_z or abs(z - unique_z[-1]) > 0.5:
            unique_z.append(z)
    n_layers = len(unique_z)

    # Create info table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Bulk", style="green")
    table.add_column("Slab", style="yellow")

    table.add_row("Formula", bulk.composition.formula, slab.composition.formula)
    table.add_row("Sites", str(bulk.num_sites), str(slab.num_sites))
    table.add_row("Lattice a (Å)", f"{bulk.lattice.a:.3f}", f"{slab.lattice.a:.3f}")
    table.add_row("Lattice b (Å)", f"{bulk.lattice.b:.3f}", f"{slab.lattice.b:.3f}")
    table.add_row("Lattice c (Å)", f"{bulk.lattice.c:.3f}", f"{slab.lattice.c:.3f}")
    table.add_row("—", "—", "—")
    table.add_row("Number of layers", "—", str(n_layers))
    table.add_row("Slab thickness", "—", f"{slab_thickness:.3f} Å")
    table.add_row("Vacuum thickness", "—", f"{vacuum_thickness:.3f} Å")

    console.print(table)

    # Surface composition
    surface_sites = _get_surface_sites(slab)
    if surface_sites:
        species_count: dict[str, int] = {}
        for site in surface_sites[:5]:  # Top 5 surface atoms
            species = str(site.specie)
            species_count[species] = species_count.get(species, 0) + 1

        console.print("\n[cyan]Top surface composition:[/cyan]")
        for species, count in sorted(species_count.items()):
            console.print(f"  {species}: {count} atom(s)")

    # Layer information
    if show_layers:
        _display_layer_info(slab)


def _get_surface_sites(slab: Structure) -> list:
    """Get sites at the top surface."""
    coords = slab.cart_coords
    max_z = coords[:, 2].max()
    # Sites within 0.5 Å of top surface
    surface_indices = [i for i, c in enumerate(coords) if abs(c[2] - max_z) < 0.5]
    return [slab[i] for i in surface_indices]


def _display_layer_info(slab: Structure) -> None:
    """Display layer-by-layer information."""
    console.print("\n[cyan]Layer Information:[/cyan]")

    # Group atoms by z-coordinate (layers)
    coords = slab.cart_coords
    z_coords = coords[:, 2]

    # Find unique layers (tolerance 0.5 Å)
    unique_z: list[float] = []
    for z in sorted(z_coords):
        if not unique_z or abs(z - unique_z[-1]) > 0.5:
            unique_z.append(z)

    console.print(f"  Found {len(unique_z)} layers")

    # Display first 5 and last 5 layers
    n_show = min(5, len(unique_z))
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Layer", style="cyan")
    table.add_column("Z (Å)", style="green")
    table.add_column("Atoms", style="yellow")
    table.add_column("Species", style="blue")

    for i in range(n_show):
        z = unique_z[i]
        layer_indices = [j for j, zz in enumerate(z_coords) if abs(zz - z) < 0.5]
        layer_sites = [slab[j] for j in layer_indices]
        species = ", ".join(sorted({str(s.specie) for s in layer_sites}))

        table.add_row(str(i), f"{z:.3f}", str(len(layer_indices)), species)

    if len(unique_z) > 2 * n_show:
        table.add_row("...", "...", "...", "...")

    if len(unique_z) > n_show:
        for i in range(max(n_show, len(unique_z) - n_show), len(unique_z)):
            z = unique_z[i]
            layer_indices = [j for j, zz in enumerate(z_coords) if abs(zz - z) < 0.5]
            layer_sites = [slab[j] for j in layer_indices]
            species = ", ".join(sorted({str(s.specie) for s in layer_sites}))

            table.add_row(str(i), f"{z:.3f}", str(len(layer_indices)), species)

    console.print(table)


def _generate_vdw_slab(
    bulk: Structure, n_layers: int, vacuum: float
) -> Structure | None:
    """Generate slab for layered vdW materials with complete layers only.

    This function detects vdW gaps in the bulk structure and creates slabs
    with exactly the specified number of complete layers (e.g., MoS2 trilayers).
    No extra atoms are added at surfaces.

    Args:
        bulk: Bulk Structure object
        n_layers: Number of vdW layers (e.g., MoS2 trilayers)
        vacuum: Vacuum thickness in Angstroms

    Returns
    -------
        Structure object for the slab, or None if vdW layers cannot be detected
    """
    import numpy as np
    from pymatgen.core import Lattice

    # Get lattice parameters
    a, b = bulk.lattice.a, bulk.lattice.b
    gamma = bulk.lattice.gamma

    # Find vdW gaps by analyzing z-positions
    z_coords = sorted([site.coords[2] for site in bulk])
    if len(z_coords) < 2:
        return None

    # Calculate gaps between adjacent atoms
    gaps = [(z_coords[i + 1] - z_coords[i], i) for i in range(len(z_coords) - 1)]
    gaps.sort(reverse=True)

    # The largest gap is the vdW gap (must be > 2.5 Å to be considered vdW)
    vdw_gap = gaps[0][0]
    if vdw_gap < 2.5:
        console.print(
            f"[yellow]Warning: Largest gap ({vdw_gap:.2f} Å) "
            f"may not be a vdW gap[/yellow]"
        )

    # Replicate bulk enough times to have enough layers
    n_vdw_in_bulk = sum(1 for g, _ in gaps if g > 2.5) + 1
    if n_vdw_in_bulk == 0:
        n_vdw_in_bulk = 1
    n_repeat = int(np.ceil(n_layers / n_vdw_in_bulk)) + 1

    supercell = bulk.copy()
    supercell.make_supercell([1, 1, n_repeat])

    # Group atoms into vdW layers by z-coordinate
    atoms_by_z = sorted(
        [
            (site.coords[2], site.specie, site.coords[:2], i)
            for i, site in enumerate(supercell)
        ],
        key=lambda x: x[0],
    )

    # Find layer groups (atoms separated by vdW gaps)
    vdw_layers = []
    current_layer = []
    prev_z = None

    for z, specie, xy, idx in atoms_by_z:
        if prev_z is not None and (z - prev_z) > 2.5:  # vdW gap threshold
            if current_layer:
                vdw_layers.append(current_layer)
            current_layer = []
        current_layer.append((z, specie, xy, idx))
        prev_z = z
    if current_layer:
        vdw_layers.append(current_layer)

    if len(vdw_layers) < n_layers:
        console.print(
            f"[red]Error: Only found {len(vdw_layers)} vdW layers, "
            f"need {n_layers}[/red]"
        )
        return None

    # Select n_layers from center
    start_idx = (len(vdw_layers) - n_layers) // 2
    selected_layers = vdw_layers[start_idx : start_idx + n_layers]

    # Calculate slab thickness
    all_z = [z for layer in selected_layers for z, _, _, _ in layer]
    z_min, z_max = min(all_z), max(all_z)
    slab_thickness = z_max - z_min

    # New cell c parameter
    slab_c = slab_thickness + vacuum

    # Build new structure
    new_lattice = Lattice.from_parameters(a, b, slab_c, 90, 90, gamma)

    # Center slab in cell
    z_shift = (slab_c - slab_thickness) / 2 - z_min

    new_species = []
    new_coords = []
    for layer in selected_layers:
        for z, specie, xy, _ in layer:
            new_species.append(specie)
            new_z = z + z_shift
            new_coords.append([xy[0], xy[1], new_z])

    return Structure(new_lattice, new_species, new_coords, coords_are_cartesian=True)


if __name__ == "__main__":
    slab()
