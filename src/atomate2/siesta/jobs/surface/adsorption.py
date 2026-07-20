"""Jobs for adsorption site scanning and optimization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from ase.build import add_adsorbate
from jobflow import job
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from atomate2.siesta.schemas.adsorption import (
    AdsorptionScanDocument,
    AdsorptionSiteResult,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def add_adsorbate_to_slab(
    slab: Structure,
    adsorbate: Structure | Molecule,
    position: tuple[float, float],
    height: float,
    placement: str = "top",
) -> Structure:
    """
    Add adsorbate to slab at specified position.

    Parameters
    ----------
    slab : Structure
        Slab structure.
    adsorbate : Structure | Molecule
        Adsorbate structure or molecule.
    position : tuple[float, float]
        (x, y) position in fractional coordinates.
    height : float
        Height above surface in Angstroms.
    placement : str
        Placement of adsorbate: 'top' or 'bottom' of slab (default: 'top').

    Returns
    -------
    Structure
        Combined slab+adsorbate structure.
    """
    adaptor = AseAtomsAdaptor()

    # Convert to ASE
    slab_ase = adaptor.get_atoms(slab)

    # Convert adsorbate
    if isinstance(adsorbate, Molecule):
        # For molecules, just use the sites directly
        ads_ase = adaptor.get_atoms(adsorbate)
    else:
        ads_ase = adaptor.get_atoms(adsorbate)

    # Convert fractional position to Cartesian
    x_frac, y_frac = position
    cell = slab_ase.get_cell()
    cart_pos = np.dot([x_frac, y_frac, 0], cell)
    position_cartesian = (cart_pos[0], cart_pos[1])

    if placement == "top":
        # Standard top placement
        add_adsorbate(slab_ase, ads_ase, height, position=position_cartesian)
    elif placement == "bottom":
        # Bottom placement: flip slab, add adsorbate, flip back
        # Get z-positions of all slab atoms
        slab_positions = slab_ase.get_positions()
        z_positions = slab_positions[:, 2]
        z_min = z_positions.min()
        z_max = z_positions.max()

        # Flip slab so bottom becomes top
        slab_ase.positions[:, 2] = z_max + z_min - slab_ase.positions[:, 2]

        # Add adsorbate to what is now the "top" (original bottom)
        add_adsorbate(slab_ase, ads_ase, height, position=position_cartesian)

        # Flip everything back
        slab_ase.positions[:, 2] = z_max + z_min + height - slab_ase.positions[:, 2]
    else:
        raise ValueError(f"Invalid placement '{placement}'. Must be 'top' or 'bottom'.")

    # Convert back to pymatgen
    return adaptor.get_structure(slab_ase)


@job
def generate_adsorption_sites(
    grid_size: tuple[int, int] = (4, 4),
) -> list[tuple[float, float]]:
    """
    Generate grid of adsorption sites.

    Uses fractional coordinates for universal applicability to both
    orthogonal and non-orthogonal cells (e.g., hexagonal slabs).

    Parameters
    ----------
    grid_size : tuple[int, int]
        Grid dimensions (nx, ny).

    Returns
    -------
    list[tuple[float, float]]
        List of (x, y) positions in fractional coordinates.
    """
    total_sites = grid_size[0] * grid_size[1]
    logger.info(
        f"[PROGRESS] Generating {total_sites} adsorption sites ({grid_size[0]}×{grid_size[1]} grid)"
    )

    nx, ny = grid_size

    # Generate grid points at the CENTER of each grid cell for full coverage
    x_positions = (np.arange(nx) + 0.5) / nx
    y_positions = (np.arange(ny) + 0.5) / ny

    sites = []
    for x in x_positions:
        for y in y_positions:
            sites.append((x, y))

    logger.info(f"Generated {len(sites)} adsorption sites")
    return sites


@job
def calculate_adsorption_energy_single_site(
    slab: Structure,
    adsorbate: Structure | Molecule,
    site: tuple[float, float],
    height: float,
    slab_energy: float,
    adsorbate_energy: float,
    slab_maker,
) -> AdsorptionSiteResult:
    """
    Calculate adsorption energy at a single site.

    Parameters
    ----------
    slab : Structure
        Clean slab structure.
    adsorbate : Structure | Molecule
        Adsorbate structure.
    site : tuple[float, float]
        (x, y) position in fractional coordinates.
    height : float
        Initial height above surface (Å).
    slab_energy : float
        Pre-calculated energy of clean slab (eV).
    adsorbate_energy : float
        Pre-calculated energy of isolated adsorbate (eV).
    slab_maker : Maker
        Maker for calculating slab+adsorbate energy.

    Returns
    -------
    AdsorptionSiteResult
        Adsorption results for this site.
    """
    x_frac, y_frac = site
    logger.info(f"Calculating adsorption energy at site ({x_frac:.3f}, {y_frac:.3f})")

    # Note: This is a placeholder function.
    # In a real workflow, this would create slab+adsorbate system and calculate energy.
    # For now, this needs to be filled in by the workflow.
    raise NotImplementedError(
        "This job needs to be called from a workflow that provides the total energy"
    )


@job
def analyze_adsorption_scan(
    slab: Structure,
    adsorbate: Structure | Molecule,
    site_energies: list[dict],
    slab_energy: float,
    adsorbate_energy: float,
    grid_size: tuple[int, int],
    heights: list[float],  # Changed from single height to list
    miller_indices: tuple[int, int, int] | None = None,
) -> AdsorptionScanDocument:
    """
    Analyze all adsorption site scan results across multiple heights.

    Parameters
    ----------
    slab : Structure
        Clean slab structure.
    adsorbate : Structure | Molecule
        Adsorbate structure.
    site_energies : list[dict]
        List of dictionaries with 'site', 'height', 'total_energy' for each site at each height.
    slab_energy : float
        Energy of clean slab (eV).
    adsorbate_energy : float
        Energy of isolated adsorbate (eV).
    grid_size : tuple[int, int]
        Grid size used for scanning.
    heights : list[float]
        List of heights that were scanned (Å).
    miller_indices : tuple[int, int, int], optional
        Miller indices of surface.

    Returns
    -------
    AdsorptionScanDocument
        Complete adsorption scan results.
    """
    total_sites = len(site_energies)
    n_heights = len(heights)
    xy_sites = grid_size[0] * grid_size[1]
    logger.info(
        f"[PROGRESS] Analyzing {total_sites} results ({xy_sites} xy sites × {n_heights} heights)"
    )

    # Calculate surface area and slab thickness
    cell = slab.lattice.matrix
    surface_area = np.linalg.norm(np.cross(cell[0], cell[1]))

    # Calculate slab thickness (extent in z-direction)
    adaptor = AseAtomsAdaptor()
    slab_ase = adaptor.get_atoms(slab)
    z_positions = slab_ase.get_positions()[:, 2]
    slab_thickness = float(z_positions.max() - z_positions.min())

    # Process each site (now includes height information)
    site_results = []
    for site_dict in site_energies:
        site = site_dict["site"]
        height = site_dict["height"]  # Get height from the site_dict
        total_energy = site_dict["total_energy"]

        x_frac, y_frac = site

        # Calculate Cartesian position
        cart_pos = np.dot([x_frac, y_frac, 0], cell)

        # Adsorption energy: E_ads = E_total - E_slab - E_adsorbate
        adsorption_energy = total_energy - slab_energy - adsorbate_energy
        adsorption_energy_per_area = adsorption_energy / surface_area

        site_result = AdsorptionSiteResult(
            site_x=x_frac,
            site_y=y_frac,
            site_x_cart=cart_pos[0],
            site_y_cart=cart_pos[1],
            adsorption_energy=adsorption_energy,
            adsorption_energy_per_area=adsorption_energy_per_area,
            total_energy=total_energy,
            slab_energy=slab_energy,
            adsorbate_energy=adsorbate_energy,
            surface_area=surface_area,
            height=height,  # Use height from site_dict
            n_atoms=len(slab)
            + (
                len(adsorbate)
                if isinstance(adsorbate, Structure)
                else len(adsorbate.sites)
            ),
            n_slab_atoms=len(slab),
            n_adsorbate_atoms=len(adsorbate)
            if isinstance(adsorbate, Structure)
            else len(adsorbate.sites),
        )
        site_results.append(site_result)

    # Find best site
    best_site_result = min(site_results, key=lambda x: x.adsorption_energy)

    # Calculate statistics
    energies = [r.adsorption_energy for r in site_results]
    mean_energy = np.mean(energies)
    std_energy = np.std(energies)
    energy_range = max(energies) - min(energies)

    # Get formulas
    slab_formula = slab.composition.reduced_formula
    if isinstance(adsorbate, Molecule):
        # Use Hill formula for molecules to avoid pymatgen's diatomic element bug
        # (reduced_formula incorrectly shows N→N2, O→O2, H→H2 for single atoms)
        adsorbate_formula = adsorbate.composition.hill_formula
    else:
        adsorbate_formula = adsorbate.composition.reduced_formula

    return AdsorptionScanDocument(
        slab_formula=slab_formula,
        adsorbate_formula=adsorbate_formula,
        miller_indices=miller_indices,
        grid_size=grid_size,
        initial_height=height,
        surface_area=surface_area,
        slab_thickness=slab_thickness,
        total_sites_scanned=len(site_results),
        slab_energy=slab_energy,
        adsorbate_energy=adsorbate_energy,
        best_site_position=(best_site_result.site_x, best_site_result.site_y),
        best_adsorption_energy=best_site_result.adsorption_energy,
        best_energy_per_area=best_site_result.adsorption_energy_per_area,
        mean_adsorption_energy=float(mean_energy),
        std_adsorption_energy=float(std_energy),
        energy_range=float(energy_range),
        site_results=site_results,
    )


@job
def plot_adsorption_sites(
    scan_doc: AdsorptionScanDocument,
    output_dir: str | Path = ".",
    filename: str = "adsorption_sites.png",
) -> dict[str, str]:
    """
    Create 2D heatmap of adsorption energies with height information.

    For height-scanned data (multiple heights per xy site), this plots the
    minimum energy at each xy position and shows which height was optimal.

    Parameters
    ----------
    scan_doc : AdsorptionScanDocument
        Adsorption scan results.
    output_dir : str | Path
        Output directory.
    filename : str
        Output filename.

    Returns
    -------
    dict
        Dictionary with plot file path.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    logger.info("Creating adsorption site heatmap")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # For height-scanned data, find best energy at each unique xy position
    # Group by (x, y) and keep minimum energy
    from collections import defaultdict

    xy_data: dict[tuple[float, float], dict[str, list]] = defaultdict(
        lambda: {"energies": [], "heights": []}
    )
    for s in scan_doc.site_results:
        key = (s.site_x, s.site_y)
        xy_data[key]["energies"].append(s.adsorption_energy)
        xy_data[key]["heights"].append(s.height)

    # Extract best energy and corresponding height at each xy position
    x_coords = []
    y_coords = []
    energies = []
    best_heights = []

    for (x, y), data in xy_data.items():
        x_coords.append(x)
        y_coords.append(y)
        # Find index of minimum energy
        min_idx = np.argmin(data["energies"])
        energies.append(data["energies"][min_idx])
        best_heights.append(data["heights"][min_idx])

    # Check if multiple heights were scanned
    # Look at ALL heights in scan_doc, not just the best ones
    all_heights = set([s.height for s in scan_doc.site_results])
    multiple_heights = len(all_heights) > 1

    # Create grid for interpolation
    nx, ny = scan_doc.grid_size
    xi = np.linspace(0, 1, nx * 10)
    yi = np.linspace(0, 1, ny * 10)
    XI, YI = np.meshgrid(xi, yi)

    # Interpolate energies
    from scipy.interpolate import griddata

    ZI = griddata((x_coords, y_coords), energies, (XI, YI), method="cubic")

    # If multiple heights, also interpolate optimal heights
    if multiple_heights:
        HI = griddata((x_coords, y_coords), best_heights, (XI, YI), method="cubic")

    # Create figure (3 plots if height-scanned, 2 otherwise)
    if multiple_heights:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Heatmap with contours
    cmap = LinearSegmentedColormap.from_list(
        "adsorption", ["green", "yellow", "orange", "red"]
    )
    contourf = ax1.contourf(XI, YI, ZI, levels=50, cmap=cmap)
    contour = ax1.contour(
        XI, YI, ZI, levels=10, colors="black", alpha=0.3, linewidths=0.5
    )
    ax1.clabel(contour, inline=True, fontsize=8, fmt="%.3f")

    # Mark best site
    best = scan_doc.best_site_position
    ax1.plot(
        best[0],
        best[1],
        "w*",
        markersize=20,
        markeredgecolor="black",
        markeredgewidth=2,
    )
    ax1.text(
        best[0],
        best[1] + 0.05,
        f"Best: {scan_doc.best_adsorption_energy:.3f} eV",
        ha="center",
        va="bottom",
        color="white",
        fontweight="bold",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
    )

    ax1.set_xlabel("X (fractional)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Y (fractional)", fontsize=12, fontweight="bold")

    # Update title to indicate height scanning if used
    title = (
        f"Adsorption Energy Map\n{scan_doc.slab_formula} + {scan_doc.adsorbate_formula}"
    )
    if multiple_heights:
        unique_heights = sorted(set(best_heights))
        if len(unique_heights) <= 5:
            heights_str = ", ".join(f"{h:.1f}" for h in unique_heights)
        else:
            heights_str = f"{min(unique_heights):.1f}-{max(unique_heights):.1f}"
        title += f"\n(Height scanned: {heights_str} Å)"

    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_aspect("equal")
    cbar = plt.colorbar(contourf, ax=ax1)
    cbar.set_label("Adsorption Energy (eV)", fontsize=11, fontweight="bold")

    # Plot 2: Scatter plot with size indicating energy
    # Normalize energies for marker sizing
    norm_energies = np.array(energies) - min(energies)
    max_norm = max(norm_energies) if max(norm_energies) > 0 else 1
    sizes = 500 - (norm_energies / max_norm) * 400  # Larger markers for lower energy

    scatter = ax2.scatter(
        x_coords,
        y_coords,
        c=energies,
        s=sizes,
        cmap=cmap,
        alpha=0.8,
        edgecolors="black",
        linewidths=1,
    )
    ax2.plot(
        best[0],
        best[1],
        "w*",
        markersize=20,
        markeredgecolor="black",
        markeredgewidth=2,
    )

    ax2.set_xlabel("X (fractional)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Y (fractional)", fontsize=12, fontweight="bold")

    # Update scatter plot title with height info
    scatter_title = f"Discrete Adsorption Sites\n{scan_doc.grid_size[0]}×{scan_doc.grid_size[1]} Grid"
    if multiple_heights:
        n_heights = len(set([s.height for s in scan_doc.site_results]))
        scatter_title += f" × {n_heights} heights"

    ax2.set_title(scatter_title, fontsize=14, fontweight="bold")
    ax2.set_aspect("equal")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    cbar2 = plt.colorbar(scatter, ax=ax2)
    cbar2.set_label("Adsorption Energy (eV)", fontsize=11, fontweight="bold")

    # Plot 3: Optimal height map (only if multiple heights scanned)
    if multiple_heights:
        height_cmap = LinearSegmentedColormap.from_list(
            "height", ["blue", "cyan", "yellow", "orange", "red"]
        )
        contourf_h = ax3.contourf(XI, YI, HI, levels=50, cmap=height_cmap)
        contour_h = ax3.contour(
            XI, YI, HI, levels=8, colors="black", alpha=0.3, linewidths=0.5
        )
        ax3.clabel(contour_h, inline=True, fontsize=8, fmt="%.2f Å")

        # Mark sites with their heights as text labels
        for x, y, h in zip(x_coords, y_coords, best_heights):
            ax3.text(
                x,
                y,
                f"{h:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="circle", facecolor="black", alpha=0.6, pad=0.3),
            )

        # Mark best site
        best = scan_doc.best_site_position
        ax3.plot(
            best[0],
            best[1],
            "w*",
            markersize=20,
            markeredgecolor="black",
            markeredgewidth=2,
        )

        ax3.set_xlabel("X (fractional)", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Y (fractional)", fontsize=12, fontweight="bold")
        ax3.set_title(
            "Optimal Height Map\nBest height at each xy position",
            fontsize=14,
            fontweight="bold",
        )
        ax3.set_aspect("equal")
        cbar3 = plt.colorbar(contourf_h, ax=ax3)
        cbar3.set_label("Optimal Height (Å)", fontsize=11, fontweight="bold")

    plt.tight_layout()

    output_file = output_dir / filename
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved adsorption site plot to: {output_file}")

    return {"plot_file": str(output_file)}


@job
def write_adsorption_summary(
    scan_doc: AdsorptionScanDocument,
    output_dir: str | Path = ".",
    filename: str = "adsorption_summary.txt",
) -> dict[str, str]:
    """
    Write comprehensive text summary of adsorption results.

    Parameters
    ----------
    scan_doc : AdsorptionScanDocument
        Adsorption scan results.
    output_dir : str | Path
        Output directory.
    filename : str
        Output filename.

    Returns
    -------
    dict
        Dictionary with summary file path.
    """
    from datetime import datetime
    from pathlib import Path

    logger.info("Writing adsorption summary")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 80)
    lines.append("ADSORPTION SITE SCAN SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Generated: {timestamp}")
    lines.append("")

    # System information
    lines.append("SYSTEM INFORMATION")
    lines.append("-" * 80)
    lines.append(f"  Slab formula:         {scan_doc.slab_formula}")
    lines.append(f"  Adsorbate formula:    {scan_doc.adsorbate_formula}")
    if scan_doc.miller_indices:
        miller_h, miller_k, miller_l = scan_doc.miller_indices
        lines.append(f"  Miller indices:       ({miller_h} {miller_k} {miller_l})")
    lines.append(
        f"  Grid size:            {scan_doc.grid_size[0]} × {scan_doc.grid_size[1]}"
    )
    lines.append(f"  Initial height:       {scan_doc.initial_height:.2f} Å")
    lines.append(f"  Surface area:         {scan_doc.surface_area:.3f} Ų")
    lines.append(f"  Total sites scanned:  {scan_doc.total_sites_scanned}")
    lines.append("")

    # Energy statistics
    lines.append("ENERGY STATISTICS")
    lines.append("-" * 80)
    lines.append(f"  Best energy:          {scan_doc.best_adsorption_energy:.6f} eV")
    lines.append(f"  Mean energy:          {scan_doc.mean_adsorption_energy:.6f} eV")
    lines.append(f"  Std deviation:        {scan_doc.std_adsorption_energy:.6f} eV")
    lines.append(f"  Energy range:         {scan_doc.energy_range:.6f} eV")
    lines.append("")

    # Best site
    lines.append("BEST ADSORPTION SITE")
    lines.append("-" * 80)
    lines.append(
        f"  Position (frac):      ({scan_doc.best_site_position[0]:.4f}, {scan_doc.best_site_position[1]:.4f})"
    )
    lines.append(f"  Adsorption energy:    {scan_doc.best_adsorption_energy:.6f} eV")
    lines.append(f"  Energy per area:      {scan_doc.best_energy_per_area:.6f} eV/Ų")
    lines.append("")

    # Top 5 sites
    lines.append("TOP 5 ADSORPTION SITES")
    lines.append("-" * 80)
    lines.append(
        f"  {'Rank':<6} {'Position (frac)':<20} {'E_ads (eV)':<14} {'E/A (eV/Ų)':<12}"
    )
    lines.append("  " + "-" * 75)

    for i, site in enumerate(scan_doc.top_5_sites, 1):
        lines.append(
            f"  {i:<6} ({site.site_x:.4f}, {site.site_y:.4f}){'':<8} "
            f"{site.adsorption_energy:>13.6f} {site.adsorption_energy_per_area:>11.6f}"
        )

    lines.append("")

    # Detailed site information - ALL SITES
    lines.append("DETAILED SITE INFORMATION (ALL SITES)")
    lines.append("=" * 80)
    lines.append("")

    # Sort sites by adsorption energy for easy reading
    sorted_sites = sorted(scan_doc.site_results, key=lambda x: x.adsorption_energy)

    for rank, site in enumerate(sorted_sites, 1):
        lines.append(f"Site #{rank}")
        lines.append("-" * 80)
        lines.append(
            f"  Position (fractional):    ({site.site_x:.6f}, {site.site_y:.6f})"
        )
        lines.append(
            f"  Position (Cartesian):     ({site.site_x_cart:.6f}, {site.site_y_cart:.6f}) Å"
        )
        lines.append("")
        lines.append(f"  Adsorption energy:        {site.adsorption_energy:>12.6f} eV")
        lines.append(
            f"  Energy per area:          {site.adsorption_energy_per_area:>12.6f} eV/Ų"
        )
        lines.append("")
        lines.append(f"  Total energy:             {site.total_energy:>12.6f} eV")
        lines.append(f"  Slab energy:              {site.slab_energy:>12.6f} eV")
        lines.append(f"  Adsorbate energy:         {site.adsorbate_energy:>12.6f} eV")
        lines.append("")
        lines.append(f"  Initial height:           {site.height:>12.2f} Å")
        lines.append(f"  Surface area:             {site.surface_area:>12.3f} Ų")
        lines.append("")
        lines.append(f"  Total atoms:              {site.n_atoms}")
        lines.append(f"  Slab atoms:               {site.n_slab_atoms}")
        lines.append(f"  Adsorbate atoms:          {site.n_adsorbate_atoms}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("")

    # Notes
    lines.append("NOTES")
    lines.append("-" * 80)
    lines.append("  • Negative adsorption energy indicates favorable adsorption")
    lines.append("  • Adsorption energy = E_total - E_slab - E_adsorbate")
    lines.append("  • Positions given in fractional coordinates of the slab cell")
    lines.append("")
    lines.append("=" * 80)

    content = "\n".join(lines)

    # Add standard footer
    from atomate2.siesta.utils.text_output import get_standard_footer

    footer = get_standard_footer(
        width=80,
        additional_info={
            "Analysis type": "Adsorption site scanning",
            "Number of sites": str(scan_doc.total_sites_scanned),
        },
    )

    output_file = output_dir / filename
    with open(output_file, "w") as f:
        f.write(content)
        f.write("\n" + footer)

    logger.info(f"Saved adsorption summary to: {output_file}")

    return {"summary_file": str(output_file)}
