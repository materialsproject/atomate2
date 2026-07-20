"""Jobs for surface energy calculations."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from jobflow import job

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


@job
def calculate_surface_energies(
    bulk_energy: float,
    bulk_structure: Structure,
    slab_data: list[dict[str, Any]],
    miller_indices: tuple[int, int, int],
    formula_units_per_cell: int | None = None,
) -> dict[str, Any]:
    """
    Calculate surface energy for each termination.

    Parameters
    ----------
    bulk_energy : float
        Total energy of bulk unit cell (eV).
    bulk_structure : Structure
        Bulk structure for composition reference.
    slab_data : list[dict]
        List of dictionaries containing:
        - 'termination': str
        - 'slab_energy': float
        - 'slab_structure': Structure
        - 'metadata': dict (from validation JSON)
    miller_indices : tuple[int, int, int]
        Miller indices (h, k, l).
    formula_units_per_cell : int, optional
        Number of formula units in bulk cell. Auto-detected if None.

    Returns
    -------
    dict
        SurfaceEnergyDocument data.
    """
    logger.info("calculate_surface_energies.__init__()")

    # Get bulk composition
    bulk_composition = bulk_structure.composition.reduced_composition
    bulk_atoms_per_formula = bulk_composition.num_atoms

    if formula_units_per_cell is None:
        # Auto-detect from bulk structure
        formula_units_per_cell = int(
            bulk_structure.composition.num_atoms / bulk_atoms_per_formula
        )
        logger.info(f"Auto-detected formula_units_per_cell = {formula_units_per_cell}")

    bulk_energy_per_formula = bulk_energy / formula_units_per_cell
    bulk_energy_per_atom = bulk_energy / len(bulk_structure)

    results = []

    for slab_dict in slab_data:
        termination = slab_dict["termination"]
        slab_energy = slab_dict["slab_energy"]
        slab_structure = slab_dict["slab_structure"]
        metadata = slab_dict.get("metadata", {})

        logger.info(f"Processing termination: {termination}")

        # Calculate surface area (xy plane)
        cell = slab_structure.lattice.matrix
        surface_area = np.linalg.norm(np.cross(cell[0], cell[1]))

        # Count formula units in slab
        slab_composition = slab_structure.composition
        n_formula_units = slab_composition.num_atoms / bulk_atoms_per_formula

        # Surface energy (1 surface for asymmetric slabs)
        # γ = (E_slab - N × E_bulk) / A  # noqa: RUF003
        gamma_eV_A2 = (  # noqa: N806
            slab_energy - n_formula_units * bulk_energy_per_formula
        ) / surface_area

        # Convert to J/m² (1 eV/Ų = 16.0218 J/m²)
        gamma_Jm2 = gamma_eV_A2 * 16.0218  # noqa: N806

        # Get slab thickness
        positions = slab_structure.cart_coords
        thickness = positions[:, 2].max() - positions[:, 2].min()

        results.append(
            {
                "termination": termination,
                "surface_energy": gamma_eV_A2,
                "surface_energy_Jm2": gamma_Jm2,
                "slab_energy": slab_energy,
                "n_formula_units": n_formula_units,
                "surface_area": surface_area,
                "n_atoms": len(slab_structure),
                "thickness": thickness,
                "composition": dict(slab_composition.as_dict()),
                "bottom_composition": metadata.get("bottom_composition", {}),
                "top_composition": metadata.get("top_composition", {}),
                "is_symmetric": metadata.get("is_symmetric", False),
                "z_position": metadata.get("z_position", 0.0),
            }
        )

        logger.info(f"  Surface energy: {gamma_eV_A2:.4f} eV/Ų ({gamma_Jm2:.2f} J/m²)")

    # Find lowest energy termination
    if not results:
        raise ValueError("No termination data to process!")

    min_gamma = min(r["surface_energy"] for r in results)
    max_gamma = max(r["surface_energy"] for r in results)

    for r in results:
        r["relative_energy"] = r["surface_energy"] - min_gamma
        r["is_lowest"] = abs(r["surface_energy"] - min_gamma) < 1e-6

    lowest_term = next(r for r in results if r["is_lowest"])["termination"]

    return {
        "bulk_energy": bulk_energy,
        "bulk_energy_per_atom": bulk_energy_per_atom,
        "miller_indices": miller_indices,
        "terminations": results,
        "lowest_termination": lowest_term,
        "formula_units_per_cell": formula_units_per_cell,
        "n_terminations": len(results),
        "energy_spread": max_gamma - min_gamma,
    }


@job
def plot_surface_energies(
    surface_doc: dict[str, Any],
    output_dir: str | Path = ".",
    filename: str = "surface_energies.png",
    figsize: tuple[float, float] = (14, 6),
    dpi: int = 300,
) -> dict[str, str]:
    """
    Create publication-quality surface energy plots.

    Parameters
    ----------
    surface_doc : dict
        SurfaceEnergyDocument data.
    output_dir : str | Path
        Output directory.
    filename : str
        Output filename.
    figsize : tuple
        Figure size (width, height) in inches.
    dpi : int
        Resolution in dots per inch.

    Returns
    -------
    dict
        Dictionary with plot file path.
    """
    import matplotlib.pyplot as plt

    logger.info("plot_surface_energies.__init__()")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    terminations = surface_doc["terminations"]

    # Sort by surface energy
    sorted_terms = sorted(terminations, key=lambda x: x["surface_energy"])

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Extract data
    names = [t["termination"] for t in sorted_terms]
    energies_eV = [t["surface_energy"] for t in sorted_terms]  # noqa: N806
    energies_Jm2 = [t["surface_energy_Jm2"] for t in sorted_terms]  # noqa: N806
    colors = ["green" if t["is_lowest"] else "steelblue" for t in sorted_terms]

    # Plot 1: Bar chart (eV/Ų)
    ax1.bar(
        names, energies_eV, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
    )
    ax1.set_ylabel("Surface Energy (eV/Ų)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Termination", fontsize=12, fontweight="bold")
    miller_h, miller_k, miller_l = surface_doc["miller_indices"]
    ax1.set_title(
        f"Surface Energy - ({miller_h}{miller_k}{miller_l})",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.tick_params(axis="both", labelsize=10)

    # Add value labels on bars
    for i, (_name, energy) in enumerate(zip(names, energies_eV, strict=False)):
        ax1.text(
            i,
            energy,
            f"{energy:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Plot 2: Bar chart (J/m²)
    ax2.bar(
        names, energies_Jm2, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
    )
    ax2.set_ylabel("Surface Energy (J/m²)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Termination", fontsize=12, fontweight="bold")
    ax2.set_title(
        f"Surface Energy - ({miller_h}{miller_k}{miller_l})",
        fontsize=14,
        fontweight="bold",
    )
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.tick_params(axis="both", labelsize=10)

    # Add value labels on bars
    for i, (_name, energy) in enumerate(zip(names, energies_Jm2, strict=False)):
        ax2.text(
            i,
            energy,
            f"{energy:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()

    output_file = output_dir / filename
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved surface energy plot to: {output_file}")

    return {"plot_file": str(output_file)}


@job
def write_surface_energy_summary(
    surface_doc: dict[str, Any],
    output_dir: str | Path = ".",
    filename: str = "surface_energy_summary.txt",
) -> dict[str, str]:
    """
    Write comprehensive text summary of surface energy results.

    Parameters
    ----------
    surface_doc : dict
        SurfaceEnergyDocument data.
    output_dir : str | Path
        Output directory.
    filename : str
        Output filename.

    Returns
    -------
    dict
        Dictionary with summary file path.
    """
    logger.info("write_surface_energy_summary.__init__()")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # noqa: DTZ005

    lines = []
    lines.append("=" * 80)
    lines.append("SURFACE ENERGY CALCULATION SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Generated: {timestamp}")
    lines.append("")

    # Bulk properties
    lines.append("BULK PROPERTIES")
    lines.append("-" * 80)
    lines.append(f"  Bulk unit cell energy:    {surface_doc['bulk_energy']:.6f} eV")
    lines.append(
        f"  Energy per atom:          {surface_doc['bulk_energy_per_atom']:.6f} eV"
    )
    lines.append(f"  Formula units per cell:   {surface_doc['formula_units_per_cell']}")
    lines.append("")

    # Surface information
    lines.append("SURFACE INFORMATION")
    lines.append("-" * 80)
    miller_h, miller_k, miller_l = surface_doc["miller_indices"]
    lines.append(f"  Miller indices:           ({miller_h} {miller_k} {miller_l})")
    lines.append(f"  Number of terminations:   {surface_doc['n_terminations']}")
    lines.append(f"  Lowest energy termination: {surface_doc['lowest_termination']}")
    lines.append(f"  Energy spread:            {surface_doc['energy_spread']:.4f} eV/Ų")
    lines.append("")

    # Termination energies table
    lines.append("TERMINATION ENERGIES")
    lines.append("-" * 80)
    lines.append(
        f"  {'Termination':<15} {'γ (eV/Ų)':<12} {'γ (J/m²)':<12} "  # noqa: RUF001
        f"{'Δγ (eV/Ų)':<12} {'Status'}"
    )
    lines.append("  " + "-" * 75)

    for t in sorted(surface_doc["terminations"], key=lambda x: x["surface_energy"]):
        status = "LOWEST ✓" if t["is_lowest"] else ""
        lines.append(
            f"  {t['termination']:<15} {t['surface_energy']:>11.4f} "
            f"{t['surface_energy_Jm2']:>11.2f} {t['relative_energy']:>11.4f}  {status}"
        )

    lines.append("")

    # Detailed breakdown
    lines.append("DETAILED BREAKDOWN")
    lines.append("-" * 80)

    for t in sorted(surface_doc["terminations"], key=lambda x: x["surface_energy"]):
        lines.append(f"\n  Termination: {t['termination']}")
        lines.append(f"    Slab energy:       {t['slab_energy']:.6f} eV")
        lines.append(f"    Formula units:     {t['n_formula_units']:.2f}")
        lines.append(f"    Surface area:      {t['surface_area']:.4f} Ų")
        lines.append(f"    Slab thickness:    {t['thickness']:.4f} Å")
        lines.append(f"    Number of atoms:   {t['n_atoms']}")
        lines.append(f"    Composition:       {t['composition']}")

        if t.get("bottom_composition") and t.get("top_composition"):
            lines.append(f"    Bottom layer:      {t['bottom_composition']}")
            lines.append(f"    Top layer:         {t['top_composition']}")

        lines.append(
            f"    Surface energy:    {t['surface_energy']:.4f} eV/Ų "
            f"({t['surface_energy_Jm2']:.2f} J/m²)"
        )
        lines.append(f"    Relative energy:   {t['relative_energy']:.4f} eV/Ų")

    lines.append("")

    # Convergence notes
    lines.append("CONVERGENCE NOTES")
    lines.append("-" * 80)
    lines.append("  To ensure converged surface energies:")
    lines.append("    • Slab thickness: Test 4, 6, 8, 10 layers (< 0.01 eV/Ų change)")
    lines.append("    • Vacuum spacing: Test 15, 20, 25 Å (< 0.001 eV/atom change)")
    lines.append("    • K-points (in-plane): Test [4,4,1], [6,6,1], [8,8,1]")
    lines.append("    • Mesh cutoff: Same as bulk convergence (typically 300-400 Ry)")
    lines.append("")

    # References
    lines.append("REFERENCES")
    lines.append("-" * 80)
    lines.append(
        "  1. Fiorentini & Methfessel (1996). J. Phys.: Condens. Matter 8, 6525"
    )
    lines.append("  2. Sun et al. (2016). npj Comput. Mater. 2, 16026")
    lines.append("")
    lines.append("=" * 80)

    content = "\n".join(lines)

    # Add standard footer
    from atomate2.siesta.utils.text_output import get_standard_footer

    miller_h, miller_k, miller_l = surface_doc["miller_indices"]
    footer = get_standard_footer(
        width=80,
        additional_info={
            "Analysis type": "Surface energy calculation",
            "Miller index": f"({miller_h} {miller_k} {miller_l})",
        },
    )

    output_file = output_dir / filename
    with open(output_file, "w") as f:
        f.write(content)
        f.write("\n" + footer)

    logger.info(f"Saved surface energy summary to: {output_file}")

    return {"summary_file": str(output_file)}
