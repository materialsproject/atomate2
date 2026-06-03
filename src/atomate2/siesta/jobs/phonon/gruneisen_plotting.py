"""Grüneisen parameter plotting and analysis utilities for SIESTA calculations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jobflow import job

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


def _get_attr(obj, key, default=None):
    """
    Get attribute from either dict or Pydantic object.

    Parameters
    ----------
    obj : dict or pydantic model
        Object to get attribute from
    key : str
        Key/attribute name
    default : any
        Default value if not found

    Returns
    -------
    Value of attribute or default
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@job
def plot_gruneisen_band_structure(
    gruneisen_doc: dict[str, Any],
    output_dir: str | Path = ".",
    filename: str = "gruneisen_bands.png",
    figsize: tuple[float, float] = (12, 8),
    dpi: int = 300,
) -> dict[str, str]:
    """
    Plot Grüneisen parameter as a function of phonon mode along high-symmetry path.

    Creates a dual-panel plot showing both phonon frequencies and mode-dependent
    Grüneisen parameters along the same high-symmetry path.

    Parameters
    ----------
    gruneisen_doc : dict
        Grüneisen calculation results from GruneisenParameterDocument
    output_dir : str | Path
        Directory to save plots
    filename : str
        Output filename for band structure plot
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        Resolution of saved figure

    Returns
    -------
    dict
        Dictionary with paths to generated files

    Examples
    --------
    >>> from atomate2.siesta.jobs.gruneisen_plotting import plot_gruneisen_band_structure
    >>> from jobflow import run_locally
    >>> plot_job = plot_gruneisen_band_structure(gruneisen_doc, output_dir="plots")
    >>> run_locally(plot_job)
    """
    try:
        import matplotlib.pyplot as plt
        from pymatgen.phonon.gruneisen import GruneisenPhononBandStructureSymmLine
    except ImportError as e:
        logger.error("matplotlib or pymatgen not available for plotting")
        raise ImportError("Install matplotlib and pymatgen for plotting") from e

    logger.info("Plotting Grüneisen parameter band structure")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract Grüneisen band structure
    grun_bs = _get_attr(gruneisen_doc, "gruneisen_band_structure")
    if grun_bs is None or not isinstance(grun_bs, GruneisenPhononBandStructureSymmLine):
        logger.warning("No Grüneisen band structure data available")
        return {"gruneisen_bands_plot": "not_available"}

    # Get formula for title (structure field doesn't exist in StructureMetadata)
    formula = _get_attr(gruneisen_doc, "formula_pretty") or _get_attr(
        _get_attr(gruneisen_doc, "structure"), "composition.formula", "Unknown"
    )

    # Create dual-panel plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Extract branch data
    distances = []
    frequencies = []
    gruneisen_params = []
    special_points = [0]
    labels = []

    for i, branch in enumerate(grun_bs.branches):
        branch_dists = [p.distance for p in branch["phonon_frequencies"]]
        branch_freqs = np.array([p.frequency for p in branch["phonon_frequencies"]])
        branch_gru = np.array([p.gruneisen for p in branch["phonon_frequencies"]])

        distances.extend(branch_dists)
        frequencies.append(branch_freqs)
        gruneisen_params.append(branch_gru)

        if i < len(grun_bs.branches) - 1:
            special_points.append(branch_dists[-1])
            labels.append(branch["name"])

    # Add final label
    if grun_bs.branches:
        labels.append(grun_bs.branches[-1]["end_name"])

    # Concatenate all branches
    all_freqs = np.concatenate(frequencies, axis=0)
    all_gru = np.concatenate(gruneisen_params, axis=0)

    # Plot phonon frequencies (top panel)
    for band_idx in range(all_freqs.shape[1]):
        ax1.plot(distances, all_freqs[:, band_idx], "b-", linewidth=1.5, alpha=0.7)

    # Add vertical lines at special points
    for sp in special_points:
        ax1.axvline(x=sp, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
        ax2.axvline(x=sp, color="k", linewidth=0.5, linestyle="--", alpha=0.5)

    # Format top panel (frequencies)
    ax1.set_ylabel("Frequency (THz)", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Grüneisen Parameters - {formula}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.axhline(y=0, color="r", linewidth=0.8, linestyle="-", alpha=0.7)
    ax1.set_xlim(distances[0], distances[-1])

    # Plot Grüneisen parameters (bottom panel)
    # Use colormap for better visualization
    cmap = plt.cm.get_cmap("RdYlBu_r")
    for band_idx in range(all_gru.shape[1]):
        gru_values = all_gru[:, band_idx]
        # Normalize colors based on Grüneisen values
        norm_gru = np.clip((gru_values + 3) / 6, 0, 1)  # Typical range: -3 to +3
        for i in range(len(distances) - 1):
            ax2.plot(
                distances[i : i + 2],
                gru_values[i : i + 2],
                color=cmap(norm_gru[i]),
                linewidth=1.5,
                alpha=0.7,
            )

    # Format bottom panel (Grüneisen parameters)
    ax2.set_xlabel("Wave vector", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Grüneisen parameter γ", fontsize=12, fontweight="bold")
    ax2.set_xlim(distances[0], distances[-1])
    ax2.set_xticks(special_points)
    ax2.set_xticklabels([label.replace("GAMMA", "Γ") for label in labels], fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.axhline(y=0, color="k", linewidth=1.0, linestyle="-", alpha=0.5)

    # Add colorbar for Grüneisen scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-3, vmax=3))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, orientation="vertical", pad=0.02, aspect=30)
    cbar.set_label("Grüneisen parameter γ", fontsize=10, fontweight="bold")

    plt.tight_layout()

    # Save figure
    band_file = output_dir / filename
    plt.savefig(band_file, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Grüneisen band structure saved to {band_file}")
    return {"gruneisen_bands_plot": str(band_file)}


@job
def plot_gruneisen_vs_frequency(
    gruneisen_doc: dict[str, Any],
    output_dir: str | Path = ".",
    filename: str = "gruneisen_vs_frequency.png",
    figsize: tuple[float, float] = (10, 7),
    dpi: int = 300,
) -> dict[str, str]:
    """
    Plot Grüneisen parameter vs phonon frequency scatter plot.

    This plot helps identify relationships between mode frequencies and their
    Grüneisen parameters, useful for understanding anharmonicity.

    Parameters
    ----------
    gruneisen_doc : dict
        Grüneisen calculation results
    output_dir : str | Path
        Directory to save plots
    filename : str
        Output filename
    figsize : tuple
        Figure size (width, height)
    dpi : int
        Resolution

    Returns
    -------
    dict
        Paths to generated files
    """
    try:
        import matplotlib.pyplot as plt
        from pymatgen.phonon.gruneisen import GruneisenParameter
    except ImportError as e:
        raise ImportError("Install matplotlib and pymatgen for plotting") from e

    logger.info("Plotting Grüneisen parameter vs frequency")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract Grüneisen parameter object
    grun_param = _get_attr(gruneisen_doc, "gruneisen_parameter")
    if grun_param is None or not isinstance(grun_param, GruneisenParameter):
        logger.warning("No Grüneisen parameter data available")
        return {"gruneisen_vs_freq_plot": "not_available"}

    # Get formula for title
    formula = _get_attr(gruneisen_doc, "formula_pretty") or _get_attr(
        _get_attr(gruneisen_doc, "structure"), "composition.formula", "Unknown"
    )

    # Extract frequencies and Grüneisen parameters from mesh
    frequencies = grun_param.frequencies.flatten()
    gruneisen_values = grun_param.gruneisen.flatten()

    # Remove zero/negative frequencies (acoustic modes at gamma)
    mask = frequencies > 0.1  # THz threshold
    frequencies = frequencies[mask]
    gruneisen_values = gruneisen_values[mask]

    # Create scatter plot
    fig, ax = plt.subplots(figsize=figsize)

    # Color points by Grüneisen value
    scatter = ax.scatter(
        frequencies,
        gruneisen_values,
        c=gruneisen_values,
        cmap="RdYlBu_r",
        s=20,
        alpha=0.6,
        edgecolors="black",
        linewidths=0.3,
    )

    # Add horizontal line at γ = 0
    ax.axhline(y=0, color="k", linewidth=1.0, linestyle="--", alpha=0.5)

    # Calculate and plot average
    avg_grun = np.mean(gruneisen_values)
    ax.axhline(
        y=avg_grun,
        color="red",
        linewidth=2.0,
        linestyle="-",
        alpha=0.7,
        label=f"Average: γ = {avg_grun:.3f}",
    )

    # Format plot
    ax.set_xlabel("Frequency (THz)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Grüneisen parameter γ", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Mode Grüneisen Parameters - {formula}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(fontsize=11, loc="best", framealpha=0.9)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label="Grüneisen parameter γ")
    cbar.set_label("Grüneisen parameter γ", fontsize=10, fontweight="bold")

    # Add statistics text box
    stats_text = (
        f"Statistics:\n"
        f"Mean: {np.mean(gruneisen_values):.3f}\n"
        f"Std: {np.std(gruneisen_values):.3f}\n"
        f"Min: {np.min(gruneisen_values):.3f}\n"
        f"Max: {np.max(gruneisen_values):.3f}\n"
        f"N modes: {len(gruneisen_values)}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    # Save figure
    freq_file = output_dir / filename
    plt.savefig(freq_file, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Grüneisen vs frequency plot saved to {freq_file}")
    return {"gruneisen_vs_freq_plot": str(freq_file)}


@job
def plot_gruneisen_distribution(
    gruneisen_doc: dict[str, Any],
    output_dir: str | Path = ".",
    filename: str = "gruneisen_distribution.png",
    figsize: tuple[float, float] = (10, 6),
    dpi: int = 300,
    bins: int = 50,
) -> dict[str, str]:
    """
    Plot histogram of Grüneisen parameter distribution.

    Shows the distribution of mode Grüneisen parameters, useful for
    identifying typical values and outliers.

    Parameters
    ----------
    gruneisen_doc : dict
        Grüneisen calculation results
    output_dir : str | Path
        Directory to save plots
    filename : str
        Output filename
    figsize : tuple
        Figure size
    dpi : int
        Resolution
    bins : int
        Number of histogram bins

    Returns
    -------
    dict
        Paths to generated files
    """
    try:
        import matplotlib.pyplot as plt
        from pymatgen.phonon.gruneisen import GruneisenParameter
    except ImportError as e:
        raise ImportError("Install matplotlib and pymatgen for plotting") from e

    logger.info("Plotting Grüneisen parameter distribution")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grun_param = _get_attr(gruneisen_doc, "gruneisen_parameter")
    if grun_param is None or not isinstance(grun_param, GruneisenParameter):
        logger.warning("No Grüneisen parameter data available")
        return {"gruneisen_dist_plot": "not_available"}

    # Get formula for title
    formula = _get_attr(gruneisen_doc, "formula_pretty") or _get_attr(
        _get_attr(gruneisen_doc, "structure"), "composition.formula", "Unknown"
    )

    # Extract Grüneisen parameters
    gruneisen_values = grun_param.gruneisen.flatten()
    frequencies = grun_param.frequencies.flatten()

    # Filter physical modes
    mask = frequencies > 0.1  # THz
    gruneisen_values = gruneisen_values[mask]

    # Create histogram
    fig, ax = plt.subplots(figsize=figsize)

    counts, bin_edges, patches = ax.hist(
        gruneisen_values, bins=bins, color="steelblue", alpha=0.7, edgecolor="black"
    )

    # Color bars by value
    cmap = plt.cm.get_cmap("RdYlBu_r")
    norm = plt.Normalize(vmin=-3, vmax=3)
    for i, patch in enumerate(patches):
        bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
        patch.set_facecolor(cmap(norm(bin_center)))

    # Add vertical lines for statistics
    mean_gru = np.mean(gruneisen_values)
    median_gru = np.median(gruneisen_values)

    ax.axvline(
        x=mean_gru,
        color="red",
        linewidth=2.5,
        linestyle="-",
        label=f"Mean: {mean_gru:.3f}",
        alpha=0.8,
    )
    ax.axvline(
        x=median_gru,
        color="green",
        linewidth=2.5,
        linestyle="--",
        label=f"Median: {median_gru:.3f}",
        alpha=0.8,
    )
    ax.axvline(x=0, color="black", linewidth=1.0, linestyle=":", alpha=0.5)

    # Format plot
    ax.set_xlabel("Grüneisen parameter γ", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of modes", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Grüneisen Parameter Distribution - {formula}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y", linestyle=":")

    # Add statistics text
    stats_text = (
        f"Statistics:\n"
        f"Mean: {mean_gru:.3f}\n"
        f"Median: {median_gru:.3f}\n"
        f"Std: {np.std(gruneisen_values):.3f}\n"
        f"Range: [{np.min(gruneisen_values):.2f}, {np.max(gruneisen_values):.2f}]\n"
        f"Total modes: {len(gruneisen_values)}"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    dist_file = output_dir / filename
    plt.savefig(dist_file, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Grüneisen distribution plot saved to {dist_file}")
    return {"gruneisen_dist_plot": str(dist_file)}


@job
def write_gruneisen_summary(
    gruneisen_doc: dict[str, Any],
    output_dir: str | Path = ".",
    filename: str = "gruneisen_summary.txt",
) -> dict[str, str]:
    """
    Write comprehensive Grüneisen parameter results to text file.

    Parameters
    ----------
    gruneisen_doc : dict
        Grüneisen calculation results
    output_dir : str | Path
        Directory to save file
    filename : str
        Output filename

    Returns
    -------
    dict
        Path to generated file
    """
    from datetime import datetime

    logger.info("Writing Grüneisen summary to text file")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / filename

    with open(summary_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GRÜNEISEN PARAMETER CALCULATION SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        code = _get_attr(gruneisen_doc, "code", "Unknown")
        f.write(f"Code: {code}\n\n")

        # Structure information
        f.write("-" * 80 + "\n")
        f.write("STRUCTURE INFORMATION\n")
        f.write("-" * 80 + "\n")

        # Check if we have structure object (dict case) or metadata fields (Pydantic case)
        structure = _get_attr(gruneisen_doc, "structure")
        if structure:
            # Dict case: has structure object
            f.write(f"Formula: {structure.composition.formula}\n")
            f.write(f"Reduced Formula: {structure.composition.reduced_formula}\n")
            f.write(f"Number of atoms: {len(structure)}\n")
            f.write(f"Space group: {structure.get_space_group_info()}\n")
            f.write("Lattice parameters:\n")
            f.write(f"  a = {structure.lattice.a:.6f} Å\n")
            f.write(f"  b = {structure.lattice.b:.6f} Å\n")
            f.write(f"  c = {structure.lattice.c:.6f} Å\n")
            f.write(f"  α = {structure.lattice.alpha:.3f}°\n")
            f.write(f"  β = {structure.lattice.beta:.3f}°\n")
            f.write(f"  γ = {structure.lattice.gamma:.3f}°\n")
            f.write(f"  Volume = {structure.lattice.volume:.6f} Å³\n\n")
        else:
            # Pydantic case: extract from metadata fields
            formula = _get_attr(gruneisen_doc, "formula_pretty", "Unknown")
            nsites = _get_attr(gruneisen_doc, "nsites", "Unknown")
            volume = _get_attr(gruneisen_doc, "volume", "Unknown")
            density = _get_attr(gruneisen_doc, "density", "Unknown")
            symmetry = _get_attr(gruneisen_doc, "symmetry", {})

            f.write(f"Formula: {formula}\n")
            f.write(f"Number of atoms: {nsites}\n")
            if symmetry:
                spg = _get_attr(symmetry, "symbol", "Unknown")
                spg_num = _get_attr(symmetry, "number", "Unknown")
                f.write(f"Space group: {spg} ({spg_num})\n")
            if volume != "Unknown":
                f.write(f"  Volume = {volume:.6f} Å³\n")
            if density != "Unknown":
                f.write(f"  Density = {density:.4f} g/cm³\n")
            f.write("\n")

        # Derived properties
        derived = _get_attr(gruneisen_doc, "derived_properties")
        if derived:
            f.write("-" * 80 + "\n")
            f.write("GRÜNEISEN PARAMETERS\n")
            f.write("-" * 80 + "\n")

            avg_grun = _get_attr(derived, "average_gruneisen")
            if avg_grun is not None:
                f.write(f"Average Grüneisen parameter: {avg_grun:.4f}\n")

            thermal_cond = _get_attr(derived, "thermal_conductivity_slack")
            if thermal_cond is not None:
                f.write(f"Thermal conductivity (Slack): {thermal_cond:.2f} W/(m·K)\n")
            f.write("\n")

        # Imaginary modes information
        imaginary_info = _get_attr(gruneisen_doc, "phonon_runs_has_imaginary_modes")
        if imaginary_info:
            f.write("-" * 80 + "\n")
            f.write("PHONON STABILITY CHECK\n")
            f.write("-" * 80 + "\n")
            # Iterate over fields (works for both dict and Pydantic object)
            for key in ["ground", "plus", "minus"]:
                has_imaginary = _get_attr(imaginary_info, key)
                if has_imaginary is not None:
                    status = "⚠️  Yes" if has_imaginary else "✓ No"
                    f.write(f"{key.capitalize()} structure imaginary modes: {status}\n")
            f.write("\n")

        # Physical interpretation
        f.write("-" * 80 + "\n")
        f.write("PHYSICAL INTERPRETATION\n")
        f.write("-" * 80 + "\n")
        f.write("The Grüneisen parameter γᵢ quantifies how phonon frequencies\n")
        f.write("change with volume:\n\n")
        f.write("    γᵢ = -V/ωᵢ · ∂ωᵢ/∂V\n\n")
        f.write("Physical significance:\n")
        f.write("  γ > 0: Mode frequency decreases with expansion (typical)\n")
        f.write("  γ < 0: Mode frequency increases with expansion (unusual)\n")
        f.write("  γ ≈ 2: Typical for most materials\n")
        f.write("  γ >> 2: Strong anharmonicity\n\n")

        if avg_grun is not None:
            f.write(f"Your material (γ_avg = {avg_grun:.3f}):\n")
            if abs(avg_grun) < 0.5:
                f.write("  → Very weak volume dependence of phonons\n")
            elif 0.5 <= abs(avg_grun) < 1.5:
                f.write("  → Weak to moderate anharmonicity\n")
            elif 1.5 <= abs(avg_grun) < 2.5:
                f.write("  → Typical anharmonicity\n")
            else:
                f.write(
                    "  → Strong anharmonicity (potential for low thermal conductivity)\n"
                )
            f.write("\n")

        # Thermal expansion relationship
        f.write("-" * 80 + "\n")
        f.write("THERMAL EXPANSION\n")
        f.write("-" * 80 + "\n")
        f.write("The Grüneisen parameter is directly related to thermal expansion:\n\n")
        f.write("    α = γ · Cv / (B · V)\n\n")
        f.write("where α is thermal expansion, Cv is heat capacity,\n")
        f.write("B is bulk modulus, and V is volume.\n\n")
        f.write("Materials with larger γ typically show:\n")
        f.write("  • Higher thermal expansion coefficients\n")
        f.write("  • Lower thermal conductivity\n")
        f.write("  • Stronger temperature dependence of elastic properties\n\n")

        # Convergence recommendations
        f.write("-" * 80 + "\n")
        f.write("CONVERGENCE RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        f.write("For reliable Grüneisen parameters, ensure:\n\n")
        f.write("1. Volume changes:\n")
        f.write("   • Test ±0.5%, ±1%, ±2% volume changes\n")
        f.write("   • Check linearity of frequency vs volume\n")
        f.write("   • Criterion: γ converged within 0.05\n\n")
        f.write("2. Phonon convergence (for all 3 volumes):\n")
        f.write("   • Supercell: 2×2×2 minimum, 3×3×3 recommended\n")
        f.write("   • K-points: Dense mesh (8×8×8 or higher)\n")
        f.write("   • Force convergence: < 0.001 eV/Å\n")
        f.write("   • Displacement: 0.01 Å\n\n")
        f.write("3. Structure optimization:\n")
        f.write("   • Relax ground state to high precision\n")
        f.write("   • Check all three structures have same space group\n\n")

        # Add standard footer
        from atomate2.siesta.utils.text_output import get_standard_footer

        structure = _get_attr(gruneisen_doc, "structure")
        formula = (
            structure.composition.reduced_formula
            if structure
            else _get_attr(gruneisen_doc, "formula_pretty", "Unknown")
        )

        f.write(
            get_standard_footer(
                width=80,
                additional_info={
                    "Analysis type": "Grüneisen parameter calculation",
                    "Formula": formula,
                },
            )
        )

    logger.info(f"Grüneisen summary written to {summary_file}")
    return {"summary_file": str(summary_file)}


@job
def calculate_thermal_expansion(
    gruneisen_doc: dict[str, Any],
    bulk_modulus: float | None = None,
    temperature_range: tuple[float, float] = (0, 1000),
    n_points: int = 101,
) -> dict[str, Any]:
    """
    Calculate volumetric thermal expansion coefficient from Grüneisen parameters.

    Uses the relationship: α_V = γ · C_V / (B · V)
    where γ is the average Grüneisen parameter, C_V is heat capacity,
    B is bulk modulus, and V is volume.

    Parameters
    ----------
    gruneisen_doc : dict
        Grüneisen calculation results
    bulk_modulus : float or None
        Bulk modulus in GPa. If None, attempts to extract from elastic tensor
        or uses estimated value
    temperature_range : tuple
        (T_min, T_max) in Kelvin
    n_points : int
        Number of temperature points to calculate

    Returns
    -------
    dict
        Dictionary containing:
        - temperatures: array of temperatures (K)
        - alpha_v: volumetric thermal expansion coefficients (K⁻¹)
        - alpha_l: linear thermal expansion coefficients (K⁻¹)
        - bulk_modulus: bulk modulus used (GPa)

    Notes
    -----
    If bulk modulus is not provided, the function will attempt to estimate it
    using typical values for the material type, but this is less accurate.
    For quantitative results, always provide the bulk modulus from elastic
    constant calculations.
    """
    try:
        from pymatgen.phonon.gruneisen import GruneisenParameter
    except ImportError as e:
        raise ImportError("Install pymatgen for thermal expansion calculation") from e

    logger.info("Calculating thermal expansion from Grüneisen parameters")

    # Extract necessary data
    grun_param = _get_attr(gruneisen_doc, "gruneisen_parameter")
    if grun_param is None or not isinstance(grun_param, GruneisenParameter):
        raise ValueError("No Grüneisen parameter data available")

    # Get volume (from structure object or metadata field)
    structure = _get_attr(gruneisen_doc, "structure")
    if structure:
        volume = structure.volume  # Å³
        n_atoms = len(structure)
    else:
        # Pydantic case: get from metadata
        volume = _get_attr(gruneisen_doc, "volume")
        n_atoms = _get_attr(gruneisen_doc, "nsites")
        if volume is None:
            raise ValueError("Volume not available in Grüneisen document")
        if n_atoms is None:
            raise ValueError("Number of atoms not available in Grüneisen document")

    derived = _get_attr(gruneisen_doc, "derived_properties")
    avg_gruneisen = _get_attr(derived, "average_gruneisen") if derived else None

    if avg_gruneisen is None:
        raise ValueError("Average Grüneisen parameter not available")

    # Get or estimate bulk modulus
    if bulk_modulus is None:
        # Try to estimate based on material type
        # This is very approximate - user should provide real value
        logger.warning(
            "Bulk modulus not provided. Using rough estimate. "
            "For accurate results, calculate elastic constants."
        )
        # Very rough estimates based on typical materials
        density = (
            structure.density if structure else _get_attr(gruneisen_doc, "density", 3.0)
        )
        if density < 2:  # Light elements
            bulk_modulus = 50  # GPa
        elif density < 5:  # Medium
            bulk_modulus = 100
        else:  # Heavy/transition metals
            bulk_modulus = 150

    logger.info(f"Using bulk modulus: {bulk_modulus:.1f} GPa")

    # Convert bulk modulus to eV/Å³
    # 1 GPa = 0.00624 eV/Å³
    bulk_modulus_ev = bulk_modulus * 0.00624

    # Generate temperature array
    temperatures = np.linspace(temperature_range[0], temperature_range[1], n_points)

    # Calculate heat capacity at each temperature
    # Using Debye model approximation based on phonon frequencies
    cv_array = np.zeros(n_points)

    # Get phonon frequencies and calculate Debye temperature roughly
    frequencies = grun_param.frequencies.flatten()
    # Filter physical modes
    mask = frequencies > 0.1
    frequencies = frequencies[mask]

    # Estimate Debye temperature from max frequency
    # ω_D = k_B * T_D / ℏ, with ω in THz: T_D ≈ 47.99 * ω_max (THz)
    omega_max = np.max(frequencies)
    debye_temp = 47.99 * omega_max  # K

    logger.info(f"Estimated Debye temperature: {debye_temp:.1f} K")

    # Calculate heat capacity using Debye model
    k_B = 8.617333e-5  # eV/K (Boltzmann constant)

    for i, T in enumerate(temperatures):
        if T < 1e-6:  # Avoid division by zero
            cv_array[i] = 0
        else:
            # Simplified Debye model
            x = debye_temp / T
            if x < 0.01:  # High temperature limit
                cv_array[i] = 3 * n_atoms * k_B  # Dulong-Petit limit
            else:
                # Numerical integration would be more accurate
                # Using approximate formula
                cv_array[i] = (
                    3
                    * n_atoms
                    * k_B
                    * (
                        12
                        * (T / debye_temp) ** 3
                        * np.sum(
                            [(i / (np.exp(i) - 1)) for i in np.linspace(0.01, x, 100)]
                        )
                        / 100
                    )
                )

    # Calculate volumetric thermal expansion: α_V = γ · C_V / (B · V)
    alpha_v = avg_gruneisen * cv_array / (bulk_modulus_ev * volume)  # K⁻¹

    # Calculate linear thermal expansion: α_L = α_V / 3
    alpha_l = alpha_v / 3  # K⁻¹

    result = {
        "temperatures": temperatures.tolist(),
        "alpha_v": alpha_v.tolist(),  # Volumetric (K⁻¹)
        "alpha_l": alpha_l.tolist(),  # Linear (K⁻¹)
        "bulk_modulus": bulk_modulus,  # GPa
        "debye_temperature": debye_temp,  # K
        "average_gruneisen": avg_gruneisen,
    }

    logger.info("Thermal expansion calculation complete")
    return result


@job
def plot_thermal_expansion(
    thermal_expansion_data: dict[str, Any],
    output_dir: str | Path = ".",
    filename: str = "thermal_expansion.png",
    figsize: tuple[float, float] = (10, 7),
    dpi: int = 300,
) -> dict[str, str]:
    """
    Plot thermal expansion coefficient vs temperature.

    Parameters
    ----------
    thermal_expansion_data : dict
        Output from calculate_thermal_expansion
    output_dir : str | Path
        Directory to save plots
    filename : str
        Output filename
    figsize : tuple
        Figure size
    dpi : int
        Resolution

    Returns
    -------
    dict
        Path to generated file
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Install matplotlib for plotting") from e

    logger.info("Plotting thermal expansion vs temperature")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temps = np.array(thermal_expansion_data["temperatures"])
    alpha_v = np.array(thermal_expansion_data["alpha_v"]) * 1e6  # Convert to 10⁻⁶ K⁻¹
    alpha_l = np.array(thermal_expansion_data["alpha_l"]) * 1e6  # Convert to 10⁻⁶ K⁻¹

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(temps, alpha_v, "r-", linewidth=2.5, label="Volumetric (α_V)", alpha=0.8)
    ax.plot(temps, alpha_l, "b-", linewidth=2.5, label="Linear (α_L)", alpha=0.8)

    # Mark room temperature
    idx_300 = np.argmin(np.abs(temps - 300))
    ax.plot(
        temps[idx_300],
        alpha_v[idx_300],
        "ro",
        markersize=10,
        label=f"α_V(300K) = {alpha_v[idx_300]:.2f} × 10⁻⁶ K⁻¹",
    )
    ax.plot(
        temps[idx_300],
        alpha_l[idx_300],
        "bo",
        markersize=10,
        label=f"α_L(300K) = {alpha_l[idx_300]:.2f} × 10⁻⁶ K⁻¹",
    )

    # Format plot
    ax.set_xlabel("Temperature (K)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Thermal Expansion (10⁻⁶ K⁻¹)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Thermal Expansion Coefficient vs Temperature", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim(temps[0], temps[-1])

    # Add parameters text
    bulk_mod = thermal_expansion_data.get("bulk_modulus")
    avg_grun = thermal_expansion_data.get("average_gruneisen")
    debye_temp = thermal_expansion_data.get("debye_temperature")

    params_text = (
        f"Parameters:\n"
        f"γ_avg = {avg_grun:.3f}\n"
        f"B = {bulk_mod:.1f} GPa\n"
        f"Θ_D ≈ {debye_temp:.0f} K"
    )
    ax.text(
        0.02,
        0.98,
        params_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    expansion_file = output_dir / filename
    plt.savefig(expansion_file, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Thermal expansion plot saved to {expansion_file}")
    return {"thermal_expansion_plot": str(expansion_file)}
