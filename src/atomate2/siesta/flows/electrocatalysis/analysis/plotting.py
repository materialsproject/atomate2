"""Plotting functions for electrocatalysis analysis.

This module provides plotting utilities for visualizing:
- Free energy diagrams (ΔG vs. reaction coordinate)
- Volcano plots (overpotential vs. binding energy)
- Bifunctional performance maps
- Pathway comparison plots

All plots are publication-quality with proper labeling and formatting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


def plot_free_energy_diagram(
    step_labels: Sequence[str],
    cumulative_G: Sequence[float],  # noqa: N803
    delta_G: Sequence[float],  # noqa: N803
    pathway_type: str = "ORR",
    filename: str | Path = "free_energy_diagram.png",
    show_values: bool = True,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """
    Plot free energy diagram for electrochemical reaction pathway.

    This is the most important visualization for electrocatalysis analysis.
    Shows ΔG for each step along the reaction coordinate.

    Parameters
    ----------
    step_labels : Sequence[str]
        Labels for each reaction step.
    cumulative_G : Sequence[float]
        Cumulative free energies (eV).
    delta_G : Sequence[float]
        Free energy changes for each step (eV).
    pathway_type : str
        Type of pathway ('ORR', 'OER', 'HER', etc.).
    filename : str | Path
        Output filename.
    show_values : bool
        Whether to show ΔG values on arrows.
    figsize : tuple[float, float]
        Figure size (width, height) in inches.

    Returns
    -------
    Path
        Path to saved plot file.

    Examples
    --------
    >>> step_labels = ["*", "O2*", "OOH*", "O*", "OH*", "H2O"]
    >>> cumulative_G = [0.0, 0.45, 1.65, 2.45, 3.05, 4.92]
    >>> delta_G = [0.45, 1.20, 0.80, 0.60, 1.87]
    >>> plot_free_energy_diagram(step_labels, cumulative_G, delta_G, "ORR")
    """
    _fig, ax = plt.subplots(figsize=figsize)

    # Ensure step_labels includes initial state
    # cumulative_G has n+1 elements (initial state + n steps)
    # step_labels has n elements (just the steps)
    # Prepend '*' for initial state if needed
    if len(step_labels) == len(cumulative_G) - 1:
        step_labels = ["*", *list(step_labels)]
    elif len(step_labels) != len(cumulative_G):
        raise ValueError(
            f"step_labels length ({len(step_labels)}) must equal "
            f"cumulative_G length ({len(cumulative_G)}) or be 1 less"
        )

    # Create reaction coordinate (x-axis)
    n_steps = len(cumulative_G)
    x_coords = np.arange(n_steps)

    # Plot free energy profile as step function
    for i in range(n_steps - 1):
        # Horizontal line for current state
        ax.plot(
            [x_coords[i], x_coords[i] + 0.8],
            [cumulative_G[i], cumulative_G[i]],
            "b-",
            linewidth=2,
        )

        # Vertical transition to next state
        ax.plot(
            [x_coords[i] + 0.8, x_coords[i] + 0.8],
            [cumulative_G[i], cumulative_G[i + 1]],
            "b--",
            linewidth=1.5,
            alpha=0.5,
        )

        # Arrow showing ΔG
        mid_x = x_coords[i] + 0.9
        mid_y = (cumulative_G[i] + cumulative_G[i + 1]) / 2
        dG = delta_G[i]  # noqa: N806

        # Color arrows by uphill/downhill
        arrow_color = "red" if dG > 0 else "green"

        if show_values:
            ax.annotate(
                f"ΔG = {dG:+.2f} eV",
                xy=(mid_x, cumulative_G[i + 1]),
                xytext=(mid_x + 0.3, mid_y),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5),
                fontsize=9,
                color=arrow_color,
            )

    # Final horizontal line
    ax.plot(
        [x_coords[-1], x_coords[-1] + 0.2],
        [cumulative_G[-1], cumulative_G[-1]],
        "b-",
        linewidth=2,
    )

    # Mark initial and final states
    ax.scatter(
        x_coords[0],
        cumulative_G[0],
        s=100,
        c="green",
        marker="o",
        zorder=5,
        label="Initial state",
    )
    ax.scatter(
        x_coords[-1],
        cumulative_G[-1],
        s=100,
        c="blue",
        marker="s",
        zorder=5,
        label="Final state",
    )

    # Add reference line at ΔG = 0
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    # Formatting
    ax.set_xlabel("Reaction Coordinate", fontsize=12, fontweight="bold")
    ax.set_ylabel("Free Energy, ΔG (eV)", fontsize=12, fontweight="bold")
    ax.set_title(f"{pathway_type} Free Energy Diagram", fontsize=14, fontweight="bold")

    # Set x-axis labels
    ax.set_xticks(x_coords)
    ax.set_xticklabels(step_labels, rotation=45, ha="right")

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--")  # noqa: FBT003
    ax.legend(loc="best", fontsize=10)

    # Tight layout
    plt.tight_layout()

    # Save
    output_path = Path(filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved free energy diagram to {output_path}")
    return output_path


def plot_overpotential_summary(
    pathway_type: str,
    overpotential: float,
    rls_label: str,
    rls_delta_G: float,  # noqa: N803
    U_onset: float,  # noqa: N803
    filename: str | Path = "overpotential_summary.png",
    figsize: tuple[float, float] = (8, 6),
) -> Path:
    """
    Plot overpotential summary with visual performance indicator.

    Creates a bar chart showing overpotential with color-coded performance rating.

    Parameters
    ----------
    pathway_type : str
        Type of pathway ('ORR', 'OER', 'HER').
    overpotential : float
        Overpotential value (V).
    rls_label : str
        Rate-limiting step label.
    rls_delta_G : float
        RLS free energy barrier (eV).
    U_onset : float
        Onset potential (V vs. SHE).
    filename : str | Path
        Output filename.
    figsize : tuple[float, float]
        Figure size.

    Returns
    -------
    Path
        Path to saved plot.
    """
    _fig, ax = plt.subplots(figsize=figsize)

    # Performance rating colors
    if pathway_type == "ORR":
        # ORR: excellent < 0.3, good < 0.5, moderate < 0.8, poor > 0.8
        if overpotential < 0.3:
            color, rating = "darkgreen", "Excellent"
        elif overpotential < 0.5:
            color, rating = "green", "Good"
        elif overpotential < 0.8:
            color, rating = "orange", "Moderate"
        else:
            color, rating = "red", "Poor"
    elif pathway_type == "OER":
        # OER: excellent < 0.4, good < 0.6, moderate < 0.8, poor > 0.8
        if overpotential < 0.4:
            color, rating = "darkgreen", "Excellent"
        elif overpotential < 0.6:
            color, rating = "green", "Good"
        elif overpotential < 0.8:
            color, rating = "orange", "Moderate"
        else:
            color, rating = "red", "Poor"
    # HER: excellent < 0.1, good < 0.2, moderate < 0.4, poor > 0.4
    elif overpotential < 0.1:
        color, rating = "darkgreen", "Excellent"
    elif overpotential < 0.2:
        color, rating = "green", "Good"
    elif overpotential < 0.4:
        color, rating = "orange", "Moderate"
    else:
        color, rating = "red", "Poor"

    # Bar plot
    ax.bar(
        [pathway_type],
        [overpotential],
        color=color,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )

    # Add value label on bar
    ax.text(
        0,
        overpotential + 0.05,
        f"η = {overpotential:.3f} V",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

    # Add rating
    ax.text(
        0,
        overpotential / 2,
        rating,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="white",
    )

    # Formatting
    ax.set_ylabel("Overpotential, η (V)", fontsize=12, fontweight="bold")
    ax.set_title(f"{pathway_type} Performance Summary", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(1.0, overpotential * 1.2))

    # Add info text box
    info_text = (
        f"Rate-Limiting Step: {rls_label}\n"
        f"ΔG_RLS = {rls_delta_G:.3f} eV\n"
        f"U_onset = {U_onset:.3f} V vs. SHE"
    )
    ax.text(
        0.95,
        0.95,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Grid
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")  # noqa: FBT003

    plt.tight_layout()

    output_path = Path(filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved overpotential summary to {output_path}")
    return output_path


def plot_bifunctional_comparison(
    eta_ORR: float,  # noqa: N803
    eta_OER: float,  # noqa: N803
    gap: float,
    filename: str | Path = "bifunctional_comparison.png",
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """
    Plot bifunctional catalyst performance comparison.

    Shows ORR and OER overpotentials side-by-side with gap visualization.

    Parameters
    ----------
    eta_ORR : float
        ORR overpotential (V).
    eta_OER : float
        OER overpotential (V).
    gap : float
        Overpotential gap (η_ORR + η_OER) in V.
    filename : str | Path
        Output filename.
    figsize : tuple[float, float]
        Figure size.

    Returns
    -------
    Path
        Path to saved plot.
    """
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Individual overpotentials
    reactions = ["ORR", "OER"]
    overpotentials = [eta_ORR, eta_OER]
    colors = ["blue", "red"]

    bars = ax1.bar(
        reactions,
        overpotentials,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )

    # Add value labels
    for bar, eta in zip(bars, overpotentials, strict=False):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.05,
            f"{eta:.3f} V",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax1.set_ylabel("Overpotential, η (V)", fontsize=12, fontweight="bold")
    ax1.set_title("Individual Overpotentials", fontsize=14, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")  # noqa: FBT003
    ax1.set_ylim(0, max(overpotentials) * 1.3)

    # Right plot: Bifunctional gap
    gap_color = (
        "darkgreen"
        if gap < 0.4
        else "green"
        if gap < 0.6
        else "orange"
        if gap < 0.8
        else "red"
    )

    gap_rating = (
        "Excellent"
        if gap < 0.4
        else "Very Good"
        if gap < 0.6
        else "Good"
        if gap < 0.8
        else "Moderate"
        if gap < 1.0
        else "Poor"
    )

    bar = ax2.bar(
        ["Gap"], [gap], color=gap_color, alpha=0.7, edgecolor="black", linewidth=2
    )

    ax2.text(
        0,
        gap + 0.1,
        f"{gap:.3f} V",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )
    ax2.text(
        0,
        gap / 2,
        gap_rating,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="white",
    )

    ax2.set_ylabel(
        "Overpotential Gap, η_ORR + η_OER (V)", fontsize=12, fontweight="bold"
    )
    ax2.set_title("Bifunctional Performance", fontsize=14, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3, linestyle="--")  # noqa: FBT003
    ax2.set_ylim(0, max(2.0, gap * 1.3))

    # Add performance guide
    guide_text = (
        "Performance Guide:\n"
        "Gap < 0.4 V: Excellent\n"
        "Gap < 0.6 V: Very Good\n"
        "Gap < 0.8 V: Good\n"
        "Gap > 1.0 V: Poor"
    )
    ax2.text(
        0.95,
        0.95,
        guide_text,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()

    output_path = Path(filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved bifunctional comparison to {output_path}")
    return output_path


def write_analysis_summary(
    pathway_type: str,
    surface_name: str,
    overpotential: float,
    rls_label: str,
    rls_delta_G: float,  # noqa: N803
    step_labels: Sequence[str],
    delta_G: Sequence[float],  # noqa: N803
    filename: str | Path = "analysis_summary.txt",
    dry_run: bool = False,
) -> Path:
    """
    Write text summary of electrocatalysis analysis.

    Parameters
    ----------
    pathway_type : str
        Type of pathway ('ORR', 'OER', 'HER').
    surface_name : str
        Name of catalyst surface.
    overpotential : float
        Overpotential (V).
    rls_label : str
        Rate-limiting step label.
    rls_delta_G : float
        RLS free energy barrier (eV).
    step_labels : Sequence[str]
        Labels for all steps.
    delta_G : Sequence[float]
        Free energy changes (eV).
    filename : str | Path
        Output filename.
    dry_run : bool
        Whether this is from a dry-run (no actual calculations). Default: False.

    Returns
    -------
    Path
        Path to saved summary file.
    """
    output_path = Path(filename)

    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"{pathway_type} ELECTROCATALYSIS ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        # Add prominent dry-run warning
        if dry_run:
            f.write("!" * 70 + "\n")
            f.write(
                "!!! WARNING: DRY-RUN MODE - NO ACTUAL CALCULATIONS PERFORMED !!!\n"
            )
            f.write("!" * 70 + "\n\n")
            f.write("This summary is generated from DRY-RUN data where:\n")
            f.write("  • NO SIESTA calculations were executed\n")
            f.write("  • All DFT energies are ZERO (0.0 eV)\n")
            f.write("  • Free energies shown are ONLY from thermodynamic corrections\n")
            f.write(
                "  • Overpotentials are NOT realistic - they only reflect standard\n"
            )
            f.write("    ZPE/entropy corrections, NOT actual binding energies!\n\n")
            f.write("To perform actual calculations:\n")
            f.write("  1. Set dry_run=False in your workflow maker\n")
            f.write("  2. Run the workflow with SIESTA\n")
            f.write("  3. Compare results with this dry-run preview\n\n")
            f.write("=" * 70 + "\n\n")

        f.write(f"Catalyst Surface: {surface_name}\n")
        f.write(f"Reaction: {pathway_type}\n\n")

        # Add reaction description
        f.write("-" * 70 + "\n")
        f.write("REACTION OVERVIEW\n")
        f.write("-" * 70 + "\n\n")

        if pathway_type == "ORR":
            f.write("Oxygen Reduction Reaction (ORR)\n")
            f.write("Cathode reaction in fuel cells and metal-air batteries:\n")
            f.write("  O₂ + 4H⁺ + 4e⁻ → 2H₂O  (E° = 1.23 V vs. SHE)\n\n")
            f.write("ORR Pathway (4-electron reduction):\n")
            f.write("  1. * + O₂ → O₂*\n")
            f.write("  2. O₂* + H⁺ + e⁻ → OOH*\n")
            f.write("  3. OOH* + H⁺ + e⁻ → O* + H₂O\n")
            f.write("  4. O* + H⁺ + e⁻ → OH*\n")
            f.write("  5. OH* + H⁺ + e⁻ → * + H₂O\n\n")
        elif pathway_type == "OER":
            f.write("Oxygen Evolution Reaction (OER)\n")
            f.write("Anode reaction in water electrolysis:\n")
            f.write("  2H₂O → O₂ + 4H⁺ + 4e⁻  (E° = 1.23 V vs. SHE)\n\n")
            f.write("OER Pathway (4-electron oxidation):\n")
            f.write("  1. * + H₂O → OH* + H⁺ + e⁻\n")
            f.write("  2. OH* → O* + H⁺ + e⁻\n")
            f.write("  3. O* + H₂O → OOH* + H⁺ + e⁻\n")
            f.write("  4. OOH* → O₂* + H⁺ + e⁻\n")
            f.write("  5. O₂* → O₂(g)\n\n")
        else:  # HER
            f.write("Hydrogen Evolution Reaction (HER)\n")
            f.write("Cathode reaction in water electrolysis:\n")
            f.write("  Acidic:   2H⁺ + 2e⁻ → H₂         (E° = 0.00 V vs. SHE)\n")
            f.write("  Alkaline: 2H₂O + 2e⁻ → H₂ + 2OH⁻ (E° = -0.83 V vs. SHE)\n\n")
            f.write("HER Pathway (Volmer-Heyrovsky or Volmer-Tafel):\n")
            f.write("  1. * + H⁺ + e⁻ → H*     (Volmer step)\n")
            f.write("  2a. H* + H⁺ + e⁻ → H₂   (Heyrovsky step)\n")
            f.write("  2b. 2H* → H₂            (Tafel step)\n\n")

        f.write("-" * 70 + "\n")
        f.write("COMPUTATIONAL METHODOLOGY\n")
        f.write("-" * 70 + "\n\n")
        f.write("Computational Hydrogen Electrode (CHE) Model:\n")
        f.write("  - μ(H⁺ + e⁻) = ½μ(H₂) - eU - k_B T ln(10) × pH\n")  # noqa: RUF001
        f.write("  - Free energies calculated at T = 298.15 K, p = 1 atm\n")
        f.write("  - Zero-point energy (ZPE) and entropy corrections included\n\n")

        if pathway_type in ["ORR", "OER"]:
            f.write("For ORR/OER:\n")
            f.write("  - At U = 1.23 V, reaction is thermoneutral (ΔG = 0)\n")
            f.write("  - Overpotential η = max(ΔG_i) for rate-limiting step\n")
            f.write("  - Lower overpotential = better catalyst\n\n")
        else:  # HER
            f.write("For HER:\n")
            f.write("  - Optimal catalyst: ΔG_H* ≈ 0 eV (thermoneutral)\n")
            f.write("  - Overpotential η = |ΔG_H*| (Sabatier principle)\n")
            f.write("  - Volcano plot: η minimized when ΔG_H* = 0\n\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 70 + "\n\n")

        f.write(f"Overpotential: η = {overpotential:.3f} V\n")
        f.write(f"Rate-Limiting Step: {rls_label}\n")
        f.write(f"RLS Barrier: ΔG_RLS = {rls_delta_G:.3f} eV\n\n")

        # Performance rating
        if pathway_type == "ORR":
            if overpotential < 0.3:
                rating = "EXCELLENT (η < 0.3 V)"
            elif overpotential < 0.5:
                rating = "GOOD (η < 0.5 V)"
            elif overpotential < 0.8:
                rating = "MODERATE (η < 0.8 V)"
            else:
                rating = "POOR (η > 0.8 V)"
        elif pathway_type == "OER":
            if overpotential < 0.4:
                rating = "EXCELLENT (η < 0.4 V)"
            elif overpotential < 0.6:
                rating = "GOOD (η < 0.6 V)"
            elif overpotential < 0.8:
                rating = "MODERATE (η < 0.8 V)"
            else:
                rating = "POOR (η > 0.8 V)"
        elif overpotential < 0.1:
            rating = "EXCELLENT (η < 0.1 V)"
        elif overpotential < 0.2:
            rating = "GOOD (η < 0.2 V)"
        elif overpotential < 0.4:
            rating = "MODERATE (η < 0.4 V)"
        else:
            rating = "POOR (η > 0.4 V)"

        f.write(f"Performance Rating: {rating}\n\n")

        f.write("-" * 70 + "\n")
        f.write("FREE ENERGY PATHWAY\n")
        f.write("-" * 70 + "\n\n")

        for i, (label, dG) in enumerate(zip(step_labels, delta_G, strict=False)):  # noqa: N806
            arrow = "↑" if dG > 0 else "↓"
            f.write(f"Step {i + 1}: {label:20s} ΔG = {dG:+.3f} eV {arrow}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("INTERPRETATION\n")
        f.write("-" * 70 + "\n\n")

        f.write(f"The reaction is limited by the {rls_label} step.\n")
        f.write(
            f"This step has an uphill free energy change of {rls_delta_G:.3f} eV.\n\n"
        )

        if pathway_type == "ORR":
            f.write("Benchmark Catalysts:\n")
            f.write("  - Pt(111): η ≈ 0.45 V (state-of-the-art)\n")
            f.write("  - PtNi alloys: η ≈ 0.30 V (best reported)\n")
            f.write("  - Non-precious: Fe-N-C, Co-N-C (η ≈ 0.5-0.7 V)\n\n")
            f.write("Scaling Relations (Nørskov et al., 2004):\n")
            f.write("  - ΔG_OOH* ≈ ΔG_OH* + 3.2 eV (universal scaling)\n")
            f.write("  - Optimal Pt-like: ΔG_OH* ≈ 1.0 eV\n")
            f.write("  - Breaking scaling: dual-site or confined catalysts\n\n")
            f.write("Design Strategies:\n")
            f.write(
                "  - Weaken O* binding: reduce overpotential on strong-binding leg\n"
            )
            f.write(
                "  - Strengthen OH* binding: reduce overpotential on weak-binding leg\n"
            )
            f.write("  - Consider d-band center shifts (strain, alloying, doping)\n")
        elif pathway_type == "OER":
            f.write("Benchmark Catalysts:\n")
            f.write("  - RuO₂/IrO₂: η ≈ 0.35-0.40 V (state-of-the-art, precious)\n")
            f.write("  - Perovskites (Ba₀.₅Sr₀.₅Co₀.₈Fe₀.₂O₃): η ≈ 0.40 V\n")
            f.write("  - Layered hydroxides (NiFe-LDH): η ≈ 0.30 V (alkaline)\n\n")
            f.write("Scaling Relations (Man et al., ChemCatChem 2011):\n")
            f.write("  - ΔG_OOH* ≈ ΔG_OH* + 3.2 eV (universal constraint)\n")
            f.write("  - Ideal catalyst: ΔG_O* - ΔG_OH* = 1.6 eV\n")
            f.write("  - Minimum theoretical η ≈ 0.37 V (fundamental limit)\n\n")
            f.write("Design Strategies:\n")
            f.write(
                "  - Overcome scaling: dual-site catalysts, lattice oxygen mechanism\n"
            )
            f.write("  - Stabilize OOH* independently of OH*\n")
            f.write("  - Consider oxide/hydroxide surfaces (dynamic restructuring)\n")
        else:  # HER
            f.write("Benchmark Catalysts:\n")
            f.write("  - Pt: ΔG_H* ≈ -0.09 eV, η ≈ 0.09 V (volcano peak)\n")
            f.write("  - MoS₂ edges: ΔG_H* ≈ +0.08 eV, η ≈ 0.08 V (excellent!)\n")
            f.write("  - Ni₂P: ΔG_H* ≈ -0.03 eV (near-optimal)\n")
            f.write("  - Strong binding: W, Mo, Nb (ΔG_H* < -0.5 eV, poor)\n")
            f.write("  - Weak binding: Au, Ag, Cu (ΔG_H* > +0.5 eV, poor)\n\n")
            f.write("Volcano Plot Analysis (Nørskov et al., 2005):\n")
            f.write("  - Optimal: |ΔG_H*| < 0.1 eV (thermoneutral)\n")
            f.write("  - Strong binding leg: H* poisoning (desorption limited)\n")
            f.write("  - Weak binding leg: no H adsorption (adsorption limited)\n\n")
            f.write("Design Strategies:\n")
            f.write("  - Target ΔG_H* ≈ 0 eV (Sabatier principle)\n")
            f.write("  - Modify d-band center: alloying, strain engineering\n")
            f.write("  - Edge sites or defects (MoS₂, WS₂, phosphides)\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 70 + "\n\n")

        if pathway_type == "ORR":
            f.write("Next Steps for ORR Optimization:\n")
            f.write(
                "  1. Analyze adsorption site geometries (coordination, distances)\n"
            )
            f.write("  2. Test alloying/doping to tune binding energies\n")
            f.write("  3. Calculate descriptor: ΔG_O* - ΔG_OH* for volcano position\n")
            f.write("  4. Consider surface modifications (defects, edges, dopants)\n")
            f.write("  5. Validate with microkinetic modeling (coverage effects)\n")
        elif pathway_type == "OER":
            f.write("Next Steps for OER Optimization:\n")
            f.write("  1. Calculate ΔG_O* - ΔG_OH* descriptor (target 1.6 eV)\n")
            f.write("  2. Investigate dual-site catalysts to break scaling\n")
            f.write("  3. Test oxide/hydroxide surfaces (dynamic effects)\n")
            f.write("  4. Consider lattice oxygen mechanism (alternative pathway)\n")
            f.write("  5. Stability analysis (dissolution, reconstruction)\n")
        else:  # HER
            f.write("Next Steps for HER Optimization:\n")
            f.write("  1. Plot volcano curve: η vs ΔG_H* for material series\n")
            f.write("  2. Test different sites (edge, terrace, defects)\n")
            f.write("  3. Alloying/doping to shift ΔG_H* toward 0 eV\n")
            f.write("  4. Consider pH effects (alkaline vs acidic HER)\n")
            f.write("  5. Microkinetic modeling (Tafel vs Heyrovsky mechanism)\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("REFERENCES\n")
        f.write("-" * 70 + "\n\n")

        if pathway_type == "ORR":
            f.write("Key Publications:\n")
            f.write("  - Nørskov et al., J. Phys. Chem. B 108, 17886 (2004)\n")
            f.write("    Scaling relations for ORR/OER intermediates\n")
            f.write("  - Stamenkovic et al., Science 315, 493 (2007)\n")
            f.write("    PtNi alloy catalysts (record ORR activity)\n")
            f.write("  - Kulkarni et al., Chem. Rev. 118, 2302 (2018)\n")
            f.write("    Comprehensive review of ORR descriptors\n")
        elif pathway_type == "OER":
            f.write("Key Publications:\n")
            f.write("  - Man et al., ChemCatChem 3, 1159 (2011)\n")
            f.write("    Universal scaling relations and minimum overpotential\n")
            f.write("  - Suntivich et al., Science 334, 1383 (2011)\n")
            f.write("    Perovskite OER catalysts (e_g occupancy descriptor)\n")
            f.write("  - Grimaud et al., Nat. Energy 2, 16189 (2017)\n")
            f.write("    Lattice oxygen mechanism for breaking scaling\n")
        else:  # HER
            f.write("Key Publications:\n")
            f.write("  - Nørskov et al., J. Electrochem. Soc. 152, J23 (2005)\n")
            f.write("    HER volcano plot and activity trends\n")
            f.write("  - Greeley et al., Nat. Mater. 5, 909 (2006)\n")
            f.write("    Computational screening for HER catalysts\n")
            f.write("  - Zheng et al., Science 338, 1321 (2012)\n")
            f.write("    MoS₂ edge sites as excellent HER catalyst\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n")

    logger.info(f"Saved analysis summary to {output_path}")
    return output_path
