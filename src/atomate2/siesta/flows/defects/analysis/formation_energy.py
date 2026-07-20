"""Formation energy diagram analysis and plotting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
from jobflow import Flow, job
from matplotlib.patches import Rectangle
from monty.json import MSONable

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from jobflow import Job
    from pymatgen.core import Structure

    from atomate2.siesta.flows.defects.schemas import DefectDocument

logger = logging.getLogger(__name__)


@dataclass
class DefectFormationEnergyData(MSONable):
    """
    Data for a single defect type at various charge states.

    Parameters
    ----------
    defect_name : str
        Name of the defect (e.g., "V_O", "V_Mg", "Li_Mg")
    defect_type : str
        Type of defect ("vacancy", "substitution", "interstitial")
    charge_states : list[int]
        List of charge states
    formation_energies : list[float]
        Formation energies at each charge state (in eV)
    corrected : bool
        If True, energies include finite-size corrections
    """

    defect_name: str
    defect_type: str
    charge_states: list[int]
    formation_energies: list[float]
    corrected: bool = True

    def as_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "defect_name": self.defect_name,
            "defect_type": self.defect_type,
            "charge_states": self.charge_states,
            "formation_energies": self.formation_energies,
            "corrected": self.corrected,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DefectFormationEnergyData:
        """Deserialize from dictionary."""
        return cls(
            defect_name=d["defect_name"],
            defect_type=d["defect_type"],
            charge_states=d["charge_states"],
            formation_energies=d["formation_energies"],
            corrected=d.get("corrected", True),
        )


@dataclass
class ChargeTransitionLevel(MSONable):
    """
    Charge transition level (CTL) between two charge states.

    The CTL ε(q₁/q₂) is the Fermi level at which the formation energies
    of charge states q₁ and q₂ are equal.

    Parameters
    ----------
    defect_name : str
        Name of the defect
    q1 : int
        First charge state
    q2 : int
        Second charge state
    fermi_level : float
        Fermi level position of the transition (in eV)
    formation_energy : float
        Formation energy at the transition point (in eV)
    """

    defect_name: str
    q1: int
    q2: int
    fermi_level: float
    formation_energy: float

    def as_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "defect_name": self.defect_name,
            "q1": self.q1,
            "q2": self.q2,
            "fermi_level": self.fermi_level,
            "formation_energy": self.formation_energy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ChargeTransitionLevel:
        """Deserialize from dictionary."""
        return cls(
            defect_name=d["defect_name"],
            q1=d["q1"],
            q2=d["q2"],
            fermi_level=d["fermi_level"],
            formation_energy=d["formation_energy"],
        )


@dataclass
class FormationEnergyDiagram(MSONable):
    """
    Formation energy diagram for multiple defects.

    This class holds formation energy data for multiple defects and
    provides methods to plot formation energy diagrams and calculate
    charge transition levels.

    Parameters
    ----------
    defects : list[DefectFormationEnergyData]
        List of defect data
    bandgap : float
        Band gap of the host material (in eV)
    vbm_energy : float, optional
        Valence band maximum energy (in eV). If None, VBM is set to 0.
    """

    defects: list[DefectFormationEnergyData] = field(default_factory=list)
    bandgap: float = 0.0
    vbm_energy: float = 0.0

    def as_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "defects": [d.as_dict() for d in self.defects],
            "bandgap": self.bandgap,
            "vbm_energy": self.vbm_energy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FormationEnergyDiagram:
        """Deserialize from dictionary."""
        return cls(
            defects=[DefectFormationEnergyData.from_dict(dd) for dd in d["defects"]],
            bandgap=d["bandgap"],
            vbm_energy=d.get("vbm_energy", 0.0),
        )

    @classmethod
    def from_defect_documents(
        cls,
        defect_docs: list[DefectDocument],
        bandgap: float,
        vbm_energy: float = 0.0,
        use_corrected: bool = True,
    ) -> FormationEnergyDiagram:
        """
        Create FormationEnergyDiagram from DefectDocument objects.

        Parameters
        ----------
        defect_docs : list[DefectDocument]
            List of DefectDocument objects from defect calculations
        bandgap : float
            Band gap of the host material (in eV)
        vbm_energy : float
            Valence band maximum energy (in eV). Default: 0.0
        use_corrected : bool
            If True, use corrected formation energies. Default: True

        Returns
        -------
        FormationEnergyDiagram
            Formation energy diagram object
        """
        # Group defects by name
        defect_groups: dict[str, list[DefectDocument]] = {}
        for doc in defect_docs:
            name = doc.defect_species or doc.defect_type
            if name not in defect_groups:
                defect_groups[name] = []
            defect_groups[name].append(doc)

        # Create DefectFormationEnergyData for each group
        defects = []
        for name, docs in defect_groups.items():
            # Sort by charge state
            sorted_docs = sorted(docs, key=lambda d: d.charge_state)

            charge_states = [d.charge_state for d in sorted_docs]
            if use_corrected:
                formation_energies = [d.corrected_formation_energy for d in sorted_docs]
            else:
                formation_energies = [d.raw_formation_energy for d in sorted_docs]

            defect_type = sorted_docs[0].defect_type

            defect_data = DefectFormationEnergyData(
                defect_name=name,
                defect_type=defect_type,
                charge_states=charge_states,
                formation_energies=formation_energies,
                corrected=use_corrected,
            )
            defects.append(defect_data)

        return cls(defects=defects, bandgap=bandgap, vbm_energy=vbm_energy)

    def calculate_charge_transition_levels(self) -> list[ChargeTransitionLevel]:
        """
        Calculate all charge transition levels.

        Returns
        -------
        list[ChargeTransitionLevel]
            List of charge transition levels
        """
        ctls = []
        for defect in self.defects:
            defect_ctls = calculate_charge_transition_levels(
                defect.charge_states, defect.formation_energies, defect.defect_name
            )
            ctls.extend(defect_ctls)
        return ctls

    def plot(
        self,
        fermi_range: tuple[float, float] | None = None,
        show_ctls: bool = True,
        show_stable_regions: bool = True,
        figsize: tuple[float, float] = (10, 7),
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot formation energy diagram.

        Parameters
        ----------
        fermi_range : tuple[float, float], optional
            Range of Fermi level to plot (E_F_min, E_F_max) in eV.
            If None, uses (VBM, VBM + bandgap)
        show_ctls : bool
            If True, mark charge transition levels. Default: True
        show_stable_regions : bool
            If True, shade stable charge state regions. Default: True
        figsize : tuple[float, float]
            Figure size (width, height). Default: (10, 7)
        save_path : str, optional
            Path to save figure. If None, figure is not saved.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            Matplotlib figure and axes objects
        """
        return plot_formation_energy_diagram(
            self,
            fermi_range=fermi_range,
            show_ctls=show_ctls,
            show_stable_regions=show_stable_regions,
            figsize=figsize,
            save_path=save_path,
        )


def calculate_charge_transition_levels(
    charge_states: list[int],
    formation_energies: list[float],
    defect_name: str = "",
) -> list[ChargeTransitionLevel]:
    """
    Calculate charge transition levels for a defect.

    The charge transition level ε(q₁/q₂) is the Fermi level at which
    the formation energies of charge states q₁ and q₂ are equal:

        E_formation(q₁, ε) = E_formation(q₂, ε)

    Since E_formation(q, E_F) = E_formation(q, 0) + q·E_F, we have:

        ε(q₁/q₂) = [E_formation(q₂, 0) - E_formation(q₁, 0)] / (q₁ - q₂)

    Parameters
    ----------
    charge_states : list[int]
        List of charge states (e.g., [0, +1, +2])
    formation_energies : list[float]
        Formation energies at E_F = 0 (in eV)
    defect_name : str
        Name of the defect (for labeling)

    Returns
    -------
    list[ChargeTransitionLevel]
        List of charge transition levels

    Example
    -------
    >>> charge_states = [0, +1, +2]
    >>> formation_energies = [1.0, 1.5, 2.5]
    >>> ctls = calculate_charge_transition_levels(charge_states, formation_energies)
    >>> for ctl in ctls:
    ...     print(f"ε({ctl.q1}/{ctl.q2}) = {ctl.fermi_level:.3f} eV")
    """
    if len(charge_states) != len(formation_energies):
        raise ValueError("charge_states and formation_energies must have same length")

    # Sort by charge state
    sorted_data = sorted(zip(charge_states, formation_energies, strict=False))
    charge_states_sorted = [d[0] for d in sorted_data]
    energies_sorted = [d[1] for d in sorted_data]

    ctls = []
    n = len(charge_states_sorted)

    for i in range(n - 1):
        q1 = charge_states_sorted[i]
        q2 = charge_states_sorted[i + 1]
        E1 = energies_sorted[i]  # noqa: N806
        E2 = energies_sorted[i + 1]  # noqa: N806

        # ε(q1/q2) = [E_formation(q2, 0) - E_formation(q1, 0)] / (q1 - q2)
        if q1 == q2:
            logger.warning(f"Duplicate charge state {q1}, skipping CTL calculation")
            continue

        fermi_level = (E2 - E1) / (q1 - q2)

        # Formation energy at transition point
        formation_energy = E1 + q1 * fermi_level

        ctl = ChargeTransitionLevel(
            defect_name=defect_name,
            q1=q1,
            q2=q2,
            fermi_level=fermi_level,
            formation_energy=formation_energy,
        )
        ctls.append(ctl)

    return ctls


def plot_formation_energy_diagram(
    diagram: FormationEnergyDiagram,
    fermi_range: tuple[float, float] | None = None,
    show_ctls: bool = True,
    show_stable_regions: bool = True,
    figsize: tuple[float, float] = (10, 7),
    save_path: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot formation energy diagram (E_formation vs E_F).

    This creates a formation energy diagram showing how defect formation
    energies vary with Fermi level position. Charge transition levels
    are marked, and stable charge state regions can be highlighted.

    Parameters
    ----------
    diagram : FormationEnergyDiagram
        Formation energy diagram object
    fermi_range : tuple[float, float], optional
        Range of Fermi level to plot (E_F_min, E_F_max) in eV.
        If None, uses (VBM, VBM + bandgap)
    show_ctls : bool
        If True, mark charge transition levels. Default: True
    show_stable_regions : bool
        If True, shade stable charge state regions. Default: True
    figsize : tuple[float, float]
        Figure size (width, height). Default: (10, 7)
    save_path : str, optional
        Path to save figure. If None, figure is not saved.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes objects

    Example
    -------
    >>> from atomate2.siesta.flows.defects.analysis import FormationEnergyDiagram
    >>> diagram = FormationEnergyDiagram.from_defect_documents(defect_docs, bandgap=3.8)
    >>> fig, ax = diagram.plot(show_ctls=True, save_path="formation_energy.png")
    """
    # Set Fermi range
    if fermi_range is None:
        E_F_min = diagram.vbm_energy  # noqa: N806
        E_F_max = diagram.vbm_energy + diagram.bandgap  # noqa: N806
    else:
        E_F_min, E_F_max = fermi_range  # noqa: N806

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color palette for defects
    colors = plt.cm.tab10(np.linspace(0, 1, len(diagram.defects)))

    # Define line styles for different charge states
    linestyles = {0: "-", 1: "--", 2: "-.", 3: ":", -1: "--", -2: "-.", -3: ":"}

    # Plot each defect
    for defect, base_color in zip(diagram.defects, colors, strict=False):
        # For each charge state, plot E_formation(E_F) = E_formation(0) + q·E_F
        E_F_range = np.linspace(E_F_min, E_F_max, 100)  # noqa: N806

        for q, E_formation_0 in zip(  # noqa: N806
            defect.charge_states, defect.formation_energies, strict=False
        ):
            E_formation = E_formation_0 + q * E_F_range  # noqa: N806

            # Color based on charge: positive = red, negative = blue,
            # neutral = base_color
            if q > 0:
                color = "red"
                label = f"{defect.defect_name} (q={q:+d})"
            elif q < 0:
                color = "blue"
                label = f"{defect.defect_name} (q={q:d})"
            else:
                color = base_color
                label = f"{defect.defect_name} (q=0)"

            # Line style based on absolute charge value
            linestyle = linestyles.get(abs(q), "-")

            ax.plot(
                E_F_range,
                E_formation,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                alpha=0.8,
            )

    # Calculate and plot charge transition levels
    if show_ctls:
        ctls = diagram.calculate_charge_transition_levels()
        for ctl in ctls:
            # Only show CTLs within the band gap
            if E_F_min <= ctl.fermi_level <= E_F_max:
                ax.axvline(
                    ctl.fermi_level,
                    color="gray",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                )
                # Add label
                ax.text(
                    ctl.fermi_level,
                    ax.get_ylim()[1] * 0.95,
                    f"ε({ctl.q1:+d}/{ctl.q2:+d})",
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=9,
                    color="gray",
                )

    # Shade stable regions (optional)
    if show_stable_regions:
        # Find convex hull for each defect to determine stable charge states
        for defect, base_color in zip(diagram.defects, colors, strict=False):
            E_F_range = np.linspace(E_F_min, E_F_max, 1000)  # noqa: N806

            # Calculate E_formation for all charge states at each E_F
            E_formation_matrix = np.zeros((len(defect.charge_states), len(E_F_range)))  # noqa: N806
            for i, (q, E_formation_0) in enumerate(  # noqa: N806
                zip(defect.charge_states, defect.formation_energies, strict=False)
            ):
                E_formation_matrix[i] = E_formation_0 + q * E_F_range

            # Find minimum (stable) charge state at each E_F
            stable_q_indices = np.argmin(E_formation_matrix, axis=0)

            # Identify regions where charge state changes
            transitions = np.where(np.diff(stable_q_indices) != 0)[0]

            # Shade each stable region with color based on charge
            prev_transition = 0
            for trans_idx in transitions:
                stable_q_idx = stable_q_indices[prev_transition]
                stable_q = defect.charge_states[stable_q_idx]

                # Color based on charge state
                if stable_q > 0:
                    shade_color = "red"
                elif stable_q < 0:
                    shade_color = "blue"
                else:
                    shade_color = base_color

                E_F_left = E_F_range[prev_transition]  # noqa: N806
                E_F_right = E_F_range[trans_idx]  # noqa: N806

                rect = Rectangle(
                    (E_F_left, ax.get_ylim()[0]),
                    E_F_right - E_F_left,
                    ax.get_ylim()[1] - ax.get_ylim()[0],
                    facecolor=shade_color,
                    alpha=0.1,
                    edgecolor="none",
                )
                ax.add_patch(rect)

                prev_transition = trans_idx + 1

            # Last region
            stable_q_idx = stable_q_indices[prev_transition]
            stable_q = defect.charge_states[stable_q_idx]

            if stable_q > 0:
                shade_color = "red"
            elif stable_q < 0:
                shade_color = "blue"
            else:
                shade_color = base_color

            E_F_left = E_F_range[prev_transition]  # noqa: N806
            E_F_right = E_F_range[-1]  # noqa: N806
            rect = Rectangle(
                (E_F_left, ax.get_ylim()[0]),
                E_F_right - E_F_left,
                ax.get_ylim()[1] - ax.get_ylim()[0],
                facecolor=shade_color,
                alpha=0.1,
                edgecolor="none",
            )
            ax.add_patch(rect)

    # Mark VBM and CBM
    ax.axvline(
        diagram.vbm_energy, color="darkgreen", linestyle="-", linewidth=2, label="VBM"
    )
    ax.axvline(
        diagram.vbm_energy + diagram.bandgap,
        color="darkorange",
        linestyle="-",
        linewidth=2,
        label="CBM",
    )

    # Formatting
    ax.set_xlabel("Fermi Level (E$_F$ - E$_{VBM}$) [eV]", fontsize=14)
    ax.set_ylabel("Formation Energy E$_f$ [eV]", fontsize=14)
    ax.set_title("Defect Formation Energy Diagram", fontsize=16, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)  # noqa: FBT003
    ax.set_xlim(E_F_min, E_F_max)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Formation energy diagram saved to {save_path}")

    return fig, ax


def plot_formation_energy_diagram_plotly(
    diagram: FormationEnergyDiagram,
    fermi_range: tuple[float, float] | None = None,
    show_ctls: bool = True,
    show_band_edges: bool = True,
    width: int = 900,
    height: int = 600,
    save_path: str | None = None,
) -> go.Figure:
    """
    Plot interactive formation energy diagram using Plotly.

    Creates an interactive formation energy diagram with hover tooltips,
    zoom/pan capabilities, and professional styling. Superior to matplotlib
    version for exploratory analysis and presentations.

    Parameters
    ----------
    diagram : FormationEnergyDiagram
        Formation energy diagram object
    fermi_range : tuple[float, float], optional
        Range of Fermi level to plot (E_F_min, E_F_max) in eV.
        If None, uses (VBM, VBM + bandgap)
    show_ctls : bool
        If True, mark charge transition levels with annotations. Default: True
    show_band_edges : bool
        If True, mark VBM and CBM positions. Default: True
    width : int
        Figure width in pixels. Default: 900
    height : int
        Figure height in pixels. Default: 600
    save_path : str, optional
        Path to save HTML file. If None, returns figure object only.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure

    Example
    -------
    >>> from atomate2.siesta.flows.defects.analysis import FormationEnergyDiagram
    >>> diagram = FormationEnergyDiagram.from_defect_documents(defect_docs, bandgap=3.8)
    >>> fig = plot_formation_energy_diagram_plotly(
    ...     diagram, save_path="formation_energy_interactive.html"
    ... )
    >>> fig.show()  # Opens in browser
    """
    try:
        import plotly.graph_objects as go
    except ImportError as err:
        raise ImportError(
            "Plotly is required for interactive plots. Install with: pip install plotly"
        ) from err

    # Set Fermi range
    if fermi_range is None:
        E_F_min = diagram.vbm_energy  # noqa: N806
        E_F_max = diagram.vbm_energy + diagram.bandgap  # noqa: N806
    else:
        E_F_min, E_F_max = fermi_range  # noqa: N806

    # Create figure
    fig = go.Figure()

    # Color palette
    colors_positive = ["#FF6B6B", "#EE5A6F", "#DC4C64"]  # Reds for positive charges
    colors_negative = ["#4ECDC4", "#44A9C0", "#3A86BC"]  # Blues for negative charges
    colors_neutral = ["#95E1D3", "#FFD93D", "#F38181"]  # Mixed for neutral

    # Line styles (plotly uses dash parameter)
    dash_styles = {0: "solid", 1: "dash", 2: "dot", 3: "dashdot"}

    # Fermi level array
    E_F_range = np.linspace(E_F_min, E_F_max, 200)  # noqa: N806

    # Plot each defect and charge state
    for defect_idx, defect in enumerate(diagram.defects):
        for q, E_formation_0 in zip(  # noqa: N806
            defect.charge_states, defect.formation_energies, strict=False
        ):
            # Calculate formation energy vs Fermi level
            E_formation = E_formation_0 + q * E_F_range  # noqa: N806

            # Select color based on charge
            if q > 0:
                color = colors_positive[min(abs(q) - 1, len(colors_positive) - 1)]
                charge_label = f"q={q:+d}"
            elif q < 0:
                color = colors_negative[min(abs(q) - 1, len(colors_negative) - 1)]
                charge_label = f"q={q:d}"
            else:
                color = colors_neutral[defect_idx % len(colors_neutral)]
                charge_label = "q=0"

            # Dash style based on charge magnitude
            dash = dash_styles.get(abs(q), "solid")

            # Create hover text with detailed information
            hover_text = [
                f"<b>{defect.defect_name} ({charge_label})</b><br>"
                f"E_F = {ef:.3f} eV<br>"
                f"E_formation = {formation:.3f} eV<br>"
                f"Formation energy at E_F=0: {E_formation_0:.3f} eV<br>"
                f"Charge state: {q:+d}"
                for ef, formation in zip(E_F_range, E_formation, strict=False)
            ]

            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=E_F_range,
                    y=E_formation,
                    mode="lines",
                    name=f"{defect.defect_name} ({charge_label})",
                    line=dict(color=color, width=3, dash=dash),
                    hovertext=hover_text,
                    hoverinfo="text",
                    showlegend=True,
                )
            )

    # Add charge transition levels
    if show_ctls:
        ctls = diagram.calculate_charge_transition_levels()
        for ctl in ctls:
            if E_F_min <= ctl.fermi_level <= E_F_max:
                # Add vertical line
                fig.add_vline(
                    x=ctl.fermi_level,
                    line_dash="dash",
                    line_color="gray",
                    line_width=2,
                    opacity=0.6,
                    annotation_text=f"ε({ctl.q1:+d}/{ctl.q2:+d})",
                    annotation_position="top",
                    annotation_font_size=10,
                    annotation_font_color="gray",
                )

    # Add band edges
    if show_band_edges:
        # VBM
        fig.add_vline(
            x=diagram.vbm_energy,
            line_dash="solid",
            line_color="darkgreen",
            line_width=3,
            annotation_text="VBM",
            annotation_position="bottom left",
            annotation_font_size=12,
            annotation_font_color="darkgreen",
        )
        # CBM
        fig.add_vline(
            x=diagram.vbm_energy + diagram.bandgap,
            line_dash="solid",
            line_color="darkorange",
            line_width=3,
            annotation_text="CBM",
            annotation_position="bottom right",
            annotation_font_size=12,
            annotation_font_color="darkorange",
        )
        # Shade band gap
        fig.add_vrect(
            x0=diagram.vbm_energy,
            x1=diagram.vbm_energy + diagram.bandgap,
            fillcolor="lightgray",
            opacity=0.1,
            layer="below",
            line_width=0,
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text="Defect Formation Energy Diagram (Interactive)",
            font=dict(size=18, family="Arial Black"),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Fermi Level (E<sub>F</sub> - E<sub>VBM</sub>) [eV]",
            title_font=dict(size=14),
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5,
            range=[E_F_min, E_F_max],
        ),
        yaxis=dict(
            title="Formation Energy E<sub>f</sub> [eV]",
            title_font=dict(size=14),
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5,
        ),
        width=width,
        height=height,
        hovermode="closest",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
        ),
        template="plotly_white",
        plot_bgcolor="white",
    )

    # Save if requested
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Interactive formation energy diagram saved to {save_path}")

    return fig


def export_formation_energy_json(
    diagram: FormationEnergyDiagram,
    filename: str = "formation_energy_data.json",
    include_metadata: bool = True,
) -> Path:
    """
    Export formation energy diagram data to JSON format.

    Exports all formation energy data, CTLs, and metadata to a JSON file
    for external analysis, plotting, or integration with other tools.

    Parameters
    ----------
    diagram : FormationEnergyDiagram
        Formation energy diagram object
    filename : str
        Output JSON filename. Default: "formation_energy_data.json"
    include_metadata : bool
        If True, include calculation metadata. Default: True

    Returns
    -------
    Path
        Path to exported JSON file

    Example
    -------
    >>> from atomate2.siesta.flows.defects.analysis import FormationEnergyDiagram
    >>> diagram = FormationEnergyDiagram.from_defect_documents(defect_docs, bandgap=3.8)
    >>> json_path = export_formation_energy_json(diagram)
    """
    import json

    # Build data structure
    data: dict[str, Any] = {
        "bandgap": diagram.bandgap,
        "vbm_energy": diagram.vbm_energy,
        "cbm_energy": diagram.vbm_energy + diagram.bandgap,
        "defects": [],
    }

    # Add defect data
    for defect in diagram.defects:
        defect_data: dict[str, Any] = {
            "defect_name": defect.defect_name,
            "defect_type": defect.defect_type,
            "charge_states": defect.charge_states,
            "formation_energies_at_vbm": defect.formation_energies,
            "corrected": defect.corrected,
            "formation_energy_vs_fermi": {},
        }

        # Calculate formation energies across Fermi level range
        E_F_range = np.linspace(  # noqa: N806
            diagram.vbm_energy, diagram.vbm_energy + diagram.bandgap, 100
        )
        for q, E_formation_0 in zip(  # noqa: N806
            defect.charge_states, defect.formation_energies, strict=False
        ):
            E_formation = E_formation_0 + q * E_F_range  # noqa: N806
            defect_data["formation_energy_vs_fermi"][f"q={q:+d}"] = {
                "fermi_levels": E_F_range.tolist(),
                "formation_energies": E_formation.tolist(),
            }

        # Add metadata if available
        if include_metadata and hasattr(defect, "corrections"):
            defect_data["corrections"] = defect.corrections
        if include_metadata and hasattr(defect, "chemical_potentials"):
            defect_data["chemical_potentials"] = defect.chemical_potentials

        data["defects"].append(defect_data)

    # Add charge transition levels
    ctls = diagram.calculate_charge_transition_levels()
    data["charge_transition_levels"] = [
        {
            "transition": f"ε({ctl.q1:+d}/{ctl.q2:+d})",
            "fermi_level": float(ctl.fermi_level),
            "defect": ctl.defect_name,
            "in_gap": diagram.vbm_energy
            <= ctl.fermi_level
            <= diagram.vbm_energy + diagram.bandgap,
        }
        for ctl in ctls
    ]

    # Add metadata
    if include_metadata:
        from datetime import datetime

        data["metadata"] = {
            "export_date": datetime.now().isoformat(),  # noqa: DTZ005
            "code": "atomate2siesta",
            "description": "Formation energy diagram data for defect analysis",
        }

    # Write JSON
    output_path = Path(filename)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Formation energy data exported to {output_path}")

    return output_path


@job
def plot_formation_energy_job(
    defect_documents: list[DefectDocument],
    bandgap: float,
    vbm_energy: float = 0.0,
    use_corrected: bool = True,
    filename: str = "formation_energy_diagram.png",
) -> Path:
    """
    Job to create and save formation energy diagram.

    This job function creates a formation energy diagram from DefectDocument
    objects and saves it to the current working directory (job folder).

    Parameters
    ----------
    defect_documents : list[DefectDocument]
        List of DefectDocument objects from defect calculations
    bandgap : float
        Band gap of the host material (in eV)
    vbm_energy : float
        Valence band maximum energy (in eV). Default: 0.0
    use_corrected : bool
        If True, use corrected formation energies. Default: True
    filename : str
        Output filename. Default: "formation_energy_diagram.png"

    Returns
    -------
    Path
        Path to the saved plot file
    """
    # Create formation energy diagram
    diagram = FormationEnergyDiagram.from_defect_documents(
        defect_documents,
        bandgap=bandgap,
        vbm_energy=vbm_energy,
        use_corrected=use_corrected,
    )

    # Calculate and log charge transition levels
    ctls = diagram.calculate_charge_transition_levels()
    if ctls:
        logger.info(f"Calculated {len(ctls)} charge transition level(s):")
        for ctl in ctls:
            # Determine position relative to band gap
            if 0 <= ctl.fermi_level <= bandgap:
                frac = ctl.fermi_level / bandgap * 100
                position = f"{frac:.1f}% through gap"
            elif ctl.fermi_level < 0:
                position = f"{abs(ctl.fermi_level):.3f} eV below VBM"
            else:
                position = f"{ctl.fermi_level - bandgap:.3f} eV above CBM"

            logger.info(
                f"  ε({ctl.q1:+d}/{ctl.q2:+d}) = {ctl.fermi_level:.4f} eV ({position})"
            )

    # Plot will be saved in the current working directory (job folder)
    save_path = Path(filename)

    diagram.plot(
        show_ctls=True,
        show_stable_regions=True,
        figsize=(10, 7),
        save_path=str(save_path),
    )

    logger.info(f"Formation energy diagram saved to {save_path}")

    return save_path


def write_defect_summary(
    defect_doc: DefectDocument,
    defect_type: str,
    charge_state: int,
    defect_species: str | None,
    mu_defect: float,
) -> None:
    """
    Write human-readable defect summary to text file.

    Parameters
    ----------
    defect_doc : DefectDocument
        Defect document containing calculation results
    defect_type : str
        Type of defect
    charge_state : int
        Charge state of defect
    defect_species : str, optional
        Species of the defect
    mu_defect : float
        Chemical potential contribution (in eV)
    """
    from pathlib import Path

    from atomate2.siesta.utils.text_output import get_standard_footer

    # Create unique filename per charge state
    if charge_state == 0:
        summary_file = Path("defect_summary_q=0.txt")
    else:
        summary_file = Path(f"defect_summary_q={charge_state:+d}.txt")

    with open(summary_file, "w") as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("DEFECT CALCULATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Defect information
        f.write("DEFECT INFORMATION\n")
        f.write("-" * 80 + "\n")
        defect_label = f"{defect_type.upper()}"
        if defect_species:
            if defect_type == "vacancy":
                defect_label = f"V_{defect_species}"
            elif defect_type == "interstitial":
                defect_label = f"{defect_species}_i"
        if charge_state != 0:
            defect_label += f" (q={charge_state:+d})"

        f.write(f"Defect:              {defect_label}\n")
        f.write(f"Type:                {defect_type}\n")
        if defect_species:
            f.write(f"Species:             {defect_species}\n")
        f.write(f"Charge state:        {charge_state:+d}\n")

        # Add position
        if defect_doc.defect_site is not None:
            pos_str = (
                f"[{defect_doc.defect_site[0]:.4f}, "
                f"{defect_doc.defect_site[1]:.4f}, "
                f"{defect_doc.defect_site[2]:.4f}]"
            )
            f.write(f"Position (frac):     {pos_str}\n")

        f.write(f"Supercell size:      {defect_doc.supercell_natoms} atoms\n")
        f.write("\n")

        # Formation energy
        f.write("FORMATION ENERGY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Uncorrected:         {defect_doc.raw_formation_energy:.4f} eV\n")

        if charge_state != 0:
            f.write(
                f"Correction:          {defect_doc.correction_energy:.4f} eV "
                f"({defect_doc.correction_scheme})\n"
            )
            f.write(
                f"Corrected:           "
                f"{defect_doc.corrected_formation_energy:.4f} eV @ E_F = 0 (VBM)\n"
            )
        else:
            f.write("Correction:          N/A (neutral defect)\n")
            f.write(
                f"Corrected:           {defect_doc.corrected_formation_energy:.4f} eV\n"
            )
        f.write("\n")

        # Energy breakdown
        f.write("ENERGY BREAKDOWN\n")
        f.write("-" * 80 + "\n")
        f.write(f"Defect energy (E_defect):    {defect_doc.defect_energy:.4f} eV\n")
        f.write(f"Host energy (E_host):        {defect_doc.host_energy:.4f} eV\n")

        # Energy difference
        energy_diff = defect_doc.defect_energy - defect_doc.host_energy
        f.write(f"E_defect - E_host:           {energy_diff:.4f} eV\n")

        # Show chemical potential contribution
        if defect_type == "substitution":
            # For substitution: show both μ_removed and μ_added
            if abs(defect_doc.mu_removed) > 0.001 or abs(defect_doc.mu_added) > 0.001:
                removed_sp = defect_doc.removed_species or "removed"
                added_sp = defect_species or "added"
                f.write(
                    f"Chemical potential (μ_{removed_sp}): "
                    f"{defect_doc.mu_removed:+.4f} eV  (removed)\n"
                )
                f.write(
                    f"Chemical potential (μ_{added_sp}):   "
                    f"{defect_doc.mu_added:+.4f} eV  (added)\n"
                )
                f.write(f"Net Δμ (μ_removed - μ_added): {mu_defect:+.4f} eV\n")
        # For vacancy/interstitial: show single μ
        elif abs(mu_defect) > 0.001:
            mu_label = f"μ_{defect_species}" if defect_species else "μ"
            f.write(f"Chemical potential ({mu_label}): {mu_defect:+.4f} eV\n")

        f.write(
            f"Raw E_formation (incl. μ):           "
            f"{defect_doc.raw_formation_energy:.4f} eV\n"
        )
        if charge_state != 0:
            f.write(
                f"Correction (E_corr):                 "
                f"{defect_doc.correction_energy:+.4f} eV\n"
            )
            f.write(
                f"Final E_formation (corrected):       "
                f"{defect_doc.corrected_formation_energy:.4f} eV\n"
            )
        f.write("\n")

        # Correction metadata (if available)
        if charge_state != 0 and hasattr(defect_doc, "correction_metadata"):
            f.write("CORRECTION DETAILS\n")
            f.write("-" * 80 + "\n")
            metadata = defect_doc.correction_metadata
            f.write(f"Scheme:              {defect_doc.correction_scheme}\n")
            if "madelung_constant" in metadata:
                alpha_M = metadata["madelung_constant"]  # noqa: N806
                f.write(f"Madelung constant:   {alpha_M:.4f}")
                if metadata.get("madelung_citation"):
                    f.write(f" ({metadata['madelung_citation']})")
                f.write("\n")
            if "characteristic_length_angstrom" in metadata:
                f.write(
                    f"Characteristic length: "
                    f"{metadata['characteristic_length_angstrom']:.2f} Å\n"
                )
            if "gaussian_width_angstrom" in metadata:
                f.write(
                    f"Gaussian width (σ):    "  # noqa: RUF001
                    f"{metadata['gaussian_width_angstrom']:.2f} Å\n"
                )
            if "lattice_term_eV" in metadata:
                f.write(f"Lattice term:        {metadata['lattice_term_eV']:.4f} eV\n")
            if "alignment_energy_eV" in metadata:
                f.write(
                    f"Alignment term:      {metadata['alignment_energy_eV']:.4f} eV\n"
                )
            if "quadrupole_term_eV" in metadata:
                f.write(
                    f"Quadrupole term:     {metadata['quadrupole_term_eV']:.4f} eV\n"
                )
            if metadata.get("alignment_plot"):
                f.write(f"Alignment plot:      {metadata['alignment_plot']}\n")
            f.write(f"Total correction:    {defect_doc.correction_energy:.4f} eV\n")
            f.write("\n")

        # Formation energy formula
        f.write("FORMATION ENERGY FORMULA\n")
        f.write("-" * 80 + "\n")
        if defect_type == "vacancy":
            f.write("For vacancy defects:\n")
            f.write(
                "  E_formation = E_defect - E_host + μ_removed + q × E_F + E_corr\n\n"  # noqa: RUF001
            )
            f.write("Where:\n")
            f.write("  E_defect = Total energy of defect supercell\n")
            f.write("  E_host   = Total energy of pristine supercell\n")
            f.write("  μ        = Chemical potential of removed atom\n")
            f.write("  q        = Charge state of defect\n")
            f.write("  E_F      = Fermi energy (referenced to VBM)\n")
            f.write("  E_corr   = Finite-size correction energy\n")
        elif defect_type == "substitution":
            f.write("For substitution defects:\n")
            f.write(
                "  E_formation = E_defect - E_host + "
                "(μ_removed - μ_added) + q × E_F + E_corr\n\n"  # noqa: RUF001
            )
            f.write("Where:\n")
            f.write("  E_defect  = Total energy of defect supercell\n")
            f.write("  E_host    = Total energy of pristine supercell\n")
            f.write("  μ_removed = Chemical potential of removed atom\n")
            f.write("  μ_added   = Chemical potential of added atom (dopant)\n")
            f.write("  q         = Charge state of defect\n")
            f.write("  E_F       = Fermi energy (referenced to VBM)\n")
            f.write("  E_corr    = Finite-size correction energy\n")
        elif defect_type == "interstitial":
            f.write("For interstitial defects:\n")
            f.write(
                "  E_formation = E_defect - E_host - μ_added + q × E_F + E_corr\n\n"  # noqa: RUF001
            )
            f.write("Where:\n")
            f.write("  E_defect = Total energy of defect supercell\n")
            f.write("  E_host   = Total energy of pristine supercell\n")
            f.write("  μ_added  = Chemical potential of added atom\n")
            f.write("  q        = Charge state of defect\n")
            f.write("  E_F      = Fermi energy (referenced to VBM)\n")
            f.write("  E_corr   = Finite-size correction energy\n")
        f.write("\n")

        # Notes
        f.write("NOTES\n")
        f.write("-" * 80 + "\n")
        f.write("• Formation energy at E_F = 0 (referenced to VBM)\n")
        f.write("• For charged defects: E_formation(E_F) = E_formation(0) + q × E_F\n")  # noqa: RUF001
        if defect_type == "vacancy" and "ghost" in str(
            defect_doc.defect_structure.site_properties.get("ghost_tags", [])
        ):
            f.write("• Vacancy calculated with ghost atoms (SIESTA-specific)\n")
        f.write("\n")

        # Add standard footer
        f.write(
            get_standard_footer(
                width=80,
                additional_info={
                    "Defect": defect_label,
                    "Charge": f"{charge_state:+d}",
                    "E_formation": f"{defect_doc.corrected_formation_energy:.4f} eV",
                },
            )
        )

    logger.info(f"Defect summary written to {summary_file}")


@job
def write_combined_defect_summary(
    defect_documents: list[DefectDocument],
    filename: str = "defect_summary_all_charges.txt",
    bandgap: float | None = None,
    vbm_energy: float = 0.0,  # noqa: ARG001
) -> Path:
    """
    Job to write a combined summary for all charge states.

    This job function creates a single summary file showing information
    for all charge states of a defect, rather than separate files for each.

    Parameters
    ----------
    defect_documents : list[DefectDocument]
        List of DefectDocument objects from defect calculations
    filename : str
        Output filename. Default: "defect_summary_all_charges.txt"
    bandgap : float, optional
        Band gap of the host material (in eV). If provided, CTL positions
        will be reported relative to VBM/CBM.
    vbm_energy : float
        Valence band maximum energy. Default: 0.0

    Returns
    -------
    Path
        Path to the saved summary file
    """
    from pathlib import Path

    from atomate2.siesta.utils.text_output import get_standard_footer

    summary_file = Path(filename)

    # Group defects by name
    defect_groups: dict[str, list[DefectDocument]] = {}
    for doc in defect_documents:
        name = doc.defect_species or doc.defect_type
        if name not in defect_groups:
            defect_groups[name] = []
        defect_groups[name].append(doc)

    with open(summary_file, "w") as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("DEFECT CALCULATION SUMMARY - ALL CHARGE STATES\n")
        f.write("=" * 80 + "\n\n")

        # Process each defect type
        for defect_name, docs in defect_groups.items():
            # Sort by charge state
            docs = sorted(docs, key=lambda d: d.charge_state)  # noqa: PLW2901

            # Defect information
            f.write("DEFECT INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Defect:              {defect_name}\n")
            f.write(f"Type:                {docs[0].defect_type}\n")
            if docs[0].defect_species:
                f.write(f"Species:             {docs[0].defect_species}\n")
            f.write(f"Supercell size:      {docs[0].supercell_natoms} atoms\n")
            f.write(f"Number of charge states: {len(docs)}\n")
            f.write(
                f"Charge states:       "
                f"{', '.join([f'{d.charge_state:+d}' for d in docs])}\n"
            )
            f.write("\n")

            # Formation energies table
            f.write("FORMATION ENERGIES (at E_F = 0)\n")
            f.write("-" * 120 + "\n")
            f.write(
                f"{'Charge':>8}  {'Position (frac)':>26}  "
                f"{'Uncorrected':>12}  {'Correction':>12}  "
                f"{'Corrected':>12}  {'Scheme':>15}\n"
            )
            f.write(
                f"{'-' * 8}  {'-' * 26}  {'-' * 12}  "
                f"{'-' * 12}  {'-' * 12}  {'-' * 15}\n"
            )

            for doc in docs:
                # Format position
                if doc.defect_site is not None:
                    pos_str = (
                        f"[{doc.defect_site[0]:6.4f}, "
                        f"{doc.defect_site[1]:6.4f}, "
                        f"{doc.defect_site[2]:6.4f}]"
                    )
                else:
                    pos_str = "N/A"

                if doc.charge_state != 0:
                    f.write(
                        f"{doc.charge_state:+8d}  "
                        f"{pos_str:>26}  "
                        f"{doc.raw_formation_energy:>12.4f}  "
                        f"{doc.correction_energy:>12.4f}  "
                        f"{doc.corrected_formation_energy:>12.4f}  "
                        f"{doc.correction_scheme:>15}\n"
                    )
                else:
                    f.write(
                        f"{doc.charge_state:+8d}  "
                        f"{pos_str:>26}  "
                        f"{doc.raw_formation_energy:>12.4f}  "
                        f"{'N/A':>12}  "
                        f"{doc.corrected_formation_energy:>12.4f}  "
                        f"{doc.correction_scheme:>15}\n"
                    )
            f.write("\n")

            # Charge transition levels (if multiple charge states)
            if len(docs) > 1:
                f.write("CHARGE TRANSITION LEVELS\n")
                f.write("-" * 80 + "\n")

                # Deduplicate charge states (important for use_symmetry=False)
                # Group by charge state and take first defect of each charge
                unique_charges = {}
                for doc in docs:
                    q = doc.charge_state
                    if q not in unique_charges:
                        unique_charges[q] = doc

                # Only calculate CTLs if we have multiple UNIQUE charge states
                if len(unique_charges) > 1:
                    charge_states = list(unique_charges.keys())
                    formation_energies = [
                        unique_charges[q].corrected_formation_energy
                        for q in charge_states
                    ]

                    ctls = calculate_charge_transition_levels(
                        charge_states, formation_energies, defect_name
                    )
                else:
                    ctls = []

                if ctls:
                    # Table header (with or without position column)
                    if bandgap is not None:
                        f.write(
                            f"{'Transition':>12}  {'ε (eV)':>10}  "
                            f"{'Position':>25}  {'E_formation (eV)':>16}\n"
                        )
                        f.write(f"{'-' * 12}  {'-' * 10}  {'-' * 25}  {'-' * 16}\n")
                    else:
                        f.write(
                            f"{'Transition':>12}  {'ε (eV)':>10}  "
                            f"{'E_formation (eV)':>16}\n"
                        )
                        f.write(f"{'-' * 12}  {'-' * 10}  {'-' * 16}\n")

                    for ctl in ctls:
                        # Format transition label
                        transition_label = f"ε({ctl.q1:+d}/{ctl.q2:+d})"

                        # Determine position relative to band gap (if bandgap provided)
                        if bandgap is not None:
                            if ctl.fermi_level < 0:
                                position_str = (
                                    f"{abs(ctl.fermi_level):.3f} eV below VBM"
                                )
                            elif ctl.fermi_level < bandgap:
                                frac = ctl.fermi_level / bandgap * 100
                                position_str = f"{frac:.1f}% through gap"
                            else:
                                position_str = (
                                    f"{ctl.fermi_level - bandgap:.3f} eV above CBM"
                                )

                            f.write(
                                f"{transition_label:>12}  "
                                f"{ctl.fermi_level:>10.4f}  "
                                f"{position_str:>25}  "
                                f"{ctl.formation_energy:>10.4f}\n"
                            )
                        else:
                            f.write(
                                f"{transition_label:>12}  "
                                f"{ctl.fermi_level:>10.4f}  "
                                f"{ctl.formation_energy:>10.4f}\n"
                            )

                    f.write("\n")
                    f.write("Notes:\n")
                    f.write(
                        "• ε(q₁/q₂) is the Fermi level where charge states "
                        "q₁ and q₂ have equal E_formation\n"
                    )
                    f.write("• At E_F < ε(q₁/q₂): charge state q₁ is more stable\n")
                    f.write("• At E_F > ε(q₁/q₂): charge state q₂ is more stable\n")
                    f.write("• All energies referenced to VBM (valence band maximum)\n")
                else:
                    f.write(
                        "No charge transition levels "
                        "(only one charge state or duplicate charges)\n"
                    )

                f.write("\n")

            # Energy breakdown for each charge state
            f.write("ENERGY BREAKDOWN\n")
            f.write("-" * 80 + "\n")
            for doc in docs:
                charge_label = (
                    f"q={doc.charge_state:+d}" if doc.charge_state != 0 else "q=0"
                )
                f.write(f"\n[{charge_label}]\n")

                # Add position
                if doc.defect_site is not None:
                    pos_str = (
                        f"[{doc.defect_site[0]:.4f}, "
                        f"{doc.defect_site[1]:.4f}, "
                        f"{doc.defect_site[2]:.4f}]"
                    )
                    f.write(f"  Position (frac):   {pos_str}\n")

                f.write(f"  Defect energy:     {doc.defect_energy:.4f} eV\n")
                f.write(f"  Host energy:       {doc.host_energy:.4f} eV\n")
                energy_diff = doc.defect_energy - doc.host_energy
                f.write(f"  E_defect - E_host: {energy_diff:.4f} eV\n")
                f.write(
                    f"  Raw E_formation:           {doc.raw_formation_energy:.4f} eV\n"
                )
                if doc.charge_state != 0:
                    f.write(
                        f"  Correction:                "
                        f"{doc.correction_energy:+.4f} eV\n"
                    )
                    f.write(
                        f"  Corrected E_formation:     "
                        f"{doc.corrected_formation_energy:.4f} eV\n"
                    )
            f.write("\n")

            # Correction details (if available)
            has_corrections = any(
                doc.charge_state != 0 and hasattr(doc, "correction_metadata")
                for doc in docs
            )
            if has_corrections:
                f.write("CORRECTION DETAILS\n")
                f.write("-" * 80 + "\n")
                for doc in docs:
                    if doc.charge_state != 0 and hasattr(doc, "correction_metadata"):
                        charge_label = f"q={doc.charge_state:+d}"
                        f.write(f"\n[{charge_label}]\n")

                        # Add position
                        if doc.defect_site is not None:
                            pos_str = (
                                f"[{doc.defect_site[0]:.4f}, "
                                f"{doc.defect_site[1]:.4f}, "
                                f"{doc.defect_site[2]:.4f}]"
                            )
                            f.write(f"  Position (frac): {pos_str}\n")

                        metadata = doc.correction_metadata
                        if "madelung_constant" in metadata:
                            alpha_M = metadata["madelung_constant"]  # noqa: N806
                            f.write(f"  Madelung constant: {alpha_M:.4f}")
                            if metadata.get("madelung_citation"):
                                f.write(f" ({metadata['madelung_citation']})")
                            f.write("\n")
                        if "supercell_length" in metadata:
                            f.write(
                                f"  Supercell length:  "
                                f"{metadata['supercell_length']:.2f} Å\n"
                            )
                        if "gaussian_width_angstrom" in metadata:
                            f.write(
                                f"  Gaussian width (σ): "  # noqa: RUF001
                                f"{metadata['gaussian_width_angstrom']:.2f} Å\n"
                            )
                        if "lattice_term" in metadata:
                            f.write(
                                f"  Lattice term:      "
                                f"{metadata.get('lattice_term', 0.0):.4f} eV\n"
                            )
                        if "alignment_energy" in metadata:
                            f.write(
                                f"  Alignment energy:  "
                                f"{metadata.get('alignment_energy', 0.0):.4f} eV\n"
                            )
                f.write("\n")

            # Formation energy formula
            f.write("FORMATION ENERGY FORMULA\n")
            f.write("-" * 80 + "\n")
            defect_type = docs[0].defect_type
            if defect_type == "vacancy":
                f.write("For vacancy defects:\n")
                f.write(
                    "  E_formation = E_defect - E_host + "
                    "μ_removed + q × E_F + E_corr\n\n"  # noqa: RUF001
                )
                f.write("Where:\n")
                f.write("  E_defect = Total energy of defect supercell\n")
                f.write("  E_host   = Total energy of pristine supercell\n")
                f.write("  μ        = Chemical potential of removed atom\n")
                f.write("  q        = Charge state of defect\n")
                f.write("  E_F      = Fermi energy (referenced to VBM)\n")
                f.write("  E_corr   = Finite-size correction energy\n")
            elif defect_type == "substitution":
                f.write("For substitution defects:\n")
                f.write(
                    "  E_formation = E_defect - E_host + "
                    "(μ_removed - μ_added) + q × E_F + E_corr\n\n"  # noqa: RUF001
                )
                f.write("Where:\n")
                f.write("  E_defect  = Total energy of defect supercell\n")
                f.write("  E_host    = Total energy of pristine supercell\n")
                f.write("  μ_removed = Chemical potential of removed atom\n")
                f.write("  μ_added   = Chemical potential of added atom (dopant)\n")
                f.write("  q         = Charge state of defect\n")
                f.write("  E_F       = Fermi energy (referenced to VBM)\n")
                f.write("  E_corr    = Finite-size correction energy\n")
            elif defect_type == "interstitial":
                f.write("For interstitial defects:\n")
                f.write(
                    "  E_formation = E_defect - E_host - μ_added + q × E_F + E_corr\n\n"  # noqa: RUF001
                )
                f.write("Where:\n")
                f.write("  E_defect = Total energy of defect supercell\n")
                f.write("  E_host   = Total energy of pristine supercell\n")
                f.write("  μ_added  = Chemical potential of added atom\n")
                f.write("  q        = Charge state of defect\n")
                f.write("  E_F      = Fermi energy (referenced to VBM)\n")
                f.write("  E_corr   = Finite-size correction energy\n")
            f.write("\n")

            # Notes
            f.write("NOTES\n")
            f.write("-" * 80 + "\n")
            f.write("• Formation energies shown at E_F = 0 (referenced to VBM)\n")
            f.write(
                "• For charged defects: E_formation(E_F) = E_formation(0) + q × E_F\n"  # noqa: RUF001
            )
            if defect_type == "vacancy":
                f.write("• Vacancy calculated with ghost atoms (SIESTA-specific)\n")
            f.write("\n\n")

        # Add standard footer
        f.write(
            get_standard_footer(
                width=80,
                additional_info={
                    "Defects": f"{len(defect_groups)} type(s)",
                    "Total calculations": str(len(defect_documents)),
                },
            )
        )

    logger.info(f"Combined defect summary written to {summary_file}")

    return summary_file


@job
def plot_formation_energy_diagram_plotly_job(
    defect_documents: list[DefectDocument],
    bandgap: float,
    vbm_energy: float = 0.0,
    use_corrected: bool = True,
    filename: str = "formation_energy_interactive.html",
    show_ctls: bool = True,
    show_band_edges: bool = True,
) -> Path:
    """
    Job to create interactive Plotly formation energy diagram.

    Parameters
    ----------
    defect_documents : list[DefectDocument]
        List of DefectDocument objects from defect calculations
    bandgap : float
        Band gap of the host material (in eV)
    vbm_energy : float
        Valence band maximum energy. Default: 0.0
    use_corrected : bool
        If True, use corrected formation energies. Default: True
    filename : str
        Output HTML filename. Default: "formation_energy_interactive.html"
    show_ctls : bool
        Show charge transition levels. Default: True
    show_band_edges : bool
        Show VBM and CBM positions. Default: True

    Returns
    -------
    Path
        Path to saved HTML file
    """
    from pathlib import Path

    # Create diagram
    diagram = FormationEnergyDiagram.from_defect_documents(
        defect_documents,
        bandgap=bandgap,
        vbm_energy=vbm_energy,
        use_corrected=use_corrected,
    )

    # Plot and save
    plot_formation_energy_diagram_plotly(
        diagram,
        show_ctls=show_ctls,
        show_band_edges=show_band_edges,
        save_path=filename,
    )

    logger.info(f"Interactive formation energy diagram saved to {filename}")

    return Path(filename)


@job
def export_formation_energy_json_job(
    defect_documents: list[DefectDocument],
    bandgap: float,
    vbm_energy: float = 0.0,
    use_corrected: bool = True,
    filename: str = "formation_energy_data.json",
    include_metadata: bool = True,
) -> Path:
    """
    Job to export formation energy data to JSON.

    Parameters
    ----------
    defect_documents : list[DefectDocument]
        List of DefectDocument objects from defect calculations
    bandgap : float
        Band gap of the host material (in eV)
    vbm_energy : float
        Valence band maximum energy. Default: 0.0
    use_corrected : bool
        If True, use corrected formation energies. Default: True
    filename : str
        Output JSON filename. Default: "formation_energy_data.json"
    include_metadata : bool
        Include calculation metadata. Default: True

    Returns
    -------
    Path
        Path to exported JSON file
    """
    # Create diagram
    diagram = FormationEnergyDiagram.from_defect_documents(
        defect_documents,
        bandgap=bandgap,
        vbm_energy=vbm_energy,
        use_corrected=use_corrected,
    )

    # Export to JSON
    json_path = export_formation_energy_json(
        diagram, filename=filename, include_metadata=include_metadata
    )

    logger.info(f"Formation energy data exported to {json_path}")

    return json_path


# ============================================================================
# Helper jobs for FormationEnergyDiagramFlowMaker
# ============================================================================
# These are defined at module level to ensure proper serialization for
# cluster execution (avoid nested @job decorators inside class methods).
# ============================================================================


@job
def _extract_bandgap_from_defect_docs(defect_docs: list) -> float:
    """
    Extract bandgap from defect document host data.

    Parameters
    ----------
    defect_docs : list
        List of DefectDocument objects

    Returns
    -------
    float
        Extracted bandgap in eV, or 0.0 if not found
    """
    # Get bandgap from first defect document's host data
    if defect_docs and len(defect_docs) > 0:
        first_doc = defect_docs[0]
        if hasattr(first_doc, "host_bandgap") and first_doc.host_bandgap is not None:
            bg = first_doc.host_bandgap
            logger.info(f"Extracted bandgap from calculations: {bg:.3f} eV")
            return bg
        if hasattr(first_doc, "bandgap") and first_doc.bandgap is not None:
            bg = first_doc.bandgap
            logger.info(f"Extracted bandgap from calculations: {bg:.3f} eV")
            return bg
    logger.warning("Could not extract bandgap from calculations, using default 0.0 eV")
    return 0.0


@job
def _extract_effective_masses_from_defect_docs(
    defect_docs: list,  # noqa: ARG001
) -> tuple[float, float]:
    """
    Extract effective masses from band structure calculations.

    Parameters
    ----------
    defect_docs : list
        List of DefectDocument objects

    Returns
    -------
    tuple[float, float]
        (m_e, m_h) effective masses for electrons and holes

    Notes
    -----
    Currently returns default values (1.0, 1.0). Full implementation requires
    band structure calculations and parabolic fitting near band edges.
    """
    logger.warning(
        "Automatic effective mass extraction not yet implemented. "
        "Using default values (1.0, 1.0). "
        "For accurate SRH analysis, provide effective masses manually or "
        "run band structure calculations with effective mass extraction."
    )
    # TODO: Implement effective mass extraction from band structure
    # This requires:
    # 1. Band structure calculation near band edges
    # 2. Parabolic fit to extract m*
    return 1.0, 1.0


@job
def _build_formation_diagram_and_ctls(
    defect_docs: list,
    bg: float,
    vbm: float,
) -> tuple[FormationEnergyDiagram, list[ChargeTransitionLevel]]:
    """
    Build formation energy diagram and calculate charge transition levels.

    Parameters
    ----------
    defect_docs : list
        List of DefectDocument objects
    bg : float
        Band gap in eV
    vbm : float
        Valence band maximum energy in eV

    Returns
    -------
    tuple[FormationEnergyDiagram, list[ChargeTransitionLevel]]
        (diagram, ctls)
    """
    diagram = FormationEnergyDiagram.from_defect_documents(
        defect_docs,
        bandgap=bg,
        vbm_energy=vbm,
        use_corrected=True,
    )
    ctls = diagram.calculate_charge_transition_levels()
    return diagram, ctls


@dataclass
class FormationEnergyDiagramFlowMaker:
    """
    Complete defect formation energy diagram workflow.

    Generates defects, runs calculations, creates plots and summaries.

    Parameters
    ----------
    defect_type : str
        Type of defect ("vacancy", "substitution", "interstitial")
    species : str, optional
        Species for the defect
    dopants : str or list[str], optional
        Dopant species (for substitution)
    supercell_matrix : list
        Supercell transformation matrix
    charge_states : list[int]
        List of charge states to calculate
    epsilon_static : float
        Static dielectric constant
    bandgap : float
        Band gap of the host material (in eV)
    vbm_energy : float
        Valence band maximum energy. Default: 0.0
    dry_run : bool
        If True, skip actual calculations. Default: False
    skip_relax : bool
        If True, use static calculations instead of relaxation. Default: False
    defect_relax_maker : Maker, optional
        Custom maker for defect relaxation
    host_static_maker : Maker, optional
        Custom maker for host static calculation
    auto_calculate_chemical_potentials : bool
        If True, automatically calculate chemical potentials. Default: False
    chemical_potentials : dict, optional
        Manual chemical potentials (eV)
    include_concentration_analysis : bool
        If True, perform defect concentration analysis. Default: False
    include_srh_analysis : bool
        If True, perform SRH recombination analysis. Default: False
    temperature : float
        Temperature for concentration/SRH analysis (K). Default: 300.0
    effective_mass_electron : float
        Effective electron mass (m*/m_e) for SRH. Default: 1.0
    effective_mass_hole : float
        Effective hole mass (m*/m_e) for SRH. Default: 1.0
    capture_parameters : dict, optional
        Custom capture parameters for SRH (defect_name -> CaptureParameters)
    """

    defect_type: str
    species: str | None = None
    dopants: str | list[str] | None = None
    supercell_matrix: list = field(
        default_factory=lambda: [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    )
    charge_states: list[int] = field(default_factory=lambda: [0])
    epsilon_static: float = 10.0
    bandgap: float | None = None
    vbm_energy: float = 0.0
    auto_bandgap: bool = True
    dry_run: bool = False
    skip_relax: bool = False
    defect_relax_maker: Any = None
    host_static_maker: Any = None
    auto_calculate_chemical_potentials: bool = False
    chemical_potentials: dict[str, float] | None = None
    include_concentration_analysis: bool = False
    include_srh_analysis: bool = False
    temperature: float = 300.0
    effective_mass_electron: float | None = None
    effective_mass_hole: float | None = None
    auto_effective_masses: bool = False
    capture_parameters: dict | None = None

    def make(self, structure: Structure) -> Flow:
        """
        Create complete formation energy diagram workflow.

        Parameters
        ----------
        structure : Structure
            Pristine structure

        Returns
        -------
        Flow
            Complete workflow: defect calculations + diagram + summaries
        """
        from atomate2.siesta.flows.defects.core import DefectFlowMaker

        # Generate all defect flows
        defect_flows = DefectFlowMaker.from_pristine_structure(
            structure,
            defect_type=self.defect_type,
            species=self.species,
            dopants=self.dopants,
            supercell_matrix=self.supercell_matrix,
            charge_states=self.charge_states,
            epsilon_static=self.epsilon_static,
            dry_run=self.dry_run,
            skip_relax=self.skip_relax,
            defect_relax_maker=self.defect_relax_maker,
            host_static_maker=self.host_static_maker,
            auto_calculate_chemical_potentials=self.auto_calculate_chemical_potentials,
            chemical_potentials=self.chemical_potentials,
        )

        # Collect outputs from defect flows
        # Handle both single Flow and list of Flows from from_pristine_structure()
        if isinstance(defect_flows, list):
            # List of individual flows (single defect case)
            defect_outputs = [flow.output for flow in defect_flows]
        else:
            # Parent flow wrapping multiple flows (multiple defects case)
            # Extract outputs from child Flow objects (skip non-Flow jobs like summary)
            from jobflow import Flow

            defect_outputs = [
                job.output for job in defect_flows.jobs if isinstance(job, Flow)
            ]

        # Extract bandgap from calculations if requested
        if self.auto_bandgap and self.bandgap is None:
            bandgap_job = _extract_bandgap_from_defect_docs(defect_outputs)
            # jobflow resolves the lazy output reference to a float at run time
            bandgap_value = cast("float", bandgap_job.output)
        else:
            bandgap_value = self.bandgap if self.bandgap is not None else 0.0

        # Extract effective masses if requested
        if self.auto_effective_masses:
            eff_mass_job = _extract_effective_masses_from_defect_docs(defect_outputs)
            # jobflow resolves the lazy output references to floats at run time
            m_e = cast("float", eff_mass_job.output[0])
            m_h = cast("float", eff_mass_job.output[1])
        else:
            m_e = (
                self.effective_mass_electron
                if self.effective_mass_electron is not None
                else 1.0
            )
            m_h = (
                self.effective_mass_hole
                if self.effective_mass_hole is not None
                else 1.0
            )

        # Create analysis jobs
        plot_job = plot_formation_energy_job(
            defect_documents=defect_outputs,
            bandgap=bandgap_value,
            vbm_energy=self.vbm_energy,
        )

        summary_job = write_combined_defect_summary(
            defect_documents=defect_outputs,
            bandgap=bandgap_value,
            vbm_energy=self.vbm_energy,
        )

        # Collect jobs (include extraction jobs if created)
        # Handle both single Flow and list of Flows from from_pristine_structure()
        jobs: list[Flow | Job]
        if isinstance(defect_flows, list):
            jobs = [*defect_flows, plot_job, summary_job]
        else:
            jobs = [defect_flows, plot_job, summary_job]

        if self.auto_bandgap and self.bandgap is None:
            jobs.append(bandgap_job)
        if self.auto_effective_masses:
            jobs.append(eff_mass_job)

        flow_output = {
            "plot": plot_job.output,
            "summary": summary_job.output,
            "defect_outputs": defect_outputs,
        }
        if self.auto_bandgap and self.bandgap is None:
            flow_output["extracted_bandgap"] = bandgap_value

        # Add concentration analysis if requested
        if self.include_concentration_analysis or self.include_srh_analysis:
            from atomate2.siesta.flows.defects.analysis.concentration import (
                calculate_defect_concentrations_job,
                write_concentration_summary,
            )

            # Concentration analysis
            conc_job = calculate_defect_concentrations_job(
                defect_documents=defect_outputs,
                bandgap=bandgap_value,
                temperature=self.temperature,
                vbm_energy=self.vbm_energy,
            )

            conc_summary_job = write_concentration_summary(
                result=conc_job.output,
                bandgap=bandgap_value,
                filename="concentration_summary.txt",
            )

            jobs.extend([conc_job, conc_summary_job])
            flow_output["concentration_result"] = conc_job.output

        # Add SRH analysis if requested
        if self.include_srh_analysis:
            from atomate2.siesta.flows.defects.analysis.srh import (
                calculate_srh_analysis_job,
                write_srh_summary_job,
            )

            # Build formation energy diagram and calculate CTLs
            diagram_ctls_job = _build_formation_diagram_and_ctls(
                defect_docs=defect_outputs,
                bg=bandgap_value,
                vbm=self.vbm_energy,
            )

            # SRH analysis
            srh_job = calculate_srh_analysis_job(
                formation_diagram=diagram_ctls_job.output[0],  # diagram
                concentration_result=conc_job.output,
                ctls=diagram_ctls_job.output[1],  # ctls
                bandgap=bandgap_value,
                effective_mass_electron=m_e,
                effective_mass_hole=m_h,
                capture_params=self.capture_parameters,
            )

            srh_summary_job = write_srh_summary_job(
                srh_result=srh_job.output,
                concentration_result=conc_job.output,
                effective_mass_electron=m_e,
                effective_mass_hole=m_h,
                directory="srh_analysis",
            )

            jobs.extend([diagram_ctls_job, srh_job, srh_summary_job])
            flow_output["formation_diagram"] = diagram_ctls_job.output[0]
            flow_output["ctls"] = diagram_ctls_job.output[1]
            flow_output["srh_result"] = srh_job.output
            flow_output["srh_files"] = srh_summary_job.output

        # Create flow with all jobs
        return Flow(
            jobs,
            output=flow_output,
            name=f"formation_energy_diagram_{self.defect_type}",
        )
