"""
Defect concentration calculations and Fermi level solver.

This module implements defect concentration calculations based on
thermodynamic equilibrium, including self-consistent Fermi level determination
through charge neutrality.

Key concepts:
- Defect concentration: [D^q] = N_sites × exp(-E_formation(q, E_F) / k_B T)
- Charge neutrality: Σ q[D^q] + n - p = 0
- Self-consistent Fermi level solver

References
----------
    - Freysoldt et al., Rev. Mod. Phys. 86, 253 (2014)
    - Van de Walle & Neugebauer, J. Appl. Phys. 95, 3851 (2004)
"""  # noqa: RUF002

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jobflow import job
from monty.json import MSONable
from scipy.optimize import brentq

if TYPE_CHECKING:
    from atomate2.siesta.flows.defects.schemas import DefectDocument

logger = logging.getLogger(__name__)

# Physical constants
K_B = 8.617333e-5  # Boltzmann constant in eV/K


@dataclass
class DefectConcentration(MSONable):
    """
    Concentration data for a single defect charge state.

    Parameters
    ----------
    defect_name : str
        Name of the defect (e.g., "O_vacancy")
    charge_state : int
        Charge state (e.g., 0, +1, -1)
    concentration : float
        Equilibrium concentration in cm^-3
    formation_energy : float
        Formation energy at the equilibrium Fermi level (eV)
    """

    defect_name: str
    charge_state: int
    concentration: float
    formation_energy: float

    def as_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "defect_name": self.defect_name,
            "charge_state": self.charge_state,
            "concentration": self.concentration,
            "formation_energy": self.formation_energy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DefectConcentration:
        """Deserialize from dictionary."""
        return cls(
            defect_name=d["defect_name"],
            charge_state=d["charge_state"],
            concentration=d["concentration"],
            formation_energy=d["formation_energy"],
        )


@dataclass
class ConcentrationResult(MSONable):
    """
    Results from defect concentration calculation.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin
    fermi_level : float
        Self-consistent Fermi level in eV (referenced to VBM)
    electron_concentration : float
        Electron concentration in cm^-3
    hole_concentration : float
        Hole concentration in cm^-3
    defect_concentrations : list[DefectConcentration]
        List of defect concentrations
    charge_neutrality_error : float
        Residual charge neutrality error (should be ~0)
    fermi_level_converged : bool
        Whether Fermi level solver converged (False if fallback was used)
    """

    temperature: float
    fermi_level: float
    electron_concentration: float
    hole_concentration: float
    defect_concentrations: list[DefectConcentration]
    charge_neutrality_error: float = 0.0
    fermi_level_converged: bool = True

    @property
    def total_defect_concentration(self) -> float:
        """Total concentration of all defects (all charge states)."""
        return sum(d.concentration for d in self.defect_concentrations)

    @property
    def dominant_charge_states(self) -> dict[str, int]:
        """
        Determine dominant charge state for each defect.

        Returns
        -------
        dict[str, int]
            Mapping of defect name to dominant charge state
        """
        # Group by defect name
        defect_groups: dict[str, list[DefectConcentration]] = {}
        for dc in self.defect_concentrations:
            if dc.defect_name not in defect_groups:
                defect_groups[dc.defect_name] = []
            defect_groups[dc.defect_name].append(dc)

        # Find dominant charge state for each defect
        dominant = {}
        for name, concentrations in defect_groups.items():
            # Sort by concentration, highest first
            sorted_conc = sorted(
                concentrations, key=lambda x: x.concentration, reverse=True
            )
            dominant[name] = sorted_conc[0].charge_state

        return dominant

    def as_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "temperature": self.temperature,
            "fermi_level": self.fermi_level,
            "electron_concentration": self.electron_concentration,
            "hole_concentration": self.hole_concentration,
            "defect_concentrations": [
                dc.as_dict() for dc in self.defect_concentrations
            ],
            "charge_neutrality_error": self.charge_neutrality_error,
            "fermi_level_converged": self.fermi_level_converged,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ConcentrationResult:
        """Deserialize from dictionary."""
        return cls(
            temperature=d["temperature"],
            fermi_level=d["fermi_level"],
            electron_concentration=d["electron_concentration"],
            hole_concentration=d["hole_concentration"],
            defect_concentrations=[
                DefectConcentration.from_dict(dc) for dc in d["defect_concentrations"]
            ],
            charge_neutrality_error=d.get("charge_neutrality_error", 0.0),
            fermi_level_converged=d.get("fermi_level_converged", True),
        )


@dataclass
class BrouwerDiagramData(MSONable):
    """
    Data for Brouwer diagram (log concentration vs. thermodynamic variable).

    Brouwer diagrams show defect concentrations as a function of chemical
    potential (Fermi level), temperature, or other thermodynamic variables.
    They are useful for identifying dominant defects under different conditions.

    Parameters
    ----------
    variable_name : str
        Name of the independent variable (e.g., "Fermi level", "Temperature")
    variable_values : np.ndarray
        Values of the independent variable
    variable_unit : str
        Unit of the independent variable (e.g., "eV", "K")
    defect_names : list[str]
        Names of defects (with charge states)
    concentrations : np.ndarray
        Concentrations array, shape (n_points, n_defects)
    electron_concentrations : np.ndarray
        Electron concentrations at each point
    hole_concentrations : np.ndarray
        Hole concentrations at each point
    """

    variable_name: str
    variable_values: np.ndarray
    variable_unit: str
    defect_names: list[str]
    concentrations: np.ndarray
    electron_concentrations: np.ndarray
    hole_concentrations: np.ndarray

    def as_dict(self) -> dict:
        """Serialize to dict for JSON/MongoDB storage."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "variable_name": self.variable_name,
            "variable_values": self.variable_values.tolist(),
            "variable_unit": self.variable_unit,
            "defect_names": self.defect_names,
            "concentrations": self.concentrations.tolist(),
            "electron_concentrations": self.electron_concentrations.tolist(),
            "hole_concentrations": self.hole_concentrations.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> BrouwerDiagramData:
        """Deserialize from dict."""
        return cls(
            variable_name=d["variable_name"],
            variable_values=np.array(d["variable_values"]),
            variable_unit=d["variable_unit"],
            defect_names=d["defect_names"],
            concentrations=np.array(d["concentrations"]),
            electron_concentrations=np.array(d["electron_concentrations"]),
            hole_concentrations=np.array(d["hole_concentrations"]),
        )


@dataclass
class DefectConcentrationAnalyzer:
    """
    Analyzer for defect concentrations and Fermi level determination.

    This class calculates equilibrium defect concentrations by solving
    the charge neutrality condition to determine the self-consistent
    Fermi level position.

    Parameters
    ----------
    defect_documents : list[DefectDocument]
        List of defect calculation results
    bandgap : float
        Band gap in eV
    vbm_energy : float
        Valence band maximum energy in eV. Default: 0.0
    temperature : float
        Temperature in Kelvin. Default: 300 K
    n_sites : float
        Number of available defect sites (typically ~number of atoms).
        If None, estimated from supercell size.
    effective_density_of_states : dict, optional
        Effective DOS: {"N_C": ..., "N_V": ...} in cm^-3.
        If None, uses standard values.

    Examples
    --------
    >>> analyzer = DefectConcentrationAnalyzer(
    ...     defect_documents=defect_docs,
    ...     bandgap=7.8,
    ...     temperature=300,
    ... )
    >>> result = analyzer.calculate_concentrations()
    >>> print(f"Fermi level: {result.fermi_level:.3f} eV")
    >>> print(f"Total defect conc: {result.total_defect_concentration:.2e} cm^-3")
    """

    defect_documents: list[DefectDocument]
    bandgap: float
    vbm_energy: float = 0.0
    temperature: float = 300.0
    n_sites: float | None = None
    effective_density_of_states: dict[str, float] | None = None

    def __post_init__(self):
        """Initialize derived quantities."""
        # Estimate n_sites if not provided
        if self.n_sites is None:
            # For now: 1 site per atom, typical solid ~10^22 atoms/cm^3
            # More accurate: should use actual cell volume from structure
            self.n_sites = 5e22  # Typical order of magnitude
            logger.warning(
                f"n_sites not provided, using default {self.n_sites:.2e} cm^-3. "
                "For accurate results, provide n_sites based on crystal structure."
            )

        # Set effective DOS if not provided
        if self.effective_density_of_states is None:
            # Standard values for many semiconductors at 300 K
            # N_C, N_V ~ 10^19 cm^-3 (temperature dependent)
            # More accurate: N_C = 2 × (2π m_e* k_B T / h^2)^(3/2)  # noqa: RUF003
            self.effective_density_of_states = {
                "N_C": 1e19,  # Conduction band
                "N_V": 1e19,  # Valence band
            }
            logger.info(
                "Using default effective DOS: N_C = N_V = 1e19 cm^-3. "
                "For accurate results, provide material-specific values."
            )

    def calculate_carrier_concentrations(
        self, fermi_level: float
    ) -> tuple[float, float]:
        """
        Calculate electron and hole concentrations.

        Parameters
        ----------
        fermi_level : float
            Fermi level in eV (referenced to VBM)

        Returns
        -------
        tuple[float, float]
            (electron_concentration, hole_concentration) in cm^-3
        """
        N_C = self.effective_density_of_states["N_C"]  # noqa: N806
        N_V = self.effective_density_of_states["N_V"]  # noqa: N806

        # CBM is VBM + bandgap
        E_C = self.vbm_energy + self.bandgap  # noqa: N806
        E_V = self.vbm_energy  # noqa: N806

        # Thermal energy
        kT = K_B * self.temperature  # noqa: N806

        # Electron concentration: n = N_C × exp(-(E_C - E_F) / kT)  # noqa: RUF003
        n = N_C * np.exp(-(E_C - fermi_level) / kT)

        # Hole concentration: p = N_V × exp(-(E_F - E_V) / kT)  # noqa: RUF003
        p = N_V * np.exp(-(fermi_level - E_V) / kT)

        return n, p

    def calculate_defect_concentration(
        self, formation_energy_at_zero: float, charge_state: int, fermi_level: float
    ) -> float:
        """
        Calculate concentration for a single defect charge state.

        Parameters
        ----------
        formation_energy_at_zero : float
            Formation energy at E_F = 0 (eV)
        charge_state : int
            Charge state of the defect
        fermi_level : float
            Fermi level in eV (referenced to VBM)

        Returns
        -------
        float
            Defect concentration in cm^-3
        """
        # Formation energy at given Fermi level
        E_formation = formation_energy_at_zero + charge_state * fermi_level  # noqa: N806

        # Thermal energy
        kT = K_B * self.temperature  # noqa: N806

        # Concentration: [D^q] = N_sites × exp(-E_formation / kT)  # noqa: RUF003
        concentration = self.n_sites * np.exp(-E_formation / kT)

        return concentration

    def charge_neutrality_residual(self, fermi_level: float) -> float:
        """
        Calculate charge neutrality residual.

        The residual is: Σ q[D^q] + n - p
        At equilibrium, this should equal zero.

        Parameters
        ----------
        fermi_level : float
            Fermi level in eV (referenced to VBM)

        Returns
        -------
        float
            Charge neutrality residual in cm^-3
        """
        # Calculate carrier concentrations
        n, p = self.calculate_carrier_concentrations(fermi_level)

        # Calculate total charge from defects
        defect_charge = 0.0
        for doc in self.defect_documents:
            q = doc.charge_state
            E_formation_0 = doc.corrected_formation_energy  # noqa: N806
            concentration = self.calculate_defect_concentration(
                E_formation_0, q, fermi_level
            )
            defect_charge += q * concentration

        # Charge neutrality condition
        residual = defect_charge + n - p

        return residual

    def solve_fermi_level(
        self, ef_min: float | None = None, ef_max: float | None = None
    ) -> float:
        """
        Solve for self-consistent Fermi level.

        Uses Brent's method to find the Fermi level that satisfies
        charge neutrality.

        Parameters
        ----------
        ef_min : float, optional
            Minimum E_F for search (eV). Default: VBM - 1 eV
        ef_max : float, optional
            Maximum E_F for search (eV). Default: CBM + 1 eV

        Returns
        -------
        float
            Self-consistent Fermi level in eV (referenced to VBM)
        """
        # Set search range
        if ef_min is None:
            ef_min = self.vbm_energy - 1.0  # 1 eV below VBM
        if ef_max is None:
            ef_max = self.vbm_energy + self.bandgap + 1.0  # 1 eV above CBM

        logger.info(f"Solving for Fermi level in range [{ef_min:.3f}, {ef_max:.3f}] eV")

        try:
            # Use Brent's method for root finding
            fermi_level = brentq(
                self.charge_neutrality_residual,
                ef_min,
                ef_max,
                xtol=1e-6,  # 1 µeV tolerance
                rtol=1e-6,
            )
            logger.info(f"Converged Fermi level: {fermi_level:.6f} eV")
        except ValueError as e:
            logger.exception(
                f"Failed to find Fermi level in range [{ef_min:.3f}, {ef_max:.3f}] eV"
            )
            logger.exception(
                f"Residual at bounds: f({ef_min:.3f}) = "
                f"{self.charge_neutrality_residual(ef_min):.2e}, "
                f"f({ef_max:.3f}) = {self.charge_neutrality_residual(ef_max):.2e}"
            )
            raise ValueError(
                "Could not find Fermi level. Charge neutrality residuals at bounds "
                "have same sign. Try adjusting search range or check defect data."
            ) from e

        return fermi_level

    def calculate_concentrations(
        self, fermi_level: float | None = None
    ) -> ConcentrationResult:
        """
        Calculate equilibrium defect concentrations.

        If fermi_level is None, solves for self-consistent Fermi level.
        Otherwise, uses provided Fermi level.

        Parameters
        ----------
        fermi_level : float, optional
            Fermi level in eV (referenced to VBM). If None, solves self-consistently.

        Returns
        -------
        ConcentrationResult
            Concentration results
        """
        # Solve for Fermi level if not provided
        fermi_level_converged = True
        if fermi_level is None:
            try:
                fermi_level = self.solve_fermi_level()
            except ValueError as e:
                logger.warning(
                    f"Could not solve Fermi level ({e}). Using midgap as fallback. "
                    "This may indicate unrealistic formation energies (e.g., dry-run mode)."
                )
                fermi_level = self.vbm_energy + self.bandgap / 2
                fermi_level_converged = False

        # Calculate carrier concentrations
        n, p = self.calculate_carrier_concentrations(fermi_level)

        logger.info(f"Electron concentration: {n:.2e} cm^-3")
        logger.info(f"Hole concentration: {p:.2e} cm^-3")

        # Calculate defect concentrations
        defect_concentrations = []
        for doc in self.defect_documents:
            q = doc.charge_state
            E_formation_0 = doc.corrected_formation_energy  # noqa: N806
            concentration = self.calculate_defect_concentration(
                E_formation_0, q, fermi_level
            )

            # Formation energy at this Fermi level
            E_formation = E_formation_0 + q * fermi_level  # noqa: N806

            defect_conc = DefectConcentration(
                defect_name=doc.defect_species or doc.defect_type,
                charge_state=q,
                concentration=concentration,
                formation_energy=E_formation,
            )
            defect_concentrations.append(defect_conc)

            logger.debug(
                f"{defect_conc.defect_name} (q={q:+d}): "
                f"[D] = {concentration:.2e} cm^-3, E_formation = {E_formation:.3f} eV"
            )

        # Calculate charge neutrality error
        charge_error = self.charge_neutrality_residual(fermi_level)

        return ConcentrationResult(
            temperature=self.temperature,
            fermi_level=fermi_level,
            electron_concentration=n,
            hole_concentration=p,
            defect_concentrations=defect_concentrations,
            charge_neutrality_error=charge_error,
            fermi_level_converged=fermi_level_converged,
        )

    def calculate_brouwer_vs_fermi_level(
        self,
        ef_min: float | None = None,
        ef_max: float | None = None,
        n_points: int = 100,
    ) -> BrouwerDiagramData:
        """
        Calculate Brouwer diagram data vs. Fermi level.

        This scans defect concentrations as a function of Fermi level position,
        showing how charge state populations change across the band gap.

        Parameters
        ----------
        ef_min : float, optional
            Minimum Fermi level (eV). Default: VBM - 0.5 eV
        ef_max : float, optional
            Maximum Fermi level (eV). Default: CBM + 0.5 eV
        n_points : int
            Number of points to sample. Default: 100

        Returns
        -------
        BrouwerDiagramData
            Brouwer diagram data
        """
        # Set scan range
        if ef_min is None:
            ef_min = self.vbm_energy - 0.5
        if ef_max is None:
            ef_max = self.vbm_energy + self.bandgap + 0.5

        # Create Fermi level array
        fermi_levels = np.linspace(ef_min, ef_max, n_points)

        # Initialize arrays
        n_defects = len(self.defect_documents)
        concentrations = np.zeros((n_points, n_defects))
        electron_concs = np.zeros(n_points)
        hole_concs = np.zeros(n_points)

        # Create defect labels
        defect_names = []
        for doc in self.defect_documents:
            name = doc.defect_species or doc.defect_type
            q = doc.charge_state
            defect_names.append(f"{name} (q={q:+d})")

        logger.info(
            f"Calculating Brouwer diagram: scanning {n_points} Fermi level points "
            f"from {ef_min:.3f} to {ef_max:.3f} eV"
        )

        # Scan over Fermi levels
        for i, ef in enumerate(fermi_levels):
            # Calculate carrier concentrations
            n, p = self.calculate_carrier_concentrations(ef)
            electron_concs[i] = n
            hole_concs[i] = p

            # Calculate defect concentrations
            for j, doc in enumerate(self.defect_documents):
                q = doc.charge_state
                E_formation_0 = doc.corrected_formation_energy  # noqa: N806
                conc = self.calculate_defect_concentration(E_formation_0, q, ef)
                concentrations[i, j] = conc

        return BrouwerDiagramData(
            variable_name="Fermi level",
            variable_values=fermi_levels,
            variable_unit="eV",
            defect_names=defect_names,
            concentrations=concentrations,
            electron_concentrations=electron_concs,
            hole_concentrations=hole_concs,
        )

    def calculate_brouwer_vs_temperature(
        self,
        temp_min: float = 200.0,
        temp_max: float = 1000.0,
        n_points: int = 100,
        solve_fermi_level: bool = True,
    ) -> BrouwerDiagramData:
        """
        Calculate Brouwer diagram data vs. temperature.

        This scans defect concentrations as a function of temperature,
        showing thermal activation of defects.

        Parameters
        ----------
        temp_min : float
            Minimum temperature (K). Default: 200 K
        temp_max : float
            Maximum temperature (K). Default: 1000 K
        n_points : int
            Number of temperature points. Default: 100
        solve_fermi_level : bool
            If True, solve for self-consistent Fermi level at each temperature.
            If False, use fixed Fermi level (midgap). Default: True

        Returns
        -------
        BrouwerDiagramData
            Brouwer diagram data
        """
        # Create temperature array
        temperatures = np.linspace(temp_min, temp_max, n_points)

        # Initialize arrays
        n_defects = len(self.defect_documents)
        concentrations = np.zeros((n_points, n_defects))
        electron_concs = np.zeros(n_points)
        hole_concs = np.zeros(n_points)

        # Create defect labels
        defect_names = []
        for doc in self.defect_documents:
            name = doc.defect_species or doc.defect_type
            q = doc.charge_state
            defect_names.append(f"{name} (q={q:+d})")

        # Store original temperature
        original_temp = self.temperature

        logger.info(
            f"Calculating Brouwer diagram: scanning {n_points} temperature points "
            f"from {temp_min:.1f} to {temp_max:.1f} K"
        )
        if solve_fermi_level:
            logger.info("Solving for self-consistent Fermi level at each temperature")
        else:
            logger.info("Using fixed Fermi level (midgap)")

        # Scan over temperatures
        for i, temp in enumerate(temperatures):
            # Update temperature
            self.temperature = temp

            # Determine Fermi level
            if solve_fermi_level:
                try:
                    fermi_level = self.solve_fermi_level()
                except ValueError:
                    logger.warning(
                        f"Could not solve Fermi level at T={temp:.1f} K, using midgap"
                    )
                    fermi_level = self.vbm_energy + self.bandgap / 2
            else:
                # Use midgap
                fermi_level = self.vbm_energy + self.bandgap / 2

            # Calculate carrier concentrations
            n, p = self.calculate_carrier_concentrations(fermi_level)
            electron_concs[i] = n
            hole_concs[i] = p

            # Calculate defect concentrations
            for j, doc in enumerate(self.defect_documents):
                q = doc.charge_state
                E_formation_0 = doc.corrected_formation_energy  # noqa: N806
                conc = self.calculate_defect_concentration(
                    E_formation_0, q, fermi_level
                )
                concentrations[i, j] = conc

        # Restore original temperature
        self.temperature = original_temp

        return BrouwerDiagramData(
            variable_name="Temperature",
            variable_values=temperatures,
            variable_unit="K",
            defect_names=defect_names,
            concentrations=concentrations,
            electron_concentrations=electron_concs,
            hole_concentrations=hole_concs,
        )


def plot_brouwer_diagram(
    data: BrouwerDiagramData,
    filename: str | Path | None = None,
    show_carriers: bool = True,
    log_scale: bool = True,
    vbm_energy: float = 0.0,
    bandgap: float | None = None,
    figsize: tuple[float, float] = (10, 7),
    dpi: int = 300,
) -> Path | None:
    """
    Plot Brouwer diagram.

    Creates a publication-quality plot showing defect concentrations
    as a function of a thermodynamic variable (Fermi level, temperature, etc.).

    Parameters
    ----------
    data : BrouwerDiagramData
        Brouwer diagram data to plot
    filename : str or Path, optional
        Output filename. If None, display plot interactively
    show_carriers : bool
        Whether to show electron/hole concentrations. Default: True
    log_scale : bool
        Use logarithmic scale for y-axis. Default: True
    vbm_energy : float
        VBM energy for marking band edges (only for Fermi level plots). Default: 0.0
    bandgap : float, optional
        Band gap for marking band edges (only for Fermi level plots)
    figsize : tuple
        Figure size in inches. Default: (10, 7)
    dpi : int
        Figure DPI for saved image. Default: 300

    Returns
    -------
    Path or None
        Path to saved figure, or None if displayed interactively
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    _fig, ax = plt.subplots(figsize=figsize)

    # Get colormap
    n_lines = len(data.defect_names)
    if show_carriers:
        n_lines += 2  # electrons and holes
    colors = cm.tab20(np.linspace(0, 1, n_lines))

    # Plot defect concentrations
    for i, name in enumerate(data.defect_names):
        conc = data.concentrations[:, i]
        # Only plot if concentration is non-zero somewhere
        if np.any(conc > 0):
            if log_scale:
                # Replace zeros with small number for log plot
                conc = np.where(conc > 0, conc, 1e-30)
            ax.plot(
                data.variable_values,
                conc,
                label=name,
                linewidth=2,
                color=colors[i],
            )

    # Plot carrier concentrations
    if show_carriers:
        idx = len(data.defect_names)
        # Electrons
        n = data.electron_concentrations
        if log_scale:
            n = np.where(n > 0, n, 1e-30)
        ax.plot(
            data.variable_values,
            n,
            label="electrons (n)",
            linewidth=2,
            linestyle="--",
            color=colors[idx],
        )

        # Holes
        p = data.hole_concentrations
        if log_scale:
            p = np.where(p > 0, p, 1e-30)
        ax.plot(
            data.variable_values,
            p,
            label="holes (p)",
            linewidth=2,
            linestyle="--",
            color=colors[idx + 1],
        )

    # Set y-axis scale
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Concentration (cm$^{-3}$, log scale)", fontsize=12)
    else:
        ax.set_ylabel("Concentration (cm$^{-3}$)", fontsize=12)

    # Set x-axis label
    xlabel = f"{data.variable_name}"
    if data.variable_unit:
        xlabel += f" ({data.variable_unit})"
    ax.set_xlabel(xlabel, fontsize=12)

    # Mark band edges for Fermi level plots
    if data.variable_name == "Fermi level" and bandgap is not None:
        # VBM line
        ax.axvline(
            vbm_energy,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label="VBM",
        )
        # CBM line
        cbm = vbm_energy + bandgap
        ax.axvline(
            cbm, color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="CBM"
        )

        # Shade gap region
        ax.axvspan(vbm_energy, cbm, alpha=0.1, color="gray")

    # Title
    title = f"Brouwer Diagram: Defect Concentrations vs. {data.variable_name}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Legend
    ax.legend(loc="best", fontsize=10, framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Tight layout
    plt.tight_layout()

    # Save or show
    if filename:
        filepath = Path(filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        logger.info(f"Brouwer diagram saved to {filepath}")
        plt.close()
        return filepath
    plt.show()
    return None


# Job functions


@job
def calculate_defect_concentrations_job(
    defect_documents: list[DefectDocument],
    bandgap: float,
    vbm_energy: float = 0.0,
    temperature: float = 300.0,
    n_sites: float | None = None,
    effective_dos: dict[str, float] | None = None,
) -> ConcentrationResult:
    """
    Job to calculate defect concentrations.

    Parameters
    ----------
    defect_documents : list[DefectDocument]
        List of defect calculation results
    bandgap : float
        Band gap in eV
    vbm_energy : float
        VBM energy in eV. Default: 0.0
    temperature : float
        Temperature in K. Default: 300 K
    n_sites : float, optional
        Number of defect sites in cm^-3
    effective_dos : dict, optional
        Effective density of states {"N_C": ..., "N_V": ...}

    Returns
    -------
    ConcentrationResult
        Concentration calculation results
    """
    analyzer = DefectConcentrationAnalyzer(
        defect_documents=defect_documents,
        bandgap=bandgap,
        vbm_energy=vbm_energy,
        temperature=temperature,
        n_sites=n_sites,
        effective_density_of_states=effective_dos,
    )

    result = analyzer.calculate_concentrations()

    logger.info(f"Defect concentration calculation complete at T = {temperature} K")
    logger.info(f"Self-consistent Fermi level: {result.fermi_level:.4f} eV")
    logger.info(
        f"Total defect concentration: {result.total_defect_concentration:.2e} cm^-3"
    )

    return result


@job
def write_concentration_summary(
    result: ConcentrationResult,
    bandgap: float,
    vbm_energy: float = 0.0,
    filename: str = "defect_concentrations.txt",
) -> Path:
    """
    Job to write defect concentration summary to file.

    Parameters
    ----------
    result : ConcentrationResult
        Concentration calculation results
    bandgap : float
        Band gap in eV
    vbm_energy : float
        VBM energy in eV. Default: 0.0
    filename : str
        Output filename

    Returns
    -------
    Path
        Path to summary file
    """
    from atomate2.siesta.utils.text_output import get_standard_footer

    summary_file = Path(filename)

    with open(summary_file, "w") as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("DEFECT CONCENTRATION CALCULATION\n")
        f.write("=" * 80 + "\n\n")

        # Add warning if Fermi level did not converge
        if not result.fermi_level_converged:
            f.write("⚠️  WARNING: DATA MAY NOT BE ACCURATE\n")
            f.write("-" * 80 + "\n")
            f.write("Fermi level solver did not converge - using midgap fallback.\n")
            f.write("This typically indicates:\n")
            f.write("  • Dry-run mode with synthetic/unrealistic formation energies\n")
            f.write("  • Formation energies that don't satisfy charge neutrality\n")
            f.write("  • Unphysical defect concentrations or carrier densities\n")
            f.write("\n")
            f.write(
                "For realistic results, run actual DFT calculations (dry_run=False).\n"
            )
            f.write("=" * 80 + "\n\n")

        # Calculation parameters
        f.write("CALCULATION PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Temperature:           {result.temperature:.1f} K\n")
        f.write(f"Band gap:              {bandgap:.3f} eV\n")
        f.write(f"VBM energy:            {vbm_energy:.3f} eV\n")
        cbm = vbm_energy + bandgap
        f.write(f"CBM energy:            {cbm:.3f} eV\n")
        f.write(f"Thermal energy (kT):   {K_B * result.temperature:.6f} eV\n")
        f.write("\n")

        # Self-consistent Fermi level
        f.write("SELF-CONSISTENT FERMI LEVEL\n")
        f.write("-" * 80 + "\n")
        f.write(f"Fermi level (E_F):     {result.fermi_level:.6f} eV\n")

        # Position relative to bands
        if result.fermi_level < vbm_energy:
            f.write(
                f"Position:              {abs(result.fermi_level - vbm_energy):.3f} eV below VBM\n"
            )
        elif result.fermi_level < cbm:
            frac = (result.fermi_level - vbm_energy) / bandgap * 100
            f.write(f"Position:              {frac:.1f}% through gap\n")
        else:
            f.write(
                f"Position:              {result.fermi_level - cbm:.3f} eV above CBM\n"
            )

        f.write(f"Charge neutrality:     {result.charge_neutrality_error:.2e} cm^-3\n")
        f.write("\n")

        # Carrier concentrations
        f.write("CARRIER CONCENTRATIONS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Electrons (n):         {result.electron_concentration:.4e} cm^-3\n")
        f.write(f"Holes (p):             {result.hole_concentration:.4e} cm^-3\n")
        intrinsic = np.sqrt(result.electron_concentration * result.hole_concentration)
        f.write(f"Intrinsic (n_i):       {intrinsic:.4e} cm^-3\n")
        f.write("\n")

        # Defect concentrations
        f.write("DEFECT CONCENTRATIONS\n")
        f.write("-" * 80 + "\n")

        # Group by defect name
        defect_groups: dict[str, list[DefectConcentration]] = {}
        for dc in result.defect_concentrations:
            if dc.defect_name not in defect_groups:
                defect_groups[dc.defect_name] = []
            defect_groups[dc.defect_name].append(dc)

        # Write table for each defect
        for defect_name, concentrations in defect_groups.items():
            f.write(f"\n{defect_name}:\n")
            f.write(
                f"{'Charge':>8}  {'[D^q] (cm^-3)':>15}  {'E_formation (eV)':>16}  {'Fraction':>10}\n"
            )
            f.write(f"{'-' * 8}  {'-' * 15}  {'-' * 16}  {'-' * 10}\n")

            # Sort by concentration (highest first)
            concentrations = sorted(
                concentrations, key=lambda x: x.concentration, reverse=True
            )

            total_conc = sum(c.concentration for c in concentrations)

            for dc in concentrations:
                fraction = dc.concentration / total_conc if total_conc > 0 else 0
                f.write(
                    f"{dc.charge_state:+8d}  "
                    f"{dc.concentration:>15.4e}  "
                    f"{dc.formation_energy:>10.4f}  "
                    f"{fraction:>10.1%}\n"
                )

            f.write(f"{'Total':>8}  {total_conc:>15.4e}\n")

        f.write("\n")

        # Dominant charge states
        f.write("DOMINANT CHARGE STATES\n")
        f.write("-" * 80 + "\n")
        dominant = result.dominant_charge_states
        for defect_name, charge in dominant.items():
            f.write(f"{defect_name:30} q = {charge:+d}\n")
        f.write("\n")

        # Total concentrations
        f.write("TOTAL CONCENTRATIONS\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"Total defects:         {result.total_defect_concentration:.4e} cm^-3\n"
        )
        f.write(f"Electrons:             {result.electron_concentration:.4e} cm^-3\n")
        f.write(f"Holes:                 {result.hole_concentration:.4e} cm^-3\n")
        f.write("\n")

        # Conductivity type
        if result.electron_concentration > result.hole_concentration:
            majority = "n-type (electron-dominated)"
        elif result.hole_concentration > result.electron_concentration:
            majority = "p-type (hole-dominated)"
        else:
            majority = "intrinsic (n ≈ p)"
        f.write(f"Conductivity type:     {majority}\n")
        f.write("\n")

        # Notes
        f.write("NOTES\n")
        f.write("-" * 80 + "\n")
        f.write("• Concentrations calculated at thermodynamic equilibrium\n")
        f.write("• Fermi level determined by charge neutrality condition\n")
        f.write("• Formation energies include finite-size corrections\n")
        f.write("• Assumes dilute defect limit and non-interacting defects\n")
        f.write("\n")

        # Footer
        f.write(
            get_standard_footer(
                width=80,
                additional_info={
                    "Temperature": f"{result.temperature:.1f} K",
                    "Fermi level": f"{result.fermi_level:.4f} eV",
                },
            )
        )

    logger.info(f"Concentration summary written to {summary_file}")

    return summary_file


@job
def calculate_brouwer_vs_fermi_level_job(
    defect_documents: list[DefectDocument],
    bandgap: float,
    vbm_energy: float = 0.0,
    temperature: float = 300.0,
    n_sites: float | None = None,
    effective_dos: dict[str, float] | None = None,
    ef_min: float | None = None,
    ef_max: float | None = None,
    n_points: int = 100,
) -> BrouwerDiagramData:
    """
    Job to calculate Brouwer diagram data vs. Fermi level.

    Parameters
    ----------
    defect_documents : list[DefectDocument]
        List of defect calculation results
    bandgap : float
        Band gap in eV
    vbm_energy : float
        VBM energy in eV. Default: 0.0
    temperature : float
        Temperature in K. Default: 300 K
    n_sites : float, optional
        Number of defect sites in cm^-3
    effective_dos : dict, optional
        Effective density of states {"N_C": ..., "N_V": ...}
    ef_min : float, optional
        Minimum Fermi level (eV)
    ef_max : float, optional
        Maximum Fermi level (eV)
    n_points : int
        Number of Fermi level points. Default: 100

    Returns
    -------
    BrouwerDiagramData
        Brouwer diagram data
    """
    analyzer = DefectConcentrationAnalyzer(
        defect_documents=defect_documents,
        bandgap=bandgap,
        vbm_energy=vbm_energy,
        temperature=temperature,
        n_sites=n_sites,
        effective_density_of_states=effective_dos,
    )

    data = analyzer.calculate_brouwer_vs_fermi_level(
        ef_min=ef_min, ef_max=ef_max, n_points=n_points
    )

    logger.info(
        f"Brouwer diagram calculation complete: {n_points} Fermi level points, "
        f"{len(data.defect_names)} defects"
    )

    return data


@job
def calculate_brouwer_vs_temperature_job(
    defect_documents: list[DefectDocument],
    bandgap: float,
    vbm_energy: float = 0.0,
    n_sites: float | None = None,
    effective_dos: dict[str, float] | None = None,
    temp_min: float = 200.0,
    temp_max: float = 1000.0,
    n_points: int = 100,
    solve_fermi_level: bool = True,
) -> BrouwerDiagramData:
    """
    Job to calculate Brouwer diagram data vs. temperature.

    Parameters
    ----------
    defect_documents : list[DefectDocument]
        List of defect calculation results
    bandgap : float
        Band gap in eV
    vbm_energy : float
        VBM energy in eV. Default: 0.0
    n_sites : float, optional
        Number of defect sites in cm^-3
    effective_dos : dict, optional
        Effective density of states {"N_C": ..., "N_V": ...}
    temp_min : float
        Minimum temperature (K). Default: 200 K
    temp_max : float
        Maximum temperature (K). Default: 1000 K
    n_points : int
        Number of temperature points. Default: 100
    solve_fermi_level : bool
        Solve for self-consistent Fermi level at each T. Default: True

    Returns
    -------
    BrouwerDiagramData
        Brouwer diagram data
    """
    # Use midpoint temperature for analyzer initialization
    temp_init = (temp_min + temp_max) / 2

    analyzer = DefectConcentrationAnalyzer(
        defect_documents=defect_documents,
        bandgap=bandgap,
        vbm_energy=vbm_energy,
        temperature=temp_init,
        n_sites=n_sites,
        effective_density_of_states=effective_dos,
    )

    data = analyzer.calculate_brouwer_vs_temperature(
        temp_min=temp_min,
        temp_max=temp_max,
        n_points=n_points,
        solve_fermi_level=solve_fermi_level,
    )

    logger.info(
        f"Brouwer diagram calculation complete: {n_points} temperature points, "
        f"{len(data.defect_names)} defects"
    )

    return data


@job
def plot_brouwer_diagram_job(
    data: BrouwerDiagramData,
    filename: str = "brouwer_diagram.png",
    show_carriers: bool = True,
    vbm_energy: float = 0.0,
    bandgap: float | None = None,
) -> Path:
    """
    Job to plot Brouwer diagram.

    Parameters
    ----------
    data : BrouwerDiagramData
        Brouwer diagram data to plot
    filename : str
        Output filename. Default: "brouwer_diagram.png"
    show_carriers : bool
        Show electron/hole concentrations. Default: True
    vbm_energy : float
        VBM energy for band edge markers. Default: 0.0
    bandgap : float, optional
        Band gap for band edge markers

    Returns
    -------
    Path
        Path to saved figure
    """
    filepath = plot_brouwer_diagram(
        data=data,
        filename=filename,
        show_carriers=show_carriers,
        vbm_energy=vbm_energy,
        bandgap=bandgap,
    )

    if filepath is None:
        raise RuntimeError("Failed to save Brouwer diagram")

    return filepath
