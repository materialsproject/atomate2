"""
Shockley-Read-Hall (SRH) recombination analysis for defects.

This module implements SRH recombination rate calculations for defect-mediated
carrier recombination in semiconductors. It provides tools to calculate:
- Electron and hole capture cross-sections
- Carrier lifetimes (minority and majority)
- SRH recombination rates
- Generation-recombination lifetimes

The SRH recombination rate describes carrier recombination through defect
states in the bandgap. It is the dominant recombination mechanism in many
semiconductors and is critical for understanding:
- Minority carrier lifetimes
- Device performance (solar cells, LEDs, transistors)
- Defect characterization via DLTS, photoluminescence

Key equations:
    R_SRH = (n*p - n_i²) / [τ_p(n + n_1) + τ_n(p + p_1)]

    where:
    - τ_n, τ_p: electron and hole lifetimes
    - n_1 = n_i * exp((E_T - E_i) / k_B T)
    - p_1 = n_i * exp((E_i - E_T) / k_B T)
    - E_T: defect energy level (from CTL)
    - E_i: intrinsic Fermi level

References
----------
    - Shockley & Read, Phys. Rev. 87, 835 (1952)
    - Hall, Phys. Rev. 87, 387 (1952)
    - Schroder, "Semiconductor Material and Device Characterization", 3rd ed.
    - Rein & Glunz, Appl. Phys. Lett. 82, 1054 (2003) - parameterized defects
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from jobflow import job
from monty.json import MSONable

if TYPE_CHECKING:
    from atomate2.siesta.flows.defects.analysis.concentration import ConcentrationResult
    from atomate2.siesta.flows.defects.analysis.formation_energy import (
        ChargeTransitionLevel,
        FormationEnergyDiagram,
    )

logger = logging.getLogger(__name__)

# Physical constants
K_B = 8.617333e-5  # Boltzmann constant in eV/K
H_BAR = 1.054571817e-34  # Reduced Planck constant in J·s
Q_E = 1.602176634e-19  # Elementary charge in C
M_E = 9.1093837015e-31  # Electron mass in kg


@dataclass
class CaptureParameters(MSONable):
    """
    Capture cross-section parameters for electron and hole capture.

    Parameters
    ----------
    sigma_n : float
        Electron capture cross-section in cm²
    sigma_p : float
        Hole capture cross-section in cm²
    method : str
        Method used to determine cross-sections:
        - "default": Default values (σ_n = σ_p = 1e-15 cm²)
        - "empirical": Empirical values from literature
        - "calculated": DFT-calculated (future implementation)
    temperature : float
        Temperature in Kelvin
    """  # noqa: RUF002

    sigma_n: float
    sigma_p: float
    method: str = "default"
    temperature: float = 300.0

    def as_dict(self) -> dict:
        """Return dict representation for serialization."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "sigma_n": self.sigma_n,
            "sigma_p": self.sigma_p,
            "method": self.method,
            "temperature": self.temperature,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CaptureParameters:
        """Create from dict representation."""
        return cls(
            sigma_n=d["sigma_n"],
            sigma_p=d["sigma_p"],
            method=d.get("method", "default"),
            temperature=d.get("temperature", 300.0),
        )

    @classmethod
    def from_defaults(cls, temperature: float = 300.0) -> CaptureParameters:
        """
        Create default capture parameters.

        Uses typical values: σ_n = σ_p = 1e-15 cm²
        This is a reasonable estimate for many defects in semiconductors.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin

        Returns
        -------
        CaptureParameters
            Default capture parameters
        """  # noqa: RUF002
        return cls(
            sigma_n=1e-15,
            sigma_p=1e-15,
            method="default",
            temperature=temperature,
        )

    @classmethod
    def from_empirical(
        cls,
        material: str,
        defect_type: str,
        temperature: float = 300.0,
    ) -> CaptureParameters:
        """
        Create capture parameters from empirical literature values.

        Parameters
        ----------
        material : str
            Material name (e.g., "Si", "GaAs", "GaN")
        defect_type : str
            Defect type (e.g., "vacancy", "interstitial")
        temperature : float
            Temperature in Kelvin

        Returns
        -------
        CaptureParameters
            Empirical capture parameters

        Notes
        -----
        Currently returns default values. Future implementation will include
        literature database of measured capture cross-sections.
        """
        # TODO: Add empirical database for common materials/defects
        logger.warning(
            f"Empirical data not yet available for {material}/{defect_type}. "
            "Using default values."
        )
        return cls.from_defaults(temperature)


@dataclass
class SRHLifetimes(MSONable):
    """
    SRH carrier lifetimes for a defect.

    Parameters
    ----------
    tau_n : float
        Electron lifetime in seconds
    tau_p : float
        Hole lifetime in seconds
    tau_n0 : float
        Electron SRH lifetime (low-injection) in seconds
    tau_p0 : float
        Hole SRH lifetime (low-injection) in seconds
    defect_name : str
        Name of the defect
    defect_concentration : float
        Defect concentration in cm⁻³
    capture_params : CaptureParameters
        Capture cross-section parameters
    """

    tau_n: float
    tau_p: float
    tau_n0: float
    tau_p0: float
    defect_name: str
    defect_concentration: float
    capture_params: CaptureParameters

    @property
    def tau_eff(self) -> float:
        """
        Effective SRH lifetime (harmonic mean).

        For ambipolar recombination:
            1/τ_eff = 1/τ_n + 1/τ_p
        """
        return 1.0 / (1.0 / self.tau_n + 1.0 / self.tau_p)

    @property
    def minority_carrier_lifetime(self) -> float:
        """
        Minority carrier lifetime (minimum of τ_n, τ_p).

        This is typically the limiting factor for device performance.
        """
        return min(self.tau_n, self.tau_p)

    def as_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "tau_n": self.tau_n,
            "tau_p": self.tau_p,
            "tau_n0": self.tau_n0,
            "tau_p0": self.tau_p0,
            "defect_name": self.defect_name,
            "defect_concentration": self.defect_concentration,
            "capture_params": self.capture_params.as_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SRHLifetimes:
        """Deserialize from dictionary."""
        return cls(
            tau_n=d["tau_n"],
            tau_p=d["tau_p"],
            tau_n0=d["tau_n0"],
            tau_p0=d["tau_p0"],
            defect_name=d["defect_name"],
            defect_concentration=d["defect_concentration"],
            capture_params=CaptureParameters.from_dict(d["capture_params"]),
        )


@dataclass
class SRHRecombinationRate(MSONable):
    """
    SRH recombination rate data.

    Parameters
    ----------
    defect_name : str
        Name of the defect
    defect_level : float
        Defect energy level (eV, referenced to VBM)
    temperature : float
        Temperature in Kelvin
    electron_concentration : float
        Electron concentration in cm⁻³
    hole_concentration : float
        Hole concentration in cm⁻³
    recombination_rate : float
        Net SRH recombination rate in cm⁻³ s⁻¹
    generation_rate : float
        Thermal generation rate in cm⁻³ s⁻¹
    lifetimes : SRHLifetimes
        Carrier lifetimes
    """

    defect_name: str
    defect_level: float
    temperature: float
    electron_concentration: float
    hole_concentration: float
    recombination_rate: float
    generation_rate: float
    lifetimes: SRHLifetimes

    def as_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "defect_name": self.defect_name,
            "defect_level": self.defect_level,
            "temperature": self.temperature,
            "electron_concentration": self.electron_concentration,
            "hole_concentration": self.hole_concentration,
            "recombination_rate": self.recombination_rate,
            "generation_rate": self.generation_rate,
            "lifetimes": self.lifetimes.as_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SRHRecombinationRate:
        """Deserialize from dictionary."""
        return cls(
            defect_name=d["defect_name"],
            defect_level=d["defect_level"],
            temperature=d["temperature"],
            electron_concentration=d["electron_concentration"],
            hole_concentration=d["hole_concentration"],
            recombination_rate=d["recombination_rate"],
            generation_rate=d["generation_rate"],
            lifetimes=SRHLifetimes.from_dict(d["lifetimes"]),
        )


@dataclass
class SRHAnalysisResult(MSONable):
    """
    Complete SRH analysis results.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin
    bandgap : float
        Band gap in eV
    intrinsic_carrier_concentration : float
        Intrinsic carrier concentration (n_i) in cm⁻³
    defect_results : list[SRHRecombinationRate]
        SRH results for each defect
    total_recombination_rate : float
        Total recombination rate (sum over all defects) in cm⁻³ s⁻¹
    dominant_defect : str
        Name of defect with highest recombination rate
    """

    temperature: float
    bandgap: float
    intrinsic_carrier_concentration: float
    defect_results: list[SRHRecombinationRate]
    total_recombination_rate: float
    dominant_defect: str = ""

    def __post_init__(self):
        """Determine dominant defect after initialization."""
        if not self.dominant_defect and self.defect_results:
            # Find defect with maximum recombination rate
            max_rate = max(d.recombination_rate for d in self.defect_results)
            for defect in self.defect_results:
                if defect.recombination_rate == max_rate:
                    self.dominant_defect = defect.defect_name
                    break

    def as_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "temperature": self.temperature,
            "bandgap": self.bandgap,
            "intrinsic_carrier_concentration": self.intrinsic_carrier_concentration,
            "defect_results": [dr.as_dict() for dr in self.defect_results],
            "total_recombination_rate": self.total_recombination_rate,
            "dominant_defect": self.dominant_defect,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SRHAnalysisResult:
        """Deserialize from dictionary."""
        return cls(
            temperature=d["temperature"],
            bandgap=d["bandgap"],
            intrinsic_carrier_concentration=d["intrinsic_carrier_concentration"],
            defect_results=[
                SRHRecombinationRate.from_dict(dr) for dr in d["defect_results"]
            ],
            total_recombination_rate=d["total_recombination_rate"],
            dominant_defect=d.get("dominant_defect", ""),
        )


class SRHAnalyzer:
    """
    Analyzer for SRH recombination calculations.

    This class provides methods to calculate SRH recombination rates,
    carrier lifetimes, and generation rates for defects in semiconductors.

    Parameters
    ----------
    formation_diagram : FormationEnergyDiagram
        Formation energy diagram with CTL data
    bandgap : float
        Band gap in eV
    effective_mass_electron : float, optional
        Effective electron mass (in units of free electron mass)
        Default: 1.0
    effective_mass_hole : float, optional
        Effective hole mass (in units of free electron mass)
        Default: 1.0
    """

    def __init__(
        self,
        formation_diagram: FormationEnergyDiagram,
        bandgap: float,
        effective_mass_electron: float = 1.0,
        effective_mass_hole: float = 1.0,
    ) -> None:
        self.formation_diagram = formation_diagram
        self.bandgap = bandgap
        self.m_e_eff = effective_mass_electron
        self.m_h_eff = effective_mass_hole

    def calculate_intrinsic_carrier_concentration(
        self,
        temperature: float,
    ) -> float:
        """
        Calculate intrinsic carrier concentration n_i.

        Uses: n_i = sqrt(N_c * N_v) * exp(-E_g / (2 k_B T))

        where N_c and N_v are effective density of states in
        conduction and valence bands.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin

        Returns
        -------
        float
            Intrinsic carrier concentration in cm⁻³
        """
        # Effective density of states (cm⁻³)
        # N_c = 2 * (2π m_e* k_B T / h²)^(3/2)
        prefactor = 2.0 * (2.0 * np.pi * K_B * Q_E * temperature / (H_BAR**2)) ** 1.5

        # Convert from m⁻³ to cm⁻³
        N_c = prefactor * (self.m_e_eff * M_E) ** 1.5 * 1e-6  # noqa: N806
        N_v = prefactor * (self.m_h_eff * M_E) ** 1.5 * 1e-6  # noqa: N806

        # Intrinsic concentration
        n_i = np.sqrt(N_c * N_v) * np.exp(-self.bandgap / (2.0 * K_B * temperature))

        return n_i

    def calculate_thermal_velocity(
        self,
        temperature: float,
        mass_ratio: float = 1.0,
    ) -> float:
        """
        Calculate thermal velocity of carriers.

        Uses: v_th = sqrt(3 k_B T / m*)

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
        mass_ratio : float
            Effective mass ratio (m*/m_e)

        Returns
        -------
        float
            Thermal velocity in cm/s
        """
        # v_th in m/s
        v_th_si = np.sqrt(3.0 * K_B * Q_E * temperature / (mass_ratio * M_E))

        # Convert to cm/s
        return v_th_si * 100.0

    def calculate_lifetimes(
        self,
        defect_concentration: float,
        capture_params: CaptureParameters,
        defect_name: str = "defect",
    ) -> SRHLifetimes:
        """
        Calculate SRH carrier lifetimes.

        τ_n = 1 / (σ_n * v_th * N_T)
        τ_p = 1 / (σ_p * v_th * N_T)

        Parameters
        ----------
        defect_concentration : float
            Defect concentration N_T in cm⁻³
        capture_params : CaptureParameters
            Capture cross-section parameters
        defect_name : str
            Name of the defect

        Returns
        -------
        SRHLifetimes
            Carrier lifetime data
        """  # noqa: RUF002
        T = capture_params.temperature  # noqa: N806

        # Thermal velocities (cm/s)
        v_th_n = self.calculate_thermal_velocity(T, self.m_e_eff)
        v_th_p = self.calculate_thermal_velocity(T, self.m_h_eff)

        # SRH lifetimes (seconds)
        # τ = 1 / (σ * v_th * N_T)  # noqa: RUF003
        tau_n0 = 1.0 / (capture_params.sigma_n * v_th_n * defect_concentration)
        tau_p0 = 1.0 / (capture_params.sigma_p * v_th_p * defect_concentration)

        return SRHLifetimes(
            tau_n=tau_n0,
            tau_p=tau_p0,
            tau_n0=tau_n0,
            tau_p0=tau_p0,
            defect_name=defect_name,
            defect_concentration=defect_concentration,
            capture_params=capture_params,
        )

    def calculate_srh_rate(
        self,
        defect_level: float,
        electron_concentration: float,
        hole_concentration: float,
        lifetimes: SRHLifetimes,
        temperature: float,
    ) -> tuple[float, float]:
        """
        Calculate SRH recombination rate.

        R = (n*p - n_i²) / [τ_p(n + n_1) + τ_n(p + p_1)]

        Parameters
        ----------
        defect_level : float
            Defect energy level in eV (referenced to VBM)
        electron_concentration : float
            Electron concentration n in cm⁻³
        hole_concentration : float
            Hole concentration p in cm⁻³
        lifetimes : SRHLifetimes
            Carrier lifetimes
        temperature : float
            Temperature in Kelvin

        Returns
        -------
        tuple[float, float]
            (recombination_rate, generation_rate) in cm⁻³ s⁻¹
        """
        n_i = self.calculate_intrinsic_carrier_concentration(temperature)

        # Intrinsic level (mid-gap)
        E_i = self.bandgap / 2.0  # noqa: N806

        # Electron and hole concentrations when E_F = E_T
        # n_1 = n_i * exp((E_T - E_i) / k_B T)
        # p_1 = n_i * exp((E_i - E_T) / k_B T)
        n_1 = n_i * np.exp((defect_level - E_i) / (K_B * temperature))
        p_1 = n_i * np.exp((E_i - defect_level) / (K_B * temperature))

        # SRH recombination rate
        numerator = electron_concentration * hole_concentration - n_i**2
        denominator = lifetimes.tau_p * (
            electron_concentration + n_1
        ) + lifetimes.tau_n * (hole_concentration + p_1)

        recombination_rate = numerator / denominator

        # Thermal generation rate (R when n = p = 0)
        generation_rate = -(n_i**2) / (lifetimes.tau_p * n_1 + lifetimes.tau_n * p_1)

        return recombination_rate, abs(generation_rate)

    def analyze_from_concentration_result(
        self,
        concentration_result: ConcentrationResult,
        ctls: list[ChargeTransitionLevel],
        capture_params: dict[str, CaptureParameters] | None = None,
    ) -> SRHAnalysisResult:
        """
        Perform SRH analysis from concentration calculation results.

        Parameters
        ----------
        concentration_result : ConcentrationResult
            Defect concentration results with Fermi level
        ctls : list[ChargeTransitionLevel]
            Charge transition levels for defects
        capture_params : dict[str, CaptureParameters], optional
            Capture parameters for each defect. If None, uses defaults.

        Returns
        -------
        SRHAnalysisResult
            Complete SRH analysis results
        """
        T = concentration_result.temperature  # noqa: N806
        n = concentration_result.electron_concentration
        p = concentration_result.hole_concentration
        n_i = self.calculate_intrinsic_carrier_concentration(T)

        # Use default capture params if not provided
        if capture_params is None:
            capture_params = {}

        defect_results = []
        total_rate = 0.0

        # Group CTLs by defect
        defect_levels: dict[str, float] = {}
        for ctl in ctls:
            # Use CTL energy as defect level
            if ctl.defect_name not in defect_levels:
                defect_levels[ctl.defect_name] = ctl.fermi_level

        # Calculate SRH rate for each defect
        for dc in concentration_result.defect_concentrations:
            defect_name = dc.defect_name

            # Get defect level (use CTL or mid-gap as fallback)
            if defect_name in defect_levels:
                E_T = defect_levels[defect_name]  # noqa: N806
            else:
                E_T = self.bandgap / 2.0  # noqa: N806
                logger.warning(f"No CTL found for {defect_name}, using mid-gap level")

            # Get capture parameters
            if defect_name not in capture_params:
                capture_params[defect_name] = CaptureParameters.from_defaults(T)

            cp = capture_params[defect_name]

            # Calculate lifetimes
            lifetimes = self.calculate_lifetimes(
                defect_concentration=dc.concentration,
                capture_params=cp,
                defect_name=defect_name,
            )

            # Calculate SRH rate
            rec_rate, gen_rate = self.calculate_srh_rate(
                defect_level=E_T,
                electron_concentration=n,
                hole_concentration=p,
                lifetimes=lifetimes,
                temperature=T,
            )

            defect_results.append(
                SRHRecombinationRate(
                    defect_name=f"{defect_name}_q_{dc.charge_state:+d}",
                    defect_level=E_T,
                    temperature=T,
                    electron_concentration=n,
                    hole_concentration=p,
                    recombination_rate=rec_rate,
                    generation_rate=gen_rate,
                    lifetimes=lifetimes,
                )
            )

            total_rate += rec_rate

        return SRHAnalysisResult(
            temperature=T,
            bandgap=self.bandgap,
            intrinsic_carrier_concentration=n_i,
            defect_results=defect_results,
            total_recombination_rate=total_rate,
        )


def plot_srh_lifetimes(
    srh_result: SRHAnalysisResult,
    filename: str | Path = "srh_lifetimes.png",
) -> None:
    """
    Plot carrier lifetimes for all defects.

    Parameters
    ----------
    srh_result : SRHAnalysisResult
        SRH analysis results
    filename : str or Path
        Output filename for the plot
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        defect_names = [d.defect_name for d in srh_result.defect_results]
        tau_n = np.array(
            [d.lifetimes.tau_n * 1e9 for d in srh_result.defect_results]
        )  # Convert to ns
        tau_p = np.array([d.lifetimes.tau_p * 1e9 for d in srh_result.defect_results])

        # Replace inf/nan with very small positive value for plotting
        tau_n = np.where(np.isfinite(tau_n), tau_n, 1e-100)
        tau_p = np.where(np.isfinite(tau_p), tau_p, 1e-100)

        # Replace zeros with small positive value for log scale
        tau_n = np.where(tau_n > 0, tau_n, 1e-100)
        tau_p = np.where(tau_p > 0, tau_p, 1e-100)

        # Check if values are too extreme (all below 1e-80)
        use_log = np.any(tau_n > 1e-80) or np.any(tau_p > 1e-80)

        x = np.arange(len(defect_names))
        width = 0.35

        ax.bar(x - width / 2, tau_n, width, label="Electron lifetime (τₙ)", alpha=0.8)
        ax.bar(x + width / 2, tau_p, width, label="Hole lifetime (τₚ)", alpha=0.8)

        ax.set_xlabel("Defect", fontsize=12)
        ax.set_ylabel("Lifetime (ns)", fontsize=12)
        ax.set_title(
            f"SRH Carrier Lifetimes (T = {srh_result.temperature:.0f} K)", fontsize=14
        )
        ax.set_xticks(x)
        ax.set_xticklabels(defect_names, rotation=45, ha="right")
        ax.legend()

        # Only use log scale if we have reasonable values
        if use_log:
            ax.set_yscale("log")

        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved SRH lifetime plot to {filename}")
    except (ValueError, OverflowError) as e:
        logger.warning(f"Could not create SRH lifetime plot: {e}")
        # Create a simple placeholder plot
        _fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "Plot unavailable\n(values too extreme for visualization)",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_title(
            f"SRH Carrier Lifetimes (T = {srh_result.temperature:.0f} K)", fontsize=14
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved placeholder SRH lifetime plot to {filename}")


def plot_srh_recombination_rates(
    srh_result: SRHAnalysisResult,
    filename: str | Path = "srh_recombination_rates.png",
) -> None:
    """
    Plot recombination rates for all defects.

    Parameters
    ----------
    srh_result : SRHAnalysisResult
        SRH analysis results
    filename : str or Path
        Output filename for the plot
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        defect_names = [d.defect_name for d in srh_result.defect_results]
        rec_rates = np.array([d.recombination_rate for d in srh_result.defect_results])

        # Handle negative rates (take absolute value for log scale)
        rec_rates_abs = np.abs(rec_rates)

        # Replace inf/nan with very small positive value for plotting
        rec_rates_abs = np.where(np.isfinite(rec_rates_abs), rec_rates_abs, 1e-100)

        # Replace zeros with small positive value for log scale
        rec_rates_abs = np.where(rec_rates_abs > 0, rec_rates_abs, 1e-100)

        # Check if values are too extreme
        use_log = np.any(rec_rates_abs > 1e-80)

        # Create bar plot
        bars = ax.bar(defect_names, rec_rates_abs, alpha=0.8, color="steelblue")

        # Highlight dominant defect
        for i, bar in enumerate(bars):
            if defect_names[i] == srh_result.dominant_defect:
                bar.set_color("darkred")
                bar.set_alpha(1.0)

        ax.set_xlabel("Defect", fontsize=12)
        ax.set_ylabel("Recombination Rate (cm⁻³ s⁻¹)", fontsize=12)
        ax.set_title(
            f"SRH Recombination Rates (T = {srh_result.temperature:.0f} K)", fontsize=14
        )
        ax.tick_params(axis="x", rotation=45)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Only use log scale if we have reasonable values
        if use_log:
            ax.set_yscale("log")

        ax.grid(True, alpha=0.3, axis="y")

        # Add text for dominant defect
        ax.text(
            0.02,
            0.98,
            f"Dominant: {srh_result.dominant_defect}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved SRH recombination rate plot to {filename}")
    except (ValueError, OverflowError) as e:
        logger.warning(f"Could not create SRH recombination rate plot: {e}")
        # Create a simple placeholder plot
        _fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "Plot unavailable\n(values too extreme for visualization)",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_title(
            f"SRH Recombination Rates (T = {srh_result.temperature:.0f} K)", fontsize=14
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved placeholder SRH recombination rate plot to {filename}")


def write_srh_summary(
    srh_result: SRHAnalysisResult,
    filename: str | Path = "srh_summary.txt",
    fermi_level_converged: bool = True,
    concentration_result: ConcentrationResult | None = None,
    effective_mass_electron: float = 1.0,
    effective_mass_hole: float = 1.0,
) -> None:
    """
    Write SRH analysis summary to text file.

    Parameters
    ----------
    srh_result : SRHAnalysisResult
        SRH analysis results
    filename : str or Path
        Output filename
    fermi_level_converged : bool
        Whether Fermi level solver converged (from ConcentrationResult)
    concentration_result : ConcentrationResult, optional
        Concentration results for additional carrier information
    effective_mass_electron : float
        Effective electron mass ratio
    effective_mass_hole : float
        Effective hole mass ratio
    """
    from datetime import datetime

    with open(filename, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 20 + "SRH RECOMBINATION ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # Add warning if using fallback data
        if not fermi_level_converged:
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
        f.write(f"Temperature:                    {srh_result.temperature:>10.2f} K\n")
        K_B = 8.617333e-5  # eV/K  # noqa: N806
        thermal_energy = K_B * srh_result.temperature
        f.write(f"Thermal energy (kT):            {thermal_energy:>10.6f} eV\n")
        f.write(f"Band gap:                       {srh_result.bandgap:>10.3f} eV\n")
        f.write(
            f"Effective mass (electron):      {effective_mass_electron:>10.3f} m₀\n"
        )
        f.write(f"Effective mass (hole):          {effective_mass_hole:>10.3f} m₀\n")
        f.write(
            f"Number of defect states:        {len(srh_result.defect_results):>10d}\n"
        )
        f.write("\n")

        # System properties
        f.write("SYSTEM PROPERTIES\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"Intrinsic carrier conc. (n_i):  {srh_result.intrinsic_carrier_concentration:>10.3e} cm⁻³\n"
        )

        if concentration_result is not None:
            f.write(
                f"Electron concentration (n):     {concentration_result.electron_concentration:>10.3e} cm⁻³\n"
            )
            f.write(
                f"Hole concentration (p):         {concentration_result.hole_concentration:>10.3e} cm⁻³\n"
            )
            f.write(
                f"Fermi level (E_F):              {concentration_result.fermi_level:>10.3f} eV (from VBM)\n"
            )
            fermi_frac = concentration_result.fermi_level / srh_result.bandgap * 100
            f.write(
                f"Fermi level position:           {fermi_frac:>10.1f} % through gap\n"
            )
        f.write("\n")

        # Recombination summary
        f.write("RECOMBINATION SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"Total recombination rate:       {srh_result.total_recombination_rate:>10.3e} cm⁻³ s⁻¹\n"
        )
        f.write(f"Dominant defect:                {srh_result.dominant_defect}\n")

        # Find min/max lifetimes
        all_tau_n = [d.lifetimes.tau_n for d in srh_result.defect_results]
        all_tau_p = [d.lifetimes.tau_p for d in srh_result.defect_results]
        if all_tau_n and all_tau_p:
            min_tau = min(min(all_tau_n), min(all_tau_p))
            max_tau = max(max(all_tau_n), max(all_tau_p))
            f.write(
                f"Lifetime range:                 {min_tau:>10.3e} - {max_tau:>10.3e} s\n"
            )
        f.write("\n")

        f.write("DEFECT-BY-DEFECT ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        for dr in srh_result.defect_results:
            f.write(f"Defect: {dr.defect_name}\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"  Defect level:              {dr.defect_level:>10.3f} eV (from VBM)\n"
            )
            f.write(
                f"  Defect concentration:      {dr.lifetimes.defect_concentration:>10.3e} cm⁻³\n"
            )
            f.write("\n")
            f.write("  Capture cross-sections:\n")
            f.write(
                f"    σ_n (electron):          {dr.lifetimes.capture_params.sigma_n:>10.3e} cm²\n"  # noqa: RUF001
            )
            f.write(
                f"    σ_p (hole):              {dr.lifetimes.capture_params.sigma_p:>10.3e} cm²\n"  # noqa: RUF001
            )
            f.write(
                f"    Method:                  {dr.lifetimes.capture_params.method}\n"
            )
            f.write("\n")
            f.write("  Carrier lifetimes:\n")

            # Only show ns conversion for reasonable timescales (1 ps to 1 s)
            # Beyond this range, nanosecond display is not meaningful
            tau_n_s = dr.lifetimes.tau_n
            tau_p_s = dr.lifetimes.tau_p
            tau_eff_s = dr.lifetimes.tau_eff
            tau_min_s = dr.lifetimes.minority_carrier_lifetime

            # Show ns if in range 1e-12 to 1.0 seconds (1 ps to 1 s)
            if 1e-12 <= tau_n_s <= 1.0:
                f.write(
                    f"    τ_n (electron):          {tau_n_s:>10.3e} s  ({tau_n_s * 1e9:>10.3f} ns)\n"
                )
            else:
                f.write(f"    τ_n (electron):          {tau_n_s:>10.3e} s\n")

            if 1e-12 <= tau_p_s <= 1.0:
                f.write(
                    f"    τ_p (hole):              {tau_p_s:>10.3e} s  ({tau_p_s * 1e9:>10.3f} ns)\n"
                )
            else:
                f.write(f"    τ_p (hole):              {tau_p_s:>10.3e} s\n")

            if 1e-12 <= tau_eff_s <= 1.0:
                f.write(
                    f"    τ_eff (effective):       {tau_eff_s:>10.3e} s  ({tau_eff_s * 1e9:>10.3f} ns)\n"
                )
            else:
                f.write(f"    τ_eff (effective):       {tau_eff_s:>10.3e} s\n")

            if 1e-12 <= tau_min_s <= 1.0:
                f.write(
                    f"    τ_min (minority):        {tau_min_s:>10.3e} s  ({tau_min_s * 1e9:>10.3f} ns)\n"
                )
            else:
                f.write(f"    τ_min (minority):        {tau_min_s:>10.3e} s\n")
            f.write("\n")
            f.write("  Recombination:\n")
            f.write(
                f"    Recombination rate:      {dr.recombination_rate:>10.3e} cm⁻³ s⁻¹\n"
            )
            f.write(
                f"    Generation rate:         {dr.generation_rate:>10.3e} cm⁻³ s⁻¹\n"
            )
            f.write("    Carrier concentrations:\n")
            f.write(
                f"      n (electron):          {dr.electron_concentration:>10.3e} cm⁻³\n"
            )
            f.write(
                f"      p (hole):              {dr.hole_concentration:>10.3e} cm⁻³\n"
            )
            f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("Generated by atomate2siesta defects workflow\n")
        f.write("=" * 80 + "\n")

    logger.info(f"Wrote SRH summary to {filename}")


# Job functions for integration with jobflow


@job
def calculate_srh_analysis_job(
    formation_diagram: FormationEnergyDiagram,
    concentration_result: ConcentrationResult,
    ctls: list[ChargeTransitionLevel],
    bandgap: float,
    effective_mass_electron: float = 1.0,
    effective_mass_hole: float = 1.0,
    capture_params: dict[str, CaptureParameters] | None = None,
) -> SRHAnalysisResult:
    """
    Calculate SRH recombination analysis (job function).

    Parameters
    ----------
    formation_diagram : FormationEnergyDiagram
        Formation energy diagram
    concentration_result : ConcentrationResult
        Defect concentration results
    ctls : list[ChargeTransitionLevel]
        Charge transition levels
    bandgap : float
        Band gap in eV
    effective_mass_electron : float
        Effective electron mass ratio
    effective_mass_hole : float
        Effective hole mass ratio
    capture_params : dict[str, CaptureParameters], optional
        Capture parameters for each defect

    Returns
    -------
    SRHAnalysisResult
        SRH analysis results
    """
    analyzer = SRHAnalyzer(
        formation_diagram=formation_diagram,
        bandgap=bandgap,
        effective_mass_electron=effective_mass_electron,
        effective_mass_hole=effective_mass_hole,
    )

    result = analyzer.analyze_from_concentration_result(
        concentration_result=concentration_result,
        ctls=ctls,
        capture_params=capture_params,
    )

    logger.info(
        f"SRH analysis complete. Total rate: {result.total_recombination_rate:.3e} cm⁻³ s⁻¹"
    )
    logger.info(f"Dominant defect: {result.dominant_defect}")

    return result


@job
def write_srh_summary_job(
    srh_result: SRHAnalysisResult,
    concentration_result: ConcentrationResult,
    effective_mass_electron: float = 1.0,
    effective_mass_hole: float = 1.0,
    directory: str | Path = ".",
) -> dict[str, str]:
    """
    Write SRH analysis summary and plots (job function).

    Parameters
    ----------
    srh_result : SRHAnalysisResult
        SRH analysis results
    concentration_result : ConcentrationResult
        Concentration results (to check convergence status)
    effective_mass_electron : float
        Effective electron mass ratio
    effective_mass_hole : float
        Effective hole mass ratio
    directory : str or Path
        Output directory

    Returns
    -------
    dict[str, str]
        Dictionary with output filenames
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Write text summary
    summary_file = directory / "srh_summary.txt"
    write_srh_summary(
        srh_result,
        summary_file,
        fermi_level_converged=concentration_result.fermi_level_converged,
        concentration_result=concentration_result,
        effective_mass_electron=effective_mass_electron,
        effective_mass_hole=effective_mass_hole,
    )

    # Plot lifetimes
    lifetime_plot = directory / "srh_lifetimes.png"
    plot_srh_lifetimes(srh_result, lifetime_plot)

    # Plot recombination rates
    rate_plot = directory / "srh_recombination_rates.png"
    plot_srh_recombination_rates(srh_result, rate_plot)

    return {
        "summary": str(summary_file),
        "lifetime_plot": str(lifetime_plot),
        "rate_plot": str(rate_plot),
    }
