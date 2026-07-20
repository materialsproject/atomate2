"""Hydrogen Evolution Reaction (HER) workflows for electrocatalysis.

This module provides automated workflows for calculating HER activity on
catalyst surfaces.
The HER is the cathode reaction in water electrolysis:

    2H⁺ + 2e⁻ → H₂  (acidic media, E° = 0.00 V vs. SHE)
    2H₂O + 2e⁻ → H₂ + 2OH⁻  (alkaline media, E° = -0.83 V vs. SHE)

HER follows the Sabatier principle: optimal catalysts have thermoneutral H binding
(ΔG_H* ≈ 0 eV), creating a volcano plot with Pt at the peak.

The workflow calculates:
1. Hydrogen adsorption energy (ΔG_H*)
2. HER overpotential based on |ΔG_H*|
3. Position on HER volcano plot

References
----------
- Nørskov et al., J. Electrochem. Soc. 152, J23 (2005): HER volcano plot
- Greeley et al., Nat. Mater. 5, 909 (2006): Computational screening
- Zheng et al., Science 338, 1321 (2012): MoS₂ edge sites
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, job
from pymatgen.core import Molecule

from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.flows.molecular.gas_phase import GasPhaseMoleculeMaker
from atomate2.siesta.flows.surface.adsorption import AdsorptionScanFlowMaker
from atomate2.siesta.jobs.core import StaticMaker

if TYPE_CHECKING:
    from pymatgen.core import Structure

    from atomate2.siesta.schemas.adsorption import AdsorptionScanDocument

logger = logging.getLogger(__name__)


@job
def _analyze_her_pathway(
    clean_surface_energy: float,
    h2_gas_energy: float,
    h_ads_doc: AdsorptionScanDocument,
    temperature: float = 298.15,
    pressure: float = 101325.0,
    ph: float = 0.0,
    plot_results: bool = True,
    write_summary: bool = True,
    surface_name: str = "HER_catalyst",
) -> dict:
    """
    Analyze HER pathway and calculate overpotential from H binding energy.

    HER is simpler than ORR/OER: only one intermediate (H*).
    The Sabatier principle states optimal HER catalysts have ΔG_H* ≈ 0 eV.

    Parameters
    ----------
    clean_surface_energy : float
        Clean surface DFT energy (eV).
    h2_gas_energy : float
        H₂ gas-phase energy (eV).
    h_ads_doc : dict
        H adsorption scan results.
    temperature : float
        Temperature (K).
    pressure : float
        Pressure (Pa).
    ph : float
        pH of electrolyte.

    Returns
    -------
    dict
        HER analysis results.
    """
    from atomate2.siesta.flows.electrocatalysis.analysis import (
        calculate_free_energy_corrections,
        calculate_her_overpotential,
    )

    # Extract best H adsorption site energy
    h_best_energy = h_ads_doc.best_site.total_energy

    # Calculate ΔG_H* = G(H*) - G(*) - 0.5 × G(H₂)  # noqa: RUF003
    # G(H*) = E(surface + H) + corrections
    # G(*) = E(clean surface)
    # G(H₂) = E(H₂ gas) + corrections

    # Get thermodynamic corrections
    corrections = calculate_free_energy_corrections(
        temperature=temperature, pressure=pressure
    )

    # ΔG_H* calculation
    delta_G_H = (  # noqa: N806
        h_best_energy  # E(H*)
        - clean_surface_energy  # E(*)
        - 0.5 * h2_gas_energy  # -0.5 × E(H₂)  # noqa: RUF003
        - 0.5 * corrections.get("H2", 0.0)  # -0.5 × correction(H₂)  # noqa: RUF003
        + corrections.get("H*", 0.0)  # correction(H*)
    )

    # pH correction (affects all proton transfers)
    # ΔG_pH = -k_B T ln(10) × pH  # noqa: RUF003
    KB = 8.617333262e-5  # eV/K  # noqa: N806
    ph_correction = -KB * temperature * 2.303 * ph  # ln(10) ≈ 2.303
    delta_G_H += ph_correction  # noqa: N806

    # Calculate HER overpotential
    her_result = calculate_her_overpotential(delta_G_H)

    # Create analysis document
    analysis = {
        "delta_G_H": delta_G_H,
        "eta_HER": her_result["eta_HER"],
        "U_equilibrium": her_result["U_equilibrium"],
        "h_best_site": h_ads_doc["best_site"],
        "temperature": temperature,
        "pressure": pressure,
        "ph": ph,
        # Volcano plot analysis
        "binding_category": (
            "optimal"
            if abs(delta_G_H) < 0.1
            else "weak"
            if delta_G_H > 0.1
            else "strong"
        ),
        "performance_rating": (
            "excellent"
            if her_result["eta_HER"] < 0.1
            else "good"
            if her_result["eta_HER"] < 0.2
            else "moderate"
            if her_result["eta_HER"] < 0.4
            else "poor"
        ),
    }

    logger.info(
        f"HER analysis complete: ΔG_H* = {delta_G_H:+.3f} eV, "
        f"η = {her_result['eta_HER']:.3f} V"
    )

    # Generate plots and summary files
    output_files = {}
    if plot_results:
        from atomate2.siesta.flows.electrocatalysis.analysis.plotting import (
            plot_overpotential_summary,
        )

        # Overpotential summary plot (HER volcano plot)
        summary_plot = plot_overpotential_summary(
            pathway_type="HER",
            overpotential=her_result["eta_HER"],
            rls_label="H adsorption",
            rls_delta_G=delta_G_H,
            U_onset=her_result["U_equilibrium"],
            filename="her_overpotential_summary.png",
        )
        output_files["overpotential_summary"] = str(summary_plot)
        logger.info(f"✓ Generated overpotential summary: {summary_plot}")

    if write_summary:
        from atomate2.siesta.flows.electrocatalysis.analysis.plotting import (
            write_analysis_summary,
        )

        # Detect dry-run mode: all input energies are zero
        is_dry_run = (
            abs(clean_surface_energy) < 1e-6
            and abs(h2_gas_energy) < 1e-6
            and abs(h_ads_doc.best_adsorption_energy) < 1e-6
        )

        # For HER, we have a single step
        summary_file = write_analysis_summary(
            pathway_type="HER",
            surface_name=surface_name,
            overpotential=her_result["eta_HER"],
            rls_label="H adsorption",
            rls_delta_G=delta_G_H,
            step_labels=["H*"],
            delta_G=[delta_G_H],
            filename="her_analysis_summary.txt",
            dry_run=is_dry_run,
        )
        output_files["analysis_summary"] = str(summary_file)
        logger.info(f"✓ Generated analysis summary: {summary_file}")

    analysis["output_files"] = output_files
    return analysis


@dataclass
class HERFlowMaker(BaseSiestaFlowMaker):
    """
    Complete HER workflow for electrocatalyst screening.

    This workflow automates the calculation of HER activity on catalyst surfaces:

    1. **Gas-phase reference**: H₂ (using GasPhaseMoleculeMaker)
    2. **Clean surface**: Static calculation
    3. **H adsorption scan**: Find optimal H binding site
    4. **HER analysis**: Calculate ΔG_H* and overpotential

    HER follows a simple 1-step mechanism:
        H⁺ + e⁻ → ½H₂  (or 2H⁺ + 2e⁻ → H₂)

    The key descriptor is the H binding free energy (ΔG_H*):
    - ΔG_H* > 0: Weak binding (H adsorption limited, Au-like)
    - ΔG_H* < 0: Strong binding (H₂ desorption limited, W-like)
    - ΔG_H* ≈ 0: Optimal (Pt-like)

    This creates a volcano plot with η_HER = |ΔG_H*|.

    Parameters
    ----------
    name : str
        Workflow name (default: 'her_workflow').
    gas_phase_maker : GasPhaseMoleculeMaker
        Maker for H₂ gas-phase calculation.
    surface_static_maker : StaticMaker
        Maker for clean surface calculation.
    adsorption_maker : AdsorptionScanFlowMaker
        Maker for H adsorption site scanning.
    grid_size : tuple[int, int]
        Grid size for H adsorption scanning (default: (4, 4)).
    height : float
        H atom height above surface (Å, default: 1.5).
    temperature : float
        Temperature (K, default: 298.15).
    pressure : float
        Pressure (Pa, default: 101325 = 1 atm).
    ph : float
        pH of electrolyte (default: 0.0 for acidic).

    Examples
    --------
    Basic HER workflow:

    >>> from pymatgen.core import Structure
    >>> from atomate2.siesta.flows.electrocatalysis import HERFlowMaker
    >>>
    >>> surface = Structure.from_file("MoS2_edge.cif")
    >>> maker = HERFlowMaker()
    >>> flow = maker.make(surface)

    With custom parameters:

    >>> from atomate2.siesta.jobs.core import StaticMaker
    >>> from atomate2.siesta.sets.core import StaticSetGenerator
    >>>
    >>> slab_params = {"PAO.BasisSize": "DZP", "kpts": [6, 6, 1]}
    >>> slab_maker = StaticMaker(
    ...     input_set_generator=StaticSetGenerator(user_params=slab_params)
    ... )
    >>>
    >>> maker = HERFlowMaker(
    ...     surface_static_maker=slab_maker,
    ...     grid_size=(6, 6),  # Finer grid
    ...     ph=14.0,  # Alkaline conditions
    ... )
    >>> flow = maker.make(surface)

    Notes
    -----
    **HER mechanism**:
        Acidic: 2H⁺ + 2e⁻ → H₂
        Alkaline: 2H₂O + 2e⁻ → H₂ + 2OH⁻

    **Sabatier principle**:
        - Optimal catalyst: ΔG_H* = 0 eV (thermoneutral)
        - Too negative: Strong H binding (poisoning)
        - Too positive: Weak H binding (no adsorption)

    **Volcano plot**:
        - Peak: Pt, Pd (ΔG_H* ≈ 0 eV, η ≈ 0 V)
        - Weak binding leg: Au, Ag, Cu (ΔG_H* > 0.2 eV)
        - Strong binding leg: W, Mo, Nb (ΔG_H* < -0.2 eV)

    **Non-metal catalysts**:
        - MoS₂ edges: ΔG_H* ≈ 0.08 eV (excellent!)
        - N-doped graphene: ΔG_H* ≈ 0.2 eV (good)
        - Fe-N-C: ΔG_H* ≈ 0.15 eV (good)

    **Output**:
        - ΔG_H*: H binding free energy (eV)
        - η_HER: Overpotential (V) = |ΔG_H*|
        - Volcano plot position (weak/optimal/strong)
        - Best H adsorption site

    See Also
    --------
    ORRFlowMaker : Oxygen Reduction Reaction workflow
    OERFlowMaker : Oxygen Evolution Reaction workflow

    References
    ----------
    - Nørskov et al., J. Electrochem. Soc. 152, J23 (2005): HER volcano
    - Greeley et al., Nat. Mater. 5, 909 (2006): Computational screening
    - Zheng et al., Science 338, 1321 (2012): MoS₂ as HER catalyst
    """

    name: str = "her_workflow"
    gas_phase_maker: GasPhaseMoleculeMaker = field(
        default_factory=GasPhaseMoleculeMaker
    )
    surface_static_maker: StaticMaker = field(default_factory=StaticMaker)
    adsorption_maker: AdsorptionScanFlowMaker = field(
        default_factory=AdsorptionScanFlowMaker
    )
    grid_size: tuple[int, int] = (4, 4)
    height: float = 1.5  # Å (lower than ORR/OER since H is smaller)
    temperature: float = 298.15  # K
    pressure: float = 101325.0  # Pa
    ph: float = 0.0  # Acidic conditions (typical for HER)
    plot_results: bool = True  # Generate plots
    write_summary: bool = True  # Write text summary

    # Note: dry_run, use_custodian, custodian_handlers, custodian_max_errors,
    # and tier support inherited from BaseSiestaFlowMaker

    def make(self, surface: Structure) -> Flow:
        """
        Create HER workflow for a given catalyst surface.

        Parameters
        ----------
        surface : Structure
            Catalyst surface slab structure.

        Returns
        -------
        Flow
            Jobflow Flow with complete HER workflow.
        """
        jobs = []

        # Step 1: Gas-phase H₂ reference
        logger.info("Setting up H₂ gas-phase reference")

        h2_molecule = Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])
        h2_job = self.gas_phase_maker.make(h2_molecule)
        h2_job.name = f"{self.name}_H2_gas"
        jobs.append(h2_job)

        # Step 2: Clean surface calculation
        logger.info("Setting up clean surface calculation")
        clean_surface_job = self.surface_static_maker.make(surface)
        clean_surface_job.name = f"{self.name}_clean_surface"
        jobs.append(clean_surface_job)

        # Step 3: H adsorption scan
        logger.info("Setting up H adsorption scan")

        h_atom = Molecule(["H"], [[0, 0, 0]])

        ads_maker = AdsorptionScanFlowMaker(
            grid_size=self.grid_size,
            height=self.height,
            slab_static_maker=self.adsorption_maker.slab_static_maker,
            adsorbate_static_maker=self.adsorption_maker.adsorbate_static_maker,
        )

        h_ads_job = ads_maker.make(surface, h_atom)
        h_ads_job.name = f"{self.name}_H_adsorption"
        jobs.append(h_ads_job)

        # Step 4: Analyze HER pathway
        logger.info("Setting up HER analysis")
        analysis_job = _analyze_her_pathway(
            clean_surface_energy=clean_surface_job.output.output.energy,
            h2_gas_energy=h2_job.output.total_energy,
            h_ads_doc=h_ads_job.output,
            temperature=self.temperature,
            pressure=self.pressure,
            ph=self.ph,
            plot_results=self.plot_results,
            write_summary=self.write_summary,
            surface_name=f"{surface.composition.reduced_formula}_surface",
        )
        analysis_job.name = f"{self.name}_analysis"
        jobs.append(analysis_job)

        logger.info(f"HER workflow created with {len(jobs)} jobs")

        return Flow(jobs, output=analysis_job.output, name=self.name)
