"""Oxygen Evolution Reaction (OER) workflows for electrocatalysis.

This module provides automated workflows for calculating OER activity on
catalyst surfaces.
The OER is the anode reaction in water electrolysis and metal-air battery charging:

    2H₂O → O₂ + 4H⁺ + 4e⁻  (acidic media, E° = 1.23 V vs. SHE)
    4OH⁻ → O₂ + 2H₂O + 4e⁻  (alkaline media, E° = 0.40 V vs. SHE)

The workflow calculates:
1. Adsorption energies of OER intermediates (OH*, O*, OOH*, O₂*)
2. Free energy diagram using the Computational Hydrogen Electrode (CHE) model
3. Overpotential and rate-limiting step identification
4. Bifunctional activity (combined ORR/OER performance)

References
----------
- Man et al., ChemCatChem 3, 1159 (2011): OER scaling relations
- Montoya et al., Nat. Mater. 16, 70 (2017): OER volcano plot
- Suntivich et al., Science 334, 1383 (2011): Descriptor-based analysis
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
def _analyze_oer_pathway(
    clean_surface_energy: float,
    o2_gas_energy: float,
    h2o_gas_energy: float,
    h2_gas_energy: float,
    oh_ads_doc: AdsorptionScanDocument,
    o_ads_doc: AdsorptionScanDocument,
    ooh_ads_doc: AdsorptionScanDocument,
    o2_ads_doc: AdsorptionScanDocument,
    temperature: float = 298.15,
    pressure: float = 101325.0,
    ph: float = 0.0,
    potential: float = 1.23,
    plot_results: bool = True,
    write_summary: bool = True,
    surface_name: str = "OER_catalyst",
) -> dict:
    """
    Analyze OER pathway and calculate overpotential.

    OER is the reverse of ORR, so the pathway goes:
        H₂O → OH* → O* → OOH* → O₂

    Parameters
    ----------
    clean_surface_energy : float
        Clean surface DFT energy (eV).
    o2_gas_energy : float
        O₂ gas-phase energy (eV).
    h2o_gas_energy : float
        H₂O gas-phase energy (eV).
    h2_gas_energy : float
        H₂ gas-phase energy (eV).
    oh_ads_doc : dict
        OH adsorption scan results.
    o_ads_doc : dict
        O adsorption scan results.
    ooh_ads_doc : dict
        OOH adsorption scan results.
    o2_ads_doc : dict
        O₂ adsorption scan results.
    temperature : float
        Temperature (K).
    pressure : float
        Pressure (Pa).
    ph : float
        pH of electrolyte.
    potential : float
        Electrode potential (V vs. SHE, default: 1.23 V equilibrium).

    Returns
    -------
    dict
        OER analysis results.
    """
    from atomate2.siesta.flows.electrocatalysis.analysis import (
        calculate_oer_overpotential,
        calculate_reaction_free_energies,
        identify_rate_limiting_step,
    )
    from atomate2.siesta.schemas.electrocatalysis import ReactionPathwayDocument

    # Extract best site energies
    oh_best_energy = oh_ads_doc.best_site.total_energy
    o_best_energy = o_ads_doc.best_site.total_energy
    ooh_best_energy = ooh_ads_doc.best_site.total_energy
    o2_best_energy = o2_ads_doc.best_site.total_energy

    # OER pathway steps (4-electron oxidation mechanism)
    # Step 1: * + H₂O → OH* + (H⁺ + e⁻)
    # Step 2: OH* → O* + (H⁺ + e⁻)
    # Step 3: O* + H₂O → OOH* + (H⁺ + e⁻)
    # Step 4: OOH* → O₂* + (H⁺ + e⁻)
    # Step 5: O₂* → O₂(g)

    pathway_steps = [
        {
            "label": "H2O + *",
            "energy": clean_surface_energy,  # Starting point (clean surface + H₂O)
            "species": "H2O",
            "n_H": 0,
            "n_e": 0,
        },
        {
            "label": "OH*",
            "energy": oh_best_energy,
            "species": "H",  # Removed H⁺ + e⁻
            "n_H": -1,  # Dehydrogenation
            "n_e": 1,  # Electron loss (oxidation)
        },
        {
            "label": "O*",
            "energy": o_best_energy,
            "species": "H",  # Another H⁺ + e⁻ removed
            "n_H": -1,
            "n_e": 1,
        },
        {
            "label": "OOH*",
            "energy": ooh_best_energy,
            "species": "H2O",  # H₂O added, then H⁺ + e⁻ removed
            "n_H": -1,
            "n_e": 1,
        },
        {
            "label": "O2*",
            "energy": o2_best_energy,
            "species": "H",  # Final H⁺ + e⁻ removed
            "n_H": -1,
            "n_e": 1,
        },
    ]

    # Calculate free energies
    gas_phase_energies = {
        "H2": h2_gas_energy,
        "H2O": h2o_gas_energy,
        "O2": o2_gas_energy,
    }

    thermo_result = calculate_reaction_free_energies(
        surface_name="OER_pathway",
        pathway_steps=pathway_steps,
        gas_phase_energies=gas_phase_energies,
        clean_surface_energy=clean_surface_energy,
        temperature=temperature,
        pressure=pressure,
        ph=ph,
        potential=potential,
    )

    # Calculate overpotential
    overpotential_result = calculate_oer_overpotential(thermo_result["delta_G"])

    # Identify rate-limiting step
    rls = identify_rate_limiting_step(
        delta_G=thermo_result["delta_G"],
        step_labels=thermo_result["step_labels"],
    )

    # Create pathway document
    pathway_doc = ReactionPathwayDocument(
        surface_name="OER_catalyst",
        pathway_type="oer",
        steps=[
            {
                "label": step["label"],
                "species": step.get("species"),
                "site_coords": None,
                "height": None,
                "energy": step["energy"],
                "structure": None,
            }
            for step in pathway_steps
        ],
        energies=thermo_result["absolute_energies"],
        delta_E=thermo_result["delta_E"],
        delta_G=thermo_result["delta_G"],
        overpotential_orr=0.0,  # Not calculated for OER-only workflow
        overpotential_oer=overpotential_result["eta_OER"],
        overpotential_gap=overpotential_result["eta_OER"],
        rate_limiting_step=rls["rls_label"],
        temperature=temperature,
        pressure=pressure,
    )

    logger.info(
        f"OER analysis complete: η = {overpotential_result['eta_OER']:.3f} V, "
        f"RLS = {rls['rls_label']}"
    )

    # Generate plots and summary files
    output_files = {}
    if plot_results:
        from atomate2.siesta.flows.electrocatalysis.analysis.plotting import (
            plot_free_energy_diagram,
            plot_overpotential_summary,
        )

        # Free energy diagram
        plot_file = plot_free_energy_diagram(
            step_labels=thermo_result["step_labels"],
            cumulative_G=thermo_result["cumulative_G"],
            delta_G=thermo_result["delta_G"],
            pathway_type="OER",
            filename="oer_free_energy_diagram.png",
        )
        output_files["free_energy_diagram"] = str(plot_file)
        logger.info(f"✓ Generated free energy diagram: {plot_file}")

        # Overpotential summary plot
        summary_plot = plot_overpotential_summary(
            pathway_type="OER",
            overpotential=overpotential_result["eta_OER"],
            rls_label=rls["rls_label"],
            rls_delta_G=rls["rls_delta_G"],
            U_onset=overpotential_result["U_onset"],
            filename="oer_overpotential_summary.png",
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
            and abs(o2_gas_energy) < 1e-6
            and abs(h2o_gas_energy) < 1e-6
            and abs(h2_gas_energy) < 1e-6
            and abs(oh_ads_doc.best_adsorption_energy) < 1e-6
            and abs(o_ads_doc.best_adsorption_energy) < 1e-6
            and abs(ooh_ads_doc.best_adsorption_energy) < 1e-6
            and abs(o2_ads_doc.best_adsorption_energy) < 1e-6
        )

        summary_file = write_analysis_summary(
            pathway_type="OER",
            surface_name=surface_name,
            overpotential=overpotential_result["eta_OER"],
            rls_label=rls["rls_label"],
            rls_delta_G=rls["rls_delta_G"],
            step_labels=thermo_result["step_labels"],
            delta_G=thermo_result["delta_G"],
            filename="oer_analysis_summary.txt",
            dry_run=is_dry_run,
        )
        output_files["analysis_summary"] = str(summary_file)
        logger.info(f"✓ Generated analysis summary: {summary_file}")

    return {
        "pathway_document": pathway_doc.dict(),
        "overpotential": overpotential_result,
        "thermodynamics": thermo_result,
        "rate_limiting_step": rls,
        "output_files": output_files,
    }


@dataclass
class OERFlowMaker(BaseSiestaFlowMaker):
    """
    Complete OER workflow for electrocatalyst screening.

    This workflow automates the calculation of OER activity on catalyst surfaces:

    1. **Gas-phase references**: H₂O, O₂, H₂ (using GasPhaseMoleculeMaker)
    2. **Clean surface**: Static calculation
    3. **Adsorption scans**: OH*, O*, OOH*, O₂* (using AdsorptionScanFlowMaker)
    4. **Thermodynamic analysis**: Free energy diagram, overpotential, RLS

    The OER is the reverse of the ORR:
        2H₂O → O₂ + 4H⁺ + 4e⁻  (E° = 1.23 V vs. SHE)

    Parameters
    ----------
    name : str
        Workflow name (default: 'oer_workflow').
    gas_phase_maker : GasPhaseMoleculeMaker
        Maker for gas-phase molecular calculations.
    surface_static_maker : StaticMaker
        Maker for clean surface calculation.
    adsorption_maker : AdsorptionScanFlowMaker
        Maker for adsorption site scanning.
    grid_size : tuple[int, int]
        Grid size for adsorption scanning (default: (4, 4)).
    height : float
        Adsorbate height above surface (Å, default: 2.0).
    temperature : float
        Temperature (K, default: 298.15).
    pressure : float
        Pressure (Pa, default: 101325 = 1 atm).
    ph : float
        pH of electrolyte (default: 0.0 for acidic).
    potential : float
        Electrode potential (V vs. SHE, default: 1.23 V equilibrium).

    Examples
    --------
    Basic OER workflow:

    >>> from pymatgen.core import Structure
    >>> from atomate2.siesta.flows.electrocatalysis import OERFlowMaker
    >>> from atomate2.siesta.jobs.core import StaticMaker
    >>>
    >>> # Load catalyst surface
    >>> surface = Structure.from_file("IrO2_110.cif")
    >>>
    >>> # Create OER workflow
    >>> maker = OERFlowMaker(
    ...     surface_static_maker=StaticMaker(
    ...         user_params={"PAO.BasisSize": "DZP", "kpts": [6, 6, 1]}
    ...     ),
    ...     ph=14.0,  # Alkaline conditions
    ... )
    >>> flow = maker.make(surface)

    Bifunctional catalyst screening (ORR + OER):

    >>> from atomate2.siesta.flows.electrocatalysis import ORRFlowMaker, OERFlowMaker
    >>>
    >>> # Run both ORR and OER workflows
    >>> orr_maker = ORRFlowMaker()
    >>> oer_maker = OERFlowMaker()
    >>>
    >>> orr_flow = orr_maker.make(surface)
    >>> oer_flow = oer_maker.make(surface)
    >>>
    >>> # Combine flows
    >>> from jobflow import Flow
    >>> bifunctional_flow = Flow([orr_flow, oer_flow])

    Notes
    -----
    **OER pathway** (4-electron oxidation):
        2H₂O → O₂ + 4H⁺ + 4e⁻

    Intermediate steps:
        1. * + H₂O → OH* + H⁺ + e⁻
        2. OH* → O* + H⁺ + e⁻
        3. O* + H₂O → OOH* + H⁺ + e⁻
        4. OOH* → O₂* + H⁺ + e⁻
        5. O₂* → O₂(g)

    **CHE Model**:
        - μ(H⁺ + e⁻) = ½μ(H₂) - eU
        - At U = 1.23 V, reaction is thermoneutral (ΔG = 0)
        - Overpotential: η = U_onset - 1.23 V

    **Output**:
        - Overpotential (η_OER in Volts)
        - Free energy diagram (ΔG for each step)
        - Rate-limiting step (RLS)
        - Best adsorption sites

    **Scaling Relations** (Man et al., ChemCatChem 2011):
        - ΔG_OOH* ≈ ΔG_OH* + 3.2 eV (universal scaling)
        - Ideal catalyst: ΔG_O* - ΔG_OH* = 1.6 eV

    See Also
    --------
    ORRFlowMaker : Oxygen Reduction Reaction workflow
    GasPhaseMoleculeMaker : Gas-phase reference calculations
    AdsorptionScanFlowMaker : Adsorption site scanning
    """

    name: str = "oer_workflow"
    gas_phase_maker: GasPhaseMoleculeMaker = field(
        default_factory=GasPhaseMoleculeMaker
    )
    surface_static_maker: StaticMaker = field(default_factory=StaticMaker)
    adsorption_maker: AdsorptionScanFlowMaker = field(
        default_factory=AdsorptionScanFlowMaker
    )
    grid_size: tuple[int, int] = (4, 4)
    height: float = 2.0  # Å
    temperature: float = 298.15  # K
    pressure: float = 101325.0  # Pa
    ph: float = 0.0  # Acidic conditions
    potential: float = 1.23  # V vs. SHE (equilibrium potential)
    plot_results: bool = True  # Generate plots
    write_summary: bool = True  # Write text summary

    # Note: dry_run, use_custodian, custodian_handlers, custodian_max_errors,
    # and tier support inherited from BaseSiestaFlowMaker

    def make(self, surface: Structure) -> Flow:
        """
        Create OER workflow for a given catalyst surface.

        Parameters
        ----------
        surface : Structure
            Catalyst surface slab structure.

        Returns
        -------
        Flow
            Jobflow Flow with complete OER workflow.
        """
        jobs = []

        # Step 1: Gas-phase references
        logger.info("Setting up gas-phase references (H₂O, O₂, H₂)")

        h2o_molecule = Molecule(
            ["O", "H", "H"],
            [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        )
        o2_molecule = Molecule(["O", "O"], [[0, 0, 0], [0, 0, 1.21]])
        h2_molecule = Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])

        h2o_job = self.gas_phase_maker.make(h2o_molecule)
        h2o_job.name = f"{self.name}_H2O_gas"
        jobs.append(h2o_job)

        o2_job = self.gas_phase_maker.make(o2_molecule)
        o2_job.name = f"{self.name}_O2_gas"
        jobs.append(o2_job)

        h2_job = self.gas_phase_maker.make(h2_molecule)
        h2_job.name = f"{self.name}_H2_gas"
        jobs.append(h2_job)

        # Step 2: Clean surface
        logger.info("Setting up clean surface calculation")
        clean_surface_job = self.surface_static_maker.make(surface)
        clean_surface_job.name = f"{self.name}_clean_surface"
        jobs.append(clean_surface_job)

        # Step 3: Adsorption scans for OER intermediates
        logger.info("Setting up adsorption scans for OER intermediates")

        ads_maker = AdsorptionScanFlowMaker(
            grid_size=self.grid_size,
            height=self.height,
            slab_static_maker=self.adsorption_maker.slab_static_maker,
            adsorbate_static_maker=self.adsorption_maker.adsorbate_static_maker,
        )

        # OH adsorption
        oh_molecule = Molecule(["O", "H"], [[0, 0, 0], [0.96, 0, 0]])
        oh_ads_job = ads_maker.make(surface, oh_molecule)
        oh_ads_job.name = f"{self.name}_OH_adsorption"
        jobs.append(oh_ads_job)

        # O adsorption
        o_molecule = Molecule(["O"], [[0, 0, 0]])
        o_ads_job = ads_maker.make(surface, o_molecule)
        o_ads_job.name = f"{self.name}_O_adsorption"
        jobs.append(o_ads_job)

        # OOH adsorption
        ooh_molecule = Molecule(
            ["O", "O", "H"],
            [[0, 0, 0], [1.33, 0, 0], [1.70, 0.90, 0]],
        )
        ooh_ads_job = ads_maker.make(surface, ooh_molecule)
        ooh_ads_job.name = f"{self.name}_OOH_adsorption"
        jobs.append(ooh_ads_job)

        # O₂ adsorption
        o2_ads_job = ads_maker.make(surface, o2_molecule)
        o2_ads_job.name = f"{self.name}_O2_adsorption"
        jobs.append(o2_ads_job)

        # Step 4: Analyze OER pathway
        logger.info("Setting up OER pathway analysis")
        analysis_job = _analyze_oer_pathway(
            clean_surface_energy=clean_surface_job.output.output.energy,
            o2_gas_energy=o2_job.output.total_energy,
            h2o_gas_energy=h2o_job.output.total_energy,
            h2_gas_energy=h2_job.output.total_energy,
            oh_ads_doc=oh_ads_job.output,
            o_ads_doc=o_ads_job.output,
            ooh_ads_doc=ooh_ads_job.output,
            o2_ads_doc=o2_ads_job.output,
            temperature=self.temperature,
            pressure=self.pressure,
            ph=self.ph,
            potential=self.potential,
            plot_results=self.plot_results,
            write_summary=self.write_summary,
            surface_name=f"{surface.composition.reduced_formula}_surface",
        )
        analysis_job.name = f"{self.name}_analysis"
        jobs.append(analysis_job)

        logger.info(f"OER workflow created with {len(jobs)} jobs")

        return Flow(jobs, output=analysis_job.output, name=self.name)
