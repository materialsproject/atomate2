"""Oxygen Reduction Reaction (ORR) workflows for electrocatalysis.

This module provides automated workflows for calculating ORR activity on catalyst surfaces.
The ORR is the cathode reaction in fuel cells and metal-air batteries:

    O₂ + 4H⁺ + 4e⁻ → 2H₂O  (acidic media, E° = 1.23 V vs. SHE)
    O₂ + 2H₂O + 4e⁻ → 4OH⁻  (alkaline media, E° = 0.40 V vs. SHE)

The workflow calculates:
1. Adsorption energies of ORR intermediates (O₂*, OOH*, O*, OH*)
2. Free energy diagram using the Computational Hydrogen Electrode (CHE) model
3. Overpotential and rate-limiting step identification
4. Thermodynamic volcano plots (optional)

References
----------
- Nørskov et al., J. Phys. Chem. B 108, 17886 (2004): CHE model
- Viswanathan et al., ACS Catal. 2, 1654 (2012): ORR volcano plot
- Kulkarni et al., Chem. Rev. 118, 2302 (2018): ORR mechanisms review
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
def _analyze_orr_pathway(
    clean_surface_energy: float,
    o2_gas_energy: float,
    h2o_gas_energy: float,
    h2_gas_energy: float,
    o2_ads_doc: AdsorptionScanDocument,
    ooh_ads_doc: AdsorptionScanDocument,
    o_ads_doc: AdsorptionScanDocument,
    oh_ads_doc: AdsorptionScanDocument,
    temperature: float = 298.15,
    pressure: float = 101325.0,
    ph: float = 0.0,
    potential: float = 0.0,
    plot_results: bool = True,
    write_summary: bool = True,
    surface_name: str = "ORR_catalyst",
):
    """
    Analyze ORR pathway and calculate overpotential.

    This is a module-level function for jobflow-remote compatibility.

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
    o2_ads_doc : dict
        O₂ adsorption scan results.
    ooh_ads_doc : dict
        OOH adsorption scan results.
    o_ads_doc : dict
        O adsorption scan results.
    oh_ads_doc : dict
        OH adsorption scan results.
    temperature : float
        Temperature (K).
    pressure : float
        Pressure (Pa).
    ph : float
        pH of electrolyte.
    potential : float
        Electrode potential (V vs. SHE).

    Returns
    -------
    dict
        ORR analysis results including overpotential, free energy diagram,
        and rate-limiting step.
    """
    from atomate2.siesta.flows.electrocatalysis.analysis import (
        calculate_orr_overpotential,
        calculate_reaction_free_energies,
        identify_rate_limiting_step,
    )
    from atomate2.siesta.schemas.electrocatalysis import ReactionPathwayDocument

    # Extract best site energies from adsorption documents
    o2_best_energy = o2_ads_doc.best_site.total_energy
    ooh_best_energy = ooh_ads_doc.best_site.total_energy
    o_best_energy = o_ads_doc.best_site.total_energy
    oh_best_energy = oh_ads_doc.best_site.total_energy

    # ORR pathway steps (4-electron mechanism)
    # Step 1: O₂(g) → O₂*
    # Step 2: O₂* + (H⁺ + e⁻) → OOH*
    # Step 3: OOH* + (H⁺ + e⁻) → O* + H₂O
    # Step 4: O* + (H⁺ + e⁻) → OH*
    # Step 5: OH* + (H⁺ + e⁻) → H₂O + *

    pathway_steps = [
        {
            "label": "O2_ads",
            "energy": o2_best_energy,
            "species": "O2",
            "n_H": 0,
            "n_e": 0,
        },
        {
            "label": "OOH*",
            "energy": ooh_best_energy,
            "species": "H",  # Added H from H⁺ + e⁻
            "n_H": 1,
            "n_e": 1,
        },
        {
            "label": "O* + H2O",
            "energy": o_best_energy,  # O* energy (H₂O is in gas phase)
            "species": "H2O",  # Product H₂O
            "n_H": 1,
            "n_e": 1,
        },
        {
            "label": "OH*",
            "energy": oh_best_energy,
            "species": "H",  # Added H from H⁺ + e⁻
            "n_H": 1,
            "n_e": 1,
        },
        {
            "label": "H2O",
            "energy": clean_surface_energy,  # Clean surface (H₂O desorbed)
            "species": "H2O",  # Final product
            "n_H": 1,
            "n_e": 1,
        },
    ]

    # Calculate free energies using CHE model (ALL DFT VALUES)
    gas_phase_energies = {
        "H2": h2_gas_energy,
        "H2O": h2o_gas_energy,  # Using DFT absolute energy
        "O2": o2_gas_energy,
    }

    thermo_result = calculate_reaction_free_energies(
        surface_name="ORR_pathway",
        pathway_steps=pathway_steps,
        gas_phase_energies=gas_phase_energies,
        clean_surface_energy=clean_surface_energy,
        temperature=temperature,
        pressure=pressure,
        ph=ph,
        potential=potential,
    )

    # Calculate overpotential
    overpotential_result = calculate_orr_overpotential(thermo_result["delta_G"])

    # Identify rate-limiting step
    rls = identify_rate_limiting_step(
        delta_G=thermo_result["delta_G"],
        step_labels=thermo_result["step_labels"],
    )

    # Create pathway document
    pathway_doc = ReactionPathwayDocument(
        surface_name="ORR_catalyst",
        pathway_type="orr",
        steps=[
            {
                "label": step["label"],
                "species": step.get("species"),
                "site_coords": None,  # Would need to extract from ads_doc
                "height": None,
                "energy": step["energy"],
                "structure": None,  # Could retrieve from database
            }
            for step in pathway_steps
        ],
        energies=thermo_result["absolute_energies"],
        delta_E=thermo_result["delta_E"],
        delta_G=thermo_result["delta_G"],
        overpotential_orr=overpotential_result["eta_ORR"],
        overpotential_oer=0.0,  # Not calculated for ORR-only workflow
        overpotential_gap=overpotential_result["eta_ORR"],  # Same as η_ORR
        rate_limiting_step=rls["rls_label"],
        temperature=temperature,
        pressure=pressure,
    )

    logger.info(
        f"ORR analysis complete: η = {overpotential_result['eta_ORR']:.3f} V, "
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
            pathway_type="ORR",
            filename="orr_free_energy_diagram.png",
        )
        output_files["free_energy_diagram"] = str(plot_file)
        logger.info(f"✓ Generated free energy diagram: {plot_file}")

        # Overpotential summary plot
        summary_plot = plot_overpotential_summary(
            pathway_type="ORR",
            overpotential=overpotential_result["eta_ORR"],
            rls_label=rls["rls_label"],
            rls_delta_G=rls["rls_delta_G"],
            U_onset=overpotential_result["U_onset"],
            filename="orr_overpotential_summary.png",
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
            and abs(o2_ads_doc.best_adsorption_energy) < 1e-6
            and abs(ooh_ads_doc.best_adsorption_energy) < 1e-6
            and abs(o_ads_doc.best_adsorption_energy) < 1e-6
            and abs(oh_ads_doc.best_adsorption_energy) < 1e-6
        )

        summary_file = write_analysis_summary(
            pathway_type="ORR",
            surface_name=surface_name,
            overpotential=overpotential_result["eta_ORR"],
            rls_label=rls["rls_label"],
            rls_delta_G=rls["rls_delta_G"],
            step_labels=thermo_result["step_labels"],
            delta_G=thermo_result["delta_G"],
            filename="orr_analysis_summary.txt",
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
class ORRFlowMaker(BaseSiestaFlowMaker):
    """
    Complete ORR workflow for electrocatalyst screening.

    This workflow automates the calculation of ORR activity on catalyst surfaces:

    1. **Gas-phase references**: O₂, H₂O, H₂ (using GasPhaseMoleculeMaker)
    2. **Clean surface**: Static calculation
    3. **Adsorption scans**: O₂*, OOH*, O*, OH* (using AdsorptionScanFlowMaker)
    4. **Thermodynamic analysis**: Free energy diagram, overpotential, RLS

    The workflow uses the Computational Hydrogen Electrode (CHE) model to convert
    DFT energies to electrochemical free energies.

    Parameters
    ----------
    name : str
        Workflow name (default: 'orr_workflow').
    gas_phase_maker : GasPhaseMoleculeMaker
        Maker for gas-phase molecular calculations (O₂, H₂O, H₂).
    surface_static_maker : StaticMaker
        Maker for clean surface calculation.
    adsorption_maker : AdsorptionScanFlowMaker
        Maker for adsorption site scanning (O₂, OOH, O, OH).
    grid_size : tuple[int, int]
        Grid size for adsorption site scanning (default: (4, 4)).
    height : float
        Adsorbate height above surface (Å, default: 2.0).
    temperature : float
        Temperature for thermodynamic corrections (K, default: 298.15).
    pressure : float
        Pressure (Pa, default: 101325 = 1 atm).
    ph : float
        pH of electrolyte (default: 0.0 for acidic).
    potential : float
        Electrode potential (V vs. SHE, default: 0.0).

    Examples
    --------
    Basic usage:

    >>> from pymatgen.core import Structure
    >>> from atomate2.siesta.flows.electrocatalysis import ORRFlowMaker
    >>> from atomate2.siesta.jobs.core import StaticMaker
    >>>
    >>> # Load catalyst surface
    >>> surface = Structure.from_file("Pt111_slab.cif")
    >>>
    >>> # Create ORR workflow
    >>> maker = ORRFlowMaker(
    ...     surface_static_maker=StaticMaker(
    ...         user_params={"PAO.BasisSize": "DZP", "kpts": [6, 6, 1]}
    ...     )
    ... )
    >>> flow = maker.make(surface)
    >>>
    >>> # Run locally
    >>> from jobflow import run_locally
    >>> results = run_locally(flow, create_folders=True)

    Custom parameters for each component:

    >>> from atomate2.siesta.flows.molecular import GasPhaseMoleculeMaker
    >>> from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker
    >>> from atomate2.siesta.jobs.core import RelaxMaker
    >>>
    >>> # Custom gas-phase maker (high accuracy)
    >>> gas_maker = GasPhaseMoleculeMaker(
    ...     relax_maker=RelaxMaker(
    ...         user_params={"PAO.BasisSize": "TZP", "Mesh.Cutoff": "400 Ry"}
    ...     )
    ... )
    >>>
    >>> # Custom adsorption maker (finer grid)
    >>> ads_maker = AdsorptionScanFlowMaker(
    ...     grid_size=(6, 6),
    ...     height=2.5,
    ...     slab_static_maker=StaticMaker(user_params={"kpts": [8, 8, 1]}),
    ... )
    >>>
    >>> # ORR workflow with custom makers
    >>> maker = ORRFlowMaker(
    ...     gas_phase_maker=gas_maker,
    ...     adsorption_maker=ads_maker,
    ...     temperature=320.0,  # Elevated temperature
    ...     ph=14.0,  # Alkaline conditions
    ... )
    >>> flow = maker.make(surface)

    Notes
    -----
    **ORR pathway** (4-electron mechanism):
        O₂ + 4H⁺ + 4e⁻ → 2H₂O

    Intermediate steps:
        1. O₂(g) → O₂* (adsorption)
        2. O₂* + H⁺ + e⁻ → OOH*
        3. OOH* + H⁺ + e⁻ → O* + H₂O
        4. O* + H⁺ + e⁻ → OH*
        5. OH* + H⁺ + e⁻ → H₂O + * (desorption)

    **CHE Model Assumptions**:
        - Chemical potential of (H⁺ + e⁻) = ½μ(H₂) - eU
        - Constant pH (adjustable via `ph` parameter)
        - Entropy and zero-point energy from gas-phase calculations

    **Output**:
        - Overpotential (η_ORR in Volts)
        - Free energy diagram (ΔG for each step)
        - Rate-limiting step (RLS)
        - Best adsorption sites for each intermediate

    See Also
    --------
    OERFlowMaker : Oxygen Evolution Reaction workflow
    HERFlowMaker : Hydrogen Evolution Reaction workflow
    GasPhaseMoleculeMaker : Gas-phase reference calculations
    AdsorptionScanFlowMaker : Adsorption site scanning
    """

    name: str = "orr_workflow"
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
    pressure: float = 101325.0  # Pa (1 atm)
    ph: float = 0.0  # Acidic conditions
    potential: float = 0.0  # V vs. SHE
    plot_results: bool = (
        True  # Generate plots (free energy diagram, overpotential summary)
    )
    write_summary: bool = True  # Write text summary file

    # Note: dry_run, use_custodian, custodian_handlers, custodian_max_errors,
    # and tier support inherited from BaseSiestaFlowMaker

    def make(self, surface: Structure) -> Flow:
        """
        Create ORR workflow for a given catalyst surface.

        Parameters
        ----------
        surface : Structure
            Catalyst surface slab structure.

        Returns
        -------
        Flow
            Jobflow Flow object with complete ORR workflow.
        """
        jobs = []

        # Step 1: Gas-phase reference calculations
        # O₂, H₂O, H₂
        logger.info("Setting up gas-phase reference calculations (O₂, H₂O, H₂)")

        o2_molecule = Molecule(["O", "O"], [[0, 0, 0], [0, 0, 1.21]])
        h2o_molecule = Molecule(
            ["O", "H", "H"],
            [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        )
        h2_molecule = Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])

        o2_job = self.gas_phase_maker.make(o2_molecule)
        o2_job.name = f"{self.name}_O2_gas"
        jobs.append(o2_job)

        h2o_job = self.gas_phase_maker.make(h2o_molecule)
        h2o_job.name = f"{self.name}_H2O_gas"
        jobs.append(h2o_job)

        h2_job = self.gas_phase_maker.make(h2_molecule)
        h2_job.name = f"{self.name}_H2_gas"
        jobs.append(h2_job)

        # Step 2: Clean surface calculation
        logger.info("Setting up clean surface calculation")
        clean_surface_job = self.surface_static_maker.make(surface)
        clean_surface_job.name = f"{self.name}_clean_surface"
        jobs.append(clean_surface_job)

        # Step 3: Adsorption scans for ORR intermediates
        # O₂*, OOH*, O*, OH*
        #
        # OPTIMIZATION: The clean slab energy is calculated once in the first
        # adsorption scan (O2), then reused for subsequent scans (OOH, O, OH)
        # via precalc_slab_energy parameter. This avoids 3 redundant slab
        # calculations that would otherwise be performed.
        logger.info("Setting up adsorption scans for ORR intermediates")

        # O₂ adsorption (calculates slab energy - will be reused)
        ads_maker_first = AdsorptionScanFlowMaker(
            grid_size=self.grid_size,
            height=self.height,
            slab_static_maker=self.adsorption_maker.slab_static_maker,
            adsorbate_static_maker=self.adsorption_maker.adsorbate_static_maker,
        )

        o2_ads_job = ads_maker_first.make(surface, o2_molecule)
        o2_ads_job.name = f"{self.name}_O2_adsorption"
        jobs.append(o2_ads_job)

        # For subsequent adsorbates, reuse slab energy from O2 scan
        # This saves 3 redundant slab calculations
        ads_maker_reuse = AdsorptionScanFlowMaker(
            grid_size=self.grid_size,
            height=self.height,
            slab_static_maker=self.adsorption_maker.slab_static_maker,
            adsorbate_static_maker=self.adsorption_maker.adsorbate_static_maker,
            precalc_slab_energy=o2_ads_job.output.slab_energy,  # Reuse from O2 scan
        )

        # OOH adsorption (reuses slab energy)
        ooh_molecule = Molecule(
            ["O", "O", "H"],
            [[0, 0, 0], [1.33, 0, 0], [1.70, 0.90, 0]],
        )
        ooh_ads_job = ads_maker_reuse.make(surface, ooh_molecule)
        ooh_ads_job.name = f"{self.name}_OOH_adsorption"
        jobs.append(ooh_ads_job)

        # O adsorption (reuses slab energy)
        o_molecule = Molecule(["O"], [[0, 0, 0]])
        o_ads_job = ads_maker_reuse.make(surface, o_molecule)
        o_ads_job.name = f"{self.name}_O_adsorption"
        jobs.append(o_ads_job)

        # OH adsorption (reuses slab energy)
        oh_molecule = Molecule(["O", "H"], [[0, 0, 0], [0.96, 0, 0]])
        oh_ads_job = ads_maker_reuse.make(surface, oh_molecule)
        oh_ads_job.name = f"{self.name}_OH_adsorption"
        jobs.append(oh_ads_job)

        # Step 4: Analyze ORR pathway
        logger.info("Setting up ORR pathway analysis")
        analysis_job = _analyze_orr_pathway(
            clean_surface_energy=clean_surface_job.output.output.energy,
            o2_gas_energy=o2_job.output.total_energy,
            h2o_gas_energy=h2o_job.output.total_energy,
            h2_gas_energy=h2_job.output.total_energy,
            o2_ads_doc=o2_ads_job.output,
            ooh_ads_doc=ooh_ads_job.output,
            o_ads_doc=o_ads_job.output,
            oh_ads_doc=oh_ads_job.output,
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

        logger.info(f"ORR workflow created with {len(jobs)} jobs")

        return Flow(jobs, output=analysis_job.output, name=self.name)
