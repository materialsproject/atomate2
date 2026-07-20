"""Bifunctional ORR/OER workflow for pre-relaxed slabs.

This module provides a combined ORR+OER workflow optimized for cases where:
1. The slab is already relaxed (no need to relax again)
2. Only adsorbates need to be relaxed (slab atoms fixed)
3. Both ORR and OER activity should be calculated

This is the recommended workflow for electrocatalysis screening because:
- Avoids redundant slab relaxations
- Ensures consistent slab geometry across all adsorption calculations
- Calculates bifunctional activity (important for metal-air batteries)

Usage:
    >>> from atomate2.siesta.flows.electrocatalysis import BifunctionalFlowMaker
    >>> from pymatgen.core import Structure
    >>>
    >>> # Load pre-relaxed slab
    >>> relaxed_slab = Structure.from_file("Pt111_relaxed.cif")
    >>>
    >>> # Create workflow (pass clean surface energy if already calculated)
    >>> maker = BifunctionalFlowMaker(
    ...     clean_surface_energy=-1234.5,  # eV, from previous calculation
    ... )
    >>> flow = maker.make(relaxed_slab)
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
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker

if TYPE_CHECKING:
    from pymatgen.core import Structure

    from atomate2.siesta.schemas.adsorption import AdsorptionScanDocument

logger = logging.getLogger(__name__)


def _get_slab_atom_indices(n_slab_atoms: int) -> list[int]:
    """Get 1-indexed list of slab atom indices for geometry constraints."""
    return list(range(1, n_slab_atoms + 1))


def _create_geometry_constraints_block(slab_atom_indices: list[int]) -> list[str]:
    """
    Create SIESTA Geometry.Constraints block to fix slab atoms.

    Parameters
    ----------
    slab_atom_indices : list[int]
        1-indexed list of slab atom indices to fix.

    Returns
    -------
    list[str]
        Lines for %block Geometry.Constraints
    """
    # SIESTA format: atom [list of atoms]
    # This fixes all coordinates (x, y, z) of the specified atoms
    constraints = []
    for idx in slab_atom_indices:
        constraints.append(f"atom {idx}")
    return constraints


@job
def _analyze_bifunctional_pathway(
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
    potential_orr: float = 0.0,
    potential_oer: float = 1.23,
    plot_results: bool = True,
    write_summary: bool = True,
    surface_name: str = "bifunctional_catalyst",
):
    """
    Analyze both ORR and OER pathways for bifunctional activity.

    This function calculates:
    - ORR overpotential (η_ORR)
    - OER overpotential (η_OER)
    - Bifunctional gap (η_OER + η_ORR)

    The bifunctional gap is important for rechargeable metal-air batteries,
    where both ORR (discharge) and OER (charge) occur on the same catalyst.
    """
    from atomate2.siesta.flows.electrocatalysis.analysis import (
        calculate_oer_overpotential,
        calculate_orr_overpotential,
        calculate_reaction_free_energies,
        identify_rate_limiting_step,
    )
    from atomate2.siesta.schemas.electrocatalysis import ReactionPathwayDocument

    # Extract best site energies
    o2_best_energy = o2_ads_doc.best_site.total_energy
    ooh_best_energy = ooh_ads_doc.best_site.total_energy
    o_best_energy = o_ads_doc.best_site.total_energy
    oh_best_energy = oh_ads_doc.best_site.total_energy

    gas_phase_energies = {
        "H2": h2_gas_energy,
        "H2O": h2o_gas_energy,
        "O2": o2_gas_energy,
    }

    # ========== ORR Analysis ==========
    orr_pathway_steps = [
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
            "species": "H",
            "n_H": 1,
            "n_e": 1,
        },
        {
            "label": "O* + H2O",
            "energy": o_best_energy,
            "species": "H2O",
            "n_H": 1,
            "n_e": 1,
        },
        {"label": "OH*", "energy": oh_best_energy, "species": "H", "n_H": 1, "n_e": 1},
        {
            "label": "H2O",
            "energy": clean_surface_energy,
            "species": "H2O",
            "n_H": 1,
            "n_e": 1,
        },
    ]

    orr_thermo = calculate_reaction_free_energies(
        surface_name="ORR_pathway",
        pathway_steps=orr_pathway_steps,
        gas_phase_energies=gas_phase_energies,
        clean_surface_energy=clean_surface_energy,
        temperature=temperature,
        pressure=pressure,
        ph=ph,
        potential=potential_orr,
    )

    orr_overpotential = calculate_orr_overpotential(orr_thermo["delta_G"])
    orr_rls = identify_rate_limiting_step(
        delta_G=orr_thermo["delta_G"],
        step_labels=orr_thermo["step_labels"],
    )

    # ========== OER Analysis ==========
    oer_pathway_steps = [
        {
            "label": "H2O + *",
            "energy": clean_surface_energy,
            "species": "H2O",
            "n_H": 0,
            "n_e": 0,
        },
        {"label": "OH*", "energy": oh_best_energy, "species": "H", "n_H": -1, "n_e": 1},
        {"label": "O*", "energy": o_best_energy, "species": "H", "n_H": -1, "n_e": 1},
        {
            "label": "OOH*",
            "energy": ooh_best_energy,
            "species": "H2O",
            "n_H": -1,
            "n_e": 1,
        },
        {"label": "O2*", "energy": o2_best_energy, "species": "H", "n_H": -1, "n_e": 1},
    ]

    oer_thermo = calculate_reaction_free_energies(
        surface_name="OER_pathway",
        pathway_steps=oer_pathway_steps,
        gas_phase_energies=gas_phase_energies,
        clean_surface_energy=clean_surface_energy,
        temperature=temperature,
        pressure=pressure,
        ph=ph,
        potential=potential_oer,
    )

    oer_overpotential = calculate_oer_overpotential(oer_thermo["delta_G"])
    oer_rls = identify_rate_limiting_step(
        delta_G=oer_thermo["delta_G"],
        step_labels=oer_thermo["step_labels"],
    )

    # ========== Bifunctional Analysis ==========
    eta_orr = orr_overpotential["eta_ORR"]
    eta_oer = oer_overpotential["eta_OER"]
    bifunctional_gap = eta_orr + eta_oer

    logger.info(
        f"Bifunctional analysis complete:\n"
        f"  η_ORR = {eta_orr:.3f} V (RLS: {orr_rls['rls_label']})\n"
        f"  η_OER = {eta_oer:.3f} V (RLS: {oer_rls['rls_label']})\n"
        f"  Bifunctional gap = {bifunctional_gap:.3f} V"
    )

    # Create pathway documents
    pathway_doc = ReactionPathwayDocument(
        surface_name=surface_name,
        pathway_type="bifunctional",
        steps=[],  # Combined steps not stored
        energies=[],
        delta_E=[],
        delta_G=[],
        overpotential_orr=eta_orr,
        overpotential_oer=eta_oer,
        overpotential_gap=bifunctional_gap,
        rate_limiting_step=f"ORR: {orr_rls['rls_label']}, OER: {oer_rls['rls_label']}",
        temperature=temperature,
        pressure=pressure,
    )

    # Generate plots and summary files
    output_files = {}
    if plot_results:
        from atomate2.siesta.flows.electrocatalysis.analysis.plotting import (
            plot_free_energy_diagram,
        )

        # ORR diagram
        orr_plot = plot_free_energy_diagram(
            step_labels=orr_thermo["step_labels"],
            cumulative_G=orr_thermo["cumulative_G"],
            delta_G=orr_thermo["delta_G"],
            pathway_type="ORR",
            filename="bifunctional_orr_diagram.png",
        )
        output_files["orr_diagram"] = str(orr_plot)

        # OER diagram
        oer_plot = plot_free_energy_diagram(
            step_labels=oer_thermo["step_labels"],
            cumulative_G=oer_thermo["cumulative_G"],
            delta_G=oer_thermo["delta_G"],
            pathway_type="OER",
            filename="bifunctional_oer_diagram.png",
        )
        output_files["oer_diagram"] = str(oer_plot)

        logger.info(f"Generated plots: {orr_plot}, {oer_plot}")

    if write_summary:
        # Write combined summary
        summary_file = _write_bifunctional_summary(
            surface_name=surface_name,
            eta_orr=eta_orr,
            eta_oer=eta_oer,
            bifunctional_gap=bifunctional_gap,
            orr_rls=orr_rls,
            oer_rls=oer_rls,
            orr_thermo=orr_thermo,
            oer_thermo=oer_thermo,
            filename="bifunctional_summary.txt",
        )
        output_files["summary"] = str(summary_file)
        logger.info(f"Generated summary: {summary_file}")

    return {
        "pathway_document": pathway_doc.dict(),
        "orr": {
            "overpotential": orr_overpotential,
            "thermodynamics": orr_thermo,
            "rate_limiting_step": orr_rls,
        },
        "oer": {
            "overpotential": oer_overpotential,
            "thermodynamics": oer_thermo,
            "rate_limiting_step": oer_rls,
        },
        "bifunctional_gap": bifunctional_gap,
        "output_files": output_files,
    }


def _write_bifunctional_summary(
    surface_name: str,
    eta_orr: float,
    eta_oer: float,
    bifunctional_gap: float,
    orr_rls: dict,
    oer_rls: dict,
    orr_thermo: dict,
    oer_thermo: dict,
    filename: str = "bifunctional_summary.txt",
) -> str:
    """Write bifunctional analysis summary to file."""
    from pathlib import Path

    lines = [
        "=" * 70,
        "BIFUNCTIONAL ORR/OER ELECTROCATALYST ANALYSIS",
        "=" * 70,
        "",
        f"Surface: {surface_name}",
        "",
        "─" * 70,
        "SUMMARY",
        "─" * 70,
        f"  ORR overpotential (η_ORR): {eta_orr:.3f} V",
        f"  OER overpotential (η_OER): {eta_oer:.3f} V",
        f"  Bifunctional gap (η_ORR + η_OER): {bifunctional_gap:.3f} V",
        "",
        "─" * 70,
        "ORR ANALYSIS (O₂ + 4H⁺ + 4e⁻ → 2H₂O)",
        "─" * 70,
        f"  Rate-limiting step: {orr_rls['rls_label']}",
        f"  RLS ΔG: {orr_rls['rls_delta_G']:.3f} eV",
        "",
        "  Step-by-step ΔG (eV):",
    ]

    for label, dg in zip(
        orr_thermo["step_labels"], orr_thermo["delta_G"], strict=False
    ):
        lines.append(f"    {label}: {dg:.3f}")

    lines.extend(
        [
            "",
            "─" * 70,
            "OER ANALYSIS (2H₂O → O₂ + 4H⁺ + 4e⁻)",
            "─" * 70,
            f"  Rate-limiting step: {oer_rls['rls_label']}",
            f"  RLS ΔG: {oer_rls['rls_delta_G']:.3f} eV",
            "",
            "  Step-by-step ΔG (eV):",
        ]
    )

    for label, dg in zip(
        oer_thermo["step_labels"], oer_thermo["delta_G"], strict=False
    ):
        lines.append(f"    {label}: {dg:.3f}")

    lines.extend(
        [
            "",
            "─" * 70,
            "INTERPRETATION",
            "─" * 70,
            "",
            "Bifunctional gap benchmark:",
            "  - Pt/C: ~0.8 V (ORR active, poor OER)",
            "  - IrO₂: ~0.4 V (OER active, poor ORR)",
            "  - Best bifunctional: ~0.6-0.7 V",
            "",
            "For rechargeable metal-air batteries:",
            "  - Lower bifunctional gap = higher round-trip efficiency",
            "  - Target: < 0.8 V for practical applications",
            "",
            "=" * 70,
        ]
    )

    output_path = Path(filename)
    output_path.write_text("\n".join(lines))
    return str(output_path)


@dataclass
class BifunctionalFlowMaker(BaseSiestaFlowMaker):
    """
    Combined ORR+OER workflow for pre-relaxed catalyst slabs.

    This workflow is optimized for electrocatalysis screening where:
    1. The slab is already relaxed (pass structure directly)
    2. Only adsorbates need geometry optimization (slab atoms fixed)
    3. Both ORR and OER activity should be evaluated

    Key Features:
    - **Pre-relaxed slab**: No redundant slab relaxation
    - **Fixed slab atoms**: Only adsorbates move during relaxation
    - **Bifunctional analysis**: Combined ORR + OER overpotentials
    - **Shared intermediates**: O*, OH*, OOH*, O₂* calculated once

    Parameters
    ----------
    name : str
        Workflow name (default: 'bifunctional_workflow').
    clean_surface_energy : float, optional
        Pre-calculated clean surface energy (eV). If None, a static
        calculation will be performed.
    gas_phase_maker : GasPhaseMoleculeMaker
        Maker for gas-phase molecular calculations.
    adsorption_relax_maker : RelaxMaker
        Maker for adsorbate relaxation (slab atoms will be fixed).
    grid_size : tuple[int, int]
        Grid size for adsorption site scanning (default: (3, 3)).
    height : float
        Initial adsorbate height (Å, default: 2.0).
    temperature : float
        Temperature (K, default: 298.15).
    ph : float
        pH of electrolyte (default: 0.0).

    Examples
    --------
    Basic usage with pre-relaxed slab:

    >>> from pymatgen.core import Structure
    >>> from atomate2.siesta.flows.electrocatalysis import BifunctionalFlowMaker
    >>>
    >>> # Load pre-relaxed slab
    >>> relaxed_slab = Structure.from_file("Pt111_relaxed.cif")
    >>>
    >>> # Create workflow
    >>> maker = BifunctionalFlowMaker(
    ...     clean_surface_energy=-1234.567,  # From previous calculation
    ... )
    >>> flow = maker.make(relaxed_slab)
    >>>
    >>> # Run
    >>> from jobflow import run_locally
    >>> results = run_locally(flow, create_folders=True)

    With diffuse basis for surface atoms:

    >>> from atomate2.siesta.sets.utils import apply_diffuse_basis_to_surface
    >>> from atomate2.siesta.jobs.core import RelaxMaker
    >>> from atomate2.siesta.sets.tiers import apply_tier_preset
    >>>
    >>> # Apply diffuse basis
    >>> labels, basis_sizes, info = apply_diffuse_basis_to_surface(
    ...     relaxed_slab, surface_basis="DZ", bulk_basis="SZ"
    ... )
    >>> relaxed_slab.add_site_property("species_label", labels)
    >>>
    >>> # Create makers with diffuse basis
    >>> ads_maker = RelaxMaker.fixed_cell_relaxation()
    >>> ads_maker = apply_tier_preset(
    ...     ads_maker,
    ...     "electrocatalysis_dirty",
    ...     override_params={"%block PAO.BasisSizes": basis_sizes},
    ... )
    >>>
    >>> # Create workflow
    >>> maker = BifunctionalFlowMaker(
    ...     adsorption_relax_maker=ads_maker,
    ...     clean_surface_energy=-1234.567,
    ... )
    >>> flow = maker.make(relaxed_slab)

    Notes
    -----
    **Workflow structure:**

    1. Gas-phase references (O₂, H₂O, H₂)
    2. Clean surface static (if energy not provided)
    3. Adsorption scans with fixed slab:
       - O₂* (O₂ on surface)
       - OOH* (peroxide intermediate)
       - O* (atomic oxygen)
       - OH* (hydroxyl)
    4. Bifunctional analysis:
       - ORR pathway: O₂ → OOH* → O* → OH* → H₂O
       - OER pathway: H₂O → OH* → O* → OOH* → O₂

    **Why fix slab atoms?**
    - Faster calculations (fewer degrees of freedom)
    - Consistent slab geometry across all adsorbates
    - Reduces artifacts from slab reconstruction
    - Standard practice in electrocatalysis DFT

    See Also
    --------
    ORRFlowMaker : ORR-only workflow
    OERFlowMaker : OER-only workflow
    AdsorptionScanFlowMaker : Adsorption site scanning
    """

    name: str = "bifunctional_workflow"
    clean_surface_energy: float | None = None
    gas_phase_maker: GasPhaseMoleculeMaker = field(
        default_factory=GasPhaseMoleculeMaker
    )
    surface_static_maker: StaticMaker = field(default_factory=StaticMaker)
    adsorption_relax_maker: RelaxMaker = field(
        default_factory=RelaxMaker.fixed_cell_relaxation
    )
    adsorbate_static_maker: StaticMaker = field(default_factory=StaticMaker)
    grid_size: tuple[int, int] = (3, 3)
    height: float = 2.0
    temperature: float = 298.15
    pressure: float = 101325.0
    ph: float = 0.0
    potential_orr: float = 0.0
    potential_oer: float = 1.23
    plot_results: bool = True
    write_summary: bool = True

    def make(self, surface: Structure) -> Flow:
        """
        Create bifunctional ORR/OER workflow for a pre-relaxed slab.

        Parameters
        ----------
        surface : Structure
            Pre-relaxed catalyst surface slab.

        Returns
        -------
        Flow
            Jobflow Flow with complete bifunctional workflow.
        """
        jobs = []
        n_slab_atoms = len(surface)

        # Create geometry constraints to fix slab atoms
        slab_indices = _get_slab_atom_indices(n_slab_atoms)
        constraints_block = _create_geometry_constraints_block(slab_indices)

        logger.info(
            f"Creating bifunctional workflow for {surface.composition.reduced_formula}"
        )
        logger.info(f"  Slab atoms to fix: {n_slab_atoms}")
        logger.info(f"  Grid size: {self.grid_size}")

        # ========== Step 1: Gas-phase references ==========
        logger.info("Setting up gas-phase references (O₂, H₂O, H₂)")

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

        # ========== Step 2: Clean surface (if energy not provided) ==========
        if self.clean_surface_energy is None:
            logger.info("Setting up clean surface static calculation")
            clean_surface_job = self.surface_static_maker.make(surface)
            clean_surface_job.name = f"{self.name}_clean_surface"
            jobs.append(clean_surface_job)
            clean_energy_ref = clean_surface_job.output.output.energy
        else:
            logger.info(
                f"Using pre-calculated clean surface energy: {self.clean_surface_energy} eV"
            )
            clean_energy_ref = self.clean_surface_energy

        # ========== Step 3: Adsorption scans (fixed slab) ==========
        logger.info("Setting up adsorption scans with fixed slab atoms")

        # Create adsorption maker with geometry constraints
        # The constraints will be added to user_params
        from copy import deepcopy

        constrained_relax_maker = deepcopy(self.adsorption_relax_maker)
        # Add constraints to the input set generator's user parameters
        # (makers keep their FDF parameters on input_set_generator)
        generator = constrained_relax_maker.input_set_generator
        if generator.user_params is None:
            generator.user_params = {}
        generator.user_params["%block Geometry.Constraints"] = constraints_block

        ads_maker = AdsorptionScanFlowMaker(
            grid_size=self.grid_size,
            height=self.height,
            slab_static_maker=constrained_relax_maker,
            adsorbate_static_maker=self.adsorbate_static_maker,
        )

        # O₂ adsorption
        o2_ads_job = ads_maker.make(surface, o2_molecule)
        o2_ads_job.name = f"{self.name}_O2_adsorption"
        jobs.append(o2_ads_job)

        # OOH adsorption
        ooh_molecule = Molecule(
            ["O", "O", "H"],
            [[0, 0, 0], [1.33, 0, 0], [1.70, 0.90, 0]],
        )
        ooh_ads_job = ads_maker.make(surface, ooh_molecule)
        ooh_ads_job.name = f"{self.name}_OOH_adsorption"
        jobs.append(ooh_ads_job)

        # O adsorption
        o_molecule = Molecule(["O"], [[0, 0, 0]])
        o_ads_job = ads_maker.make(surface, o_molecule)
        o_ads_job.name = f"{self.name}_O_adsorption"
        jobs.append(o_ads_job)

        # OH adsorption
        oh_molecule = Molecule(["O", "H"], [[0, 0, 0], [0.96, 0, 0]])
        oh_ads_job = ads_maker.make(surface, oh_molecule)
        oh_ads_job.name = f"{self.name}_OH_adsorption"
        jobs.append(oh_ads_job)

        # ========== Step 4: Bifunctional analysis ==========
        logger.info("Setting up bifunctional analysis")
        analysis_job = _analyze_bifunctional_pathway(
            clean_surface_energy=clean_energy_ref,
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
            potential_orr=self.potential_orr,
            potential_oer=self.potential_oer,
            plot_results=self.plot_results,
            write_summary=self.write_summary,
            surface_name=f"{surface.composition.reduced_formula}_bifunctional",
        )
        analysis_job.name = f"{self.name}_analysis"
        jobs.append(analysis_job)

        logger.info(f"Bifunctional workflow created with {len(jobs)} jobs")

        return Flow(jobs, output=analysis_job.output, name=self.name)
