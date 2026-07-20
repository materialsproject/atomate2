"""
Correction scheme comparison workflow.

This module provides a unique feature for comparing multiple finite-size
correction schemes on the same defect calculation, enabling uncertainty
quantification and validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Flow, Maker, job

from atomate2.siesta.flows.defects.core import (
    _enable_modules,
    extract_chemical_potential,
    generate_density_plot,
    generate_dielectric_profile_plot,
    generate_potential_plot,
    generate_radial_distribution_plot,
    get_reference_structure,
)
from atomate2.siesta.flows.defects.corrections import (
    FreysoldtCorrection,
    KumagaiCorrection,
    LanyZungerCorrection,
    MakovPayneCorrection,
)
from atomate2.siesta.flows.defects.makers import DefectRelaxMaker, DefectStaticMaker
from atomate2.siesta.flows.defects.utils import (
    find_vt_files,
    prepare_freysoldt_potential_data,
)

if TYPE_CHECKING:
    from pymatgen.core import Structure

    from atomate2.siesta.schemas.task import SiestaTaskDoc

logger = logging.getLogger(__name__)


@dataclass
class CorrectionComparisonFlowMaker(Maker):
    """
    Compare multiple finite-size correction schemes.

    This is a unique feature of atomate2siesta that runs all available
    correction schemes on the same defect calculation and compares the
    results. This provides:

    - Uncertainty quantification (spread between schemes)
    - Validation (check consistency between methods)
    - Automated recommendations (best scheme for this system)
    - Publication-ready comparison plots

    The workflow runs:
    1. Single defect relaxation (reused across all corrections)
    2. Single host static calculation (reused)
    3. Apply all correction schemes in parallel
    4. Statistical analysis and comparison
    5. Generate text summary and diagnostic plots
       - Potential alignment plots (ΔV with fit)
       - Charge density difference plots (Δρ)
       - Radial distribution plots (Δρ(r) and ΔV(r))
       - Dielectric profile plots (for slab geometries)

    Parameters
    ----------
    name : str
        Name of the flow (default: "Correction Comparison")
    defect_relax_maker : DefectRelaxMaker
        Maker for defect relaxation
    host_static_maker : DefectStaticMaker
        Maker for host static calculation
    epsilon_static : float
        Static dielectric constant for corrections (isotropic value)
    epsilon_parallel : float, optional
        In-plane dielectric constant for 2D slab corrections (ε∥)
        Only used when "slab2d" is in correction_schemes
        If not provided, uses epsilon_static
    epsilon_perpendicular : float, optional
        Out-of-plane dielectric constant for 2D slab corrections (ε⊥)
        Only used when "slab2d" is in correction_schemes
        If not provided, uses epsilon_static
    correction_schemes : list[str]
        List of correction schemes to compare.
        Available: ["lany-zunger", "makov-payne", "makov-payne-quadrupole",
                    "freysoldt", "kumagai", "slab2d"]
        Default: ["lany-zunger"]
    defect_type : str
        Type of defect (vacancy, substitutional, interstitial)
    charge_state : int
        Charge state of the defect
    dry_run : bool
        If True, enable dry-run mode for all calculations (default: False)
    skip_relax : bool
        If True, skip defect relaxation and use input structure directly.
        Useful for testing corrections on pre-relaxed or unrelaxed structures.
        Default: False (perform relaxation)
    chemical_potentials : dict, optional
        Chemical potentials for defect species (in eV). Format:
        {"O": -5.0, "Mg": -2.0, ...}
    auto_calculate_chemical_potentials : bool
        If True, automatically calculate chemical potentials from reference
        structures (O2, H2, N2 molecules; bulk metals). Default: False

    Examples
    --------
    Compare all available correction schemes:

    >>> from atomate2.siesta.flows.defects import CorrectionComparisonFlowMaker
    >>> maker = CorrectionComparisonFlowMaker(
    ...     epsilon_static=9.8,
    ...     defect_type="vacancy",
    ...     charge_state=2,
    ...     correction_schemes=["lany-zunger"],  # More to be added
    ... )
    >>> flow = maker.make(defect_structure, host_structure)

    The output will include:
    - Comparison of correction energies for all schemes
    - Statistical analysis (mean, std, range)
    - Automated recommendation
    - Diagnostic plots:
      * Potential alignment plots (ΔV vs position with polynomial fit)
      * Charge density difference plots (Δρ visualization)
      * Radial distribution plots (Δρ(r) and ΔV(r) with Gaussian model)
      * Dielectric profile plots (for Slab2D corrections)

    Notes
    -----
    This is a killer feature unique to atomate2siesta. No other defect
    workflow framework provides automated comparison of multiple correction
    schemes with uncertainty quantification.

    Available correction schemes:
    - Lany-Zunger: Simple isotropic model-charge correction
    - Makov-Payne: Basic (monopole only, Q=0)
    - Makov-Payne-Quadrupole: Full with quadrupole from .RHO (auto-enabled)
    - Freysoldt: Anisotropic with potential alignment (gold standard)
    - Kumagai: Extended Freysoldt with atomic-site sampling
    - Slab2D: 2D materials with anisotropic dielectric (requires ε∥, ε⊥)

    Phase 3 will add:
    - Kumagai-Oba (atomic-site sampling)
    - Gaussian countercharge
    """

    name: str = "Correction Comparison"
    defect_relax_maker: DefectRelaxMaker = field(
        default_factory=DefectRelaxMaker.defect_relax
    )
    host_static_maker: DefectStaticMaker = field(
        default_factory=DefectStaticMaker.defect_scf
    )
    epsilon_static: float = 10.0
    epsilon_parallel: float | None = None
    epsilon_perpendicular: float | None = None
    correction_schemes: list[str] = field(default_factory=lambda: ["lany-zunger"])
    defect_type: str = "vacancy"
    charge_state: int = 0
    dry_run: bool = False
    skip_relax: bool = False
    chemical_potentials: dict[str, float] | None = None
    auto_calculate_chemical_potentials: bool = False

    def make(
        self,
        defect_structure: Structure,
        host_structure: Structure,
        defect_site: list[float] | None = None,
        defect_species: str | None = None,
    ) -> Flow:
        """
        Create the correction comparison flow.

        Parameters
        ----------
        defect_structure : Structure
            Defect supercell structure (with ghost atoms for vacancies!)
        host_structure : Structure
            Pristine host supercell structure (same size as defect)
        defect_site : list[float], optional
            Fractional coordinates [x, y, z] of defect site
        defect_species : str, optional
            Species of the defect (e.g., 'O' for oxygen vacancy)

        Returns
        -------
        Flow
            Jobflow Flow object with comparison jobs

        Notes
        -----
        The workflow structure is:

        DefectRelax ─┐
                     ├─→ CompareCorrections ─→ Summary ─→ DiagnosticPlots
        HostStatic ──┘                                    (potential, density,
                                                           radial, dielectric)

        All correction schemes use the same defect and host calculations,
        so the comparison is fair and computational cost is minimized.

        Diagnostic plots generated (for charged defects):
        - Potential alignment plot (if Freysoldt/Kumagai/Slab2D used)
        - Charge density difference plot (always for q≠0)
        - Radial distribution plot (always for q≠0)
        - Dielectric profile plot (if Slab2D used)
        """
        # Create safe job name
        safe_name = self.name.replace(" ", "_").replace("+", "p").replace("-", "m")

        # Set dry_run mode if requested
        if self.dry_run:
            self.defect_relax_maker.dry_run = True
            self.host_static_maker.dry_run = True

        # Auto-enable .VT file writing if Freysoldt or Kumagai corrections requested
        needs_vt_files = any(
            scheme.lower()
            in ["freysoldt", "frey", "fnv", "kumagai", "kumagai-oba", "ko"]
            for scheme in self.correction_schemes
        )
        if needs_vt_files:
            # Add to defect maker
            current_defect_params = (
                self.defect_relax_maker.input_set_generator.user_params or {}
            )
            current_defect_params.pop("enabled_modules", None)
            _enable_modules(
                self.defect_relax_maker.input_set_generator, ["grids_advanced"]
            )

            self.defect_relax_maker.input_set_generator.user_params = {
                **current_defect_params,
                "SaveTotalPotential": True,
                "SaveElectrostaticPotential": True,  # Also save VH (Hartree potential)
            }

            # Add to host maker
            current_host_params = (
                self.host_static_maker.input_set_generator.user_params or {}
            )
            current_host_params.pop("enabled_modules", None)
            _enable_modules(
                self.host_static_maker.input_set_generator, ["grids_advanced"]
            )

            self.host_static_maker.input_set_generator.user_params = {
                **current_host_params,
                "SaveTotalPotential": True,
                "SaveElectrostaticPotential": True,  # Also save VH (Hartree potential)
            }

            logger.info(
                "Auto-enabled .VT and .VH file writing for "
                "Freysoldt/Kumagai corrections"
            )

        # CRITICAL: Set NetCharge for charged defects on the right maker
        saved_host_params = None
        if self.charge_state != 0:
            defect_maker = (
                self.host_static_maker if self.skip_relax else self.defect_relax_maker
            )

            current_params = defect_maker.input_set_generator.user_params or {}
            updated_params = current_params.copy()
            updated_params.pop("enabled_modules", None)
            updated_params["NetCharge"] = self.charge_state

            # Enable charge_dipole module (NetCharge is in advanced tier)
            _enable_modules(defect_maker.input_set_generator, ["charge_dipole"])

            # Save original params if modifying host_static_maker
            if self.skip_relax:
                saved_host_params = current_params

            defect_maker.input_set_generator.user_params = updated_params

            logger.info(
                f"Setting NetCharge = {self.charge_state} "
                "for charged defect calculation"
            )

        # Job 1: Defect calculation (shared by all corrections)
        if self.skip_relax:
            # Use static calculation (no relaxation)
            defect_relax_job = self.host_static_maker.make(defect_structure)
            defect_relax_job.name = f"{safe_name}_defect_static"
            logger.info("Skipping relaxation - using static calculation for defect")

            # Restore host_static_maker params (remove NetCharge before host calc)
            if saved_host_params is not None:
                self.host_static_maker.input_set_generator.user_params = (
                    saved_host_params
                )
        else:
            # Normal relaxation
            defect_relax_job = self.defect_relax_maker.make(defect_structure)
            defect_relax_job.name = f"{safe_name}_defect_relax"

        # Job 2: Static calculation on host (NetCharge removed if skip_relax)
        host_static_job = self.host_static_maker.make(host_structure)
        host_static_job.name = f"{safe_name}_host_static"

        # Job 2.5: Calculate chemical potential if requested
        if self.auto_calculate_chemical_potentials and defect_species:
            logger.info(f"Auto-calculating chemical potential for {defect_species}...")
            # Get reference structure (O2, Mg bulk, etc.)
            ref_structure, n_atoms = get_reference_structure(defect_species)

            # Run static calculation on reference
            ref_static_job = self.host_static_maker.make(ref_structure)
            ref_static_job.name = f"{safe_name}_ref_{defect_species}"

            # Extract μ from reference calculation
            mu_extract_job = extract_chemical_potential(
                task_doc=ref_static_job.output,
                species=defect_species,
                n_atoms=n_atoms,
            )
            mu_extract_job.name = f"{safe_name}_mu_{defect_species}"

            # Use extracted μ value
            chemical_potentials = {defect_species: mu_extract_job.output}
            ref_jobs = [ref_static_job, mu_extract_job]
        elif self.auto_calculate_chemical_potentials and not defect_species:
            logger.warning(
                "auto_calculate_chemical_potentials=True but defect_species "
                "not provided. Cannot calculate chemical potential!"
            )
            chemical_potentials = self.chemical_potentials
            ref_jobs = []
        else:
            chemical_potentials = self.chemical_potentials
            ref_jobs = []

        # Job 3: Apply all corrections and compare
        comparison_job = compare_all_corrections(
            defect_task_doc=defect_relax_job.output,
            host_task_doc=host_static_job.output,
            host_structure=host_structure,
            epsilon_static=self.epsilon_static,
            epsilon_parallel=self.epsilon_parallel,
            epsilon_perpendicular=self.epsilon_perpendicular,
            correction_schemes=self.correction_schemes,
            defect_type=self.defect_type,
            charge_state=self.charge_state,
            defect_site=defect_site,
            defect_species=defect_species,
            chemical_potentials=chemical_potentials,  # Use auto-calculated or manual
        )
        comparison_job.name = f"{safe_name}_compare_corrections"

        # Job 4: Generate summary and plots
        summary_job = generate_comparison_summary(
            comparison_results=comparison_job.output,
            charge_state=self.charge_state,
        )
        summary_job.name = f"{safe_name}_summary"

        # Job 5+: Generate diagnostic plots (same as DefectFlowMaker)
        plot_jobs = []

        # Determine which schemes need which plots
        scheme_lower_list = [s.lower() for s in self.correction_schemes]
        needs_potential_plot = any(
            scheme
            in [
                "freysoldt",
                "frey",
                "fnv",
                "kumagai",
                "kumagai-oba",
                "ko",
                "slab2d",
                "slab-2d",
            ]
            for scheme in scheme_lower_list
        )
        needs_dielectric_profile = any(
            scheme in ["slab2d", "slab-2d"] for scheme in scheme_lower_list
        )

        # Potential alignment plot (for potential-based corrections)
        if needs_potential_plot:
            vt_plot_job = generate_potential_plot(
                defect_task_doc=defect_relax_job.output,
                host_task_doc=host_static_job.output,
                output_name=f"{safe_name}_potential_alignment.png",
            )
            vt_plot_job.name = f"{safe_name}_vt_plot"
            plot_jobs.append(vt_plot_job)

        # Dielectric profile plot (for Slab2D)
        if needs_dielectric_profile:
            profile_plot_job = generate_dielectric_profile_plot(
                defect_task_doc=defect_relax_job.output,
                epsilon_parallel=self.epsilon_parallel or self.epsilon_static,
                epsilon_perpendicular=self.epsilon_perpendicular or self.epsilon_static,
                output_name=f"{safe_name}_dielectric_profile.png",
            )
            profile_plot_job.name = f"{safe_name}_profile_plot"
            plot_jobs.append(profile_plot_job)

        # Density plot (for ALL charged defects)
        if self.charge_state != 0:
            rho_plot_job = generate_density_plot(
                defect_task_doc=defect_relax_job.output,
                host_task_doc=host_static_job.output,
                output_name=f"{safe_name}_density_difference.png",
            )
            rho_plot_job.name = f"{safe_name}_rho_plot"
            plot_jobs.append(rho_plot_job)

        # Radial distribution plot (for ALL charged defects)
        if self.charge_state != 0:
            radial_plot_job = generate_radial_distribution_plot(
                defect_task_doc=defect_relax_job.output,
                host_task_doc=host_static_job.output,
                defect_site_frac=defect_site,
                charge_state=self.charge_state,
                output_name=f"{safe_name}_radial_distribution.png",
            )
            radial_plot_job.name = f"{safe_name}_radial_plot"
            plot_jobs.append(radial_plot_job)

        # Create flow
        if ref_jobs:
            # Include chemical potential calculation
            flow = Flow(
                [
                    defect_relax_job,
                    host_static_job,
                    *ref_jobs,
                    comparison_job,
                    summary_job,
                    *plot_jobs,
                ],
                output=summary_job.output,
                name=self.name,
            )
        else:
            # Standard flow without μ calculation
            flow = Flow(
                [
                    defect_relax_job,
                    host_static_job,
                    comparison_job,
                    summary_job,
                    *plot_jobs,
                ],
                output=summary_job.output,
                name=self.name,
            )

        return flow


@job
def compare_all_corrections(
    defect_task_doc: SiestaTaskDoc | dict,
    host_task_doc: SiestaTaskDoc | dict,
    host_structure: Structure,
    epsilon_static: float,
    correction_schemes: list[str],
    defect_type: str,
    charge_state: int,
    defect_site: list[float] | None = None,
    defect_species: str | None = None,
    chemical_potentials: dict[str, float] | None = None,
    calculation_dirs: dict[str, str] | None = None,
    plot_alignment: bool = True,  # noqa: ARG001
    epsilon_parallel: float | None = None,
    epsilon_perpendicular: float | None = None,
) -> dict:
    """
    Apply all correction schemes and collect results.

    This job applies all requested correction schemes to the same
    defect calculation and collects the results for comparison.

    Parameters
    ----------
    defect_task_doc : TaskDocument or dict
        TaskDocument from defect relaxation
    host_task_doc : TaskDocument or dict
        TaskDocument from host static calculation
    host_structure : Structure
        Pristine host structure
    epsilon_static : float
        Static dielectric constant
    correction_schemes : list[str]
        List of correction scheme names to apply
    defect_type : str
        Type of defect
    charge_state : int
        Charge state of defect
    defect_site : list[float], optional
        Fractional coordinates of defect site
    defect_species : str, optional
        Species of the defect
    chemical_potentials : dict, optional
        Chemical potentials in eV: {"O": -5.0, "Mg": -2.0, ...}
    calculation_dirs : dict, optional
        Paths to calculation directories for reading .VT files:
        {"defect": "/path/to/defect_calc", "host": "/path/to/host_calc"}
        If provided, will automatically read and use electrostatic potential
        data for Freysoldt and Kumagai corrections.
    plot_alignment : bool
        Whether to generate potential alignment plots (default: True).
        Only applies if .VT files are found.

    Returns
    -------
    dict
        Dictionary with comparison results:
        - "schemes": list of scheme names
        - "correction_energies": list of correction energies (eV)
        - "corrected_formation_energies": list of corrected E_formation (eV)
        - "raw_formation_energy": raw E_formation before correction (eV)
        - "metadata": list of metadata dicts from each scheme
        - "statistics": statistical summary
    """
    logger.info(f"Comparing {len(correction_schemes)} correction schemes...")

    # Get energies from task documents
    # Handle both real TaskDoc objects and dry-run dicts
    if isinstance(defect_task_doc, dict):
        # Dry-run mode
        defect_energy = defect_task_doc.get("output", {}).get("energy", -100.0)
        host_energy = host_task_doc.get("output", {}).get("energy", -50.0)
        defect_structure = host_structure.copy()  # Placeholder
    else:
        # Real calculation
        defect_energy = defect_task_doc.output.energy
        host_energy = host_task_doc.output.energy
        defect_structure = defect_task_doc.output.structure

        # Try to extract calculation directories from task documents if not provided
        if calculation_dirs is None:
            # Task documents may have dir_name or similar field
            defect_dir = getattr(defect_task_doc, "dir_name", None)
            host_dir = getattr(host_task_doc, "dir_name", None)

            if defect_dir and host_dir:
                calculation_dirs = {"defect": defect_dir, "host": host_dir}
                logger.info(
                    f"Extracted calculation directories from task documents: "
                    f"defect={defect_dir}, host={host_dir}"
                )

    # Calculate chemical potential contribution
    mu_defect = 0.0
    if chemical_potentials and defect_species:
        if defect_type == "vacancy":
            # For vacancy: add back energy of removed atom
            mu_defect = chemical_potentials.get(defect_species, 0.0)
            logger.info(
                f"Chemical potential for {defect_species} vacancy: "
                f"μ = {mu_defect:.4f} eV"
            )
        elif defect_type == "substitutional":
            # For substitution: μ_removed - μ_added (would need both species)
            mu_defect = chemical_potentials.get(defect_species, 0.0)
            logger.warning(
                "Substitutional defects need both removed and added "
                "species for correct μ"
            )
        # For interstitial: subtract energy of added atom (negative μ)
    elif defect_species:
        logger.warning(
            f"Chemical potentials not provided for {defect_species}. "
            f"Formation energy will be INCORRECT! "
            f"Provide chemical_potentials dict: {{'O': μ_O, 'Mg': μ_Mg, ...}}"
        )

    # Calculate raw formation energy (includes chemical potential term)
    raw_formation_energy = defect_energy - host_energy + mu_defect

    # Try to read electrostatic potential data for advanced corrections
    potential_data = None
    vt_file_paths = None
    vt_files_found = False

    if calculation_dirs is not None:
        logger.info("Attempting to read .VT files for potential alignment...")

        defect_dir = calculation_dirs.get("defect")
        host_dir = calculation_dirs.get("host")

        if defect_dir and host_dir:
            # Find .VT files in calculation directories
            defect_vt = find_vt_files(defect_dir)
            host_vt = find_vt_files(host_dir)

            if defect_vt and host_vt:
                try:
                    # Prepare potential data for corrections
                    potential_data = prepare_freysoldt_potential_data(
                        defect_vt_path=defect_vt,
                        host_vt_path=host_vt,
                    )
                    # Store VT file paths for plot generation by corrections
                    vt_file_paths = {"defect": defect_vt, "host": host_vt}
                    vt_files_found = True
                    logger.info(
                        f"Successfully read .VT files: {defect_vt.name}, {host_vt.name}"
                    )
                    logger.info(f"Potential grid shape: {potential_data['grid_shape']}")
                    logger.info(
                        "Corrections will auto-generate alignment plots if applicable"
                    )

                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to read .VT files: {e}")
                    logger.warning(
                        "Corrections will use lattice term only (dry-run mode)"
                    )
            else:
                if not defect_vt:
                    logger.info(f"No .VT file found in defect directory: {defect_dir}")
                if not host_vt:
                    logger.info(f"No .VT file found in host directory: {host_dir}")
                logger.info("Corrections will use lattice term only (dry-run mode)")

    # Apply each correction scheme
    results = {
        "schemes": [],
        "correction_energies": [],
        "corrected_formation_energies": [],
        "raw_formation_energy": raw_formation_energy,
        "metadata": [],
        "defect_type": defect_type,
        "defect_species": defect_species,
        "defect_site": defect_site,
        "charge_state": charge_state,
        "epsilon_static": epsilon_static,
        "epsilon_parallel": epsilon_parallel,
        "epsilon_perpendicular": epsilon_perpendicular,
        "vt_files_used": vt_files_found,
        "potential_alignment_available": potential_data is not None,
    }

    # Only apply corrections for charged defects
    if charge_state == 0:
        logger.info("Neutral defect (q=0) - no corrections needed")
        results["schemes"] = ["none"]
        results["correction_energies"] = [0.0]
        results["corrected_formation_energies"] = [raw_formation_energy]
        results["metadata"] = [{"note": "Neutral defect, no correction applied"}]
        results["statistics"] = {
            "mean_correction": 0.0,
            "std_correction": 0.0,
            "range_correction": 0.0,
            "mean_formation_energy": raw_formation_energy,
            "std_formation_energy": 0.0,
        }
        return results

    # Apply each correction scheme
    for scheme_name in correction_schemes:
        logger.info(f"Applying {scheme_name} correction...")

        try:
            if scheme_name.lower() in ["lany-zunger", "lz"]:
                correction_scheme = LanyZungerCorrection(epsilon_static=epsilon_static)
            elif scheme_name.lower() in [
                "makov-payne",
                "mp",
            ] or scheme_name.lower() in [
                "makov-payne-quadrupole",
                "mp-quad",
                "makov-payne-full",
            ]:
                correction_scheme = MakovPayneCorrection(epsilon_static=epsilon_static)
            elif scheme_name.lower() in ["freysoldt", "frey", "fnv"]:
                correction_scheme = FreysoldtCorrection(epsilon_static=epsilon_static)
            elif scheme_name.lower() in ["kumagai", "kumagai-oba", "ko"]:
                correction_scheme = KumagaiCorrection(epsilon_static=epsilon_static)
            elif scheme_name.lower() in ["slab2d", "slab-2d"]:
                from atomate2.siesta.flows.defects.corrections import Slab2DCorrection

                correction_scheme = Slab2DCorrection(
                    epsilon_parallel=epsilon_parallel or epsilon_static,
                    epsilon_perpendicular=epsilon_perpendicular or epsilon_static,
                )
            else:
                logger.warning(f"Unknown correction scheme: {scheme_name}. Skipping.")
                continue

            # Calculate correction
            # Pass potential_data and vt_file_paths (for plotting) if available
            correction_result = correction_scheme.calculate_correction(
                defect_structure=defect_structure,
                host_structure=host_structure,
                charge_state=charge_state,
                defect_energy=defect_energy,
                host_energy=host_energy,
                defect_site=defect_site,
                potential_data=potential_data,  # Will be None if .VT files not found
                vt_file_paths=vt_file_paths,  # For automatic plot generation
            )

            # Store results
            # Use original scheme name (not correction_result.scheme_name)
            # to differentiate variants
            results["schemes"].append(scheme_name)
            results["correction_energies"].append(correction_result.correction_energy)
            results["corrected_formation_energies"].append(
                raw_formation_energy + correction_result.correction_energy
            )
            results["metadata"].append(correction_result.metadata)

            logger.info(
                f"{scheme_name}: E_corr = {correction_result.correction_energy:.4f} eV"
            )

        except Exception as e:
            logger.exception(
                f"Failed to apply {scheme_name} correction: {e}"  # noqa: TRY401
            )
            continue

    # Calculate statistics
    if len(results["correction_energies"]) > 0:
        corr_energies = np.array(results["correction_energies"])
        form_energies = np.array(results["corrected_formation_energies"])

        results["statistics"] = {
            "mean_correction": float(np.mean(corr_energies)),
            "std_correction": float(np.std(corr_energies)),
            "range_correction": float(np.ptp(corr_energies)),  # max - min
            "mean_formation_energy": float(np.mean(form_energies)),
            "std_formation_energy": float(np.std(form_energies)),
            "range_formation_energy": float(np.ptp(form_energies)),
        }
    else:
        results["statistics"] = {
            "mean_correction": 0.0,
            "std_correction": 0.0,
            "range_correction": 0.0,
            "mean_formation_energy": raw_formation_energy,
            "std_formation_energy": 0.0,
            "range_formation_energy": 0.0,
        }

    logger.info(
        f"Comparison complete: {len(results['schemes'])} schemes, "
        f"E_corr = {results['statistics']['mean_correction']:.4f} ± "
        f"{results['statistics']['std_correction']:.4f} eV"
    )

    return results


@job
def generate_comparison_summary(
    comparison_results: dict,
    charge_state: int,
) -> dict:
    """
    Generate human-readable summary of correction comparison.

    Parameters
    ----------
    comparison_results : dict
        Results from compare_all_corrections job
    charge_state : int
        Charge state of defect

    Returns
    -------
    dict
        Summary with:
        - "summary_text": formatted text summary
        - "recommendation": recommended correction scheme
        - "comparison_results": original comparison results
    """
    logger.info("Generating correction comparison summary...")

    schemes = comparison_results["schemes"]
    corr_energies = comparison_results["correction_energies"]
    form_energies = comparison_results["corrected_formation_energies"]
    raw_e_f = comparison_results["raw_formation_energy"]
    stats = comparison_results["statistics"]

    # Build summary text
    lines = []
    lines.append("=" * 70)
    lines.append("CORRECTION SCHEME COMPARISON SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Defect: {comparison_results.get('defect_type', 'unknown')}")
    lines.append(f"Species: {comparison_results.get('defect_species', 'unknown')}")
    lines.append(f"Charge state: {charge_state:+d}")

    # Add position if available
    defect_site = comparison_results.get("defect_site")
    if defect_site is not None:
        pos_str = f"[{defect_site[0]:.4f}, {defect_site[1]:.4f}, {defect_site[2]:.4f}]"
        lines.append(f"Position (frac): {pos_str}")

    # Show dielectric constants (anisotropic if Slab2D is used)
    epsilon_parallel = comparison_results.get("epsilon_parallel")
    epsilon_perpendicular = comparison_results.get("epsilon_perpendicular")
    epsilon_static = comparison_results["epsilon_static"]

    if epsilon_parallel is not None and epsilon_perpendicular is not None:
        # Anisotropic dielectric (for Slab2D)
        lines.append(f"Dielectric constant (isotropic schemes): {epsilon_static}")
        lines.append(f"Dielectric constant ε∥ (in-plane):      {epsilon_parallel}")
        lines.append(f"Dielectric constant ε⊥ (out-of-plane):  {epsilon_perpendicular}")
    else:
        # Isotropic dielectric
        lines.append(f"Dielectric constant: {epsilon_static}")
    lines.append("")

    if charge_state == 0:
        lines.append("NEUTRAL DEFECT (q=0):")
        lines.append("  No finite-size correction needed")
        lines.append(f"  Formation energy: {raw_e_f:.4f} eV")
    else:
        lines.append("RAW FORMATION ENERGY (uncorrected):")
        lines.append(f"  E_formation(raw) = {raw_e_f:.4f} eV")
        lines.append("")
        lines.append("CORRECTION ENERGIES:")
        lines.append(f"  {'Scheme':<30} {'E_corr (eV)':>12} {'E_formation (eV)':>12}")
        lines.append("  " + "-" * 56)

        for scheme, e_corr, e_f in zip(
            schemes, corr_energies, form_energies, strict=False
        ):
            # Add annotations for Makov-Payne variants
            annotation = ""
            if scheme.lower() in ["makov-payne", "mp"]:
                annotation = "  ← Basic (Q=0)"
            elif scheme.lower() in [
                "makov-payne-quadrupole",
                "mp-quad",
                "makov-payne-full",
            ]:
                annotation = "  ← With Q from .RHO"

            # Format line with proper alignment (60 chars before annotation)
            line = f"  {scheme:<30} {e_corr:>12.4f} {e_f:>12.4f}"
            if annotation:
                line = f"{line:<60}{annotation}"
            lines.append(line)

        # Add correction metadata details
        if comparison_results.get("metadata"):
            lines.append("")
            lines.append("CORRECTION DETAILS:")
            lines.append("  " + "-" * 66)
            for scheme, metadata in zip(
                schemes, comparison_results["metadata"], strict=False
            ):
                lines.append(f"\n  [{scheme}]")
                if metadata:
                    if "madelung_constant" in metadata:
                        alpha_M = metadata["madelung_constant"]  # noqa: N806
                        lines.append(f"    Madelung constant:   {alpha_M:.4f}")
                        if metadata.get("madelung_citation"):
                            lines.append(f"      ({metadata['madelung_citation']})")
                    if "characteristic_length_angstrom" in metadata:
                        lines.append(
                            "    Characteristic length: "
                            f"{metadata['characteristic_length_angstrom']:.2f} Å"
                        )
                    if "gaussian_width_angstrom" in metadata:
                        lines.append(
                            "    Gaussian width (σ):    "  # noqa: RUF001
                            f"{metadata['gaussian_width_angstrom']:.2f} Å"
                        )
                    if "lattice_term_eV" in metadata:
                        lines.append(
                            "    Lattice term:        "
                            f"{metadata['lattice_term_eV']:.4f} eV"
                        )
                    if "alignment_energy_eV" in metadata:
                        lines.append(
                            "    Alignment term:      "
                            f"{metadata['alignment_energy_eV']:.4f} eV"
                        )
                    if "quadrupole_term_eV" in metadata:
                        lines.append(
                            "    Quadrupole term:     "
                            f"{metadata['quadrupole_term_eV']:.4f} eV"
                        )

        lines.append("")
        lines.append("STATISTICAL ANALYSIS:")
        lines.append(f"  Mean correction:     {stats['mean_correction']:>8.4f} eV")
        lines.append(f"  Std correction:      {stats['std_correction']:>8.4f} eV")
        lines.append(f"  Range (max-min):     {stats['range_correction']:>8.4f} eV")
        lines.append("")
        lines.append(
            f"  Mean formation E:    {stats['mean_formation_energy']:>8.4f} eV"
        )
        lines.append(f"  Std formation E:     {stats['std_formation_energy']:>8.4f} eV")

        # Recommendation
        lines.append("")
        lines.append("RECOMMENDATION:")

        if len(schemes) == 1:
            lines.append(f"  Use {schemes[0]} correction")
            lines.append(f"  E_formation = {form_energies[0]:.4f} eV")
            recommendation = schemes[0]
        # Recommend based on spread
        elif stats["std_correction"] < 0.1:
            lines.append("  ✓ Good agreement between schemes")
            lines.append(
                f"  Recommended: {stats['mean_formation_energy']:.4f} ± "
                f"{stats['std_formation_energy']:.4f} eV"
            )
            recommendation = "average"
        else:
            lines.append("  ⚠ Large spread between schemes")
            lines.append("  Consider larger supercell for better convergence")
            # Recommend the most conservative (largest correction)
            max_idx = corr_energies.index(max(corr_energies))
            recommendation = schemes[max_idx]
            lines.append(f"  Conservative choice: {recommendation}")
            lines.append(f"  E_formation = {form_energies[max_idx]:.4f} eV")

    lines.append("")
    lines.append("=" * 70)

    summary_text = "\n".join(lines)

    # Print to logger
    for line in lines:
        logger.info(line)

    # Write to file
    from pathlib import Path

    from atomate2.siesta.utils.text_output import get_standard_footer

    summary_file = Path("correction_comparison.txt")
    with open(summary_file, "w") as f:
        f.write(summary_text)
        f.write("\n")

        # Add standard footer
        std_correction = comparison_results["statistics"].get("std_correction", 0.0)
        f.write(
            get_standard_footer(
                width=70,
                additional_info={
                    "Defect": f"{comparison_results.get('defect_type', 'unknown')}",
                    "Schemes": f"{len(schemes)}",
                    "Spread": f"{std_correction:.4f} eV",
                },
            )
        )

    logger.info(f"Comparison summary written to {summary_file}")

    return {
        "summary_text": summary_text,
        "recommendation": recommendation if charge_state != 0 else "none",
        "comparison_results": comparison_results,
    }
