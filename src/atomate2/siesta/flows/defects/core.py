"""Core defect workflow - minimal implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from jobflow import Flow, Maker, job

from atomate2.siesta.flows.defects.corrections import (
    FreysoldtCorrection,
    KumagaiCorrection,
    LanyZungerCorrection,
    MakovPayneCorrection,
    Slab2DCorrection,
)
from atomate2.siesta.flows.defects.makers import DefectRelaxMaker, DefectStaticMaker
from atomate2.siesta.flows.defects.schemas import DefectDocument

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


def _enable_modules(input_gen, modules: list[str]) -> None:
    """Add modules to an input generator's ``enabled_modules`` list (in-place).

    Parameters
    ----------
    input_gen : SiestaInputGenerator
        The input set generator to modify.
    modules : list[str]
        Module names to enable (e.g. ``["grids_advanced", "dos_bands"]``).
    """
    existing = list(input_gen.enabled_modules or [])
    input_gen.enabled_modules = list(set(existing + modules))


@dataclass
class DefectFlowMaker(Maker):
    """
    Minimal defect workflow (Phase 1).

    This is a minimal implementation that demonstrates the basic workflow:
    1. Relax the defect supercell structure
    2. Calculate the host supercell energy (static calculation)
    3. Apply finite-size correction (Lany-Zunger)
    4. Create DefectDocument with results

    Parameters
    ----------
    name : str
        Name of the flow (default: "Defect Calculation")
    defect_relax_maker : DefectRelaxMaker
        Maker for defect relaxation
    host_static_maker : DefectStaticMaker
        Maker for host static calculation
    epsilon_static : float
        Static dielectric constant for correction (isotropic value)
    epsilon_parallel : float, optional
        In-plane dielectric constant for 2D slab corrections (ε∥)
        Only used when correction_scheme="slab2d"
        If not provided, uses epsilon_static
    epsilon_perpendicular : float, optional
        Out-of-plane dielectric constant for 2D slab corrections (ε⊥)
        Only used when correction_scheme="slab2d"
        If not provided, uses epsilon_static
    correction_scheme : str
        Finite-size correction scheme to use:
        - "lany-zunger" (default): Simple isotropic correction
        - "makov-payne": Basic Makov-Payne (monopole only, Q=0)
        - "makov-payne-quadrupole": Full Makov-Payne (monopole + quadrupole from .RHO)
        - "freysoldt": Anisotropic with potential alignment
        - "kumagai": Atomic-site sampling (SOTA for relaxed systems)
        - "slab2d": 2D slab correction with anisotropic dielectric
        When using freysoldt/kumagai/slab2d, .VT files are auto-enabled
        When using makov-payne-quadrupole, .RHO files are auto-enabled
    defect_type : str
        Type of defect (vacancy, substitutional, interstitial)
    charge_state : int or list[int]
        Charge state(s) of the defect (e.g., +2, 0, -1).
        If int: creates a single Flow for that charge state.
        If list: creates multiple Flows, one for each charge state.
    use_ghost_atoms : bool
        If True (default), automatically use ghost atoms for vacancies.
        Ghost atoms are essential for SIESTA to maintain proper basis
        set coverage at vacancy sites. Recommended: True
    dry_run : bool
        If True, enable dry-run mode for all calculations (default: False)
    skip_relax : bool
        If True, skip defect relaxation and use input structure directly.
        Useful for testing corrections on pre-relaxed or unrelaxed structures.
        Default: False (perform relaxation)
    chemical_potentials : dict, optional
        Chemical potentials for defect species (in eV). Format:
        {"O": -5.0, "Mg": -2.0, ...}
        For vacancy: μ_removed is ADDED to formation energy
        For substitution: μ_removed - μ_added
        If not provided, chemical potential term is omitted (will give wrong E_formation!)
    auto_calculate_chemical_potentials : bool
        If True, automatically calculate chemical potentials from reference
        structures (O2, H2, N2 molecules; bulk metals). Default: False
        If True, ignores manual chemical_potentials dict
    include_bandstructure : bool
        If True, enable band structure output in both the defect and host
        calculations. Injects WriteBands, BandLinesScale, and BandLines FDF
        parameters. Auto-generates k-path from each structure's symmetry
        unless custom BandLines are provided via bands_fdf_params (applied
        to the defect calculation only; the host always uses an auto-generated
        path from host_structure). Default: False
    include_pdos : bool
        If True, enable projected density of states output in both the defect
        and host calculations. Injects ProjectedDensityOfStates FDF block.
        Default: False
    bands_fdf_params : dict, optional
        Override default band structure FDF parameters. Supported keys:
        - "n_interpolations": int (default 20) - k-points between high-sym points
        - "%block BandLines": list - custom band path (skips auto-generation)
        - "BandLinesScale": str - override scale (default "ReciprocalLatticeVectors")
        - Any other valid SIESTA FDF band parameter
    pdos_fdf_params : dict, optional
        Override default PDOS FDF parameters. Default block:
        ``%block ProjectedDensityOfStates = ["EF -15.0 15.0 0.05 1000 eV"]``
        EF means energies are relative to Fermi level.
        Can override with custom energy range or add PDOS k-grid.

    Examples
    --------
    >>> from atomate2.siesta.flows.defects.core import DefectFlowMaker
    >>> from pymatgen.core import Structure
    >>> host = Structure.from_file("host_supercell.cif")
    >>> defect = Structure.from_file("defect_supercell.cif")
    >>> flow_maker = DefectFlowMaker(
    ...     epsilon_static=10.0, defect_type="vacancy", charge_state=2
    ... )
    >>> flow = flow_maker.make(defect, host)
    """

    name: str = "Defect Calculation"
    defect_relax_maker: DefectRelaxMaker = field(
        default_factory=DefectRelaxMaker.defect_relax
    )
    host_static_maker: DefectStaticMaker = field(
        default_factory=DefectStaticMaker.defect_scf
    )
    epsilon_static: float = 10.0
    epsilon_parallel: float | None = None
    epsilon_perpendicular: float | None = None
    correction_scheme: str = "lany-zunger"
    defect_type: str = "vacancy"
    charge_state: int | list[int] = 0
    use_ghost_atoms: bool = True
    dry_run: bool = False
    skip_relax: bool = False
    chemical_potentials: dict[str, float] | None = None
    auto_calculate_chemical_potentials: bool = False
    include_bandstructure: bool = False
    include_pdos: bool = False
    bands_fdf_params: dict[str, Any] | None = None
    pdos_fdf_params: dict[str, Any] | None = None

    def make(
        self,
        defect_structure: Structure,
        host_structure: Structure,
        defect_site: list[float] | None = None,
        defect_species: str | None = None,
        removed_species: str | None = None,
        host_task_doc=None,
    ) -> Flow | list[Flow]:
        """
        Create the defect calculation flow.

        Parameters
        ----------
        defect_structure : Structure
            Defect supercell structure (unrelaxed or pre-relaxed).
            For vacancies, use create_vacancy_with_ghost() to create
            the defect structure with ghost atoms (recommended for SIESTA).
        host_structure : Structure
            Pristine host supercell structure (same size as defect)
        defect_site : list[float], optional
            Fractional coordinates [x, y, z] of defect site
        defect_species : str, optional
            Species of the defect
            For vacancy/interstitial: the removed/added species
            For substitution: the added species (dopant)
        removed_species : str, optional
            For substitution defects only: the removed (host) species
            Used to calculate μ_removed - μ_added correctly
        host_task_doc : optional
            Pre-calculated host task document (from a previous calculation).
            If provided, skips host calculation and uses this energy directly.
            Useful when running multiple defects with the same host supercell.

        Returns
        -------
        Flow or list[Flow]
            If charge_state is int: Single Flow object with defect calculation jobs.
            If charge_state is list: List of Flow objects, one per charge state.

        Notes
        -----
        For vacancy defects with SIESTA, it is strongly recommended to use
        ghost atoms instead of complete atom removal. Use the helper function:

        >>> from atomate2.siesta.flows.defects.generation import (
        ...     create_vacancy_with_ghost,
        ... )
        >>> defect_structure = create_vacancy_with_ghost(host_structure, site_index=10)

        Ghost atoms maintain proper basis set coverage and grid sampling at
        the vacancy site, which is critical for convergence in SIESTA.
        """
        # Handle multiple charge states
        if isinstance(self.charge_state, (list, tuple)):
            logger.info(
                f"Multiple charge states detected: {self.charge_state}. "
                f"Creating {len(self.charge_state)} separate flows."
            )
            flows = []
            for q in self.charge_state:
                # Create a copy of the maker with single charge state
                single_maker = DefectFlowMaker(
                    name=f"{self.name}_q_{q:+d}",
                    defect_relax_maker=self.defect_relax_maker,
                    host_static_maker=self.host_static_maker,
                    epsilon_static=self.epsilon_static,
                    epsilon_parallel=self.epsilon_parallel,
                    epsilon_perpendicular=self.epsilon_perpendicular,
                    correction_scheme=self.correction_scheme,
                    defect_type=self.defect_type,
                    charge_state=q,  # Single charge state
                    use_ghost_atoms=self.use_ghost_atoms,
                    dry_run=self.dry_run,
                    skip_relax=self.skip_relax,
                    chemical_potentials=self.chemical_potentials,
                    auto_calculate_chemical_potentials=self.auto_calculate_chemical_potentials,
                    include_bandstructure=self.include_bandstructure,
                    include_pdos=self.include_pdos,
                    bands_fdf_params=self.bands_fdf_params,
                    pdos_fdf_params=self.pdos_fdf_params,
                )
                # Create flow for this charge state
                flow = single_maker.make(
                    defect_structure=defect_structure,
                    host_structure=host_structure,
                    defect_site=defect_site,
                    defect_species=defect_species,
                    removed_species=removed_species,
                )
                flows.append(flow)
            return flows

        # Single charge state - continue with normal flow creation

        # Validate chemical potentials are provided when needed
        if defect_species and not self.auto_calculate_chemical_potentials:
            if (
                not self.chemical_potentials
                or (
                    self.defect_type == "substitution"
                    and removed_species
                    and (
                        defect_species not in self.chemical_potentials
                        or removed_species not in self.chemical_potentials
                    )
                )
                or (
                    self.defect_type != "substitution"
                    and defect_species not in (self.chemical_potentials or {})
                )
            ):
                species_str = (
                    f"{defect_species}_{removed_species}"
                    if removed_species
                    else defect_species
                )
                raise ValueError(
                    f"Chemical potentials required for {species_str} defect but not provided!\n"
                    f"Formation energy calculation requires chemical potentials.\n\n"
                    f"Either:\n"
                    f"  1. Provide chemical_potentials dict with NUMERIC values (eV):\n"
                    f"     chemical_potentials={{'O': -5.12, 'Mg': -1.51}}\n"
                    f"  2. Enable automatic calculation:\n"
                    f"     auto_calculate_chemical_potentials=True"
                )

        # Create safe job name (no spaces)
        safe_name = self.name.replace(" ", "_").replace("+", "p").replace("-", "m")

        # Set dry_run mode if requested
        if self.dry_run:
            self.defect_relax_maker.dry_run = True
            self.host_static_maker.dry_run = True

        # Validate ghost atoms for vacancies
        if self.defect_type == "vacancy" and self.use_ghost_atoms:
            # Check if defect_structure has ghost_tags property
            if "ghost_tags" not in defect_structure.site_properties:
                logger.warning(
                    "use_ghost_atoms=True but defect_structure does not have "
                    "'ghost_tags' site property. For SIESTA vacancy calculations, "
                    "it is strongly recommended to use create_vacancy_with_ghost() "
                    "to create the defect structure with ghost atoms."
                )
            elif not any(defect_structure.site_properties.get("ghost_tags", [])):
                logger.warning(
                    "use_ghost_atoms=True but no ghost atoms found in defect_structure. "
                    "For SIESTA vacancy calculations, use create_vacancy_with_ghost() "
                    "to properly create ghost atoms at vacancy sites."
                )

        # Auto-enable grid file writing based on correction scheme
        # Handle both string and correction object
        if isinstance(self.correction_scheme, str):
            scheme_lower = self.correction_scheme.lower()
        else:
            # Correction object - get name attribute
            scheme_lower = getattr(self.correction_scheme, "name", "none").lower()

        # Enable .VT files for Freysoldt/Kumagai (potential alignment)
        if scheme_lower in [
            "freysoldt",
            "frey",
            "fnv",
            "kumagai",
            "kumagai-oba",
            "ko",
            "slab2d",
            "slab-2d",
        ]:
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
                "SaveElectrostaticPotential": True,
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
                "SaveElectrostaticPotential": True,
            }

            logger.info(
                f"Auto-enabled .VT file writing for {self.correction_scheme} correction"
            )

        # Enable .RHO files for Makov-Payne-Quadrupole (quadrupole calculation from density)
        # Note: basic "makov-payne" does NOT need .RHO (uses Q=0)
        if scheme_lower in ["makov-payne-quadrupole", "mp-quad", "makov-payne-full"]:
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
                "SaveRho": True,
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
                "SaveRho": True,
            }

            logger.info(
                f"Auto-enabled .RHO file writing for {self.correction_scheme} correction (quadrupole calculation from density)"
            )

        # Enable band structure / PDOS output in BOTH defect and host calculations.
        # Params are routed through regular user_params so the dos_bands
        # dataclass module handles them (preserving header and defaults).
        if self.include_bandstructure or self.include_pdos:
            # --- Configure defect maker ---
            defect_maker = (
                self.host_static_maker if self.skip_relax else self.defect_relax_maker
            )
            current_params = defect_maker.input_set_generator.user_params or {}
            current_params.pop("enabled_modules", None)
            _enable_modules(defect_maker.input_set_generator, ["dos_bands"])

            # --- Configure host maker (same bands/PDOS for host material) ---
            if not self.skip_relax:
                # In skip_relax mode, defect_maker IS host_static_maker, so
                # we only need a separate host config when skip_relax=False.
                host_params = (
                    self.host_static_maker.input_set_generator.user_params or {}
                )
                host_params.pop("enabled_modules", None)
                _enable_modules(
                    self.host_static_maker.input_set_generator, ["dos_bands"]
                )
            else:
                host_params = None

            if self.include_bandstructure:
                user_bands = dict(self.bands_fdf_params or {})
                band_lines_scale = user_bands.pop(
                    "BandLinesScale", "ReciprocalLatticeVectors"
                )

                current_params["WriteBands"] = "true"
                current_params["BandLinesScale"] = band_lines_scale

                if "%block BandLines" in user_bands:
                    # User provided custom band path - use it directly
                    custom_path = user_bands.pop("%block BandLines")
                    current_params["%block BandLines"] = custom_path
                    logger.info("Using user-provided custom band path for defect")
                else:
                    # Auto-generate k-path from defect structure symmetry
                    from atomate2.siesta.sets.bands import band_paymatgen_to_siesta

                    n_interp = user_bands.pop("n_interpolations", 20)
                    band_lines = band_paymatgen_to_siesta(
                        defect_structure, interpolations=[n_interp]
                    )
                    current_params["%block BandLines"] = band_lines
                    logger.info(
                        f"Auto-generated defect band path "
                        f"({len(band_lines)} points, interp={n_interp})"
                    )

                # Apply remaining user overrides to defect
                for k, v in user_bands.items():
                    current_params[k] = v

                # Host band structure (separate k-path from host symmetry)
                if host_params is not None:
                    from atomate2.siesta.sets.bands import band_paymatgen_to_siesta

                    host_params["WriteBands"] = "true"
                    host_params["BandLinesScale"] = band_lines_scale

                    # Always auto-generate k-path for host from host_structure
                    host_n_interp = (self.bands_fdf_params or {}).get(
                        "n_interpolations", 20
                    )
                    host_band_lines = band_paymatgen_to_siesta(
                        host_structure, interpolations=[host_n_interp]
                    )
                    host_params["%block BandLines"] = host_band_lines
                    logger.info(
                        f"Auto-generated host band path "
                        f"({len(host_band_lines)} points, "
                        f"interp={host_n_interp})"
                    )

            if self.include_pdos:
                user_pdos = dict(self.pdos_fdf_params or {})
                if "%block ProjectedDensityOfStates" not in user_pdos:
                    pdos_block = ["EF -15.0 15.0 0.05 1000 eV"]
                else:
                    pdos_block = user_pdos.pop("%block ProjectedDensityOfStates")

                current_params["%block ProjectedDensityOfStates"] = pdos_block

                # Handle PDOS k-grid block
                pdos_kgrid = None
                if "%block PDOS.kgrid.MonkhorstPack" in user_pdos:
                    pdos_kgrid = user_pdos.pop("%block PDOS.kgrid.MonkhorstPack")
                    current_params["%block PDOS.kgrid.MonkhorstPack"] = pdos_kgrid

                # Apply remaining user PDOS overrides to defect
                for k, v in user_pdos.items():
                    current_params[k] = v
                logger.info("Auto-enabled PDOS output for defect calculation")

                # Host PDOS (same energy range and k-grid)
                if host_params is not None:
                    host_params["%block ProjectedDensityOfStates"] = pdos_block
                    if pdos_kgrid is not None:
                        host_params["%block PDOS.kgrid.MonkhorstPack"] = pdos_kgrid
                    logger.info("Auto-enabled PDOS output for host calculation")

            defect_maker.input_set_generator.user_params = current_params
            if host_params is not None:
                self.host_static_maker.input_set_generator.user_params = host_params

        # Enable spin polarization for ALL defect calculations.
        # Defects (charged or neutral) commonly introduce unpaired electrons,
        # so spin-polarized calculations are essential for correct energies.
        # The host calculation is also run spin-polarized for consistency
        # (host and defect energies must use the same Hamiltonian).
        for maker in (self.defect_relax_maker, self.host_static_maker):
            mp = maker.input_set_generator.user_params or {}
            if "Spin" not in mp:
                mp = {**mp, "Spin": "polarized"}
                mp.pop("enabled_modules", None)
                _enable_modules(maker.input_set_generator, ["spin"])
                maker.input_set_generator.user_params = mp
        logger.info("Auto-enabled Spin = polarized for defect and host calculations")

        # Job 1: Defect calculation (relax or static depending on skip_relax)
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
                f"Setting NetCharge = {self.charge_state} for charged defect calculation"
            )

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
        # Skip if host_task_doc is already provided (shared host calculation)
        if host_task_doc is not None:
            host_static_job = None
            logger.info("Using pre-calculated host energy (shared host calculation)")
        else:
            host_static_job = self.host_static_maker.make(host_structure)
            host_static_job.name = f"{safe_name}_host_static"

        # Job 2.5: Calculate chemical potential if requested
        if self.auto_calculate_chemical_potentials:
            ref_jobs = []
            chemical_potentials = {}

            # For substitution: calculate both removed and added species
            if (
                self.defect_type == "substitution"
                and removed_species
                and defect_species
            ):
                logger.info(
                    f"Auto-calculating chemical potentials for {defect_species}_{removed_species} substitution..."
                )

                # Calculate μ for removed species (e.g., Mg)
                ref_structure_removed, n_atoms_removed = get_reference_structure(
                    removed_species
                )
                ref_job_removed = self.host_static_maker.make(ref_structure_removed)
                ref_job_removed.name = f"{safe_name}_ref_{removed_species}"
                mu_job_removed = extract_chemical_potential(
                    task_doc=ref_job_removed.output,
                    species=removed_species,
                    n_atoms=n_atoms_removed,
                )
                mu_job_removed.name = f"{safe_name}_mu_{removed_species}"

                # Calculate μ for added species (e.g., Li)
                ref_structure_added, n_atoms_added = get_reference_structure(
                    defect_species
                )
                ref_job_added = self.host_static_maker.make(ref_structure_added)
                ref_job_added.name = f"{safe_name}_ref_{defect_species}"
                mu_job_added = extract_chemical_potential(
                    task_doc=ref_job_added.output,
                    species=defect_species,
                    n_atoms=n_atoms_added,
                )
                mu_job_added.name = f"{safe_name}_mu_{defect_species}"

                chemical_potentials = {
                    removed_species: mu_job_removed.output,
                    defect_species: mu_job_added.output,
                }
                ref_jobs = [
                    ref_job_removed,
                    mu_job_removed,
                    ref_job_added,
                    mu_job_added,
                ]

            elif defect_species:
                # For vacancy/interstitial: only one species
                logger.info(
                    f"Auto-calculating chemical potential for {defect_species}..."
                )
                ref_structure, n_atoms = get_reference_structure(defect_species)
                ref_static_job = self.host_static_maker.make(ref_structure)
                ref_static_job.name = f"{safe_name}_ref_{defect_species}"
                mu_extract_job = extract_chemical_potential(
                    task_doc=ref_static_job.output,
                    species=defect_species,
                    n_atoms=n_atoms,
                )
                mu_extract_job.name = f"{safe_name}_mu_{defect_species}"
                chemical_potentials = {defect_species: mu_extract_job.output}
                ref_jobs = [ref_static_job, mu_extract_job]
            else:
                logger.warning(
                    "auto_calculate_chemical_potentials=True but defect_species not provided. "
                    "Cannot calculate chemical potential!"
                )
                chemical_potentials = self.chemical_potentials or {}
        else:
            chemical_potentials = self.chemical_potentials or {}
            ref_jobs = []

        # Determine host task doc source (pre-calculated or from job)
        host_output = (
            host_task_doc if host_task_doc is not None else host_static_job.output
        )

        # Job 3: Apply correction and create DefectDocument
        finalize_job = finalize_defect_calculation(
            defect_task_doc=defect_relax_job.output,
            host_task_doc=host_output,
            host_structure=host_structure,
            epsilon_static=self.epsilon_static,
            epsilon_parallel=self.epsilon_parallel,
            epsilon_perpendicular=self.epsilon_perpendicular,
            correction_scheme_name=self.correction_scheme,
            defect_type=self.defect_type,
            charge_state=self.charge_state,
            defect_site=defect_site,
            defect_species=defect_species,
            removed_species=removed_species,
            chemical_potentials=chemical_potentials,  # Use auto-calculated or manual
        )
        finalize_job.name = f"{safe_name}_finalize"

        # Job 4: Generate plots
        plot_jobs = []

        # Potential alignment plot (for potential-based corrections)
        if scheme_lower in [
            "freysoldt",
            "frey",
            "fnv",
            "kumagai",
            "kumagai-oba",
            "ko",
            "slab2d",
            "slab-2d",
        ]:
            vt_plot_job = generate_potential_plot(
                defect_task_doc=defect_relax_job.output,
                host_task_doc=host_output,
                output_name=f"{safe_name}_potential_alignment.png",
            )
            vt_plot_job.name = f"{safe_name}_vt_plot"
            plot_jobs.append(vt_plot_job)

        # Dielectric profile plot (for Slab2D)
        if scheme_lower in ["slab2d", "slab-2d"]:
            profile_plot_job = generate_dielectric_profile_plot(
                defect_task_doc=defect_relax_job.output,
                epsilon_parallel=self.epsilon_parallel or self.epsilon_static,
                epsilon_perpendicular=self.epsilon_perpendicular or self.epsilon_static,
                output_name=f"{safe_name}_dielectric_profile.png",
            )
            profile_plot_job.name = f"{safe_name}_profile_plot"
            plot_jobs.append(profile_plot_job)

        # Density plot (for ALL corrections - real RHO or Gaussian approximation)
        if self.charge_state != 0:
            rho_plot_job = generate_density_plot(
                defect_task_doc=defect_relax_job.output,
                host_task_doc=host_output,
                output_name=f"{safe_name}_density_difference.png",
            )
            rho_plot_job.name = f"{safe_name}_rho_plot"
            plot_jobs.append(rho_plot_job)

        # Radial distribution plot (for ALL charged defects)
        if self.charge_state != 0:
            radial_plot_job = generate_radial_distribution_plot(
                defect_task_doc=defect_relax_job.output,
                host_task_doc=host_output,
                defect_site_frac=defect_site,
                charge_state=self.charge_state,
                output_name=f"{safe_name}_radial_distribution.png",
            )
            radial_plot_job.name = f"{safe_name}_radial_plot"
            plot_jobs.append(radial_plot_job)

        # Create flow - exclude host_static_job if using pre-calculated host
        base_jobs = [defect_relax_job]
        if host_static_job is not None:
            base_jobs.append(host_static_job)
        jobs = base_jobs + ref_jobs + [finalize_job] + plot_jobs

        flow = Flow(
            jobs,
            output=finalize_job.output,
            name=self.name,
        )

        return flow

    @classmethod
    def from_pristine_structure(
        cls,
        structure: Structure,
        defect_type: str = "vacancy",
        species: str | list[str] | None = None,
        dopants: str | list[str] | None = None,
        supercell_matrix: list[list[int]] | None = None,
        charge_states: list[int] | None = None,
        use_symmetry: bool = True,
        tier_preset: str | None = None,
        use_custodian: bool = False,
        **kwargs,
    ) -> Flow | list[Flow]:
        """
        Generate defect flows from pristine structure automatically.

        This classmethod provides a one-liner API for generating multiple
        defect calculations with symmetry reduction. It uses the automated
        defect generators to find all symmetry-unique defect sites and
        creates flows for each.

        Parameters
        ----------
        structure : Structure
            Pristine structure (unit cell or supercell)
        defect_type : str
            Type of defect: "vacancy", "substitution", or "interstitial"
            Default: "vacancy"
        species : str or list[str], optional
            For vacancy/substitution: species to remove/replace
            For interstitial: species to insert
            If None, generate all possible defects
        dopants : str or list[str], optional
            For substitution only: dopant species to substitute
            For antisites, use dopants=None to generate all pairs
        supercell_matrix : list[list[int]], optional
            Supercell transformation matrix (3×3)
            If None, use input structure as-is
        charge_states : list[int], optional
            Charge states to generate for each defect
            If None, only neutral defects (q=0)
        use_symmetry : bool
            If True (default), use symmetry to find unique defect sites.
            If False, generate defects at ALL sites (no symmetry reduction).
            Useful for slabs, specific site selection, or testing.
        tier_preset : str, optional
            Tier preset name to apply to makers (e.g., "defect_dirty", "defect_standard")
            If provided, automatically creates makers with tier preset applied
            If None, use default makers or provide custom makers via kwargs
        use_custodian : bool
            If True, enable custodian error handling for all calculations.
            Default: False
        **kwargs
            Additional arguments passed to DefectFlowMaker constructor
            (e.g., epsilon_static, use_ghost_atoms, dry_run, defect_relax_maker, etc.)

        Returns
        -------
        Flow or list[Flow]
            - If multiple defects (>1): Returns a parent Flow containing all
              individual defect flows plus a combined summary job that writes
              "all_defects_summary.txt" with information for all defects
            - If single defect: Returns list with one Flow

        Examples
        --------
        Generate all vacancies in MgO:

        >>> from pymatgen.core import Structure
        >>> from atomate2.siesta.flows.defects import DefectFlowMaker
        >>> mgo = Structure.from_file("MgO.cif")
        >>> flows = DefectFlowMaker.from_pristine_structure(
        ...     mgo,
        ...     defect_type="vacancy",
        ...     supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        ...     charge_states=[0, +2],
        ...     epsilon_static=9.8,
        ... )
        >>> # Returns flows for V_Mg and V_O, each with q=0 and q=+2

        Generate Li dopant on Mg sites with tier preset:

        >>> flows = DefectFlowMaker.from_pristine_structure(
        ...     mgo,
        ...     defect_type="substitution",
        ...     species="Mg",
        ...     dopants="Li",
        ...     charge_states=[-1, 0],
        ...     tier_preset="defect_dirty",  # Apply tier preset
        ... )

        Generate all antisites:

        >>> flows = DefectFlowMaker.from_pristine_structure(
        ...     mgo,
        ...     defect_type="substitution",
        ...     dopants=None,  # Triggers antisite generation
        ... )

        Generate interstitials:

        >>> flows = DefectFlowMaker.from_pristine_structure(
        ...     mgo,
        ...     defect_type="interstitial",
        ...     species=["Li", "Na"],
        ...     charge_states=[+1],
        ... )

        Generate ALL vacancies (no symmetry reduction):

        >>> # For MoS2: returns 4 S vacancies instead of 1
        >>> flows = DefectFlowMaker.from_pristine_structure(
        ...     mos2,
        ...     defect_type="vacancy",
        ...     species="S",
        ...     use_symmetry=False,  # All sites, not just unique ones
        ... )

        Notes
        -----
        This method dramatically reduces code required for defect studies.
        Instead of manually creating each defect structure, you can generate
        all relevant defects with symmetry reduction in a single call.

        **Code Reduction**: Significantly fewer lines for typical defect studies
        """  # noqa: RUF002
        from atomate2.siesta.flows.defects.generation import (
            SiestaInterstitialGenerator,
            SiestaSubstitutionGenerator,
            SiestaVacancyGenerator,
        )

        # Apply tier preset and/or custodian if provided
        if tier_preset is not None or use_custodian:
            from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
            from atomate2.siesta.sets.tiers import apply_tier_preset

            if "defect_relax_maker" not in kwargs:
                relax_maker = RelaxMaker.fixed_cell_relaxation(
                    use_custodian=use_custodian
                )
                if tier_preset is not None:
                    relax_maker = apply_tier_preset(relax_maker, tier_preset)
                kwargs["defect_relax_maker"] = relax_maker
            elif use_custodian:
                # User provided custom maker but also requested custodian —
                # rebuild via dataclasses.replace() so __post_init__ runs and
                # properly initializes custodian_handlers (e.g.,
                # DEFAULT_RELAXATION_HANDLERS for RelaxMaker). This also
                # produces a clean serializable dataclass for jobflow-remote.
                # Only set use_custodian=True; preserve any user-provided
                # custodian_handlers and custodian_max_errors.
                import dataclasses as dc

                relax_maker = kwargs["defect_relax_maker"]
                if not getattr(relax_maker, "use_custodian", False):
                    logger.info(
                        "Enabling custodian on user-provided defect_relax_maker "
                        "(use_custodian=True was requested)"
                    )
                    kwargs["defect_relax_maker"] = dc.replace(
                        relax_maker, use_custodian=True
                    )

            if "host_static_maker" not in kwargs:
                static_maker = StaticMaker(use_custodian=use_custodian)
                if tier_preset is not None:
                    static_maker = apply_tier_preset(static_maker, tier_preset)
                kwargs["host_static_maker"] = static_maker
            elif use_custodian:
                # Same for host_static_maker — rebuild for clean serialization
                import dataclasses as dc

                static_maker = kwargs["host_static_maker"]
                if not getattr(static_maker, "use_custodian", False):
                    logger.info(
                        "Enabling custodian on user-provided host_static_maker "
                        "(use_custodian=True was requested)"
                    )
                    kwargs["host_static_maker"] = dc.replace(
                        static_maker, use_custodian=True
                    )

        # Generate defects based on type
        if defect_type == "vacancy":
            generator = SiestaVacancyGenerator(structure, use_symmetry=use_symmetry)
            defects = generator.generate_defects(
                species=species,
                supercell_matrix=supercell_matrix,
                charge_states=charge_states,
            )
        elif defect_type == "substitution":
            generator = SiestaSubstitutionGenerator(
                structure, use_symmetry=use_symmetry
            )
            if dopants is None:
                # Generate antisites
                defects = generator.generate_antisites(
                    supercell_matrix=supercell_matrix,
                    charge_states=charge_states,
                )
            else:
                # Generate substitutions with specified dopants
                if species is None:
                    raise ValueError(
                        "For substitution defects, 'species' (host species to replace) must be specified"
                    )
                defects = generator.generate_defects(
                    species=species,
                    dopants=dopants,
                    supercell_matrix=supercell_matrix,
                    charge_states=charge_states,
                )
        elif defect_type == "interstitial":
            generator = SiestaInterstitialGenerator(
                structure, use_symmetry=use_symmetry
            )
            if species is None:
                raise ValueError(
                    "For interstitial defects, 'species' (interstitial species) must be specified"
                )
            defects = generator.generate_defects(
                species=species,
                supercell_matrix=supercell_matrix,
                charge_states=charge_states,
            )
        else:
            raise ValueError(
                f"Unknown defect_type: {defect_type}. "
                f"Must be 'vacancy', 'substitution', or 'interstitial'"
            )

        # Create shared host calculation (calculate ONCE, share with all defects)
        shared_host_job = None
        if defects:
            # All defects share the same host supercell - calculate once
            host_structure = defects[0]["host_structure"]
            from atomate2.siesta.jobs.core import StaticMaker

            # Create host static maker (with custodian if requested)
            if "host_static_maker" in kwargs:
                host_maker = kwargs["host_static_maker"]
            else:
                host_maker = StaticMaker(use_custodian=use_custodian)
                if tier_preset is not None:
                    from atomate2.siesta.sets.tiers import apply_tier_preset

                    host_maker = apply_tier_preset(host_maker, tier_preset)

            # Propagate dry_run to shared host maker
            dry_run = kwargs.get("dry_run", False)
            if dry_run:
                host_maker.dry_run = True

            # Configure shared host with band structure / PDOS / spin params.
            # These must be applied HERE because the shared host job bypasses
            # DefectFlowMaker.make() where individual defect flows are configured.
            include_bandstructure = kwargs.get("include_bandstructure", False)
            include_pdos = kwargs.get("include_pdos", False)

            if include_bandstructure or include_pdos:
                host_params = host_maker.input_set_generator.user_params or {}
                host_params.pop("enabled_modules", None)
                _enable_modules(host_maker.input_set_generator, ["dos_bands"])

                if include_bandstructure:
                    from atomate2.siesta.sets.bands import band_paymatgen_to_siesta

                    bands_fdf = dict(kwargs.get("bands_fdf_params") or {})
                    band_lines_scale = bands_fdf.pop(
                        "BandLinesScale", "ReciprocalLatticeVectors"
                    )
                    host_params["WriteBands"] = "true"
                    host_params["BandLinesScale"] = band_lines_scale

                    if "%block BandLines" in bands_fdf:
                        host_params["%block BandLines"] = bands_fdf.pop(
                            "%block BandLines"
                        )
                    else:
                        n_interp = bands_fdf.pop("n_interpolations", 20)
                        host_band_lines = band_paymatgen_to_siesta(
                            host_structure, interpolations=[n_interp]
                        )
                        host_params["%block BandLines"] = host_band_lines
                        logger.info(
                            f"Auto-generated shared host band path "
                            f"({len(host_band_lines)} points, interp={n_interp})"
                        )

                    for k, v in bands_fdf.items():
                        host_params[k] = v

                if include_pdos:
                    pdos_fdf = dict(kwargs.get("pdos_fdf_params") or {})
                    if "%block ProjectedDensityOfStates" not in pdos_fdf:
                        pdos_block = ["EF -15.0 15.0 0.05 1000 eV"]
                    else:
                        pdos_block = pdos_fdf.pop("%block ProjectedDensityOfStates")
                    host_params["%block ProjectedDensityOfStates"] = pdos_block

                    if "%block PDOS.kgrid.MonkhorstPack" in pdos_fdf:
                        host_params["%block PDOS.kgrid.MonkhorstPack"] = pdos_fdf.pop(
                            "%block PDOS.kgrid.MonkhorstPack"
                        )

                    for k, v in pdos_fdf.items():
                        host_params[k] = v
                    logger.info("Auto-enabled PDOS for shared host calculation")

                host_maker.input_set_generator.user_params = host_params

            # Spin polarization for the shared host
            host_spin_params = host_maker.input_set_generator.user_params or {}
            if "Spin" not in host_spin_params:
                host_spin_params = {**host_spin_params, "Spin": "polarized"}
                host_spin_params.pop("enabled_modules", None)
                _enable_modules(host_maker.input_set_generator, ["spin"])
                host_maker.input_set_generator.user_params = host_spin_params
                logger.info("Auto-enabled Spin = polarized for shared host calculation")

            shared_host_job = host_maker.make(host_structure)
            shared_host_job.name = "shared_host_static"
            logger.info(
                f"Created shared host calculation for {len(defects)} defects "
                f"(saves {len(defects) - 1} redundant host calculations)"
            )

        # Create shared chemical potential calculations (instead of per-defect)
        # This reduces N_defects * N_species reference jobs to just N_species
        shared_ref_jobs = []
        if kwargs.get("auto_calculate_chemical_potentials", False) and defects:
            # Collect unique species across all defects
            unique_species = set()
            for defect_info in defects:
                defect_type_from_gen = defect_info["defect_type"]
                if defect_type_from_gen == "substitution":
                    if defect_info.get("dopant_species"):
                        unique_species.add(defect_info["dopant_species"])
                    if defect_info.get("original_species"):
                        unique_species.add(defect_info["original_species"])
                elif defect_info.get("species"):
                    unique_species.add(defect_info["species"])

            if unique_species:
                logger.info(
                    f"Creating shared chemical potential calculations for "
                    f"{len(unique_species)} unique species: {sorted(unique_species)} "
                    f"(saves {len(defects) * len(unique_species) - len(unique_species)} "
                    f"redundant reference calculations)"
                )

                # Use host_maker for reference calculations
                # (dry_run already propagated above)
                shared_chemical_potentials = {}
                for sp in sorted(unique_species):
                    ref_structure, n_atoms = get_reference_structure(sp)
                    ref_job = host_maker.make(ref_structure)
                    ref_job.name = f"shared_ref_{sp}"
                    mu_job = extract_chemical_potential(
                        task_doc=ref_job.output,
                        species=sp,
                        n_atoms=n_atoms,
                    )
                    mu_job.name = f"shared_mu_{sp}"
                    shared_ref_jobs.extend([ref_job, mu_job])
                    shared_chemical_potentials[sp] = mu_job.output

                # Override kwargs so individual flows don't create their own ref jobs
                kwargs["auto_calculate_chemical_potentials"] = False
                kwargs["chemical_potentials"] = shared_chemical_potentials

                logger.info(
                    f"Created {len(shared_ref_jobs)} shared reference jobs "
                    f"(instead of {len(defects) * len(unique_species) * 2} per-defect jobs)"
                )

        # Create flows for each defect
        flows = []
        for defect_info in defects:
            # Extract defect-specific info
            defect_structure = defect_info["structure"]
            host_structure = defect_info["host_structure"]
            frac_coords = defect_info["frac_coords"]
            charge_state = defect_info["charge_state"]
            defect_type_from_gen = defect_info["defect_type"]

            # Extract species information (different for each defect type)
            if defect_type_from_gen == "substitution":
                # Substitution: need both removed and added species
                defect_species = defect_info.get("dopant_species")  # Added species
                removed_species = defect_info.get("original_species")  # Removed species
            else:
                # Vacancy/Interstitial: only one species
                defect_species = defect_info.get("species")
                removed_species = None

            # Create flow maker with defect-specific settings
            flow_maker = cls(
                defect_type=defect_type_from_gen,
                charge_state=charge_state,
                **kwargs,
            )

            # Create flow (pass shared host if available)
            flow = flow_maker.make(
                defect_structure=defect_structure,
                host_structure=host_structure,
                defect_site=frac_coords,
                host_task_doc=shared_host_job.output if shared_host_job else None,
                defect_species=defect_species,
                removed_species=removed_species,
            )

            flows.append(flow)

        logger.info(
            f"Generated {len(flows)} defect flow(s) using from_pristine_structure()"
        )

        # Add combined summary job for all defects
        if len(flows) > 1:
            from atomate2.siesta.flows.defects.analysis.formation_energy import (
                write_combined_defect_summary,
            )

            # Collect outputs from all defect flows
            # Type assertion: flows is a list of Flow objects here
            defect_outputs = [flow.output for flow in flows]  # type: ignore[union-attr]

            # Create summary job
            summary_job = write_combined_defect_summary(
                defect_documents=defect_outputs,
                filename="all_defects_summary.txt",
            )

            # Wrap flows + summary in a parent Flow
            # Include shared_host_job and shared_ref_jobs first
            all_jobs = []
            if shared_host_job is not None:
                all_jobs.append(shared_host_job)
            all_jobs.extend(shared_ref_jobs)
            all_jobs.extend(flows)
            all_jobs.append(summary_job)

            parent_flow = Flow(all_jobs, output=summary_job.output)

            logger.info("Added combined summary job: all_defects_summary.txt")
            if shared_host_job is not None:
                logger.info(
                    f"Shared host calculation saves {len(flows) - 1} redundant calculations"
                )

            return parent_flow

        # Single defect - include shared host and ref jobs if present
        if (shared_host_job is not None or shared_ref_jobs) and flows:
            pre_jobs = []
            if shared_host_job is not None:
                pre_jobs.append(shared_host_job)
            pre_jobs.extend(shared_ref_jobs)
            return Flow(pre_jobs + flows, output=flows[0].output)  # type: ignore[union-attr]

        return flows


def get_reference_structure(species: str) -> tuple[Structure, int]:
    """
    Get reference structure for chemical potential calculation.

    For diatomic molecules (O2, H2, N2, F2, Cl2): μ = E(X2) / 2
    For metals: μ = E(bulk) / n_atoms

    Parameters
    ----------
    species : str
        Chemical species (e.g., "O", "Mg", "H")

    Returns
    -------
    tuple
        (reference_structure, n_atoms_per_formula_unit)
    """
    from pymatgen.core import Lattice, Molecule, Structure

    # Define reference structures
    # Sources:
    #   - CRC Handbook of Chemistry and Physics, 97th Edition (2016-2017)
    #   - NIST Chemistry WebBook (https://webbook.nist.gov)
    #   - Huber & Herzberg, "Molecular Spectra and Molecular Structure" (1979)
    #
    # Diatomic molecules with experimental bond lengths (Å)
    # Source: NIST Chemistry WebBook & Huber-Herzberg
    diatomic_molecules = {
        "O": ("O2", 1.2075),  # NIST: 1.20752 Å
        "H": ("H2", 0.7414),  # NIST: 0.74144 Å
        "N": ("N2", 1.0977),  # NIST: 1.09768 Å
        "F": ("F2", 1.4119),  # NIST: 1.41193 Å
        "Cl": ("Cl2", 1.9878),  # NIST: 1.9878 Å
        "Br": ("Br2", 2.2811),  # NIST: 2.2811 Å
        "I": ("I2", 2.6663),  # NIST: 2.6663 Å
    }

    # Bulk metals and semiconductors with experimental lattice parameters (Å)
    # Source: CRC Handbook of Chemistry and Physics, 97th Ed., Section 12
    # "Lattice Constants of the Elements" at 25°C (298 K)
    bulk_metals = {
        "Li": ("bcc", 3.509),  # CRC: 3.509 Å
        "Na": ("bcc", 4.2906),  # CRC: 4.2906 Å
        "K": ("bcc", 5.328),  # CRC: 5.328 Å
        "Mg": ("hcp", 3.2094),  # CRC: a=3.2094 Å, c=5.2108 Å
        "Ca": ("fcc", 5.5884),  # CRC: 5.5884 Å
        "Al": ("fcc", 4.0495),  # CRC: 4.0495 Å
        "Ti": ("hcp", 2.9508),  # CRC: a=2.9508 Å, c=4.6855 Å
        "V": ("bcc", 3.024),  # CRC: 3.024 Å
        "Cr": ("bcc", 2.8848),  # CRC: 2.8848 Å
        "Fe": ("bcc", 2.8665),  # CRC: 2.8665 Å (α-Fe)  # noqa: RUF003
        "Co": ("hcp", 2.5071),  # CRC: a=2.5071 Å, c=4.0695 Å
        "Ni": ("fcc", 3.5240),  # CRC: 3.5240 Å
        "Cu": ("fcc", 3.6149),  # CRC: 3.6149 Å
        "Zn": ("hcp", 2.6649),  # CRC: a=2.6649 Å, c=4.9468 Å
        "Mo": ("bcc", 3.1470),  # CRC: 3.1470 Å
        "Ag": ("fcc", 4.0862),  # CRC: 4.0862 Å
        "W": ("bcc", 3.1652),  # CRC: 3.1652 Å
        "Pt": ("fcc", 3.9242),  # CRC: 3.9242 Å
        "Au": ("fcc", 4.0782),  # CRC: 4.0782 Å
        "Pd": ("fcc", 3.8907),  # CRC: 3.8907 Å
        # Semiconductors
        "C": ("diamond", 3.5670),  # CRC: 3.5670 Å (diamond)
        "Si": ("diamond", 5.4310),  # CRC: 5.4310 Å
        "Ge": ("diamond", 5.6579),  # CRC: 5.6579 Å
    }

    # Chalcogens with experimental structures
    # Source: CRC Handbook, Wyckoff Crystal Structures
    special_elements = {
        "S": ("s2_molecule", 1.889),  # S-S bond in S2: NIST 1.889 Å
        "Se": ("hexagonal", 4.3662, 4.9536),  # CRC: trigonal Se
        "Te": ("hexagonal", 4.4572, 5.9290),  # CRC: trigonal Te
    }

    if species in diatomic_molecules:
        # Create diatomic molecule
        molecule_name, bond_length = diatomic_molecules[species]
        reference_structure = Molecule(
            [species, species],
            [[0, 0, 0], [0, 0, bond_length]],
        )
        # Place in large box to avoid periodic interactions
        reference_structure = reference_structure.get_boxed_structure(20, 21, 22)
        n_atoms = 2
        logger.info(f"Using {molecule_name} molecule (bond length {bond_length} Å)")

    elif species in bulk_metals:
        # Create bulk metal structure
        structure_type, lattice_param = bulk_metals[species]

        if structure_type == "fcc":
            lattice = Lattice.cubic(lattice_param)
            reference_structure = Structure(
                lattice,
                [species] * 4,
                [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
            )
            n_atoms = 4
        elif structure_type == "bcc":
            lattice = Lattice.cubic(lattice_param)
            reference_structure = Structure(
                lattice,
                [species] * 2,
                [[0, 0, 0], [0.5, 0.5, 0.5]],
            )
            n_atoms = 2
        elif structure_type == "hcp":
            a = lattice_param
            c = a * 1.633  # Ideal c/a ratio
            lattice = Lattice.hexagonal(a, c)
            reference_structure = Structure(
                lattice,
                [species] * 2,
                [[1 / 3, 2 / 3, 1 / 4], [2 / 3, 1 / 3, 3 / 4]],
            )
            n_atoms = 2
        elif structure_type == "diamond":
            lattice = Lattice.cubic(lattice_param)
            reference_structure = Structure(
                lattice,
                [species] * 8,
                [
                    [0, 0, 0],
                    [0.25, 0.25, 0.25],
                    [0.5, 0.5, 0],
                    [0.75, 0.75, 0.25],
                    [0.5, 0, 0.5],
                    [0.75, 0.25, 0.75],
                    [0, 0.5, 0.5],
                    [0.25, 0.75, 0.75],
                ],
            )
            n_atoms = 8
        else:
            raise ValueError(f"Unknown structure type: {structure_type}")

        logger.info(
            f"Using bulk {species} ({structure_type} structure, a={lattice_param} Å)"
        )

    elif species in special_elements:
        # Handle chalcogens and other special elements
        if species == "S":
            # Use S2 molecule (NIST bond length: 1.889 Å)
            # Note: Standard state is S8, but S2 is used for computational efficiency
            _, bond_length = special_elements["S"]  # type: ignore[assignment]
            reference_structure = Molecule(
                ["S", "S"],
                [[0, 0, 0], [0, 0, bond_length]],
            )
            reference_structure = reference_structure.get_boxed_structure(20, 21, 22)
            n_atoms = 2
            logger.info(f"Using S2 molecule reference (bond length {bond_length} Å)")
        elif species in ("Se", "Te"):
            # Hexagonal/trigonal structure
            _, a, c = special_elements[species]  # type: ignore[assignment]
            lattice = Lattice.hexagonal(a, c)
            reference_structure = Structure(
                lattice,
                [species] * 3,
                [[0, 0, 0], [1 / 3, 2 / 3, 1 / 3], [2 / 3, 1 / 3, 2 / 3]],
            )
            n_atoms = 3
            logger.info(f"Using trigonal {species} (a={a}, c={c} Å)")
        else:
            raise ValueError(f"Unknown special element: {species}")

    else:
        supported = (
            list(diatomic_molecules.keys())
            + list(bulk_metals.keys())
            + list(special_elements.keys())
        )
        raise ValueError(
            f"No reference structure defined for {species}. Supported: {supported}"
        )

    return reference_structure, n_atoms


@job
def extract_chemical_potential(
    task_doc,
    species: str,
    n_atoms: int,
) -> float:
    """
    Extract chemical potential from reference calculation.

    Parameters
    ----------
    task_doc : TaskDocument or dict
        TaskDocument from reference calculation
    species : str
        Chemical species
    n_atoms : int
        Number of atoms in reference structure

    Returns
    -------
    float
        Chemical potential (eV)
    """
    # Get energy from task document
    if isinstance(task_doc, dict):
        # Dry-run mode
        total_energy = -10.0  # Placeholder
    else:
        total_energy = task_doc.output.energy

    # Calculate chemical potential
    mu = total_energy / n_atoms

    logger.info(
        f"Chemical potential for {species}: "
        f"E_total = {total_energy:.4f} eV, "
        f"μ = {mu:.4f} eV"
    )

    return mu  # Return just the value, not dict


@job
def finalize_defect_calculation(
    defect_task_doc,
    host_task_doc,
    host_structure: Structure,
    epsilon_static: float,
    correction_scheme_name: str,
    defect_type: str,
    charge_state: int,
    defect_site: list[float] | None = None,
    defect_species: str | None = None,
    removed_species: str | None = None,
    chemical_potentials: dict[str, float] | None = None,
    potential_data: dict | None = None,
    vt_file_paths: dict | None = None,
    epsilon_parallel: float | None = None,
    epsilon_perpendicular: float | None = None,
) -> DefectDocument:
    """
    Finalize defect calculation by applying correction and creating document.

    Parameters
    ----------
    defect_task_doc : TaskDocument
        TaskDocument from defect relaxation
    host_task_doc : TaskDocument
        TaskDocument from host static calculation
    host_structure : Structure
        Pristine host structure
    epsilon_static : float
        Static dielectric constant
    defect_type : str
        Type of defect
    charge_state : int
        Charge state of defect
    defect_site : list[float], optional
        Fractional coordinates of defect site
    defect_species : str, optional
        Species of the defect (added species for substitution)
    removed_species : str, optional
        For substitution only: removed (host) species
    chemical_potentials : dict, optional
        Chemical potentials in eV: {"O": -5.0, "Mg": -2.0, ...}

    Returns
    -------
    DefectDocument
        Complete defect calculation document
    """
    logger.info("Finalizing defect calculation...")

    # Get energies from task documents
    # Handle both real TaskDoc objects and dry-run dicts (check each separately)
    if isinstance(defect_task_doc, dict):
        defect_energy = defect_task_doc.get("output", {}).get("energy", -100.0)
        defect_structure = host_structure.copy()  # Use host structure as placeholder
    else:
        # Check task state for SIESTA failures before extracting energy
        defect_state = getattr(defect_task_doc, "state", None)
        if defect_state is not None and str(defect_state).lower() != "successful":
            state_str = str(defect_state)
            defect_dir = getattr(defect_task_doc, "dir_name", "unknown")
            raise RuntimeError(
                f"Defect calculation did not complete successfully "
                f"(state: {state_str}). "
                f"Run directory: {defect_dir}. "
                + (
                    "SCF did not converge — consider enabling custodian with "
                    "SCFConvergenceHandler (use_custodian=True) or reducing "
                    "SCF.Mixer.Weight (e.g., 0.01-0.05)."
                    if "unconverged" in state_str.lower()
                    else "Check the SIESTA output files for details."
                )
            )
        defect_energy = defect_task_doc.output.energy
        defect_structure = defect_task_doc.output.structure

        # Guard against None energy even when state appears successful
        if defect_energy is None:
            defect_dir = getattr(defect_task_doc, "dir_name", "unknown")
            raise RuntimeError(
                f"Defect calculation returned no energy (output.energy is None). "
                f"Run directory: {defect_dir}. "
                f"The SIESTA output likely lacks a converged energy — check the "
                f".out file for SCF convergence or ABNORMAL_TERMINATION errors. "
                f"Consider enabling custodian (use_custodian=True)."
            )

    if isinstance(host_task_doc, dict):
        host_energy = host_task_doc.get("output", {}).get("energy", -50.0)
        host_bandgap = None
    else:
        host_energy = host_task_doc.output.energy
        host_bandgap = getattr(host_task_doc.output, "bandgap", None)

        if host_energy is None:
            host_dir = getattr(host_task_doc, "dir_name", "unknown")
            raise RuntimeError(
                f"Host calculation returned no energy (output.energy is None). "
                f"Run directory: {host_dir}. "
                f"The SIESTA output likely lacks a converged energy — check the "
                f".out file for SCF convergence or ABNORMAL_TERMINATION errors."
            )

    # Extract calculation directories and read grid files (.VT and .RHO) if available
    density_data = None

    if (
        not isinstance(defect_task_doc, dict)
        and not isinstance(host_task_doc, dict)
        and potential_data is None
    ):
        # Try to find .VT and .RHO files from task documents

        from atomate2.siesta.flows.defects.utils import (
            find_vt_files,
            prepare_density_data,
            prepare_freysoldt_potential_data,
        )

        defect_dir = getattr(defect_task_doc, "dir_name", None)
        host_dir = getattr(host_task_doc, "dir_name", None)

        if defect_dir and host_dir:
            # Search for .VT files (for potential alignment)
            logger.info("Searching for .VT files in calculation directories...")
            defect_vt = find_vt_files(defect_dir)
            host_vt = find_vt_files(host_dir)

            if defect_vt and host_vt:
                try:
                    potential_data = prepare_freysoldt_potential_data(
                        defect_vt_path=defect_vt,
                        host_vt_path=host_vt,
                    )
                    vt_file_paths = {"defect": str(defect_vt), "host": str(host_vt)}
                    logger.info("Successfully loaded .VT files for potential alignment")
                except Exception as e:
                    logger.warning(f"Failed to read .VT files: {e}")
                    potential_data = None
                    vt_file_paths = None
            else:
                logger.debug("No .VT files found in calculation directories")

            # Search for .RHO files (for quadrupole moment calculation)
            logger.info("Searching for .RHO files in calculation directories...")
            defect_rho = _find_rho_files(defect_dir)
            host_rho = _find_rho_files(host_dir)

            if defect_rho and host_rho:
                try:
                    density_data = prepare_density_data(
                        defect_rho_path=defect_rho,
                        host_rho_path=host_rho,
                    )
                    logger.info(
                        "Successfully loaded .RHO files for quadrupole calculation"
                    )
                except Exception as e:
                    logger.warning(f"Failed to read .RHO files: {e}")
                    density_data = None
            else:
                logger.debug("No .RHO files found in calculation directories")
        else:
            logger.debug(
                "Could not extract calculation directories from task documents"
            )

    # Apply finite-size correction (only if charged)
    if charge_state != 0:
        # Select correction scheme
        scheme_name_lower = correction_scheme_name.lower()
        if scheme_name_lower in ["lany-zunger", "lz"]:
            correction_scheme = LanyZungerCorrection(epsilon_static=epsilon_static)
        elif scheme_name_lower in ["makov-payne", "mp"]:
            # Basic Makov-Payne: monopole only (Q=0, no .RHO needed)
            correction_scheme = MakovPayneCorrection(epsilon_static=epsilon_static)
        elif scheme_name_lower in [
            "makov-payne-quadrupole",
            "mp-quad",
            "makov-payne-full",
        ]:
            # Full Makov-Payne: monopole + quadrupole from .RHO files
            correction_scheme = MakovPayneCorrection(epsilon_static=epsilon_static)
        elif scheme_name_lower in ["freysoldt", "frey", "fnv"]:
            correction_scheme = FreysoldtCorrection(epsilon_static=epsilon_static)
        elif scheme_name_lower in ["kumagai", "kumagai-oba", "ko"]:
            correction_scheme = KumagaiCorrection(epsilon_static=epsilon_static)
        elif scheme_name_lower in ["slab2d", "slab-2d"]:
            correction_scheme = Slab2DCorrection(
                epsilon_parallel=epsilon_parallel or epsilon_static,
                epsilon_perpendicular=epsilon_perpendicular or epsilon_static,
            )
        else:
            logger.warning(
                f"Unknown correction scheme: {correction_scheme_name}. "
                f"Using Lany-Zunger as fallback."
            )
            correction_scheme = LanyZungerCorrection(epsilon_static=epsilon_static)

        # Prepare kwargs for correction calculation
        correction_kwargs = {
            "defect_structure": defect_structure,
            "host_structure": host_structure,
            "charge_state": charge_state,
            "defect_energy": defect_energy,
            "host_energy": host_energy,
            "defect_site": defect_site,
        }

        # Add potential_data for schemes that support it
        if potential_data is not None:
            correction_kwargs["potential_data"] = potential_data
            correction_kwargs["vt_file_paths"] = vt_file_paths

        # Add density_data for Makov-Payne-Quadrupole (enables automatic quadrupole calculation)
        # Note: basic "makov-payne" does NOT get density_data (uses Q=0)
        if (
            scheme_name_lower
            in ["makov-payne-quadrupole", "mp-quad", "makov-payne-full"]
            and density_data is not None
        ):
            correction_kwargs["density_data"] = density_data

        correction_result = correction_scheme.calculate_correction(**correction_kwargs)
        correction_energy = correction_result.correction_energy
        correction_metadata = correction_result.metadata

        # Log correction breakdown
        logger.info(
            f"Applied {correction_scheme_name} correction: {correction_energy:.4f} eV"
        )
        if "lattice_term_eV" in correction_metadata:
            logger.info(
                f"  Lattice term: {correction_metadata['lattice_term_eV']:.4f} eV"
            )
        if "alignment_energy_eV" in correction_metadata:
            logger.info(
                f"  Alignment term: {correction_metadata['alignment_energy_eV']:.4f} eV"
            )
        if "quadrupole_term_eV" in correction_metadata:
            logger.info(
                f"  Quadrupole term: {correction_metadata['quadrupole_term_eV']:.4f} eV"
            )
    else:
        logger.info("Neutral defect - no correction needed")
        correction_energy = 0.0
        correction_metadata = {"note": "Neutral defect, no correction applied"}

    # Calculate chemical potential contribution
    mu_defect = 0.0
    mu_removed = 0.0
    mu_added = 0.0

    if chemical_potentials:
        if defect_type == "vacancy" and defect_species:
            # For vacancy: add back energy of removed atom
            mu_defect = chemical_potentials.get(defect_species, 0.0)
            mu_removed = mu_defect
            logger.info(
                f"Chemical potential for {defect_species} vacancy: μ = {mu_defect:.4f} eV"
            )
        elif defect_type == "substitution" and defect_species and removed_species:
            # For substitution: μ_removed - μ_added
            mu_removed = chemical_potentials.get(removed_species, 0.0)
            mu_added = chemical_potentials.get(defect_species, 0.0)
            mu_defect = mu_removed - mu_added
            logger.info(
                f"Substitution {defect_species}_{removed_species}: "
                f"μ_{removed_species} = {mu_removed:.4f} eV, "
                f"μ_{defect_species} = {mu_added:.4f} eV, "
                f"Δμ = {mu_defect:.4f} eV"
            )
        elif defect_type == "interstitial" and defect_species:
            # For interstitial: subtract energy of added atom (negative μ)
            mu_defect = -chemical_potentials.get(defect_species, 0.0)
            mu_added = -mu_defect
            logger.info(
                f"Chemical potential for {defect_species} interstitial: μ = {mu_defect:.4f} eV"
            )
    # Note: Missing chemical potentials are now caught early in make() method

    # Create DefectDocument
    if isinstance(defect_task_doc, dict):
        # Dry-run mode - create minimal DefectDocument
        raw_formation_energy = defect_energy - host_energy + mu_defect
        corrected_formation_energy = raw_formation_energy + correction_energy

        defect_doc = DefectDocument(
            defect_type=defect_type,
            defect_species=defect_species,
            removed_species=removed_species,
            defect_site=defect_site,
            charge_state=charge_state,
            defect_energy=defect_energy,
            host_energy=host_energy,
            raw_formation_energy=raw_formation_energy,
            correction_scheme=correction_scheme_name if charge_state != 0 else "none",
            correction_energy=correction_energy,
            corrected_formation_energy=corrected_formation_energy,
            chemical_potential=mu_defect,
            mu_removed=mu_removed,
            mu_added=mu_added,
            supercell_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            supercell_natoms=len(defect_structure),
            defect_structure=defect_structure,
            host_structure=host_structure,
            host_bandgap=host_bandgap,
            correction_metadata=correction_metadata,
        )
    else:
        # Real calculation
        defect_doc = DefectDocument.from_defect_calc(
            task_doc=defect_task_doc,
            defect_type=defect_type,
            charge_state=charge_state,
            correction_energy=correction_energy,
            correction_scheme=correction_scheme_name if charge_state != 0 else "none",
            host_energy=host_energy,
            host_structure=host_structure,
            chemical_potential=mu_defect,  # CRITICAL: Include chemical potential
            defect_site=defect_site,
            defect_species=defect_species,
            removed_species=removed_species,
            mu_removed=mu_removed,
            mu_added=mu_added,
            supercell_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Placeholder
            host_bandgap=host_bandgap,
            correction_metadata=correction_metadata,
        )

    logger.info(
        f"Defect calculation complete: {defect_type} (q={charge_state}), "
        f"E_form = {defect_doc.corrected_formation_energy:.3f} eV"
    )

    # Write text summary
    from atomate2.siesta.flows.defects.analysis.formation_energy import (
        write_defect_summary,
    )

    write_defect_summary(
        defect_doc, defect_type, charge_state, defect_species, mu_defect
    )

    return defect_doc


def _find_rho_files(
    calculation_dir: str | Path,
    system_label: str = "siesta",
) -> Path | None:
    """
    Find .RHO file in a SIESTA calculation directory.

    Parameters
    ----------
    calculation_dir : str or Path
        Directory containing SIESTA calculation outputs
    system_label : str, optional
        SIESTA SystemLabel. Default is "siesta".

    Returns
    -------
    Path or None
        Path to .RHO file if found, None otherwise
    """
    from pathlib import Path

    calc_dir = Path(calculation_dir)

    if not calc_dir.exists():
        logger.warning(f"Calculation directory not found: {calc_dir}")
        return None

    # Try common .RHO file names (uncompressed)
    rho_candidates = [
        calc_dir / f"{system_label}.RHO",
        calc_dir / "siesta.RHO",
        calc_dir / "SystemLabel.RHO",
    ]

    # Also search for any .RHO file (uncompressed)
    rho_files = list(calc_dir.glob("*.RHO"))

    for rho_path in rho_candidates + rho_files:
        if rho_path.exists():
            logger.info(f"Found .RHO file: {rho_path}")
            return rho_path

    # Try compressed .RHO.gz files in siesta_compressed subdirectory
    compressed_dir = calc_dir / "siesta_compressed"
    if compressed_dir.exists():
        rho_gz_candidates = [
            compressed_dir / f"{system_label}.RHO.gz",
            compressed_dir / "siesta.RHO.gz",
            compressed_dir / "SystemLabel.RHO.gz",
        ]
        rho_gz_files = list(compressed_dir.glob("*.RHO.gz"))

        for rho_gz_path in rho_gz_candidates + rho_gz_files:
            if rho_gz_path.exists():
                logger.info(f"Found compressed .RHO file: {rho_gz_path}")
                return rho_gz_path

    logger.debug(f"No .RHO or .RHO.gz file found in {calc_dir}")
    return None


@job
def generate_potential_plot(
    defect_task_doc,
    host_task_doc,
    output_name: str = "potential_alignment.png",
    axis: int = 2,
):
    """
    Generate potential alignment plot from VT files.

    Parameters
    ----------
    defect_task_doc : TaskDocument
        Defect calculation task document
    host_task_doc : TaskDocument
        Host calculation task document
    output_name : str
        Output filename for plot
    axis : int
        Axis for planar averaging (0=x, 1=y, 2=z)

    Returns
    -------
    dict
        Plot data with alignment information
    """
    from pathlib import Path

    from atomate2.siesta.flows.defects.utils import (
        find_vt_files,
        plot_potential_alignment,
    )

    # Skip if dry-run
    if isinstance(defect_task_doc, dict):
        logger.info("Dry-run mode - skipping potential plot generation")
        return {"status": "skipped", "reason": "dry_run"}

    # Get calculation directories
    defect_dir = getattr(defect_task_doc, "dir_name", None)
    host_dir = getattr(host_task_doc, "dir_name", None)

    if not defect_dir or not host_dir:
        logger.warning("Cannot find calculation directories for plotting")
        return {"status": "skipped", "reason": "no_directories"}

    # Find VT files
    defect_vt = find_vt_files(defect_dir)
    host_vt = find_vt_files(host_dir)

    if not defect_vt or not host_vt:
        logger.warning("VT files not found - cannot generate potential plot")
        return {"status": "skipped", "reason": "no_vt_files"}

    # Generate plot
    try:
        # Save to current job directory (plot job folder)
        output_path = Path.cwd() / output_name

        plot_data = plot_potential_alignment(
            defect_vt_path=defect_vt,
            host_vt_path=host_vt,
            axis=axis,
            output_path=output_path,
            show_plot=False,  # Don't show interactively in workflow
        )

        logger.info(f"Generated potential alignment plot: {output_path}")
        logger.info(f"Mean alignment: ΔV = {plot_data['mean_alignment']:.4f} eV")

        return {
            "status": "success",
            "output_path": str(output_path),
            "mean_alignment_eV": plot_data["mean_alignment"],
            "axis": axis,
        }

    except Exception as e:
        logger.warning(f"Failed to generate potential plot: {e}")
        return {"status": "failed", "error": str(e)}


@job
def generate_density_plot(
    defect_task_doc,
    host_task_doc,
    output_name: str = "density_difference.png",
    axis: int = 2,
):
    """
    Generate charge density difference plot.

    Shows Δρ = ρ_defect - ρ_host (real from RHO or Gaussian approximation).

    Parameters
    ----------
    defect_task_doc : TaskDocument
        Defect calculation task document
    host_task_doc : TaskDocument
        Host calculation task document
    output_name : str
        Output filename for plot
    axis : int
        Axis for planar averaging (0=x, 1=y, 2=z)

    Returns
    -------
    dict
        Plot data with density information
    """  # noqa: RUF002
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    # Skip if dry-run
    if isinstance(defect_task_doc, dict):
        logger.info("Dry-run mode - skipping density plot generation")
        return {"status": "skipped", "reason": "dry_run"}

    # Get calculation directories
    defect_dir = getattr(defect_task_doc, "dir_name", None)
    host_dir = getattr(host_task_doc, "dir_name", None)

    if not defect_dir or not host_dir:
        logger.warning("Cannot find calculation directories for plotting")
        return {"status": "skipped", "reason": "no_directories"}

    # Try to find RHO files
    defect_rho = _find_rho_files(defect_dir)
    host_rho = _find_rho_files(host_dir)

    try:
        if defect_rho and host_rho:
            # Real RHO from SIESTA
            from atomate2.siesta.flows.defects.utils import (
                calculate_planar_average,
                prepare_density_data,
            )

            density_data = prepare_density_data(defect_rho, host_rho)
            delta_rho = density_data["defect_density"] - density_data["host_density"]

            # Planar average
            positions, avg_rho = calculate_planar_average(delta_rho, axis=axis)

            title = "Charge Density Difference (from SIESTA .RHO files)"
            logger.info("Using real RHO files for density plot")
        else:
            # Gaussian approximation (fallback)
            logger.info("RHO files not found - using Gaussian approximation")

            # Create simple Gaussian model
            positions = np.linspace(0, 1, 100)
            sigma = 0.1
            center = 0.5
            avg_rho = np.exp(-((positions - center) ** 2) / (2 * sigma**2))

            title = "Charge Density Difference (Gaussian approximation)"

        # Generate plot - save to current job directory
        output_path = Path.cwd() / output_name

        plt.figure(figsize=(10, 6))
        plt.plot(positions, avg_rho, "b-", linewidth=2)
        plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        plt.xlabel(f"Fractional Coordinate ({'xyz'[axis]}-axis)", fontsize=12)
        plt.ylabel("Δρ (electrons/Ų)", fontsize=12)
        plt.title(title, fontsize=13, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Generated density plot: {output_path}")

        return {
            "status": "success",
            "output_path": str(output_path),
            "used_real_rho": defect_rho is not None and host_rho is not None,
            "axis": axis,
        }

    except Exception as e:
        logger.warning(f"Failed to generate density plot: {e}")
        return {"status": "failed", "error": str(e)}


@job
def generate_dielectric_profile_plot(
    defect_task_doc,
    epsilon_parallel: float,
    epsilon_perpendicular: float,
    output_name: str = "dielectric_profile.png",
):
    """
    Generate dielectric profile ε(z) plot for Slab2D correction.

    Shows how dielectric constant varies through slab and vacuum regions.

    Parameters
    ----------
    defect_task_doc : TaskDocument
        Defect calculation task document
    epsilon_parallel : float
        In-plane dielectric constant
    epsilon_perpendicular : float
        Out-of-plane dielectric constant
    output_name : str
        Output filename

    Returns
    -------
    dict
        Plot status
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    # Skip if dry-run
    if isinstance(defect_task_doc, dict):
        logger.info("Dry-run mode - skipping profile plot")
        return {"status": "skipped", "reason": "dry_run"}

    defect_dir = getattr(defect_task_doc, "dir_name", None)
    if not defect_dir:
        return {"status": "skipped", "reason": "no_directory"}

    try:
        # Get structure from task doc
        from atomate2.siesta.flows.defects.corrections.slab_2d import (
            DielectricProfile,
            detect_slab_geometry,
        )

        structure = defect_task_doc.output.structure
        slab_info = detect_slab_geometry(structure, vacuum_threshold=6.0)

        # Create Gaussian profile
        profile = DielectricProfile.create_gaussian_profile(
            epsilon_parallel_bulk=epsilon_parallel,
            epsilon_perpendicular_bulk=epsilon_perpendicular,
            slab_center=slab_info["slab_center"],
            slab_thickness=slab_info["slab_thickness"],
            cell_length_z=slab_info["cell_length_z"],
        )

        # Generate plot - save to current job directory
        output_path = Path.cwd() / output_name

        _fig, ax = plt.subplots(figsize=(10, 6))

        # Plot profiles
        ax.plot(
            profile.z_coords,
            profile.epsilon_parallel,
            "b-",
            linewidth=2,
            label="ε∥ (in-plane)",
        )
        ax.plot(
            profile.z_coords,
            profile.epsilon_perpendicular,
            "r-",
            linewidth=2,
            label="ε⊥ (out-of-plane)",
        )

        # Mark slab region
        slab_z_min = slab_info["slab_center"] - slab_info["slab_thickness"] / 2
        slab_z_max = slab_info["slab_center"] + slab_info["slab_thickness"] / 2
        ax.axvspan(
            slab_z_min, slab_z_max, alpha=0.2, color="green", label="Slab region"
        )

        # Formatting
        ax.set_xlabel("z-coordinate (Å)", fontsize=12)
        ax.set_ylabel("Dielectric constant ε", fontsize=12)
        ax.set_title("Dielectric Profile for 2D Slab", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(epsilon_parallel, epsilon_perpendicular) + 1)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Generated dielectric profile plot: {output_path}")

        return {
            "status": "success",
            "output_path": str(output_path),
            "epsilon_parallel": epsilon_parallel,
            "epsilon_perpendicular": epsilon_perpendicular,
            "slab_thickness": slab_info["slab_thickness"],
        }

    except Exception as e:
        logger.warning(f"Failed to generate profile plot: {e}")
        return {"status": "failed", "error": str(e)}


@job
def generate_radial_distribution_plot(
    defect_task_doc,
    host_task_doc,
    defect_site_frac: list,
    charge_state: int,
    output_name: str = "radial_distribution.png",
):
    """
    Generate radial distribution plots for Δρ(r) and ΔV(r).

    Shows how charge density and electrostatic potential differences
    decay with distance from the defect center. Useful for validating
    correction scheme assumptions.

    Parameters
    ----------
    defect_task_doc : TaskDocument
        Defect calculation task document
    host_task_doc : TaskDocument
        Host calculation task document
    defect_site_frac : list
        Defect site fractional coordinates [x, y, z]
    charge_state : int
        Defect charge state
    output_name : str
        Output filename

    Returns
    -------
    dict
        Plot status and statistics
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    # Skip if dry-run
    if isinstance(defect_task_doc, dict):
        logger.info("Dry-run mode - skipping radial plot")
        return {"status": "skipped", "reason": "dry_run"}

    defect_dir = getattr(defect_task_doc, "dir_name", None)
    host_dir = getattr(host_task_doc, "dir_name", None)

    if not defect_dir or not host_dir:
        return {"status": "skipped", "reason": "no_directory"}

    try:
        # Find RHO and VT files
        from atomate2.siesta.flows.defects.utils import (
            read_siesta_density,
            read_siesta_grid_file,
        )

        defect_rho_files = _find_rho_files(Path(defect_dir), ".RHO")
        host_rho_files = _find_rho_files(Path(host_dir), ".RHO")
        defect_vt_files = _find_rho_files(Path(defect_dir), ".VT")
        host_vt_files = _find_rho_files(Path(host_dir), ".VT")

        # Get structure
        structure = defect_task_doc.output.structure
        cell = structure.lattice.matrix

        # Convert defect position to Cartesian
        defect_pos_cart = np.dot(defect_site_frac, cell)

        _fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- Plot 1: Charge density Δρ(r) ---
        if defect_rho_files and host_rho_files:
            defect_rho_data = read_siesta_density(str(defect_rho_files))
            host_rho_data = read_siesta_density(str(host_rho_files))

            delta_rho = defect_rho_data["data"] - host_rho_data["data"]

            # Calculate radial distribution
            r_bins, rho_avg, rho_std = _calculate_radial_distribution(
                delta_rho, defect_rho_data["grid_shape"], cell, defect_pos_cart
            )

            axes[0].plot(r_bins, rho_avg, "b-", linewidth=2, label="Δρ(r)")
            axes[0].fill_between(
                r_bins,
                rho_avg - rho_std,
                rho_avg + rho_std,
                alpha=0.3,
                color="blue",
                label="±1σ",  # noqa: RUF001
            )
            axes[0].axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
            axes[0].set_xlabel("Distance from defect (Å)", fontsize=11)
            axes[0].set_ylabel("Δρ(r) (e/Å³)", fontsize=11)
            axes[0].set_title(
                "Charge Density Difference", fontsize=12, fontweight="bold"
            )
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)

        else:
            # Gaussian approximation
            logger.info("No RHO files - using Gaussian approximation for Δρ(r)")
            r = np.linspace(0, 10, 200)
            sigma = 1.5  # Default localization radius
            gaussian_rho = (
                charge_state
                / (np.sqrt(2 * np.pi) * sigma) ** 3
                * np.exp(-(r**2) / (2 * sigma**2))
            )

            axes[0].plot(
                r,
                gaussian_rho,
                "b--",
                linewidth=2,
                label=f"Gaussian (σ={sigma:.2f} Å)",  # noqa: RUF001
            )
            axes[0].axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
            axes[0].set_xlabel("Distance from defect (Å)", fontsize=11)
            axes[0].set_ylabel("ρ(r) (e/Å³)", fontsize=11)  # noqa: RUF001
            axes[0].set_title(
                f"Charge Density (Gaussian σ={sigma:.2f} Å)",  # noqa: RUF001
                fontsize=12,
                fontweight="bold",
            )
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)

        # --- Plot 2: Electrostatic potential ΔV(r) ---
        if defect_vt_files and host_vt_files:
            defect_vt_data = read_siesta_grid_file(str(defect_vt_files), file_type="VT")
            host_vt_data = read_siesta_grid_file(str(host_vt_files), file_type="VT")

            delta_V = defect_vt_data["data"] - host_vt_data["data"]  # noqa: N806

            # Calculate radial distribution
            r_bins, V_avg, V_std = _calculate_radial_distribution(  # noqa: N806
                delta_V, defect_vt_data["grid_shape"], cell, defect_pos_cart
            )

            axes[1].plot(r_bins, V_avg, "r-", linewidth=2, label="ΔV(r)")
            axes[1].fill_between(
                r_bins,
                V_avg - V_std,
                V_avg + V_std,
                alpha=0.3,
                color="red",
                label="±1σ",  # noqa: RUF001
            )
            axes[1].axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
            axes[1].set_xlabel("Distance from defect (Å)", fontsize=11)
            axes[1].set_ylabel("ΔV(r) (eV)", fontsize=11)
            axes[1].set_title(
                "Electrostatic Potential Difference", fontsize=12, fontweight="bold"
            )
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)

        else:
            # Point charge approximation: V(r) = q/(4πε₀r)
            logger.info("No VT files - using point charge approximation for V(r)")
            r = np.linspace(0.5, 10, 200)  # Avoid r=0 singularity
            epsilon_0 = 8.854187817e-12  # F/m
            e = 1.602176634e-19  # C
            ke = 1 / (4 * np.pi * epsilon_0) * e * 1e10 / e  # eV·Å/e
            V_point = charge_state * ke / r  # noqa: N806

            axes[1].plot(r, V_point, "r--", linewidth=2, label="Point charge")
            axes[1].axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
            axes[1].set_xlabel("Distance from defect (Å)", fontsize=11)
            axes[1].set_ylabel("V(r) (eV)", fontsize=11)
            axes[1].set_title(
                "Electrostatic Potential (Point)", fontsize=12, fontweight="bold"
            )
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_yscale("log")

        plt.suptitle(
            f"Radial Distribution from Defect Center (q={charge_state:+d})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save to current job directory
        output_path = Path.cwd() / output_name
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Generated radial distribution plot: {output_path}")

        return {
            "status": "success",
            "output_path": str(output_path),
            "has_real_rho": bool(defect_rho_files and host_rho_files),
            "has_real_vt": bool(defect_vt_files and host_vt_files),
        }

    except Exception as e:
        logger.warning(f"Failed to generate radial plot: {e}")
        return {"status": "failed", "error": str(e)}


def _calculate_radial_distribution(
    data_3d: np.ndarray,
    grid_shape: tuple,
    cell: np.ndarray,
    center_cart: np.ndarray,
    num_bins: int = 100,
):
    """
    Calculate radial average and std of 3D grid data.

    Parameters
    ----------
    data_3d : np.ndarray
        3D grid data (nx, ny, nz)
    grid_shape : tuple
        Grid dimensions (nx, ny, nz)
    cell : np.ndarray
        Cell matrix (3x3)
    center_cart : np.ndarray
        Center position in Cartesian coords
    num_bins : int
        Number of radial bins

    Returns
    -------
    r_bins : np.ndarray
        Bin centers (Å)
    avg : np.ndarray
        Average value in each bin
    std : np.ndarray
        Standard deviation in each bin
    """
    nx, ny, nz = grid_shape

    # Create fractional coordinate grid
    x_frac = np.linspace(0, 1, nx, endpoint=False)
    y_frac = np.linspace(0, 1, ny, endpoint=False)
    z_frac = np.linspace(0, 1, nz, endpoint=False)

    X_frac, Y_frac, Z_frac = np.meshgrid(x_frac, y_frac, z_frac, indexing="ij")  # noqa: N806

    # Convert to Cartesian
    coords_frac = np.stack([X_frac.ravel(), Y_frac.ravel(), Z_frac.ravel()], axis=1)
    coords_cart = coords_frac @ cell

    # Calculate distances from center
    distances = np.linalg.norm(coords_cart - center_cart, axis=1)

    # Flatten data
    data_flat = data_3d.ravel()

    # Bin by distance
    max_r = np.max(distances)
    r_bins_edges = np.linspace(0, max_r, num_bins + 1)
    r_bins = (r_bins_edges[:-1] + r_bins_edges[1:]) / 2

    avg = np.zeros(num_bins)
    std = np.zeros(num_bins)

    for i in range(num_bins):
        mask = (distances >= r_bins_edges[i]) & (distances < r_bins_edges[i + 1])
        if np.any(mask):
            avg[i] = np.mean(data_flat[mask])
            std[i] = np.std(data_flat[mask])

    return r_bins, avg, std
