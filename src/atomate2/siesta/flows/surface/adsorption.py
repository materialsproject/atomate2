"""Workflows for adsorption site scanning and optimization."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from jobflow import Flow, Response, job
from pymatgen.core import Molecule

from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.jobs.surface.adsorption import (
    add_adsorbate_to_slab,
    analyze_adsorption_scan,
    generate_adsorption_sites,
)

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


def _molecular_spin_setup(molecule, structure):
    """Apply the correct spin state to an isolated adsorbate molecule.

    Open-shell adsorbates (O2, O, OH, ...) have a non-singlet ground state.
    Without initial moments the isolated-adsorbate reference converges to the
    wrong spin (e.g. O2 -> singlet, 0.44 eV too high), biasing every adsorption
    energy. Detect the molecular spin configuration, apply ferromagnetic magmoms
    as a site property, and return the FDF params (Spin, Spin.Total, and
    ``a2s_magnetic_ordering="custom"`` so the applied signs are preserved).

    Returns
    -------
    tuple[Structure, dict]
        The structure with a ``magmom`` site property and the spin params to
        merge into the adsorbate static maker (empty dict if non-magnetic).
    """
    from atomate2.siesta.flows.electrocatalysis.utils.spin_config import (
        get_siesta_spin_config,
    )

    try:
        cfg = get_siesta_spin_config(molecule.composition.reduced_formula)
    except Exception:  # noqa: BLE001 - unknown molecule -> leave non-polarized
        return structure, {}

    if not cfg.get("spin_polarized") or cfg.get("init_magnetic_moments") is None:
        return structure, {}

    moments = cfg["init_magnetic_moments"]
    structure = structure.copy()
    structure.add_site_property(
        "magmom", [moments.get(site.specie.symbol, 0.0) for site in structure]
    )
    params = {"Spin": "polarized", "a2s_magnetic_ordering": "custom"}
    if cfg.get("fix_spin", False):
        params["Spin.Total"] = cfg["total_spin_moment"]
    return structure, params


# Module-level helper functions for jobflow-remote serialization


@job
def _save_best_structure(scan_doc, slab, adsorbate, placement, output_dir="."):
    """
    Save the structure with adsorbate at the best site.

    This is a module-level function to ensure proper serialization
    for jobflow-remote execution.

    Parameters
    ----------
    scan_doc : AdsorptionScanDocument
        Scan results containing best site information.
    slab : Structure
        Clean slab structure.
    adsorbate : Structure | Molecule
        Adsorbate structure.
    placement : str
        Placement location ('top' or 'bottom').
    output_dir : str | Path
        Output directory for saving the structure file.

    Returns
    -------
    dict
        Dictionary containing the structure and file path.
    """
    best_site = scan_doc.best_site
    # Use the height from the best site (which might be different from initial height)
    best_structure = add_adsorbate_to_slab(
        slab,
        adsorbate,
        (best_site.site_x, best_site.site_y),
        best_site.height,
        placement,
    )

    # Save to file in specified directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "best_adsorption_structure.cif"
    best_structure.to(filename=str(output_file))
    logger.info(
        f"✓ Saved best structure to {output_file} "
        f"(E_ads = {best_site.adsorption_energy:.3f} eV at height = {best_site.height:.2f} Å)"
    )
    return {"structure": best_structure, "structure_file": str(output_file)}


@job
def _consolidated_analysis(
    slab,
    adsorbate,
    site_energies,
    slab_energy,
    adsorbate_energy,
    grid_size,
    heights,
    miller_indices,
    placement,
    plot_results,
    write_summary,
):
    """
    Run analysis and create all output files in same directory.

    This consolidates all outputs (analysis, best structure, plots, summary)
    into a single job directory.

    Parameters
    ----------
    slab : Structure
        Clean slab structure.
    adsorbate : Structure | Molecule
        Adsorbate structure.
    site_energies : list[dict]
        List of site energy dictionaries.
    slab_energy : float
        Clean slab energy.
    adsorbate_energy : float
        Adsorbate energy.
    grid_size : tuple[int, int]
        Grid dimensions.
    heights : list[float]
        List of heights scanned.
    miller_indices : tuple[int, int, int] | None
        Miller indices.
    placement : str
        Placement location ('top' or 'bottom').
    plot_results : bool
        Whether to generate plots.
    write_summary : bool
        Whether to write summary file.

    Returns
    -------
    AdsorptionScanDocument
        Analysis results with all files created in current directory.
    """
    # Import here to avoid circular dependency
    from atomate2.siesta.jobs.surface.adsorption import (
        add_adsorbate_to_slab,
        analyze_adsorption_scan,
        plot_adsorption_sites,
        write_adsorption_summary,
    )

    # Run the analysis
    scan_doc = analyze_adsorption_scan.original(
        slab=slab,
        adsorbate=adsorbate,
        site_energies=site_energies,
        slab_energy=slab_energy,
        adsorbate_energy=adsorbate_energy,
        grid_size=grid_size,
        heights=heights,
        miller_indices=miller_indices,
    )

    # Save best structure directly (not a separate job)
    best_site = scan_doc.best_site
    best_structure = add_adsorbate_to_slab(
        slab,
        adsorbate,
        (best_site.site_x, best_site.site_y),
        best_site.height,
        placement,
    )
    output_file = Path(".") / "best_adsorption_structure.cif"
    best_structure.to(filename=str(output_file))
    logger.info(
        f"✓ Saved best structure to {output_file} "
        f"(E_ads = {best_site.adsorption_energy:.3f} eV at height = {best_site.height:.2f} Å)"
    )

    # Create plot directly (not a separate job)
    if plot_results:
        plot_adsorption_sites.original(scan_doc=scan_doc, output_dir=".")
        logger.info("✓ Created adsorption site plots")

    # Write summary directly (not a separate job)
    if write_summary:
        write_adsorption_summary.original(scan_doc=scan_doc, output_dir=".")
        logger.info("✓ Wrote adsorption summary")

    return scan_doc


@job
def _extract_site_energy(site, height, static_job_output):
    """
    Extract energy from static calculation output.

    This is a module-level function to ensure proper serialization
    for jobflow-remote execution.

    Parameters
    ----------
    site : tuple
        Site coordinates (x, y).
    height : float
        Height of adsorbate above surface (Å).
    static_job_output : dict or object
        Output from static calculation containing energy.

    Returns
    -------
    dict
        Dictionary with site coordinates, height, and total energy.
    """
    # Handle both dry_run (dict) and regular (object with .energy) cases
    if isinstance(static_job_output, dict):
        total_energy = static_job_output.get("energy", 0.0)
    else:
        total_energy = static_job_output.energy

    return {
        "site": site,
        "height": height,
        "total_energy": total_energy,
    }


@job
def _calc_site_energy_impl(
    slab,
    adsorbate,
    site,
    height,
    placement,
    slab_static_maker,
    site_index,
    grid_size,
    job_num=0,
    total_jobs=0,
):
    """
    Calculate energy for a specific adsorption site.

    This is a module-level function to ensure proper serialization
    for jobflow-remote execution.

    Parameters
    ----------
    slab : Structure
        Clean slab structure.
    adsorbate : Structure | Molecule
        Adsorbate structure.
    site : tuple
        Site coordinates (x, y).
    height : float
        Adsorbate height above surface.
    placement : str
        Placement location ('top' or 'bottom').
    slab_static_maker : StaticMaker
        Maker for static calculations.
    site_index : int
        Index of this site.
    grid_size : tuple
        Grid dimensions (nx, ny).
    job_num : int
        Current job number.
    total_jobs : int
        Total number of jobs.

    Returns
    -------
    Response
        Jobflow Response with site energy calculation flow.
    """
    total_sites = grid_size[0] * grid_size[1]
    logger.info(
        f"[PROGRESS] Site {site_index + 1}/{total_sites}: "
        f"Position ({site[0]:.3f}, {site[1]:.3f}), "
        f"Height {height:.2f} Å, "
        f"Placement: {placement}"
    )

    # Create combined structure
    ads_slab = add_adsorbate_to_slab(slab, adsorbate, site, height, placement=placement)

    # Create the static calculation job
    static_job = slab_static_maker.make(ads_slab)
    if job_num > 0:
        static_job.name = f"[{job_num}_of_{total_jobs}]_site_{site_index:03d}_static"
    else:
        static_job.name = f"site_{site_index:03d}_static"

    # Create extraction job
    extract_job_num = job_num + 1 if job_num > 0 else 0
    extract_name = (
        f"[{extract_job_num}_of_{total_jobs}]_site_{site_index:03d}_extract"
        if extract_job_num > 0
        else f"site_{site_index:03d}_extract"
    )

    # Use module-level extraction function
    analysis = _extract_site_energy(
        site=site, height=height, static_job_output=static_job.output.output
    )
    analysis.name = extract_name

    return Response(replace=Flow([static_job, analysis], output=analysis.output))


@dataclass
class AdsorptionScanFlowMaker(BaseSiestaFlowMaker):
    """
    Scan adsorption sites on a surface slab with automatic dry-run propagation.

    This workflow:

    1. Calculates energies of clean slab and isolated adsorbate (once)
    2. Generates a grid of adsorption sites
    3. For each site:

       - Places adsorbate at specified height
       - Calculates energy of slab+adsorbate
       - Computes adsorption energy

    4. Analyzes results and identifies best sites
    5. Generates plots and summary

    Inherits from BaseSiestaFlowMaker, so dry_run=True automatically propagates
    to child makers (slab_static_maker, adsorbate_static_maker).

    Parameters
    ----------
    name : str
        Name of the flow.
    slab_static_maker : StaticMaker
        Maker for slab energy calculations.
    adsorbate_static_maker : StaticMaker
        Maker for isolated adsorbate energy calculation.
    grid_size : tuple[int, int]
        Grid dimensions (nx, ny) for site scanning.
    height : float
        Initial adsorbate height above surface (Å). Used if heights is None.
    heights : list[float], optional
        Explicit list of heights to scan (Å). If provided, overrides height/height_min/height_max.
    height_min : float, optional
        Minimum height for automatic range generation (Å).
    height_max : float, optional
        Maximum height for automatic range generation (Å).
    height_step : float, optional
        Step size for height range (Å). Required if height_min/height_max provided.
    miller_indices : tuple[int, int, int], optional
        Miller indices of the surface for documentation.
    plot_results : bool
        Whether to generate adsorption site plots.
    write_summary : bool
        Whether to write text summary file.
    custom_mol_file : str, optional
        Path to custom molecule file (XYZ, CIF, etc.) for adsorbate.
    plane_atoms : list[int], optional
        List of 3 atom indices defining plane normal for molecular orientation.
    target_vector : list[float], optional
        Target direction vector [x, y, z] for molecule orientation.
    extra_rotation : float
        Additional rotation angle in degrees (default: 0.0).
    rotation_axis : list[float], optional
        Axis for additional rotation [x, y, z] (default: [0, 0, 1]).
    placement : str
        Placement of adsorbate: 'top' or 'bottom' of slab (default: 'top').
    dry_run : bool
        If True, generate and save structures without running SIESTA calculations.
        Useful for previewing adsorption geometries (inherited from BaseSiestaFlowMaker).
    dry_run_output_dir : str
        Directory to save structures when dry_run=True (default: 'preview_structures').
    dry_run_format : str
        File format for saved structures (inherited from BaseSiestaFlowMaker).

    Examples
    --------
    >>> from pymatgen.core import Structure, Molecule
    >>> from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker
    >>> from atomate2.siesta.jobs.core import StaticMaker
    >>> from atomate2.siesta.sets.core import StaticSetGenerator
    >>>
    >>> # Setup SIESTA parameters
    >>> slab_params = {
    ...     "PAO.BasisSize": "DZP",
    ...     "Mesh.Cutoff": "300 Ry",
    ...     "kpts": [6, 6, 1],
    ... }
    >>> ads_params = {"PAO.BasisSize": "DZP", "Mesh.Cutoff": "300 Ry"}
    >>>
    >>> slab_maker = StaticMaker(
    ...     input_set_generator=StaticSetGenerator(user_params=slab_params)
    ... )
    >>> ads_maker = StaticMaker(
    ...     input_set_generator=StaticSetGenerator(user_params=ads_params)
    ... )
    >>>
    >>> # Create workflow
    >>> slab = Structure.from_file("slab.cif")
    >>> adsorbate = Molecule.from_file("CO.xyz")
    >>>
    >>> maker = AdsorptionScanFlowMaker(
    ...     slab_static_maker=slab_maker,
    ...     adsorbate_static_maker=ads_maker,
    ...     grid_size=(5, 5),
    ...     height=2.0,
    ...     miller_indices=(1, 0, 0),
    ... )
    >>> flow = maker.make(slab, adsorbate)
    """

    name: str = "adsorption_scan"
    slab_static_maker: StaticMaker = field(default_factory=StaticMaker)
    adsorbate_static_maker: StaticMaker = field(default_factory=StaticMaker)
    # Precalculated reference energies (eV). When provided, the corresponding
    # static calculation is skipped and the given energy is used directly -
    # enables multi-adsorbate screening with a single slab calculation.
    precalc_slab_energy: float | None = None
    precalc_adsorbate_energy: float | None = None
    grid_size: tuple[int, int] = (4, 4)
    height: float = 2.0
    heights: list[float] | None = None  # NEW: Explicit list of heights to scan
    height_min: float | None = None  # NEW: Minimum height for range
    height_max: float | None = None  # NEW: Maximum height for range
    height_step: float | None = None  # NEW: Step size for height range
    miller_indices: tuple[int, int, int] | None = None
    plot_results: bool = True
    write_summary: bool = True
    custom_mol_file: str | None = None
    plane_atoms: list[int] | None = None
    target_vector: list[float] | None = None
    extra_rotation: float = 0.0
    rotation_axis: list[float] | None = None
    placement: str = "top"
    # Override default dry_run_output_dir from BaseSiestaFlowMaker
    dry_run_output_dir: str = "preview_structures"
    # dry_run and dry_run_format inherited from BaseSiestaFlowMaker

    def _resolve_heights(self) -> list[float]:
        """
        Resolve the list of heights to scan.

        Priority:
        1. heights (explicit list)
        2. height_min/height_max/height_step (range)
        3. height (single value)

        Returns
        -------
        list[float]
            List of heights to scan.
        """
        import numpy as np

        if self.heights is not None:
            # Explicit list provided
            return list(self.heights)
        if (
            self.height_min is not None
            and self.height_max is not None
            and self.height_step is not None
        ):
            # Generate range
            heights = np.arange(
                self.height_min,
                self.height_max + self.height_step / 2,
                self.height_step,
            )
            return heights.tolist()
        # Single height (backward compatible)
        return [self.height]

    def make(
        self,
        slab: Structure,
        adsorbate: Structure | Molecule,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """
        Create adsorption site scanning workflow.

        Parameters
        ----------
        slab : Structure
            Slab structure.
        adsorbate : Structure | Molecule
            Adsorbate structure or molecule.
        prev_dir : str | Path, optional
            Previous directory for restart.

        Returns
        -------
        Flow
            Adsorption scan workflow.
        """
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        logger.info("AdsorptionScanFlowMaker.make()")

        jobs = []

        # Resolve heights to scan
        heights_to_scan = self._resolve_heights()
        n_heights = len(heights_to_scan)

        logger.info(f"Height scanning: {n_heights} height(s) - {heights_to_scan}")

        # Calculate total number of jobs for progress tracking
        total_sites = self.grid_size[0] * self.grid_size[1]
        total_sites_with_heights = total_sites * n_heights  # 3D grid (x, y, z)
        total_jobs = (
            (1 if self.precalc_slab_energy is None else 0)  # slab energy
            + (1 if self.precalc_adsorbate_energy is None else 0)  # adsorbate energy
            + 1  # generate sites
            + total_sites_with_heights * 2  # each (x,y,z) site: calc + extract
            + 1  # analysis
            + 1  # save best structure
            + (1 if self.plot_results else 0)  # plot
            + (1 if self.write_summary else 0)  # summary
        )
        logger.info(
            f"Creating workflow with {total_jobs} total jobs "
            f"({total_sites} xy sites × {n_heights} heights = {total_sites_with_heights} total calculations)"  # noqa: RUF001
        )

        # Global counter for progress tracking
        job_counter = {"current": 0, "total": total_jobs}

        # Prepare adsorbate with orientation if specified
        from atomate2.siesta.utils.molecule_utils import (
            molecule_to_structure_in_box,
            prepare_molecule_with_orientation,
        )

        # Apply orientation if this is a Molecule and orientation parameters are provided
        if isinstance(adsorbate, Molecule) and (
            self.custom_mol_file or self.target_vector is not None
        ):
            logger.info("Applying molecule orientation...")
            adsorbate = prepare_molecule_with_orientation(
                adsorbate,
                custom_file=self.custom_mol_file,
                plane_atoms=self.plane_atoms,
                target_vector=self.target_vector,
                extra_rotation=self.extra_rotation,
                rotation_axis=self.rotation_axis,
            )

        # 1. Calculate clean slab energy (skipped when a precalculated energy
        #    is provided, e.g. reused from a previous adsorbate scan)
        slab_job = None
        if self.precalc_slab_energy is None:
            job_counter["current"] += 1
            logger.info(
                f"[{job_counter['current']}/{job_counter['total']}] Creating clean slab energy calculation..."
            )
            slab_job = self.slab_static_maker.make(slab, prev_dir=prev_dir)
            slab_job.name = (
                f"[{job_counter['current']}_of_{job_counter['total']}]_{self.name}_slab"
            )
            jobs.append(slab_job)
        else:
            logger.info("Reusing precalculated slab energy - skipping slab calculation")

        # 2. Calculate isolated adsorbate energy (skipped when precalculated)
        ads_job = None
        if self.precalc_adsorbate_energy is None:
            job_counter["current"] += 1
            logger.info(
                f"[{job_counter['current']}/{job_counter['total']}] Creating adsorbate energy calculation..."
            )
            # Convert Molecule to Structure if needed (molecules need a box for SIESTA)
            if isinstance(adsorbate, Molecule):
                adsorbate_struct = molecule_to_structure_in_box(
                    adsorbate, box_size=20.0
                )
                # Apply the correct spin state for open-shell adsorbates (O2, etc.)
                adsorbate_struct, spin_params = _molecular_spin_setup(
                    adsorbate, adsorbate_struct
                )
                ads_job = self.adsorbate_static_maker.make(adsorbate_struct)
                if spin_params:
                    from atomate2.siesta.powerups import update_user_siesta_settings

                    ads_job = update_user_siesta_settings(ads_job, spin_params)
            else:
                ads_job = self.adsorbate_static_maker.make(adsorbate)
            ads_job.name = f"[{job_counter['current']}_of_{job_counter['total']}]_{self.name}_adsorbate"
            jobs.append(ads_job)
        else:
            logger.info(
                "Reusing precalculated adsorbate energy - skipping adsorbate calculation"
            )

        # 3. Generate adsorption sites
        job_counter["current"] += 1
        logger.info(
            f"[{job_counter['current']}/{job_counter['total']}] "
            f"Generating {self.grid_size[0]}×{self.grid_size[1]} adsorption sites..."  # noqa: RUF001
        )
        sites_job = generate_adsorption_sites(grid_size=self.grid_size)
        sites_job.name = f"[{job_counter['current']}_of_{job_counter['total']}]_{self.name}_generate_sites"
        jobs.append(sites_job)

        # 4. Calculate energies for each site at each height
        logger.info(
            f"Setting up {total_sites_with_heights} site energy calculations "
            f"(jobs {job_counter['current'] + 1}-{job_counter['current'] + total_sites_with_heights * 2})..."
        )
        site_calc_jobs = []

        # Loop over heights first, then xy sites
        for height_idx, height in enumerate(heights_to_scan):
            for site_idx in range(self.grid_size[0] * self.grid_size[1]):
                # Create job that calculates energy for this site at this height
                # This job will access sites_job.output[site_idx]
                job_counter["current"] += 2  # calc + extract
                site_job = self._create_site_calc_job(
                    slab=slab,
                    adsorbate=adsorbate,
                    sites_output=sites_job.output,
                    site_index=site_idx,
                    height=height,
                    height_index=height_idx,
                    progress=f"[{job_counter['current'] - 1}-{job_counter['current']}_of_{job_counter['total']}]",
                    job_num=job_counter["current"] - 1,  # static job number
                    total_jobs=job_counter["total"],
                )
                site_job.name = f"[{job_counter['current'] - 1}_of_{job_counter['total']}]_{self.name}_h{height_idx:02d}_site_{site_idx:03d}"
                jobs.append(site_job)
                site_calc_jobs.append(site_job)

        # 5. Analyze all results and create output files in analysis job directory
        job_counter["current"] += 1
        logger.info(
            f"[{job_counter['current']}/{job_counter['total']}] Creating analysis job with consolidated outputs..."
        )
        analysis_job = self._create_consolidated_analysis_job(
            slab=slab,
            adsorbate=adsorbate,
            slab_job=slab_job,
            ads_job=ads_job,
            sites_job=sites_job,
            site_calc_jobs=site_calc_jobs,
            heights=heights_to_scan,
            placement=self.placement,
            plot_results=self.plot_results,
            write_summary=self.write_summary,
            job_counter=job_counter,
        )
        analysis_job.name = (
            f"[{job_counter['current']}_of_{job_counter['total']}]_{self.name}_analysis"
        )
        jobs.append(analysis_job)

        logger.info(f"✓ Workflow complete: {len(jobs)} jobs created")

        return Flow(jobs, output=analysis_job.output, name=self.name)

    def _create_site_calc_job(
        self,
        slab: Structure,
        adsorbate: Structure | Molecule,
        sites_output,
        site_index: int,
        height: float,
        height_index: int,
        progress: str = "",
        job_num: int = 0,
        total_jobs: int = 0,
    ):
        """
        Create job for calculating energy at a single site at specific height.

        Parameters
        ----------
        slab : Structure
            Clean slab structure.
        adsorbate : Structure | Molecule
            Adsorbate structure.
        sites_output : list
            Output from sites generation job.
        site_index : int
            Index of site in the list.
        height : float
            Height of adsorbate above surface (Å).
        height_index : int
            Index of height in the heights list.
        progress : str
            Progress indicator string for logging.
        job_num : int
            Current job number for counter.
        total_jobs : int
            Total number of jobs for counter.

        Returns
        -------
        Job
            Site energy calculation job that returns dict with site, height, and energy.
        """
        # Get the site as a reference (will be resolved at runtime)
        site_ref = sites_output[site_index]

        # Use module-level function for proper serialization with jobflow-remote
        calc_job = _calc_site_energy_impl(
            slab=slab,
            adsorbate=adsorbate,
            site=site_ref,
            height=height,  # Use the specific height for this job
            placement=self.placement,
            slab_static_maker=self.slab_static_maker,
            site_index=site_index,
            grid_size=self.grid_size,
            job_num=job_num,
            total_jobs=total_jobs,
        )

        # Set the job name
        if job_num > 0:
            calc_job.name = f"[{job_num}_of_{total_jobs}]_adsorption_scan_h{height_index:02d}_site_{site_index:03d}"
        else:
            calc_job.name = f"adsorption_scan_h{height_index:02d}_site_{site_index:03d}"

        return calc_job

    def _create_analysis_job(
        self,
        slab: Structure,
        adsorbate: Structure | Molecule,
        slab_job,
        ads_job,
        sites_job,
        site_calc_jobs: list,
        heights: list[float],
    ):
        """
        Create job that analyzes all site results.

        Parameters
        ----------
        slab : Structure
            Clean slab structure.
        adsorbate : Structure | Molecule
            Adsorbate structure.
        slab_job : Job
            Clean slab energy calculation job.
        ads_job : Job
            Adsorbate energy calculation job.
        sites_job : Job
            Sites generation job.
        site_calc_jobs : list[Job]
            List of site calculation jobs.
        heights : list[float]
            List of heights that were scanned.

        Returns
        -------
        Job
            Analysis job.
        """
        # Collect site energies as a list
        site_energies = []
        for site_job in site_calc_jobs:
            site_energies.append(site_job.output)

        # Run analysis (this is already a @job, so just call it directly)
        analysis_job = analyze_adsorption_scan(
            slab=slab,
            adsorbate=adsorbate,
            site_energies=site_energies,
            slab_energy=slab_job.output.output.energy,
            adsorbate_energy=ads_job.output.output.energy,
            grid_size=self.grid_size,
            heights=heights,  # Pass list of heights
            miller_indices=self.miller_indices,
        )

        return analysis_job

    def _create_consolidated_analysis_job(
        self,
        slab: Structure,
        adsorbate: Structure | Molecule,
        slab_job,
        ads_job,
        sites_job,
        site_calc_jobs: list,
        heights: list[float],
        placement: str,
        plot_results: bool,
        write_summary: bool,
        job_counter: dict,
    ):
        """
        Create consolidated analysis job that creates all outputs in one directory.

        All output files (best structure, plots, summary) are created directly
        in the analysis job's directory.

        Parameters
        ----------
        slab : Structure
            Clean slab structure.
        adsorbate : Structure | Molecule
            Adsorbate structure.
        slab_job : Job
            Clean slab energy calculation job.
        ads_job : Job
            Adsorbate energy calculation job.
        sites_job : Job
            Sites generation job.
        site_calc_jobs : list[Job]
            List of site calculation jobs.
        heights : list[float]
            List of heights that were scanned.
        placement : str
            Placement location ('top' or 'bottom').
        plot_results : bool
            Whether to generate plots.
        write_summary : bool
            Whether to write summary file.
        job_counter : dict
            Job counter for naming (not used, kept for compatibility).

        Returns
        -------
        Job
            Consolidated analysis job.
        """
        # Collect site energies as a list
        site_energies = []
        for site_job in site_calc_jobs:
            site_energies.append(site_job.output)

        # Reference energies: either from the static jobs or precalculated
        # values (plain floats or jobflow OutputReferences from another flow)
        slab_energy = (
            self.precalc_slab_energy
            if slab_job is None
            else slab_job.output.output.energy
        )
        adsorbate_energy = (
            self.precalc_adsorbate_energy
            if ads_job is None
            else ads_job.output.output.energy
        )

        # Create analysis job that creates all files in its directory
        analysis_job = _consolidated_analysis(
            slab=slab,
            adsorbate=adsorbate,
            site_energies=site_energies,
            slab_energy=slab_energy,
            adsorbate_energy=adsorbate_energy,
            grid_size=self.grid_size,
            heights=heights,
            miller_indices=self.miller_indices,
            placement=placement,
            plot_results=plot_results,
            write_summary=write_summary,
        )

        return analysis_job


@dataclass
class AdsorptionOptimizationFlowMaker(BaseSiestaFlowMaker):
    """
    Optimize adsorption geometry at best site from scan.

    This workflow:
    1. Takes scan results as input
    2. Places adsorbate at the best site
    3. Relaxes the structure (optionally fixing slab atoms)
    4. Calculates final adsorption energy
    5. Reports optimization results

    Inherits from BaseSiestaFlowMaker, so dry_run=True automatically propagates
    to child makers (relax_maker, final_static_maker).

    Parameters
    ----------
    name : str
        Name of the flow.
    relax_maker : RelaxMaker
        Maker for geometry optimization.
    final_static_maker : StaticMaker
        Maker for final energy calculation.
    relax_adsorbate_only : bool
        If True, fix slab atoms during relaxation.
    dry_run : bool
        If True, skip SIESTA calculations and only save structures (inherited).
    dry_run_output_dir : str
        Directory to save dry-run structures (inherited).
    dry_run_format : str
        Output format for dry-run structures (inherited).

    Examples
    --------
    >>> from atomate2.siesta.flows.surface import AdsorptionOptimizationFlowMaker
    >>> from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
    >>>
    >>> relax_maker = RelaxMaker.fixed_cell_relaxation()
    >>> static_maker = StaticMaker()
    >>>
    >>> maker = AdsorptionOptimizationFlowMaker(
    ...     relax_maker=relax_maker,
    ...     final_static_maker=static_maker,
    ...     relax_adsorbate_only=True,
    ... )
    """

    name: str = "adsorption_optimization"
    relax_maker: RelaxMaker = field(default_factory=RelaxMaker.fixed_cell_relaxation)
    final_static_maker: StaticMaker = field(default_factory=StaticMaker)
    relax_adsorbate_only: bool = True

    def make(
        self,
        slab: Structure,
        adsorbate: Structure | Molecule,
        best_site: tuple[float, float],
        height: float,
        slab_energy: float,
        adsorbate_energy: float,
        initial_adsorption_energy: float,
    ) -> Flow:
        """
        Create adsorption optimization workflow.

        Parameters
        ----------
        slab : Structure
            Clean slab structure.
        adsorbate : Structure | Molecule
            Adsorbate structure.
        best_site : tuple[float, float]
            Best adsorption site position (fractional).
        height : float
            Initial adsorbate height (Å).
        slab_energy : float
            Clean slab energy (eV).
        adsorbate_energy : float
            Isolated adsorbate energy (eV).
        initial_adsorption_energy : float
            Initial adsorption energy before optimization (eV).

        Returns
        -------
        Flow
            Adsorption optimization workflow.
        """
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        from jobflow import job

        logger.info("AdsorptionOptimizationFlowMaker.make()")

        jobs = []

        # 1. Create initial structure with adsorbate
        @job(name=f"{self.name}_create_structure")
        def create_ads_structure():
            """Create initial adsorption structure."""
            return add_adsorbate_to_slab(slab, adsorbate, best_site, height)

        initial_struct_job = create_ads_structure()
        jobs.append(initial_struct_job)

        # 2. Optionally set constraints
        @job(name=f"{self.name}_add_constraints")
        def add_constraints():
            """Add selective dynamics if needed."""
            struct = initial_struct_job.output

            if self.relax_adsorbate_only:
                # Mark slab atoms as fixed
                n_slab_atoms = len(slab)
                selective_dynamics = [
                    [False, False, False] if i < n_slab_atoms else [True, True, True]
                    for i in range(len(struct))
                ]
                struct.add_site_property("selective_dynamics", selective_dynamics)

            return struct

        constrained_struct_job = add_constraints()
        jobs.append(constrained_struct_job)

        # 3. Relax geometry
        relax_job = self.relax_maker.make(constrained_struct_job.output)
        relax_job.name = f"{self.name}_relax"
        jobs.append(relax_job)

        # 4. Final static calculation
        static_job = self.final_static_maker.make(
            relax_job.output.structure, prev_dir=relax_job.output.dir_name
        )
        static_job.name = f"{self.name}_final_static"
        jobs.append(static_job)

        # 5. Analyze optimization
        @job(name=f"{self.name}_analyze")
        def analyze_optimization():
            """Analyze optimization results."""
            from atomate2.siesta.schemas.adsorption import AdsorptionOptimizationResult
            from atomate2.siesta.schemas.calculation import TaskState

            final_energy = static_job.output.output.energy
            final_adsorption_energy = final_energy - slab_energy - adsorbate_energy

            # Get initial total energy
            initial_total_energy = (
                slab_energy + adsorbate_energy + initial_adsorption_energy
            )

            # Check convergence from relax job state
            converged = relax_job.output.state == TaskState.SUCCESS

            return AdsorptionOptimizationResult(
                initial_site=best_site,
                initial_adsorption_energy=initial_adsorption_energy,
                final_adsorption_energy=final_adsorption_energy,
                energy_improvement=final_adsorption_energy - initial_adsorption_energy,
                initial_total_energy=initial_total_energy,
                final_total_energy=final_energy,
                converged=converged,
                n_ionic_steps=relax_job.output.output.get("n_ionic_steps"),
            )

        analysis_job = analyze_optimization()
        jobs.append(analysis_job)

        return Flow(jobs, output=analysis_job.output, name=self.name)
