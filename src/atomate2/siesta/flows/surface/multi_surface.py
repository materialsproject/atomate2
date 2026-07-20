"""Workflows for multi-surface energy calculations (multiple Miller indices)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jobflow import Flow, job
from pymatgen.core import Structure

from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.jobs.core import StaticMaker

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def calculate_multi_surface_energies(
    structure: Structure,
    miller_indices: list[tuple[int, int, int]],
    bulk_maker: StaticMaker,
    slab_maker: StaticMaker,
    slab_layers: int = 4,
    vacuum_size: float = 15.0,
    symmetrize: bool = False,
    formula_units_per_cell: int | None = None,
    apply_diffuse_basis: bool = False,
    surface_basis: str = "DZP",
    bulk_basis: str = "DZ",
    n_surface_layers: int = 1,
) -> Flow:
    """
    Create workflow for multi-surface energy calculation.

    This creates a Flow with parallel slab calculations for multiple Miller indices.

    Parameters
    ----------
    structure : Structure
        Bulk structure.
    miller_indices : list[tuple]
        List of Miller indices to calculate.
    bulk_maker : StaticMaker
        Maker for bulk calculation.
    slab_maker : StaticMaker
        Maker for slab calculations.
    slab_layers : int
        Number of layers in the slab.
    vacuum_size : float
        Vacuum spacing (Å).
    symmetrize : bool
        Whether to create symmetric slabs.
    formula_units_per_cell : int, optional
        Formula units in bulk cell (auto-detect if None).
    apply_diffuse_basis : bool
        If True, apply diffuse basis to surface atoms.
    surface_basis : str
        Basis size for surface atoms (e.g., "DZP").
    bulk_basis : str
        Basis size for bulk/interior atoms (e.g., "DZ").
    n_surface_layers : int
        Number of outermost layers to treat as surface.

    Returns
    -------
    Flow
        Multi-surface energy calculation workflow.
    """
    from pymatgen.core.surface import SlabGenerator
    import numpy as np
    from collections import Counter

    logger.info("calculate_multi_surface_energies: Creating workflow")

    jobs = []

    # Calculate total number of jobs for counter
    # Calculate total number of jobs for counter (will be updated after generating slabs)

    # 1. Calculate bulk energy
    logger.info("Adding bulk calculation job...")
    bulk_job = bulk_maker.make(structure)
    bulk_job.name = "[1_of_?]_bulk_energy"  # Will update total count later
    jobs.append(bulk_job)

    # Get formula units
    bulk_composition = structure.composition.reduced_composition
    if formula_units_per_cell is None:
        formula_units_per_cell = int(
            structure.composition.num_atoms / bulk_composition.num_atoms
        )

    # 2. Generate slabs and create calculation jobs for each Miller index
    all_slab_data = []
    job_counter = 2  # Start after bulk job (which is job 1)

    for hkl in miller_indices:
        miller_h, miller_k, miller_l = hkl
        logger.info(
            f"Generating slabs for ({miller_h} {miller_k} {miller_l}) surface..."
        )

        # Calculate d-spacing for this Miller index
        d_hkl = structure.lattice.d_hkl(hkl)
        min_slab_size = slab_layers * d_hkl

        logger.info(f"  d-spacing: {d_hkl:.4f} Å, min_slab_size: {min_slab_size:.4f} Å")

        # Generate slabs using pymatgen
        slabgen = SlabGenerator(
            initial_structure=structure,
            miller_index=hkl,
            min_slab_size=min_slab_size,
            min_vacuum_size=vacuum_size,
            lll_reduce=False,
            center_slab=True,
            primitive=True,
            max_normal_search=1,
        )

        # Get slabs - when symmetrize=False, only use the first (initial) slab
        if symmetrize:
            # Symmetric slabs - get all unique terminations
            slabs = slabgen.get_slabs(
                bonds=None,
                ftol=0.1,
                tol=0.1,
                max_broken_bonds=0,
                symmetrize=True,
            )
        else:
            # Asymmetric - just use the initial slab (like Slab_initial.fdf)
            slabs = [slabgen.get_slab(shift=0)]  # Initial slab with no shift

        logger.info(f"  Generated {len(slabs)} slab(s)")

        # Create calculation jobs for each slab
        slab_jobs_for_hkl = []

        for i, slab in enumerate(slabs):
            # Get termination label
            positions = slab.cart_coords
            z_coords = positions[:, 2]
            z_max = z_coords.max()
            top_layer_indices = np.where(z_coords > z_max - 0.5)[0]
            top_species = [slab.species[idx].symbol for idx in top_layer_indices]
            species_count = Counter(top_species)
            dominant_species = species_count.most_common(1)[0][0]
            label = f"{dominant_species}_term{i + 1}"

            logger.info(f"  Adding calculation job for {label}...")

            # Apply diffuse basis to surface atoms if requested
            if apply_diffuse_basis:
                from atomate2.siesta.sets.utils import apply_diffuse_basis_to_surface
                from copy import deepcopy

                species_labels, pao_basissizes, _ = apply_diffuse_basis_to_surface(
                    slab,
                    surface_basis=surface_basis,
                    bulk_basis=bulk_basis,
                    surface_layers=n_surface_layers,
                )
                slab.add_site_property("species_label", species_labels)

                # Create a copy of slab_maker with the PAO.BasisSizes override
                slab_maker_with_basis = deepcopy(slab_maker)
                current_params = (
                    getattr(
                        slab_maker_with_basis.input_set_generator, "user_params", {}
                    )
                    or {}
                )
                current_params["%block PAO.BasisSizes"] = pao_basissizes
                slab_maker_with_basis.input_set_generator.user_params = current_params

                slab_job = slab_maker_with_basis.make(slab)
            else:
                # Create SIESTA job for this slab
                slab_job = slab_maker.make(slab)
            slab_job.name = f"[{job_counter}_of_?]_slab_{miller_h}{miller_k}{miller_l}_{label}"  # Will update total later
            jobs.append(slab_job)
            job_counter += 1

            # Calculate surface area
            cell = slab.lattice.matrix
            surface_area = np.linalg.norm(np.cross(cell[0], cell[1]))

            # Count formula units
            slab_composition = slab.composition
            n_formula_units = slab_composition.num_atoms / bulk_composition.num_atoms

            # Slab thickness
            thickness = z_coords.max() - z_coords.min()

            # Store metadata for analysis job
            slab_jobs_for_hkl.append(
                {
                    "job_output": slab_job.output,  # Store output reference, not job
                    "termination": label,
                    "surface_area": surface_area,
                    "n_formula_units": n_formula_units,
                    "n_atoms": len(slab),
                    "thickness": thickness,
                    "composition": dict(slab_composition.as_dict()),
                    "is_symmetric": symmetrize,
                }
            )

        all_slab_data.append(
            {
                "miller_index": hkl,
                "slab_jobs": slab_jobs_for_hkl,
            }
        )

    # 3. Create analysis job that collects all results
    analysis_job = analyze_multi_surface_results(
        bulk_job_output=bulk_job.output,
        all_slab_data=all_slab_data,
        bulk_composition=bulk_composition,
        formula_units_per_cell=formula_units_per_cell,
    )
    analysis_job.name = (
        f"[{job_counter}_of_?]_multi_surface_analysis"  # Will update total later
    )
    jobs.append(analysis_job)

    # Now we know the total number of jobs - update all job names
    total_jobs = len(jobs)
    logger.info(f"Created workflow with {total_jobs} total jobs")

    # Update job names with actual total count
    jobs[0].name = f"[1_of_{total_jobs}]_bulk_energy"

    job_idx = 2
    for surface_data in all_slab_data:
        for _ in surface_data["slab_jobs"]:
            # Extract the base name from the placeholder format
            old_name = jobs[job_idx - 1].name
            base_name = old_name.split("?]_", 1)[1] if "?]_" in old_name else old_name
            jobs[job_idx - 1].name = f"[{job_idx}_of_{total_jobs}]_{base_name}"
            job_idx += 1

    # Update analysis job name
    jobs[-1].name = f"[{total_jobs}_of_{total_jobs}]_multi_surface_analysis"

    return Flow(jobs, output=analysis_job.output, name="multi_surface_energy")


@job
def analyze_multi_surface_results(
    bulk_job_output: Any,
    all_slab_data: list[dict],
    bulk_composition,
    formula_units_per_cell: int,
) -> dict:
    """
    Analyze results from all slab calculations.

    Parameters
    ----------
    bulk_job_output : Any
        Output from bulk calculation job.
    all_slab_data : list[dict]
        List of slab data for each Miller index.
    bulk_composition : Composition
        Reduced composition of bulk.
    formula_units_per_cell : int
        Number of formula units in bulk cell.

    Returns
    -------
    dict
        Complete multi-surface analysis results.
    """
    from datetime import datetime
    from pymatgen.core import Composition

    logger.info("analyze_multi_surface_results: Starting analysis")

    # Handle serialized Composition
    if isinstance(bulk_composition, dict):
        bulk_composition = Composition(bulk_composition)

    # Extract bulk energy
    bulk_energy = bulk_job_output.output.energy
    bulk_energy_per_formula = bulk_energy / formula_units_per_cell

    # Analyze each Miller index
    all_results = []

    for surface_data in all_slab_data:
        hkl = surface_data["miller_index"]
        miller_h, miller_k, miller_l = hkl
        logger.info(f"Analyzing ({miller_h} {miller_k} {miller_l}) surface...")

        termination_results = []

        for slab_info in surface_data["slab_jobs"]:
            # Extract slab energy from job output reference
            slab_energy = slab_info["job_output"].output.energy

            # Calculate surface energy (1 surface for asymmetric)
            gamma_eV_A2 = (
                slab_energy - slab_info["n_formula_units"] * bulk_energy_per_formula
            ) / slab_info["surface_area"]
            gamma_Jm2 = gamma_eV_A2 * 16.0218

            termination_results.append(
                {
                    "termination": slab_info["termination"],
                    "surface_energy": gamma_eV_A2,
                    "surface_energy_Jm2": gamma_Jm2,
                    "slab_energy": slab_energy,
                    "n_formula_units": slab_info["n_formula_units"],
                    "surface_area": slab_info["surface_area"],
                    "n_atoms": slab_info["n_atoms"],
                    "thickness": slab_info["thickness"],
                    "composition": slab_info["composition"],
                    "is_symmetric": slab_info["is_symmetric"],
                }
            )

        # Find lowest for this surface
        min_gamma = min(t["surface_energy"] for t in termination_results)
        for t in termination_results:
            t["relative_energy"] = t["surface_energy"] - min_gamma
            t["is_lowest"] = abs(t["surface_energy"] - min_gamma) < 1e-6

        all_results.append(
            {
                "miller_index": hkl,
                "terminations": termination_results,
                "n_terminations": len(termination_results),
            }
        )

    # 3. Write summary file
    logger.info("Creating final summary...")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 80)
    lines.append("MULTI-SURFACE ENERGY CALCULATION SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Generated: {timestamp}")
    lines.append(f"Material: {bulk_composition.reduced_formula}")
    lines.append("")

    # THEORETICAL BACKGROUND
    lines.append("=" * 80)
    lines.append("THEORETICAL BACKGROUND")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Surface Energy Definition:")
    lines.append(
        "  Surface energy (γ) represents the excess energy per unit area required"
    )
    lines.append(
        "  to create a surface by cleaving a bulk crystal. It is a fundamental"
    )
    lines.append(
        "  thermodynamic property that determines surface stability and morphology."
    )
    lines.append("")
    lines.append("Calculation Method:")
    lines.append("  For asymmetric slabs (one surface exposed):")
    lines.append("")
    lines.append("    γ = (E_slab - N × E_bulk) / A")
    lines.append("")
    lines.append("  Where:")
    lines.append("    E_slab = Total energy of the slab (DFT calculated)")
    lines.append("    E_bulk = Energy per formula unit in bulk")
    lines.append("    N      = Number of formula units in the slab")
    lines.append("    A      = Surface area (cross-sectional area of slab)")
    lines.append("")
    lines.append("  For symmetric slabs (two identical surfaces):")
    lines.append("")
    lines.append("    γ = (E_slab - N × E_bulk) / (2A)")
    lines.append("")
    lines.append("  The factor of 2 accounts for two equivalent surfaces.")
    lines.append("")
    lines.append("Unit Conversion:")
    lines.append("  1 eV/Ų = 16.0218 J/m²")
    lines.append("")
    lines.append("Physical Interpretation:")
    lines.append("  • Lower γ → More stable surface (energetically favorable)")
    lines.append("  • Higher γ → Less stable surface (higher energy cost to form)")
    lines.append(
        "  • Crystal morphology follows Wulff construction (minimizes total surface energy)"
    )
    lines.append("  • Surfaces with lowest γ dominate equilibrium crystal shapes")
    lines.append("")

    # CALCULATION DETAILS
    lines.append("=" * 80)
    lines.append("CALCULATION DETAILS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Bulk Reference:")
    lines.append(f"  Energy per formula unit: {bulk_energy_per_formula:.6f} eV")
    lines.append(f"  Total bulk energy: {bulk_energy:.6f} eV")
    lines.append(f"  Formula units in cell: {formula_units_per_cell}")
    lines.append("")
    lines.append(f"Number of Miller indices calculated: {len(all_results)}")
    lines.append(
        f"Total slab calculations: {sum(len(r['terminations']) for r in all_results)}"
    )
    lines.append("")

    # SURFACE ENERGY COMPARISON
    lines.append("=" * 80)
    lines.append("SURFACE ENERGY COMPARISON")
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        f"  {'Surface':<10} {'Lowest γ (eV/Ų)':<18} "
        f"{'Lowest γ (J/m²)':<18} {'Best Term.':<15} {'# Terms'}"
    )
    lines.append("  " + "-" * 77)

    for result in all_results:
        hkl = result["miller_index"]
        miller_h, miller_k, miller_l = hkl
        terminations = result["terminations"]
        lowest = min(terminations, key=lambda t: t["surface_energy"])

        lines.append(
            f"  ({miller_h} {miller_k} {miller_l}){' ' * (7 - len(str(miller_h) + str(miller_k) + str(miller_l)))} "
            f"{lowest['surface_energy']:>17.4f} "
            f"{lowest['surface_energy_Jm2']:>17.2f}  "
            f"{lowest['termination']:<15} {len(terminations)}"
        )

    # Find global minimum
    all_lowest = [
        (r["miller_index"], min(r["terminations"], key=lambda t: t["surface_energy"]))
        for r in all_results
    ]
    global_min_hkl, global_min_term = min(
        all_lowest, key=lambda x: x[1]["surface_energy"]
    )
    miller_h, miller_k, miller_l = global_min_hkl

    lines.append("")
    lines.append("Global Minimum:")
    lines.append(f"  Surface: ({miller_h} {miller_k} {miller_l})")
    lines.append(f"  Termination: {global_min_term['termination']}")
    lines.append(
        f"  γ = {global_min_term['surface_energy']:.4f} eV/Ų ({global_min_term['surface_energy_Jm2']:.2f} J/m²)"
    )
    lines.append("")

    # DETAILED RESULTS FOR EACH SURFACE
    lines.append("=" * 80)
    lines.append("DETAILED RESULTS BY SURFACE")
    lines.append("=" * 80)

    for result in all_results:
        hkl = result["miller_index"]
        miller_h, miller_k, miller_l = hkl
        terminations = result["terminations"]

        lines.append("")
        lines.append("-" * 80)
        lines.append(f"Surface: ({miller_h} {miller_k} {miller_l})")
        lines.append("-" * 80)
        lines.append(f"Number of terminations calculated: {len(terminations)}")
        lines.append("")

        # Sort terminations by energy
        sorted_terms = sorted(terminations, key=lambda t: t["surface_energy"])

        for i, term in enumerate(sorted_terms, 1):
            lines.append(f"Termination {i}: {term['termination']}")
            lines.append("  Surface Energy:")
            lines.append(f"    γ = {term['surface_energy']:.6f} eV/Ų")
            lines.append(f"    γ = {term['surface_energy_Jm2']:.4f} J/m²")
            if term["is_lowest"]:
                lines.append("    ★ LOWEST for this surface")
            else:
                lines.append(
                    f"    Δγ = +{term['relative_energy']:.6f} eV/Ų (relative to lowest)"
                )
            lines.append("")
            lines.append("  Slab Properties:")
            lines.append(f"    Total energy: {term['slab_energy']:.6f} eV")
            lines.append(f"    Number of atoms: {term['n_atoms']}")
            lines.append(f"    Formula units: {term['n_formula_units']:.2f}")
            lines.append(f"    Thickness: {term['thickness']:.4f} Å")
            lines.append(f"    Surface area: {term['surface_area']:.4f} Ų")
            lines.append(f"    Symmetric: {term['is_symmetric']}")

            # Show composition
            comp_str = ", ".join(
                [f"{elem}: {count}" for elem, count in term["composition"].items()]
            )
            lines.append(f"    Composition: {comp_str}")
            lines.append("")

    lines.append("=" * 80)
    lines.append("NOTES")
    lines.append("=" * 80)
    lines.append("")
    lines.append("• Surface energies are calculated using DFT with SIESTA")
    lines.append("• Asymmetric slabs have one surface (no factor of 2 in formula)")
    lines.append("• Symmetric slabs have two identical surfaces (divide by 2A)")
    lines.append("• Lower surface energy indicates more stable surface")
    lines.append("• Termination refers to the atomic species at the surface layer")
    lines.append("• Different terminations can have significantly different energies")
    lines.append("• Convergence with respect to slab thickness should be verified")
    lines.append("• K-point sampling and energy cutoff affect accuracy")
    lines.append("")
    lines.append("=" * 80)

    summary_content = "\n".join(lines)

    # Add standard footer
    from atomate2.siesta.utils.text_output import get_standard_footer

    footer = get_standard_footer(
        width=80,
        additional_info={
            "Analysis type": "Multi-surface energy calculation",
            "Number of Miller indices": str(len(all_results)),
            "Bulk composition": bulk_composition,
        },
    )

    with open("multi_surface_summary.txt", "w") as f:
        f.write(summary_content)
        f.write("\n" + footer)

    # 4. Create comparison plots
    logger.info("Creating comparison plots...")
    plot_filename = _create_multi_surface_plot(all_results, bulk_composition)

    logger.info("Finished multi-surface analysis")

    return {
        "bulk_energy": bulk_energy,
        "bulk_energy_per_formula": bulk_energy_per_formula,
        "formula_units_per_cell": formula_units_per_cell,
        "surface_results": all_results,
        "summary_file": "multi_surface_summary.txt",
        "plot_file": plot_filename,
        "n_miller_indices": len(all_results),
    }


@dataclass
class MultiSurfaceEnergyFlowMaker(BaseSiestaFlowMaker):
    """
    Calculate surface energies for multiple Miller indices automatically.

    This workflow calculates surface energies for all specified Miller indices
    in a single integrated job. Slabs are generated internally using pymatgen.

    Inherits from BaseSiestaFlowMaker, so dry_run=True automatically propagates
    to child makers (bulk_static_maker, slab_static_maker).

    Parameters
    ----------
    name : str
        Name of the flow.
    miller_indices : list[tuple[int, int, int]]
        List of Miller indices to calculate.
    bulk_static_maker : StaticMaker
        Maker for bulk energy calculation.
    slab_static_maker : StaticMaker
        Maker for slab energy calculations.
    slab_layers : int
        Number of atomic layers in the slab.
    vacuum_size : float
        Vacuum spacing in Ångströms.
    symmetrize : bool
        Whether to generate symmetric slabs.
    formula_units_per_cell : int, optional
        Number of formula units in bulk unit cell (auto-detect if None).
    apply_diffuse_basis : bool
        If True, apply diffuse basis to surface atoms. Surface atoms get larger
        basis sets because electrons extend further into vacuum. Default: False.
    surface_basis : str
        Basis size for surface atoms when apply_diffuse_basis=True. Default: "DZP".
    bulk_basis : str
        Basis size for bulk atoms when apply_diffuse_basis=True. Default: "DZ".
    surface_layers : int
        Number of outermost layers to treat as surface when apply_diffuse_basis=True.
        Default: 1 (top and bottom layers).
    dry_run : bool
        If True, skip SIESTA calculations and only save structures (inherited).
    dry_run_output_dir : str
        Directory to save dry-run structures (inherited).
    dry_run_format : str
        Output format for dry-run structures (inherited).

    Examples
    --------
    >>> from pymatgen.core import Structure
    >>> from atomate2.siesta.flows.surface import MultiSurfaceEnergyFlowMaker
    >>> from atomate2.siesta.jobs.core import StaticMaker
    >>>
    >>> # Load bulk structure
    >>> bulk = Structure.from_file("MgO.cif")
    >>>
    >>> # Setup SIESTA parameters for bulk and slab calculations
    >>> # IMPORTANT: Use reduced k-points in vacuum direction for slabs
    >>> bulk_maker = StaticMaker(
    ...     user_params={
    ...         "PAO.BasisSize": "DZP",
    ...         "Mesh.Cutoff": "300 Ry",
    ...         "a2s_kpts": [3, 3, 3],  # Isotropic for bulk
    ...     }
    ... )
    >>> slab_maker = StaticMaker(
    ...     user_params={
    ...         "PAO.BasisSize": "DZP",
    ...         "Mesh.Cutoff": "300 Ry",
    ...         "a2s_kpts": [2, 2, 1],  # Reduced along z (vacuum direction)
    ...     }
    ... )
    >>>
    >>> # Create workflow for multiple surfaces
    >>> maker = MultiSurfaceEnergyFlowMaker(
    ...     miller_indices=[(0, 0, 1), (1, 1, 0), (1, 1, 1)],
    ...     bulk_static_maker=bulk_maker,
    ...     slab_static_maker=slab_maker,
    ...     slab_layers=4,
    ...     vacuum_size=15.0,
    ... )
    >>> flow = maker.make(bulk)
    >>>
    >>> # Run workflow
    >>> from jobflow import run_locally
    >>> results = run_locally(flow, create_folders=True)

    Notes
    -----
    • K-point sampling: Surface slabs should use reduced k-points in the vacuum
      direction (typically z). For example, if bulk uses [3,3,3], slabs should
      use [2,2,1].
    • The slab_static_maker uses smart defaults: if no k-points are specified,
      it automatically uses [2,2,1] assuming (001)-type surfaces.
    • For non-(001) surfaces or higher accuracy, explicitly set k-points via
      user_params as shown in the example above.
    """

    name: str = "multi_surface_energy"
    miller_indices: list[tuple[int, int, int]] = field(
        default_factory=lambda: [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ]
    )
    bulk_static_maker: StaticMaker = field(default_factory=StaticMaker)
    slab_static_maker: StaticMaker = field(default_factory=StaticMaker)
    slab_layers: int = 4
    vacuum_size: float = 15.0
    symmetrize: bool = False
    formula_units_per_cell: int | None = None
    plot_results: bool = (
        True  # Not used in simplified version but kept for compatibility
    )
    write_summary: bool = (
        True  # Not used in simplified version but kept for compatibility
    )
    # Diffuse basis for surface atoms
    apply_diffuse_basis: bool = False
    surface_basis: str = "DZP"
    bulk_basis: str = "DZ"
    surface_layers: int = 1

    def __post_init__(self):
        """Apply smart defaults for slab k-points if not explicitly set by user.

        If slab_static_maker was created with default_factory (no custom input_set_generator),
        override the default [4,4,4] k-points with surface-appropriate [2,2,1].
        """
        # CRITICAL: Call parent __post_init__ to propagate custodian/tier settings
        super().__post_init__()

        from atomate2.siesta.sets.core import StaticSetGenerator

        # Get current user_params
        current_generator = self.slab_static_maker.input_set_generator
        current_user_params = getattr(current_generator, "user_params", {})

        # Check if this is a default StaticSetGenerator (user didn't customize it)
        # We detect this by checking if k-points are the default [4,4,4]
        current_kpts = current_user_params.get(
            "a2s_kpts", current_user_params.get("kpts")
        )

        # If k-points are [4,4,4] (the StaticSetGenerator default), replace with [2,2,1]
        if current_kpts == [4, 4, 4]:
            logger.info(
                "Detected default k-points [4,4,4] in slab_static_maker. "
                "Applying surface-optimized default: [2,2,1] (assumes (001)-type surfaces)."
            )
            # Create new user_params dict with updated k-points
            new_user_params = dict(current_user_params)  # Copy existing params
            new_user_params["a2s_kpts"] = [2, 2, 1]

            # Create new generator with updated params
            self.slab_static_maker.input_set_generator = StaticSetGenerator(
                user_params=new_user_params
            )

    def make(self, structure: Structure, prev_dir: str | None = None) -> Flow:
        """
        Create multi-surface energy calculation workflow.

        Parameters
        ----------
        structure : Structure
            Bulk structure.
        prev_dir : str, optional
            Previous directory for restart.

        Returns
        -------
        Flow
            Multi-surface energy calculation flow.
        """
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        logger.info("MultiSurfaceEnergyFlowMaker.make()")

        flow = calculate_multi_surface_energies(
            structure=structure,
            miller_indices=self.miller_indices,
            bulk_maker=self.bulk_static_maker,
            slab_maker=self.slab_static_maker,
            slab_layers=self.slab_layers,
            vacuum_size=self.vacuum_size,
            symmetrize=self.symmetrize,
            formula_units_per_cell=self.formula_units_per_cell,
            apply_diffuse_basis=self.apply_diffuse_basis,
            surface_basis=self.surface_basis,
            bulk_basis=self.bulk_basis,
            n_surface_layers=self.surface_layers,
        )
        flow.name = self.name

        return flow


def _create_multi_surface_plot(all_results: list[dict], bulk_composition) -> str:
    """
    Create comprehensive comparison plots for multi-surface energy results.

    Parameters
    ----------
    all_results : list[dict]
        List of surface analysis results.
    bulk_composition : Composition or dict
        Bulk material composition (may be serialized as dict).

    Returns
    -------
    str
        Filename of generated plot.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from pymatgen.core import Composition
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")
        return None

    # Handle bulk_composition if it's a dict (serialized Composition)
    if isinstance(bulk_composition, dict):
        bulk_composition = Composition(bulk_composition)

    formula = bulk_composition.reduced_formula

    # Prepare data
    surfaces = []
    lowest_energies_eV = []
    lowest_energies_Jm2 = []
    n_terminations = []
    all_termination_data = []

    for result in all_results:
        hkl = result["miller_index"]
        miller_h, miller_k, miller_l = hkl
        surface_label = f"({miller_h}{miller_k}{miller_l})"
        surfaces.append(surface_label)

        terminations = result["terminations"]
        n_terminations.append(len(terminations))

        # Find lowest for bar plot
        lowest = min(terminations, key=lambda t: t["surface_energy"])
        lowest_energies_eV.append(lowest["surface_energy"])
        lowest_energies_Jm2.append(lowest["surface_energy_Jm2"])

        # Collect all terminations for scatter plot
        for t in terminations:
            all_termination_data.append(
                {
                    "surface": surface_label,
                    "hkl": hkl,
                    "termination": t["termination"],
                    "energy_eV": t["surface_energy"],
                    "energy_Jm2": t["surface_energy_Jm2"],
                    "is_lowest": t["is_lowest"],
                }
            )

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Bar chart of lowest surface energies
    x_pos = np.arange(len(surfaces))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(surfaces)))

    bars = ax1.bar(
        x_pos, lowest_energies_eV, color=colors, alpha=0.7, edgecolor="black"
    )
    ax1.set_xlabel("Miller Index", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Surface Energy (eV/Ų)", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Lowest Surface Energies - {formula}", fontsize=14, fontweight="bold"
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(surfaces, rotation=0, fontsize=11)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for i, (bar, energy) in enumerate(zip(bars, lowest_energies_eV)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{energy:.3f}\n({lowest_energies_Jm2[i]:.1f} J/m²)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 2: Scatter plot showing all terminations
    surface_indices = {surf: i for i, surf in enumerate(surfaces)}

    for term_data in all_termination_data:
        x = surface_indices[term_data["surface"]]
        y = term_data["energy_eV"]
        is_lowest = term_data["is_lowest"]

        # Use different markers for lowest vs. higher energy
        if is_lowest:
            ax2.scatter(
                x,
                y,
                s=150,
                c="red",
                marker="*",
                edgecolors="black",
                linewidths=1.5,
                zorder=3,
                label="Lowest" if x == 0 else "",
            )
        else:
            ax2.scatter(
                x,
                y,
                s=80,
                c="blue",
                marker="o",
                alpha=0.6,
                edgecolors="black",
                linewidths=0.5,
                zorder=2,
                label="Higher energy" if x == 0 and not is_lowest else "",
            )

    ax2.set_xlabel("Miller Index", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Surface Energy (eV/Ų)", fontsize=12, fontweight="bold")
    ax2.set_title(f"All Terminations - {formula}", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(surfaces, rotation=0, fontsize=11)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # Add legend (remove duplicates)
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=10)

    # Add text box with summary
    summary_text = f"Calculated {len(surfaces)} surfaces\n"
    summary_text += f"Total terminations: {sum(n_terminations)}\n"
    global_min = min(lowest_energies_eV)
    global_min_surf = surfaces[lowest_energies_eV.index(global_min)]
    summary_text += f"Global minimum: {global_min:.3f} eV/Ų\n"
    summary_text += f"  @ {global_min_surf}"

    ax2.text(
        0.02,
        0.98,
        summary_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save figure
    filename = "multi_surface_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved multi-surface comparison plot: {filename}")

    return filename
