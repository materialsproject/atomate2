"""Surface energy convergence workflow for slab thickness and vacuum testing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jobflow import Flow, job

from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.utils.common import print_docstring_in_box

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


@dataclass
class SurfaceEnergyConvergenceFlowMaker(BaseSiestaFlowMaker):
    """
    Workflow to test convergence of surface energy with slab thickness and vacuum.

    This workflow systematically tests how surface energy converges with:
    1. **Slab thickness** (number of atomic layers) - ensures bulk-like interior
    2. **Vacuum thickness** - eliminates periodic image interactions

    The workflow runs calculations in parallel for efficiency and generates:
    - Convergence plots (surface energy vs. layers, surface energy vs. vacuum)
    - Summary file with recommended parameters
    - Convergence analysis with automatic threshold detection

    Inherits from BaseSiestaFlowMaker, so dry_run=True automatically propagates
    to child makers (bulk_static_maker, slab_static_maker).

    Parameters
    ----------
    name : str
        Name of the flow.
    miller_index : tuple[int, int, int]
        Miller index for the surface (e.g., (1, 1, 1) for (111) surface).
    bulk_static_maker : StaticMaker
        Maker for bulk energy calculation.
    slab_static_maker : StaticMaker
        Maker for slab energy calculations.
    slab_layers : list[int]
        List of slab thicknesses (number of layers) to test.
        Default: [4, 6, 8, 10, 12] layers.
    vacuum_sizes : list[float]
        List of vacuum thicknesses to test (in Ångströms).
        Default: [10.0, 12.5, 15.0, 17.5, 20.0] Å.
    convergence_mode : str
        Mode for convergence testing:
        - "layers": Test slab thickness with fixed vacuum (default)
        - "vacuum": Test vacuum thickness with fixed layers
        - "both": Test both (2D grid of calculations)
    fixed_vacuum : float
        Fixed vacuum when testing layer convergence (Å). Default: 15.0.
    fixed_layers : int
        Fixed layers when testing vacuum convergence. Default: 8.
    symmetrize : bool
        Whether to generate symmetric slabs. Default: False.
    convergence_threshold : float
        Convergence threshold for surface energy (J/m²). Default: 0.01.
    dry_run : bool
        If True, skip SIESTA calculations and only save structures (inherited).
    dry_run_output_dir : str
        Directory to save dry-run structures (inherited).
    dry_run_format : str
        Output format for dry-run structures (inherited).

    Examples
    --------
    >>> from pymatgen.core import Structure
    >>> from atomate2.siesta.flows.surface.convergence import (
    ...     SurfaceEnergyConvergenceFlowMaker,
    ... )
    >>> from atomate2.siesta.jobs.core import StaticMaker
    >>>
    >>> # Load bulk structure
    >>> bulk = Structure.from_file("MgO.cif")
    >>>
    >>> # Test layer convergence for (100) surface
    >>> maker = SurfaceEnergyConvergenceFlowMaker(
    ...     miller_index=(1, 0, 0),
    ...     slab_layers=[4, 6, 8, 10, 12],
    ...     convergence_mode="layers",
    ... )
    >>> flow = maker.make(bulk)
    >>>
    >>> # Test vacuum convergence
    >>> maker_vacuum = SurfaceEnergyConvergenceFlowMaker(
    ...     miller_index=(1, 1, 1),
    ...     vacuum_sizes=[10, 12.5, 15, 17.5, 20],
    ...     convergence_mode="vacuum",
    ...     fixed_layers=8,
    ... )
    >>> flow_vacuum = maker_vacuum.make(bulk)
    >>>
    >>> # Test both (comprehensive)
    >>> maker_full = SurfaceEnergyConvergenceFlowMaker(
    ...     miller_index=(1, 1, 0),
    ...     slab_layers=[4, 6, 8, 10],
    ...     vacuum_sizes=[10, 15, 20],
    ...     convergence_mode="both",
    ... )
    >>> flow_full = maker_full.make(bulk)

    Notes
    -----
    **Typical Convergence Behavior**:

    - **Slab thickness**: Surface energy typically converges within 4-8 layers
      for simple metals, but may require 10-15 layers for oxides or surfaces
      with significant relaxation.

    - **Vacuum thickness**: Usually 12-15 Å is sufficient for most systems.
      Systems with large dipole moments may need 20+ Å.

    **Convergence Criteria**:

    - Publication quality: < 0.01 J/m² change
    - Standard calculations: < 0.05 J/m² change
    - Quick screening: < 0.1 J/m² change

    **K-point Considerations**:

    - Slab calculations should use reduced k-points in the vacuum direction.
    - If bulk uses [4,4,4], slabs typically use [4,4,1] for (001) surfaces.
    """

    name: str = "surface_energy_convergence"
    miller_index: tuple[int, int, int] = (1, 0, 0)
    bulk_static_maker: StaticMaker = field(default_factory=StaticMaker)
    slab_static_maker: StaticMaker = field(default_factory=StaticMaker)
    slab_layers: list[int] = field(default_factory=lambda: [4, 6, 8, 10, 12])
    vacuum_sizes: list[float] = field(
        default_factory=lambda: [10.0, 12.5, 15.0, 17.5, 20.0]
    )
    convergence_mode: str = "layers"  # "layers", "vacuum", or "both"
    fixed_vacuum: float = 15.0
    fixed_layers: int = 8
    symmetrize: bool = False
    convergence_threshold: float = 0.01  # J/m²

    def __post_init__(self):
        """Validate parameters and apply smart defaults."""
        super().__post_init__()

        # Validate convergence mode
        valid_modes = ["layers", "vacuum", "both"]
        if self.convergence_mode not in valid_modes:
            raise ValueError(
                f"convergence_mode must be one of {valid_modes}, "
                f"got '{self.convergence_mode}'"
            )

        # Apply smart k-point defaults for slab calculations
        from atomate2.siesta.sets.core import StaticSetGenerator

        current_generator = self.slab_static_maker.input_set_generator
        current_user_params = getattr(current_generator, "user_params", {}) or {}
        current_kpts = current_user_params.get(
            "a2s_kpts", current_user_params.get("kpts")
        )

        # If using default k-points [4,4,4], change to [2,2,1] for surfaces
        if current_kpts == [4, 4, 4]:
            logger.info(
                "Detected default k-points [4,4,4] in slab_static_maker. "
                "Applying surface-optimized default: [2,2,1]."
            )
            new_user_params = dict(current_user_params)
            new_user_params["a2s_kpts"] = [2, 2, 1]
            self.slab_static_maker.input_set_generator = StaticSetGenerator(
                user_params=new_user_params
            )

    def make(self, structure: Structure, prev_dir: str | None = None) -> Flow:
        """
        Create surface energy convergence workflow.

        Parameters
        ----------
        structure : Structure
            Bulk structure.
        prev_dir : str, optional
            Previous directory for restart.

        Returns
        -------
        Flow
            Surface energy convergence workflow.
        """
        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        logger.info(
            f"SurfaceEnergyConvergenceFlowMaker.make() - "
            f"Miller index: {self.miller_index}, mode: {self.convergence_mode}"
        )

        jobs = []
        job_counter = 1

        # 1. Bulk calculation
        bulk_job = self.bulk_static_maker.make(structure, prev_dir=prev_dir)
        bulk_job.name = f"[{job_counter}]_bulk_energy"
        jobs.append(bulk_job)
        job_counter += 1

        # Get formula units
        bulk_composition = structure.composition.reduced_composition
        formula_units_per_cell = int(
            structure.composition.num_atoms / bulk_composition.num_atoms
        )

        # 2. Generate slab calculations based on mode
        slab_jobs_metadata = []

        if self.convergence_mode == "layers":
            # Test slab thickness with fixed vacuum
            for n_layers in self.slab_layers:
                slab_job, metadata = self._create_slab_job(
                    structure=structure,
                    n_layers=n_layers,
                    vacuum=self.fixed_vacuum,
                    job_counter=job_counter,
                    bulk_composition=bulk_composition,
                )
                jobs.append(slab_job)
                slab_jobs_metadata.append(metadata)
                job_counter += 1

        elif self.convergence_mode == "vacuum":
            # Test vacuum thickness with fixed layers
            for vacuum in self.vacuum_sizes:
                slab_job, metadata = self._create_slab_job(
                    structure=structure,
                    n_layers=self.fixed_layers,
                    vacuum=vacuum,
                    job_counter=job_counter,
                    bulk_composition=bulk_composition,
                )
                jobs.append(slab_job)
                slab_jobs_metadata.append(metadata)
                job_counter += 1

        else:  # both
            # Test both parameters (2D grid)
            for n_layers in self.slab_layers:
                for vacuum in self.vacuum_sizes:
                    slab_job, metadata = self._create_slab_job(
                        structure=structure,
                        n_layers=n_layers,
                        vacuum=vacuum,
                        job_counter=job_counter,
                        bulk_composition=bulk_composition,
                    )
                    jobs.append(slab_job)
                    slab_jobs_metadata.append(metadata)
                    job_counter += 1

        # 3. Analysis job
        analysis_job = analyze_surface_convergence(
            bulk_job_output=bulk_job.output,
            slab_jobs_metadata=slab_jobs_metadata,
            bulk_composition=dict(bulk_composition.as_dict()),
            formula_units_per_cell=formula_units_per_cell,
            miller_index=self.miller_index,
            convergence_mode=self.convergence_mode,
            convergence_threshold=self.convergence_threshold,
            symmetrize=self.symmetrize,
        )
        analysis_job.name = f"[{job_counter}]_convergence_analysis"
        jobs.append(analysis_job)

        return Flow(jobs, output=analysis_job.output, name=self.name)

    def _create_slab_job(
        self,
        structure: Structure,
        n_layers: int,
        vacuum: float,
        job_counter: int,
        bulk_composition,
    ) -> tuple[Any, dict]:
        """
        Create a slab calculation job with given parameters.

        Returns
        -------
        tuple
            (job, metadata_dict)
        """
        import numpy as np
        from pymatgen.core.surface import SlabGenerator

        hkl = self.miller_index
        miller_str = f"{hkl[0]}{hkl[1]}{hkl[2]}"

        # Calculate d-spacing and minimum slab size
        d_hkl = structure.lattice.d_hkl(hkl)
        min_slab_size = n_layers * d_hkl

        logger.debug(
            f"Creating slab: {n_layers} layers, {vacuum} Å vacuum, "
            f"d-spacing: {d_hkl:.4f} Å"
        )

        # Generate slab
        slabgen = SlabGenerator(
            initial_structure=structure,
            miller_index=hkl,
            min_slab_size=min_slab_size,
            min_vacuum_size=vacuum,
            lll_reduce=False,
            center_slab=True,
            primitive=True,
            max_normal_search=1,
        )

        if self.symmetrize:
            slabs = slabgen.get_slabs(
                bonds=None,
                ftol=0.1,
                tol=0.1,
                max_broken_bonds=0,
                symmetrize=True,
            )
            slab = slabs[0] if slabs else slabgen.get_slab(shift=0)
        else:
            slab = slabgen.get_slab(shift=0)

        # Create job
        slab_job = self.slab_static_maker.make(slab)
        slab_job.name = f"[{job_counter}]_slab_{miller_str}_{n_layers}L_{vacuum:.1f}A"

        # Calculate surface area
        cell = slab.lattice.matrix
        surface_area = np.linalg.norm(np.cross(cell[0], cell[1]))

        # Calculate formula units in slab
        slab_composition = slab.composition
        n_formula_units = slab_composition.num_atoms / bulk_composition.num_atoms

        # Slab thickness
        positions = slab.cart_coords
        z_coords = positions[:, 2]
        thickness = z_coords.max() - z_coords.min()

        metadata = {
            "job_output": slab_job.output,
            "n_layers": n_layers,
            "vacuum": vacuum,
            "surface_area": surface_area,
            "n_formula_units": n_formula_units,
            "n_atoms": len(slab),
            "thickness": thickness,
            "d_hkl": d_hkl,
        }

        return slab_job, metadata


@job
def analyze_surface_convergence(
    bulk_job_output: Any,
    slab_jobs_metadata: list[dict],
    bulk_composition: dict,
    formula_units_per_cell: int,
    miller_index: tuple[int, int, int],
    convergence_mode: str,
    convergence_threshold: float,
    symmetrize: bool,
) -> dict:
    """
    Analyze surface energy convergence results.

    Parameters
    ----------
    bulk_job_output : Any
        Output from bulk calculation.
    slab_jobs_metadata : list[dict]
        Metadata for each slab calculation including job outputs.
    bulk_composition : dict
        Bulk composition as dict.
    formula_units_per_cell : int
        Formula units in bulk cell.
    miller_index : tuple
        Miller index of the surface.
    convergence_mode : str
        "layers", "vacuum", or "both".
    convergence_threshold : float
        Threshold for convergence (J/m²).
    symmetrize : bool
        Whether slabs are symmetric.

    Returns
    -------
    dict
        Convergence analysis results.
    """
    import numpy as np
    from pymatgen.core import Composition

    logger.info("Analyzing surface energy convergence...")

    # Handle serialized Composition
    if isinstance(bulk_composition, dict):
        bulk_composition = Composition(bulk_composition)

    # At this point bulk_composition is definitely a Composition object
    assert isinstance(bulk_composition, Composition)

    miller_str = f"({miller_index[0]} {miller_index[1]} {miller_index[2]})"

    # Check if this is a dry-run (outputs are dicts, not Response objects)
    is_dry_run = isinstance(bulk_job_output, dict)

    if is_dry_run:
        logger.warning(
            "Dry-run mode detected - no actual energies available. "
            "Returning placeholder results."
        )
        return {
            "miller_index": miller_index,
            "bulk_energy": None,
            "bulk_energy_per_formula": None,
            "convergence_mode": convergence_mode,
            "convergence_threshold": convergence_threshold,
            "results": [],
            "converged": None,
            "converged_at": None,
            "recommended_layers": None,
            "recommended_vacuum": None,
            "final_surface_energy_Jm2": None,
            "summary_file": None,
            "plot_files": {},
            "dry_run": True,
            "message": "Dry-run mode: no energies available for convergence analysis",
        }

    # Extract bulk energy
    bulk_energy = bulk_job_output.output.energy
    bulk_energy_per_formula = bulk_energy / formula_units_per_cell

    # Process slab results
    results = []
    for slab_info in slab_jobs_metadata:
        slab_energy = slab_info["job_output"].output.energy

        # Calculate surface energy
        # Factor of 1 for asymmetric slabs, 2 for symmetric
        n_surfaces = 2 if symmetrize else 1
        gamma_eV_A2 = (  # noqa: N806
            slab_energy - slab_info["n_formula_units"] * bulk_energy_per_formula
        ) / (n_surfaces * slab_info["surface_area"])
        gamma_Jm2 = gamma_eV_A2 * 16.0218  # Convert to J/m²  # noqa: N806

        results.append(
            {
                "n_layers": slab_info["n_layers"],
                "vacuum": slab_info["vacuum"],
                "slab_energy": slab_energy,
                "surface_energy_eV_A2": gamma_eV_A2,
                "surface_energy_Jm2": gamma_Jm2,
                "surface_area": slab_info["surface_area"],
                "n_formula_units": slab_info["n_formula_units"],
                "n_atoms": slab_info["n_atoms"],
                "thickness": slab_info["thickness"],
            }
        )

    # Sort results
    if convergence_mode == "layers":
        results = sorted(results, key=lambda x: x["n_layers"])
    elif convergence_mode == "vacuum":
        results = sorted(results, key=lambda x: x["vacuum"])
    else:
        results = sorted(results, key=lambda x: (x["n_layers"], x["vacuum"]))

    # Calculate convergence metrics
    surface_energies = np.array([r["surface_energy_Jm2"] for r in results])

    # Find converged point (difference from last value < threshold)
    reference_energy = surface_energies[-1]
    energy_diffs = np.abs(surface_energies - reference_energy)

    converged_idx = None
    for i, diff in enumerate(energy_diffs):
        if diff < convergence_threshold:
            converged_idx = i
            break

    # Generate plots
    plot_files = _create_convergence_plots(
        results=results,
        convergence_mode=convergence_mode,
        miller_str=miller_str,
        formula=bulk_composition.reduced_formula,
        convergence_threshold=convergence_threshold,
    )

    # Generate summary file
    summary_file = _write_convergence_summary(
        results=results,
        bulk_energy=bulk_energy,
        bulk_energy_per_formula=bulk_energy_per_formula,
        formula_units_per_cell=formula_units_per_cell,
        miller_str=miller_str,
        formula=bulk_composition.reduced_formula,
        convergence_mode=convergence_mode,
        convergence_threshold=convergence_threshold,
        converged_idx=converged_idx,
        symmetrize=symmetrize,
    )

    # Prepare output
    output = {
        "miller_index": miller_index,
        "bulk_energy": bulk_energy,
        "bulk_energy_per_formula": bulk_energy_per_formula,
        "convergence_mode": convergence_mode,
        "convergence_threshold": convergence_threshold,
        "results": results,
        "converged": converged_idx is not None,
        "converged_at": results[converged_idx] if converged_idx is not None else None,
        "recommended_layers": (
            results[converged_idx]["n_layers"] if converged_idx is not None else None
        ),
        "recommended_vacuum": (
            results[converged_idx]["vacuum"] if converged_idx is not None else None
        ),
        "final_surface_energy_Jm2": reference_energy,
        "summary_file": summary_file,
        "plot_files": plot_files,
    }

    logger.info(
        f"Surface energy convergence analysis complete. "
        f"Converged: {output['converged']}"
    )

    return output


def _create_convergence_plots(
    results: list[dict],
    convergence_mode: str,
    miller_str: str,
    formula: str,
    convergence_threshold: float,
) -> dict[str, str]:
    """Create convergence plots."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")
        return {}

    plot_files = {}

    if convergence_mode == "layers":
        # Plot surface energy vs. number of layers
        layers = [r["n_layers"] for r in results]
        energies = [r["surface_energy_Jm2"] for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Absolute surface energy
        ax1.plot(layers, energies, "o-", linewidth=2, markersize=10, color="#1f77b4")
        ax1.set_xlabel("Number of Layers", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Surface Energy (J/m²)", fontsize=12, fontweight="bold")
        ax1.set_title(
            f"Surface Energy vs. Slab Thickness\n{formula} {miller_str}",
            fontsize=14,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)  # noqa: FBT003
        ax1.set_xticks(layers)

        # Convergence (difference from reference)
        ref_energy = energies[-1]
        diffs = [abs(e - ref_energy) for e in energies]

        ax2.plot(layers, diffs, "s-", linewidth=2, markersize=10, color="red")
        ax2.axhline(
            y=convergence_threshold,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Threshold: {convergence_threshold} J/m²",
        )
        ax2.set_xlabel("Number of Layers", fontsize=12, fontweight="bold")
        ax2.set_ylabel("ΔSurface Energy (J/m²)", fontsize=12, fontweight="bold")
        ax2.set_title(
            f"Convergence: Δγ vs. Layers\n{formula} {miller_str}",
            fontsize=14,
            fontweight="bold",
        )
        ax2.grid(True, alpha=0.3)  # noqa: FBT003
        ax2.legend(fontsize=11)
        ax2.set_xticks(layers)

        plt.tight_layout()
        filename = "surface_convergence_layers.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        plot_files["layers"] = filename

    elif convergence_mode == "vacuum":
        # Plot surface energy vs. vacuum thickness
        vacuums = [r["vacuum"] for r in results]
        energies = [r["surface_energy_Jm2"] for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Absolute surface energy
        ax1.plot(vacuums, energies, "o-", linewidth=2, markersize=10, color="#2ca02c")
        ax1.set_xlabel("Vacuum Thickness (Å)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Surface Energy (J/m²)", fontsize=12, fontweight="bold")
        ax1.set_title(
            f"Surface Energy vs. Vacuum\n{formula} {miller_str}",
            fontsize=14,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)  # noqa: FBT003

        # Convergence
        ref_energy = energies[-1]
        diffs = [abs(e - ref_energy) for e in energies]

        ax2.plot(vacuums, diffs, "s-", linewidth=2, markersize=10, color="red")
        ax2.axhline(
            y=convergence_threshold,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Threshold: {convergence_threshold} J/m²",
        )
        ax2.set_xlabel("Vacuum Thickness (Å)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("ΔSurface Energy (J/m²)", fontsize=12, fontweight="bold")
        ax2.set_title(
            f"Convergence: Δγ vs. Vacuum\n{formula} {miller_str}",
            fontsize=14,
            fontweight="bold",
        )
        ax2.grid(True, alpha=0.3)  # noqa: FBT003
        ax2.legend(fontsize=11)

        plt.tight_layout()
        filename = "surface_convergence_vacuum.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        plot_files["vacuum"] = filename

    else:  # both
        # Create heatmap of surface energy vs. layers and vacuum
        layers = sorted(set(r["n_layers"] for r in results))
        vacuums = sorted(set(r["vacuum"] for r in results))

        # Create 2D array
        energy_grid = np.zeros((len(vacuums), len(layers)))
        for r in results:
            i = vacuums.index(r["vacuum"])
            j = layers.index(r["n_layers"])
            energy_grid[i, j] = r["surface_energy_Jm2"]

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        im = ax.imshow(
            energy_grid,
            cmap="viridis",
            aspect="auto",
            origin="lower",
        )
        plt.colorbar(im, ax=ax, label="Surface Energy (J/m²)")

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers)
        ax.set_yticks(range(len(vacuums)))
        ax.set_yticklabels([f"{v:.1f}" for v in vacuums])

        ax.set_xlabel("Number of Layers", fontsize=12, fontweight="bold")
        ax.set_ylabel("Vacuum Thickness (Å)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Surface Energy Convergence Grid\n{formula} {miller_str}",
            fontsize=14,
            fontweight="bold",
        )

        # Add value annotations
        for i in range(len(vacuums)):
            for j in range(len(layers)):
                ax.text(
                    j,
                    i,
                    f"{energy_grid[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white"
                    if energy_grid[i, j] < energy_grid.mean()
                    else "black",
                    fontsize=9,
                )

        plt.tight_layout()
        filename = "surface_convergence_grid.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        plot_files["grid"] = filename

        # Also create line plots for each
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Group by vacuum (plot energy vs. layers for each vacuum)
        for vacuum in vacuums:
            subset = [r for r in results if r["vacuum"] == vacuum]
            subset = sorted(subset, key=lambda x: x["n_layers"])
            ax1.plot(
                [r["n_layers"] for r in subset],
                [r["surface_energy_Jm2"] for r in subset],
                "o-",
                linewidth=2,
                markersize=8,
                label=f"{vacuum:.1f} Å",
            )

        ax1.set_xlabel("Number of Layers", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Surface Energy (J/m²)", fontsize=12, fontweight="bold")
        ax1.set_title(
            f"Surface Energy vs. Layers\n{formula} {miller_str}",
            fontsize=14,
            fontweight="bold",
        )
        ax1.legend(title="Vacuum", fontsize=10)
        ax1.grid(True, alpha=0.3)  # noqa: FBT003

        # Group by layers (plot energy vs. vacuum for each layer count)
        for n_layers in layers:
            subset = [r for r in results if r["n_layers"] == n_layers]
            subset = sorted(subset, key=lambda x: x["vacuum"])
            ax2.plot(
                [r["vacuum"] for r in subset],
                [r["surface_energy_Jm2"] for r in subset],
                "s-",
                linewidth=2,
                markersize=8,
                label=f"{n_layers} layers",
            )

        ax2.set_xlabel("Vacuum Thickness (Å)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Surface Energy (J/m²)", fontsize=12, fontweight="bold")
        ax2.set_title(
            f"Surface Energy vs. Vacuum\n{formula} {miller_str}",
            fontsize=14,
            fontweight="bold",
        )
        ax2.legend(title="Layers", fontsize=10)
        ax2.grid(True, alpha=0.3)  # noqa: FBT003

        plt.tight_layout()
        filename2 = "surface_convergence_lines.png"
        plt.savefig(filename2, dpi=300, bbox_inches="tight")
        plt.close()
        plot_files["lines"] = filename2

    logger.info(f"Created {len(plot_files)} convergence plots")
    return plot_files


def _write_convergence_summary(
    results: list[dict],
    bulk_energy: float,
    bulk_energy_per_formula: float,
    formula_units_per_cell: int,
    miller_str: str,
    formula: str,
    convergence_mode: str,
    convergence_threshold: float,
    converged_idx: int | None,
    symmetrize: bool,
) -> str:
    """Write convergence summary to text file."""
    from datetime import datetime

    lines = []
    lines.append("=" * 80)
    lines.append("SURFACE ENERGY CONVERGENCE ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Material: {formula}")
    lines.append(f"Surface: {miller_str}")
    lines.append(f"Convergence mode: {convergence_mode}")
    lines.append(f"Symmetric slabs: {symmetrize}")
    lines.append("")

    # Theoretical background
    lines.append("=" * 80)
    lines.append("THEORETICAL BACKGROUND")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Surface energy (γ) represents the energy cost per unit area to")  # noqa: RUF001
    lines.append("create a surface by cleaving a bulk crystal:")
    lines.append("")
    if symmetrize:
        lines.append("  γ = (E_slab - N × E_bulk) / (2A)   [symmetric slab]")  # noqa: RUF001
    else:
        lines.append("  γ = (E_slab - N × E_bulk) / A      [asymmetric slab]")  # noqa: RUF001
    lines.append("")
    lines.append("Convergence with slab thickness ensures the slab interior is")
    lines.append("bulk-like. Convergence with vacuum ensures no interaction between")
    lines.append("periodic images.")
    lines.append("")
    lines.append("Unit conversion: 1 eV/Å² = 16.0218 J/m²")
    lines.append("")

    # Bulk reference
    lines.append("=" * 80)
    lines.append("BULK REFERENCE")
    lines.append("=" * 80)
    lines.append(f"Bulk energy: {bulk_energy:.6f} eV")
    lines.append(f"Energy per formula unit: {bulk_energy_per_formula:.6f} eV")
    lines.append(f"Formula units per cell: {formula_units_per_cell}")
    lines.append("")

    # Results table
    lines.append("=" * 80)
    lines.append("CALCULATION RESULTS")
    lines.append("=" * 80)
    lines.append("")

    if convergence_mode == "layers":
        lines.append(
            f"{'Layers':<8} {'Atoms':<8} {'Thickness':<12} "
            f"{'γ (eV/Å²)':<14} {'γ (J/m²)':<12} {'Δγ (J/m²)':<12}"  # noqa: RUF001
        )
        lines.append("-" * 80)
        ref_energy = results[-1]["surface_energy_Jm2"]
        for r in results:
            diff = abs(r["surface_energy_Jm2"] - ref_energy)
            converged_mark = "✓" if diff < convergence_threshold else ""
            lines.append(
                f"{r['n_layers']:<8} {r['n_atoms']:<8} {r['thickness']:<12.4f} "
                f"{r['surface_energy_eV_A2']:<14.6f} {r['surface_energy_Jm2']:<12.4f} "
                f"{diff:<12.4f} {converged_mark}"
            )

    elif convergence_mode == "vacuum":
        lines.append(
            f"{'Vacuum':<10} {'Atoms':<8} "
            f"{'γ (eV/Å²)':<14} {'γ (J/m²)':<12} {'Δγ (J/m²)':<12}"  # noqa: RUF001
        )
        lines.append("-" * 80)
        ref_energy = results[-1]["surface_energy_Jm2"]
        for r in results:
            diff = abs(r["surface_energy_Jm2"] - ref_energy)
            converged_mark = "✓" if diff < convergence_threshold else ""
            lines.append(
                f"{r['vacuum']:<10.2f} {r['n_atoms']:<8} "
                f"{r['surface_energy_eV_A2']:<14.6f} {r['surface_energy_Jm2']:<12.4f} "
                f"{diff:<12.4f} {converged_mark}"
            )

    else:  # both
        lines.append(f"{'Layers':<8} {'Vacuum':<10} {'Atoms':<8} {'γ (J/m²)':<12}")  # noqa: RUF001
        lines.append("-" * 80)
        for r in results:
            lines.append(
                f"{r['n_layers']:<8} {r['vacuum']:<10.2f} {r['n_atoms']:<8} "
                f"{r['surface_energy_Jm2']:<12.4f}"
            )

    lines.append("")

    # Convergence analysis
    lines.append("=" * 80)
    lines.append("CONVERGENCE ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"Convergence threshold: {convergence_threshold} J/m²")
    lines.append("")

    if converged_idx is not None:
        converged_result = results[converged_idx]
        lines.append("✓ CONVERGED at:")
        if convergence_mode == "layers":
            lines.append(f"  - Number of layers: {converged_result['n_layers']}")
        elif convergence_mode == "vacuum":
            lines.append(f"  - Vacuum thickness: {converged_result['vacuum']:.1f} Å")
        else:
            lines.append(f"  - Number of layers: {converged_result['n_layers']}")
            lines.append(f"  - Vacuum thickness: {converged_result['vacuum']:.1f} Å")
        lines.append(
            f"  - Surface energy: {converged_result['surface_energy_Jm2']:.4f} J/m²"
        )
    else:
        lines.append("✗ NOT CONVERGED with tested parameters")
        lines.append("  Consider extending the parameter range.")

    lines.append("")

    # Statistics
    energies = [r["surface_energy_Jm2"] for r in results]
    lines.append("Surface Energy Statistics:")
    lines.append(f"  Minimum: {min(energies):.4f} J/m²")
    lines.append(f"  Maximum: {max(energies):.4f} J/m²")
    lines.append(f"  Range: {max(energies) - min(energies):.4f} J/m²")
    lines.append(f"  Final value: {energies[-1]:.4f} J/m²")
    lines.append("")

    # Recommendations
    lines.append("=" * 80)
    lines.append("RECOMMENDATIONS")
    lines.append("=" * 80)
    lines.append("")

    if converged_idx is not None:
        if convergence_mode == "layers":
            lines.append(
                f"• Use {results[converged_idx]['n_layers']} or more layers "
                f"for production calculations"
            )
        elif convergence_mode == "vacuum":
            lines.append(
                f"• Use {results[converged_idx]['vacuum']:.1f} Å or more vacuum "
                f"for production calculations"
            )
        else:
            lines.append(
                f"• Use {results[converged_idx]['n_layers']} layers and "
                f"{results[converged_idx]['vacuum']:.1f} Å vacuum"
            )
        lines.append(f"• Expected surface energy: {energies[-1]:.3f} J/m²")
    else:
        lines.append("• Extend parameter range to achieve convergence")
        if convergence_mode == "layers":
            lines.append(
                f"• Try more layers (current max: {max(r['n_layers'] for r in results)})"
            )
        elif convergence_mode == "vacuum":
            lines.append(
                f"• Try larger vacuum (current max: {max(r['vacuum'] for r in results):.1f} Å)"
            )

    lines.append("")
    lines.append("General guidelines:")
    lines.append("  • Publication quality: Δγ < 0.01 J/m²")
    lines.append("  • Standard calculations: Δγ < 0.05 J/m²")
    lines.append("  • Quick screening: Δγ < 0.1 J/m²")
    lines.append("")

    lines.append("=" * 80)

    # Add footer
    from atomate2.siesta.utils.text_output import get_standard_footer

    footer = get_standard_footer(
        width=80,
        additional_info={
            "Analysis type": "Surface energy convergence",
            "Mode": convergence_mode,
            "Number of calculations": str(len(results)),
        },
    )

    content = "\n".join(lines) + "\n" + footer

    filename = "surface_convergence_summary.txt"
    with open(filename, "w") as f:
        f.write(content)

    logger.info(f"Convergence summary saved to {filename}")
    return filename
