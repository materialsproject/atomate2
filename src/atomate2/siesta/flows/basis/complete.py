"""
Comprehensive Basis Set Convergence Workflows for SIESTA.

This module provides workflows for systematic testing of SIESTA basis set
convergence, combining both basis size (SZ, DZ, DZP, TZP, etc.) and basis
parameters (PAO.EnergyShift, PAO.SplitNorm) to find optimal settings for
accuracy and computational efficiency.

Key Features:
- Tests multiple basis sizes in parallel
- For each basis size, tests parameter combinations
- Comprehensive visualization of convergence trends
- Timing analysis for computational cost estimation
- Automatic recommendations based on convergence criteria

Workflows:
- BasisSizeConvergenceFlowMaker: Test different basis sizes only
- CompleteBasisConvergenceFlowMaker: Test basis sizes + parameters (comprehensive)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jobflow import Flow, job

from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.flows.basis.eos import (
    plot_basis_functions,
    plot_real_basis_functions,
)
from atomate2.siesta.jobs.core import StaticMaker

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Molecule, Structure

    from atomate2.siesta.jobs.base import BaseSiestaMaker

logger = logging.getLogger(__name__)

# Rich console for pretty output
try:
    from rich.console import Console

    console = Console()
except ImportError:
    console = None


# =============================================================================
# Basis Size Convergence Workflow
# =============================================================================


@dataclass
class CompleteBasisConvergenceFlowMaker(BaseSiestaFlowMaker):
    """
    Comprehensive basis convergence: tests both basis sizes AND parameters.

    This workflow runs the full BasisParametersConvergenceMaker for each basis size,
    providing complete convergence information for systematic basis optimization.

    For each basis size (SZ, DZ, DZP, etc.), it tests:
    - Multiple PAO.EnergyShift values
    - Multiple PAO.SplitNorm values
    - All combinations in parallel

    This is the most comprehensive basis convergence study, suitable for
    establishing optimal settings for production calculations.

    Parameters
    ----------
    name : str
        Name of the workflow
    basis_sizes : list[str]
        List of basis sizes to test
    energy_shifts : list[float]
        PAO.EnergyShift values to test (Ry)
    split_norms : list[float]
        PAO.SplitNorm values to test
    kpts : list[int] | None
        K-points grid
    static_maker : StaticMaker | None
        Maker for static calculations

    Examples
    --------
    >>> maker = CompleteBasisConvergenceFlowMaker(
    ...     basis_sizes=["DZ", "DZP", "TZP"],
    ...     energy_shifts=[0.005, 0.010, 0.015],
    ...     split_norms=[0.15, 0.20, 0.25],
    ...     kpts=[4, 4, 4],
    ... )
    >>> flow = maker.make(structure)
    >>> # This runs 3 basis × 3 shifts × 3 norms = 27 calculations
    """  # noqa: RUF002

    name: str = "Complete Basis Convergence"
    basis_sizes: list[str] = None
    energy_shifts: list[float] = None
    split_norms: list[float] = None
    kpts: list[int] | None = None
    static_maker: BaseSiestaMaker = field(default_factory=StaticMaker)

    def __post_init__(self) -> None:
        """Set defaults if not provided."""
        if self.basis_sizes is None:
            self.basis_sizes = ["DZ", "DZP", "TZP"]
        if self.energy_shifts is None:
            self.energy_shifts = [0.010, 0.015, 0.020]
        if self.split_norms is None:
            self.split_norms = [0.15, 0.20, 0.25]
        # Propagate dry_run, use_custodian, tier, manager_config to child makers
        super().__post_init__()

    def make(
        self, structure: Structure | Molecule, prev_dir: str | Path | None = None
    ) -> Flow:
        """
        Create comprehensive basis convergence flow.

        Parameters
        ----------
        structure : Structure | Molecule
            Structure to run calculations on
        prev_dir : str | Path | None
            Previous directory for continuation jobs

        Returns
        -------
        Flow
            Jobflow Flow with all basis size/parameter combinations
        """
        n_total = (
            len(self.basis_sizes) * len(self.energy_shifts) * len(self.split_norms)
        )
        logger.info(
            f"CompleteBasisConvergenceFlowMaker.make() - {n_total} total calculations"
        )
        logger.info(
            f"  {len(self.basis_sizes)} basis sizes × "  # noqa: RUF001
            f"{len(self.energy_shifts)} shifts × "  # noqa: RUF001
            f"{len(self.split_norms)} norms"
        )

        jobs = []
        all_metadata = []

        # Calculate total number of jobs
        total_jobs = (
            len(self.basis_sizes) * len(self.energy_shifts) * len(self.split_norms)
        )
        job_counter = 0

        # Create SCF jobs directly for all combinations
        from atomate2.siesta.powerups import update_user_siesta_settings

        for basis_size in self.basis_sizes:
            for energy_shift in self.energy_shifts:
                for split_norm in self.split_norms:
                    job_counter += 1

                    # Create static maker
                    maker = (
                        self.static_maker.scf()
                        if hasattr(self.static_maker, "scf")
                        else self.static_maker
                    )

                    # Propagate custodian settings
                    self.propagate_custodian_to_maker(maker)

                    # Configure basis parameters
                    siesta_updates = {
                        "PAO.BasisSize": basis_size,
                        "PAO.BasisType": "split",
                        "PAO.EnergyShift": f"{energy_shift} Ry",
                        "PAO.SplitNorm": split_norm,
                        "a2s_kpts": self.kpts if self.kpts is not None else [4, 4, 4],
                    }

                    maker = update_user_siesta_settings(
                        maker, siesta_updates, class_filter=StaticMaker
                    )

                    # Create job with counter
                    scf_job = maker.make(structure, prev_dir=prev_dir)
                    job_label = (
                        f"{basis_size} ES{energy_shift:.3f} SN{split_norm:.2f}".replace(
                            ".", "p"
                        )
                    )
                    scf_job.name = (
                        f"{self.name}_{job_label}_[{job_counter}_of_{total_jobs}]"
                    )

                    jobs.append(scf_job)

                    # Store metadata
                    all_metadata.append(
                        {
                            "uuid": scf_job.uuid,
                            "name": scf_job.name,
                            "basis_size": basis_size,
                            "energy_shift": energy_shift,
                            "split_norm": split_norm,
                        }
                    )

        logger.info(f"Created {len(jobs)} SCF jobs for complete basis convergence")

        # Create a flow from SCF jobs to get combined output
        scf_jobs_only = jobs.copy()  # Save SCF jobs list
        scf_flow = Flow(
            jobs,
            name="Complete Basis SCF Jobs",
            output={job.uuid: job.output for job in jobs},
        )

        # Collect all results
        collect_job = collect_complete_basis_data(
            [job.output for job in scf_jobs_only], all_metadata
        )
        collect_job.name = f"{self.name}-collect"

        # Plot comprehensive comparison
        plot_job = plot_complete_basis_convergence(
            collect_job.output, output_file="complete_basis_convergence.png"
        )
        plot_job.name = f"{self.name}-plot"

        # Write comprehensive summary
        summary_job = write_complete_basis_summary(
            collect_job.output, output_file="complete_basis_summary.txt"
        )
        summary_job.name = f"{self.name}-summary"

        # Create basis function visualization (orbital plots from ion.xml)
        basis_viz_job = plot_basis_functions(
            flow_results=scf_flow.output,
            job_metadata=all_metadata,
        )
        basis_viz_job.name = f"{self.name}-basis-viz"

        # Create real basis function plots - one per basis size
        real_basis_jobs = []
        unique_basis = sorted(
            {m["basis_size"] for m in all_metadata},
            key=lambda x: (
                ["SZ", "DZ", "DZP", "SZP", "DZDP", "TZ", "TZP", "TZDP"].index(x)
                if x in ["SZ", "DZ", "DZP", "SZP", "DZDP", "TZ", "TZP", "TZDP"]
                else 999
            ),
        )

        for basis_size in unique_basis:
            # Filter metadata for this basis size
            basis_metadata = [m for m in all_metadata if m["basis_size"] == basis_size]

            # Create plot for this basis size
            real_basis_job = plot_real_basis_functions(
                flow_results=scf_flow.output,
                job_metadata=basis_metadata,
                output_file=f"basis_functions_real_{basis_size}.png",
            )
            real_basis_job.name = f"{self.name}-real-basis-{basis_size}"
            real_basis_jobs.append(real_basis_job)

        # Combine all jobs
        all_jobs = [
            scf_flow,
            collect_job,
            plot_job,
            summary_job,
            basis_viz_job,
            *real_basis_jobs,
        ]

        # Create final flow
        return Flow(all_jobs, output=collect_job.output, name=self.name)


@job
def collect_complete_basis_data(
    job_outputs: list[Any], job_metadata: list[dict]
) -> dict[str, Any]:
    """
    Collect data from complete basis convergence (size + parameters).

    Parameters
    ----------
    job_outputs : list[Any]
        List of job outputs (SiestaTaskDoc objects)
    job_metadata : list[dict]
        List of job metadata with keys: uuid, name, basis_size, energy_shift, split_norm

    Returns
    -------
    dict[str, Any]
        Complete convergence data with basis_sizes, energy_shifts, split_norms,
        energies, max_forces, max_stresses, and run_times
    """
    import numpy as np

    logger.info("Collecting complete basis convergence data")

    # Collect data from all jobs
    data: dict[str, list[Any]] = {
        "basis_sizes": [],
        "energy_shifts": [],
        "split_norms": [],
        "energies": [],
        "max_forces": [],
        "max_stresses": [],
        "run_times": [],
    }

    for job_info, output in zip(job_metadata, job_outputs, strict=False):
        basis_size = job_info["basis_size"]
        energy_shift = job_info["energy_shift"]
        split_norm = job_info["split_norm"]
        job_name = job_info["name"]

        if output is None:
            logger.warning(f"No output for {job_name}")
            continue

        try:
            # Extract energy - output is SiestaTaskDoc,
            # energy is in output.output.energy
            energy = (
                output.output.energy
                if hasattr(output, "output") and output.output
                else None
            )
            if energy is None:
                logger.warning(f"No energy found for {job_name}")
                continue

            # Extract max force
            max_force = 0.0
            forces = (
                output.output.forces
                if hasattr(output, "output") and output.output
                else None
            )
            if forces is not None:
                forces_array = np.array(forces)
                max_force = np.max(np.linalg.norm(forces_array, axis=1))

            # Extract max stress
            max_stress = 0.0
            stress = (
                output.output.stress
                if hasattr(output, "output") and output.output
                else None
            )
            if stress is not None:
                stress_array = np.array(stress)
                max_stress = np.max(np.abs(stress_array))

            # Get run_time directly from schema (output.output.run_time)
            run_time = 0.0
            if hasattr(output, "output") and output.output:
                run_time = getattr(output.output, "run_time", None) or 0.0

            if run_time == 0.0:
                logger.debug(f"No run_time in schema for {job_name}")

            # Store data
            data["basis_sizes"].append(basis_size)
            data["energy_shifts"].append(energy_shift)
            data["split_norms"].append(split_norm)
            data["energies"].append(energy)
            data["max_forces"].append(max_force)
            data["max_stresses"].append(max_stress)
            data["run_times"].append(run_time)

            logger.debug(
                f"Collected data from {job_name}: E={energy:.6f} eV, t={run_time:.1f}s"
            )

        except Exception:
            logger.exception(f"Failed to extract data from {job_name}")
            import traceback

            logger.exception(traceback.format_exc())
            continue

    logger.info(f"Collected {len(data['energies'])} complete basis results")
    return data


@job
def plot_complete_basis_convergence(
    data: dict[str, Any], output_file: str = "complete_basis_convergence.png"
) -> dict[str, str]:
    """
    Plot comprehensive basis convergence analysis across sizes and parameters.

    Creates detailed multi-page PDF or multiple PNG files showing:
    - Overview: Comparison across all basis sizes
    - Per-Basis Analysis: 2D heatmaps, parameter sensitivity, quality maps
    - Timing: Computational cost analysis
    - Recommendations: Convergence assessment

    Parameters
    ----------
    data : dict[str, Any]
        Complete convergence data with basis_sizes, energy_shifts, split_norms,
        energies, max_forces, max_stresses, run_times
    output_file : str
        Output filename (will create _overview.png, _<basis>.png files)

    Returns
    -------
    dict[str, str]
        Dictionary with paths to all plot files
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm

    if console:
        console.print(
            "[green]Plotting complete basis convergence "
            "(detailed per-basis analysis)[/green]"
        )

    basis_sizes = np.array(data["basis_sizes"])
    energy_shifts = np.array(data["energy_shifts"])
    split_norms = np.array(data["split_norms"])
    energies = np.array(data["energies"])
    max_forces = np.array(data["max_forces"])
    max_stresses = np.array(data["max_stresses"])
    run_times = np.array(data.get("run_times", [0] * len(energies)))

    # Check if we have data
    if len(energies) == 0:
        logger.error("No energy data to plot - creating empty plot with error message")
        _fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "ERROR: No convergence data collected\n\n"
            "Check individual job directories for calculation outputs",
            ha="center",
            va="center",
            fontsize=14,
            color="red",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax.axis("off")
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        if console:
            console.print(f"[yellow]Empty plot saved to: {output_file}[/yellow]")
        return {"plot": output_file}

    unique_basis = sorted(
        set(basis_sizes),
        key=lambda x: (
            ["SZ", "DZ", "DZP", "SZP", "DZDP", "TZ", "TZP", "TZDP"].index(x)
            if x in ["SZ", "DZ", "DZP", "SZP", "DZDP", "TZ", "TZP", "TZDP"]
            else 999
        ),
    )
    unique_shifts = np.unique(energy_shifts)
    unique_norms = np.unique(split_norms)

    logger.info(f"Creating plots for {len(unique_basis)} basis sizes")
    plot_files = {}

    # =========================================================================
    # OVERVIEW PLOT: Comparison across all basis sizes
    # =========================================================================
    plt.figure(figsize=(18, 12))

    # Panel 1: Energy vs Basis Size (for different parameter combinations)
    ax1 = plt.subplot(2, 3, 1)
    colors = cm.viridis(np.linspace(0, 1, len(unique_shifts) * len(unique_norms)))
    color_idx = 0
    for shift in unique_shifts:
        for norm in unique_norms:
            mask = (energy_shifts == shift) & (split_norms == norm)
            if np.sum(mask) > 0:
                basis_subset = basis_sizes[mask]
                energy_subset = energies[mask]

                # Sort by basis order
                sorted_indices = []
                for ub in unique_basis:
                    if ub in basis_subset:
                        idx = np.where(basis_subset == ub)[0][0]
                        sorted_indices.append(idx)

                if sorted_indices:
                    x_pos = np.arange(len(sorted_indices))
                    sorted_e = energy_subset[sorted_indices]
                    ax1.plot(
                        x_pos,
                        sorted_e,
                        "o-",
                        color=colors[color_idx],
                        alpha=0.7,
                        label=f"ES={shift:.3f},SN={norm:.2f}",
                    )
                    color_idx += 1

    ax1.set_xlabel("Basis Size", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Energy (eV)", fontsize=11, fontweight="bold")
    ax1.set_title("Energy vs Basis Size", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(len(unique_basis)))
    ax1.set_xticklabels(unique_basis, fontsize=10)
    ax1.grid(True, alpha=0.3)  # noqa: FBT003
    if color_idx <= 6:
        ax1.legend(fontsize=8, loc="best")

    # Panel 2: Energy vs EnergyShift (for each basis)
    ax2 = plt.subplot(2, 3, 2)
    colors2 = cm.plasma(np.linspace(0, 1, len(unique_basis)))
    for i, basis in enumerate(unique_basis):
        mask = basis_sizes == basis
        if np.sum(mask) > 1:
            shifts_subset = energy_shifts[mask]
            energies_subset = energies[mask]

            # Average over split norms for clarity
            avg_energies = []
            for shift in unique_shifts:
                shift_mask = shifts_subset == shift
                if np.sum(shift_mask) > 0:
                    avg_energies.append(np.mean(energies_subset[shift_mask]))
                else:
                    avg_energies.append(np.nan)

            ax2.plot(
                unique_shifts,
                avg_energies,
                "o-",
                color=colors2[i],
                linewidth=2,
                markersize=6,
                label=basis,
            )

    ax2.set_xlabel("PAO.EnergyShift (Ry)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Energy (eV)", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Energy vs EnergyShift (avg over SplitNorm)", fontsize=12, fontweight="bold"
    )
    ax2.legend(fontsize=9, loc="best")
    ax2.grid(True, alpha=0.3)  # noqa: FBT003

    # Panel 3: Energy vs SplitNorm (for each basis)
    ax3 = plt.subplot(2, 3, 3)
    for i, basis in enumerate(unique_basis):
        mask = basis_sizes == basis
        if np.sum(mask) > 1:
            norms_subset = split_norms[mask]
            energies_subset = energies[mask]

            # Average over energy shifts
            avg_energies = []
            for norm in unique_norms:
                norm_mask = norms_subset == norm
                if np.sum(norm_mask) > 0:
                    avg_energies.append(np.mean(energies_subset[norm_mask]))
                else:
                    avg_energies.append(np.nan)

            ax3.plot(
                unique_norms,
                avg_energies,
                "s-",
                color=colors2[i],
                linewidth=2,
                markersize=6,
                label=basis,
            )

    ax3.set_xlabel("PAO.SplitNorm", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Energy (eV)", fontsize=11, fontweight="bold")
    ax3.set_title(
        "Energy vs SplitNorm (avg over EnergyShift)", fontsize=12, fontweight="bold"
    )
    ax3.legend(fontsize=9, loc="best")
    ax3.grid(True, alpha=0.3)  # noqa: FBT003

    # Panel 4: Minimum energy for each basis
    ax4 = plt.subplot(2, 3, 4)
    min_energies = []
    for basis in unique_basis:
        mask = basis_sizes == basis
        if np.sum(mask) > 0:
            min_energies.append(np.min(energies[mask]))
        else:
            min_energies.append(np.nan)

    x_pos = np.arange(len(unique_basis))
    ax4.bar(x_pos, min_energies, color=colors2, alpha=0.7, edgecolor="black")
    ax4.set_xlabel("Basis Size", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Minimum Energy (eV)", fontsize=11, fontweight="bold")
    ax4.set_title("Lowest Energy for Each Basis", fontsize=12, fontweight="bold")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(unique_basis, fontsize=10)
    ax4.grid(True, alpha=0.3, axis="y")  # noqa: FBT003

    # Panel 5: Energy range for each basis
    ax5 = plt.subplot(2, 3, 5)
    energy_ranges = []
    for basis in unique_basis:
        mask = basis_sizes == basis
        if np.sum(mask) > 1:
            e_range = (np.max(energies[mask]) - np.min(energies[mask])) * 1000  # meV
            energy_ranges.append(e_range)
        else:
            energy_ranges.append(0)

    ax5.bar(x_pos, energy_ranges, color=colors2, alpha=0.7, edgecolor="black")
    ax5.axhline(
        y=5, color="green", linestyle="--", alpha=0.5, linewidth=2, label="5 meV"
    )
    ax5.axhline(
        y=10, color="orange", linestyle="--", alpha=0.5, linewidth=2, label="10 meV"
    )
    ax5.set_xlabel("Basis Size", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Parameter Variation (meV)", fontsize=11, fontweight="bold")
    ax5.set_title("Energy Range Within Each Basis", fontsize=12, fontweight="bold")
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(unique_basis, fontsize=10)
    ax5.legend(fontsize=9, loc="best")
    ax5.grid(True, alpha=0.3, axis="y")  # noqa: FBT003

    # Panel 6: Timing and efficiency analysis
    ax6 = plt.subplot(2, 3, 6)
    if np.any(run_times > 0):
        avg_times = []
        min_times = []
        max_times = []

        for basis in unique_basis:
            mask = (basis_sizes == basis) & (run_times > 0)
            if np.sum(mask) > 0:
                valid_times = run_times[mask]
                avg_times.append(np.mean(valid_times))
                min_times.append(np.min(valid_times))
                max_times.append(np.max(valid_times))
            else:
                avg_times.append(0)
                min_times.append(0)
                max_times.append(0)

        # Bar chart with error bars showing range
        ax6.bar(x_pos, avg_times, color=colors2, alpha=0.7, edgecolor="black")

        # Add error bars showing min-max range
        errors = [
            [avg_times[i] - min_times[i] for i in range(len(avg_times))],
            [max_times[i] - avg_times[i] for i in range(len(avg_times))],
        ]
        ax6.errorbar(
            x_pos,
            avg_times,
            yerr=errors,
            fmt="none",
            ecolor="black",
            capsize=5,
            capthick=2,
            alpha=0.7,
        )

        # Add value labels on bars
        for _i, (x, y) in enumerate(zip(x_pos, avg_times, strict=False)):
            if y > 0:
                ax6.text(
                    x,
                    y,
                    f"{y:.1f}s",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        ax6.set_xlabel("Basis Size", fontsize=11, fontweight="bold")
        ax6.set_ylabel("Wall Time (seconds)", fontsize=11, fontweight="bold")
        ax6.set_title(
            "Computational Cost (avg ± range)", fontsize=12, fontweight="bold"
        )
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(unique_basis, fontsize=10)
        ax6.grid(True, alpha=0.3, axis="y")  # noqa: FBT003
        ax6.set_ylim(bottom=0)
    else:
        # No timing data available - show placeholder message
        ax6.text(
            0.5,
            0.5,
            "Timing Data Not Available\n\n"
            "(Run times will be extracted from\nSIESTA output files when available)",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax6.transAxes,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )
        ax6.set_xlabel("Basis Size", fontsize=11, fontweight="bold")
        ax6.set_ylabel("Wall Time (seconds)", fontsize=11, fontweight="bold")
        ax6.set_title("Computational Cost Comparison", fontsize=12, fontweight="bold")
        ax6.set_xticks([])
        ax6.set_yticks([])

    plt.suptitle(
        "Complete Basis Convergence Study - Overview\n"
        "(Comparison Across All Basis Sizes)",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save overview plot
    base_name = output_file.replace(".png", "")
    overview_file = f"{base_name}_overview.png"
    plt.savefig(overview_file, dpi=150, bbox_inches="tight")
    plt.close()
    plot_files["overview"] = overview_file

    logger.info(f"Overview plot saved to {overview_file}")
    if console:
        console.print(f"[green]Overview plot saved: {overview_file}[/green]")

    # =========================================================================
    # PER-BASIS DETAILED PLOTS: Like BasisParametersConvergenceMaker
    # =========================================================================
    for basis in unique_basis:
        if console:
            console.print(f"[cyan]Creating detailed plot for {basis} basis...[/cyan]")

        # Extract data for this basis size
        mask = basis_sizes == basis
        basis_shifts = energy_shifts[mask]
        basis_norms = split_norms[mask]
        basis_energies = energies[mask]
        basis_forces = max_forces[mask]
        basis_stresses = max_stresses[mask]
        run_times[mask]

        # Create 2x2 grid for this basis
        plt.figure(figsize=(16, 12))

        # Panel 1: Energy vs EnergyShift (for each SplitNorm)
        ax1 = plt.subplot(2, 2, 1)
        colors = cm.viridis(np.linspace(0, 1, len(unique_norms)))

        for i, norm in enumerate(unique_norms):
            norm_mask = basis_norms == norm
            if np.sum(norm_mask) > 0:
                shifts_filtered = basis_shifts[norm_mask]
                energies_filtered = basis_energies[norm_mask]

                # Sort by energy shift
                sort_idx = np.argsort(shifts_filtered)
                ax1.plot(
                    shifts_filtered[sort_idx],
                    energies_filtered[sort_idx],
                    "o-",
                    color=colors[i],
                    linewidth=2,
                    markersize=8,
                    label=f"SplitNorm={norm:.2f}",
                )

        ax1.set_xlabel("PAO.EnergyShift (Ry)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Total Energy (eV)", fontsize=12, fontweight="bold")
        ax1.set_title(
            f"{basis} Basis: Energy vs PAO.EnergyShift", fontsize=13, fontweight="bold"
        )
        ax1.legend(loc="best", fontsize=10)
        ax1.grid(True, alpha=0.3)  # noqa: FBT003

        # Panel 2: Energy vs SplitNorm (for each EnergyShift)
        ax2 = plt.subplot(2, 2, 2)
        colors = cm.plasma(np.linspace(0, 1, len(unique_shifts)))

        for i, shift in enumerate(unique_shifts):
            shift_mask = basis_shifts == shift
            if np.sum(shift_mask) > 0:
                norms_filtered = basis_norms[shift_mask]
                energies_filtered = basis_energies[shift_mask]

                # Sort by split norm
                sort_idx = np.argsort(norms_filtered)
                ax2.plot(
                    norms_filtered[sort_idx],
                    energies_filtered[sort_idx],
                    "s-",
                    color=colors[i],
                    linewidth=2,
                    markersize=8,
                    label=f"EnergyShift={shift:.3f}",
                )

        ax2.set_xlabel("PAO.SplitNorm", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Total Energy (eV)", fontsize=12, fontweight="bold")
        ax2.set_title(
            f"{basis} Basis: Energy vs PAO.SplitNorm", fontsize=13, fontweight="bold"
        )
        ax2.legend(loc="best", fontsize=10)
        ax2.grid(True, alpha=0.3)  # noqa: FBT003

        # Panel 3: 2D Energy Heatmap
        ax3 = plt.subplot(2, 2, 3)

        # Create grid for heatmap
        shift_grid, norm_grid = np.meshgrid(unique_shifts, unique_norms)
        energy_grid = np.zeros_like(shift_grid)

        for i in range(len(basis_energies)):
            shift_idx = np.where(unique_shifts == basis_shifts[i])[0][0]
            norm_idx = np.where(unique_norms == basis_norms[i])[0][0]
            energy_grid[norm_idx, shift_idx] = basis_energies[i]

        # Normalize to show relative energies (difference from minimum)
        energy_diff = (energy_grid - np.min(basis_energies)) * 1000  # Convert to meV

        im = ax3.contourf(
            shift_grid, norm_grid, energy_diff, levels=20, cmap="RdYlGn_r"
        )
        ax3.contour(
            shift_grid,
            norm_grid,
            energy_diff,
            levels=10,
            colors="black",
            linewidths=0.5,
            alpha=0.3,
        )

        # Mark data points
        ax3.scatter(
            basis_shifts,
            basis_norms,
            c="black",
            s=50,
            marker="o",
            edgecolors="white",
            linewidths=1,
            zorder=10,
        )

        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label("Energy above minimum (meV)", fontsize=11, fontweight="bold")

        ax3.set_xlabel("PAO.EnergyShift (Ry)", fontsize=12, fontweight="bold")
        ax3.set_ylabel("PAO.SplitNorm", fontsize=12, fontweight="bold")
        ax3.set_title(
            f"{basis} Basis: Energy Landscape (2D Heatmap)",
            fontsize=13,
            fontweight="bold",
        )

        # Panel 4: Force and Stress convergence map
        ax4 = plt.subplot(2, 2, 4)

        # Calculate energy relative to minimum for this basis
        energy_relative = (basis_energies - np.min(basis_energies)) * 1000  # meV

        # Create scatter plot: EnergyShift vs SplitNorm, colored by energy diff
        scatter = ax4.scatter(
            basis_shifts,
            basis_norms,
            c=energy_relative,
            s=200,
            cmap="RdYlGn_r",
            edgecolors="black",
            linewidths=1.5,
        )

        # Add text labels showing max force and max stress
        for i in range(len(basis_energies)):
            # Display force (top) and stress (bottom)
            label_text = f"F:{basis_forces[i]:.2f}\nσ:{basis_stresses[i]:.2f}"  # noqa: RUF001
            ax4.text(
                basis_shifts[i],
                basis_norms[i],
                label_text,
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
            )

        cbar2 = plt.colorbar(scatter, ax=ax4)
        cbar2.set_label("Energy above min (meV)", fontsize=11, fontweight="bold")

        ax4.set_xlabel("PAO.EnergyShift (Ry)", fontsize=12, fontweight="bold")
        ax4.set_ylabel("PAO.SplitNorm", fontsize=12, fontweight="bold")
        ax4.set_title(
            f"{basis} Basis: Quality Map (F=Force eV/Å, σ=Stress GPa)",  # noqa: RUF001
            fontsize=13,
            fontweight="bold",
        )

        plt.suptitle(
            f"Complete Parameter Analysis: {basis} Basis Size",
            fontsize=15,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save per-basis plot
        basis_file = f"{base_name}_{basis}.png"
        plt.savefig(basis_file, dpi=150, bbox_inches="tight")
        plt.close()
        plot_files[basis] = basis_file

        logger.info(f"Detailed plot for {basis} saved to {basis_file}")
        if console:
            console.print(f"[green]  → {basis} plot saved: {basis_file}[/green]")

    if console:
        console.print("[bold green]All plots created successfully![/bold green]")
        console.print(f"  Overview: {overview_file}")
        for basis, fpath in plot_files.items():
            if basis != "overview":
                console.print(f"  {basis}: {fpath}")

    return plot_files


@job
def write_complete_basis_summary(
    data: dict[str, Any], output_file: str = "complete_basis_summary.txt"
) -> dict[str, str]:
    """
    Write comprehensive summary of complete basis convergence study.

    Creates detailed text summary with:
    - Overview: Global comparison across all basis sizes
    - Per-Basis Analysis: Detailed results for each basis size including:
      * Parameter sensitivity (energy range for ES/SN variations)
      * Optimal parameters and minimum energy
      * Force and stress convergence
      * Computational cost
    - Recommendations: Best basis size and parameters for production

    Parameters
    ----------
    data : dict[str, Any]
        Complete convergence data with basis_sizes, energy_shifts, split_norms,
        energies, max_forces, max_stresses, run_times
    output_file : str
        Output filename for summary

    Returns
    -------
    dict[str, str]
        Dictionary with summary file path
    """
    import numpy as np

    basis_sizes = np.array(data["basis_sizes"])
    energy_shifts = np.array(data["energy_shifts"])
    split_norms = np.array(data["split_norms"])
    energies = np.array(data["energies"])
    max_forces = np.array(data["max_forces"])
    max_stresses = np.array(data["max_stresses"])
    run_times = np.array(data.get("run_times", [0] * len(energies)))

    # Check if we have data
    if len(energies) == 0:
        logger.error("No energy data collected - cannot write summary")
        error_msg = (
            "ERROR: No convergence data was collected.\n\n"
            "This usually means:\n"
            "1. The SCF calculations failed\n"
            "2. The job outputs were not properly passed to the collection function\n"
            "3. The output format was not recognized\n\n"
            "Check the individual job directories for calculation outputs.\n"
        )
        with open(output_file, "w") as f:
            f.write("=" * 90 + "\n")
            f.write("COMPLETE BASIS CONVERGENCE SUMMARY\n")
            f.write("=" * 90 + "\n\n")
            f.write(error_msg)
        return {"summary": output_file}

    logger.info(f"Writing summary for {len(energies)} results")

    unique_basis = sorted(
        set(basis_sizes),
        key=lambda x: (
            ["SZ", "DZ", "DZP", "SZP", "DZDP", "TZ", "TZP", "TZDP"].index(x)
            if x in ["SZ", "DZ", "DZP", "SZP", "DZDP", "TZ", "TZP", "TZDP"]
            else 999
        ),
    )

    with open(output_file, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("COMPLETE BASIS CONVERGENCE STUDY\n")
        f.write("Basis Size + PAO Parameters (EnergyShift & SplitNorm)\n")
        f.write("=" * 90 + "\n\n")

        f.write(
            f"Tested {len(unique_basis)} basis sizes with "
            f"{len(np.unique(energy_shifts))} "
            f"EnergyShift × {len(np.unique(split_norms))} SplitNorm values\n"  # noqa: RUF001
        )
        f.write(f"Total calculations: {len(energies)}\n\n")

        # OVERALL COMPARISON (Global table first)
        f.write("=" * 90 + "\n")
        f.write("OVERALL COMPARISON\n")
        f.write("=" * 90 + "\n\n")

        f.write("Minimum Energy for Each Basis:\n")
        f.write("-" * 90 + "\n")
        f.write(
            f"{'Basis':<10} {'Min Energy (eV)':<18} {'ΔE (meV)':<12} "
            f"{'Optimal ES':<14} {'Optimal SN':<12} {'Time (s)':<10}\n"
        )
        f.write("-" * 90 + "\n")

        # First pass: find global minimum for ΔE calculation
        global_min_e_temp = np.min(energies)

        min_energies = []
        for basis in unique_basis:
            mask = basis_sizes == basis
            if np.sum(mask) > 0:
                basis_energies = energies[mask]
                basis_shifts = energy_shifts[mask]
                basis_norms = split_norms[mask]
                basis_times = run_times[mask]

                opt_idx = np.argmin(basis_energies)
                min_e = basis_energies[opt_idx]
                min_energies.append(min_e)
                opt_es = basis_shifts[opt_idx]
                opt_sn = basis_norms[opt_idx]

                # Calculate ΔE from global minimum
                de = (min_e - global_min_e_temp) * 1000  # meV

                # Get timing for optimal parameters
                opt_time = basis_times[opt_idx]
                time_str = f"{opt_time:.1f}" if opt_time > 0 else "N/A"

                f.write(
                    f"{basis:<10} {min_e:<18.8f} {de:<12.4f} "
                    f"{opt_es:<14.6f} {opt_sn:<12.4f} {time_str:<10}\n"
                )

        f.write("\n")

        # Find global minimum
        global_min_idx = np.argmin(energies)
        global_min_e = energies[global_min_idx]

        f.write("Global Optimum:\n")
        f.write(f"  Basis Size      = {basis_sizes[global_min_idx]}\n")
        f.write(f"  PAO.EnergyShift = {energy_shifts[global_min_idx]:.6f} Ry\n")
        f.write(f"  PAO.SplitNorm   = {split_norms[global_min_idx]:.4f}\n")
        f.write(f"  Energy          = {global_min_e:.8f} eV\n\n")

        # Basis size energy differences
        f.write("Energy Differences Between Basis Sizes:\n")
        f.write("(Comparing minimum energies for each basis)\n")
        f.write("-" * 60 + "\n")

        for i, basis in enumerate(unique_basis):
            if i > 0:
                e_diff = (min_energies[i] - min_energies[0]) * 1000
                f.write(f"  {basis} vs {unique_basis[0]}: {e_diff:+.4f} meV\n")
        f.write("\n\n")

        # Results for each basis size (Per-basis detailed sections)
        for basis in unique_basis:
            mask = basis_sizes == basis
            n_calcs = np.sum(mask)

            f.write("=" * 90 + "\n")
            f.write(f"BASIS SIZE: {basis} ({n_calcs} calculations)\n")
            f.write("=" * 90 + "\n\n")

            basis_energies = energies[mask]
            basis_shifts = energy_shifts[mask]
            basis_norms = split_norms[mask]
            basis_forces = max_forces[mask]
            basis_stresses = max_stresses[mask]
            basis_times = run_times[mask]

            # Find optimal for this basis
            optimal_idx = np.argmin(basis_energies)
            e_min = basis_energies[optimal_idx]
            e_max = basis_energies.max()
            e_range = (e_max - e_min) * 1000

            f.write(f"Optimal Parameters for {basis}:\n")
            f.write(f"  PAO.EnergyShift = {basis_shifts[optimal_idx]:.6f} Ry\n")
            f.write(f"  PAO.SplitNorm   = {basis_norms[optimal_idx]:.4f}\n")
            f.write(f"  Energy          = {e_min:.8f} eV\n")
            f.write(f"  Max Force       = {basis_forces[optimal_idx]:.6f} eV/Å\n")
            f.write(f"  Max Stress      = {basis_stresses[optimal_idx]:.6f} GPa\n\n")

            f.write(f"Parameter Variation for {basis}:\n")
            f.write(f"  Energy range:   {e_range:.4f} meV\n")
            f.write(
                f"  Force range:    {basis_forces.min():.6f} - "
                f"{basis_forces.max():.6f} eV/Å\n"
            )
            f.write(
                f"  Stress range:   {basis_stresses.min():.6f} - "
                f"{basis_stresses.max():.6f} GPa\n"
            )

            if np.any(basis_times > 0):
                valid_times = basis_times[basis_times > 0]
                f.write(f"  Average time:   {np.mean(valid_times):.1f} s\n")
            f.write("\n")

            # Convergence assessment
            if e_range < 1.0:
                f.write("  ✓ EXCELLENT parameter convergence (< 1 meV)\n")
            elif e_range < 5.0:
                f.write("  ✓ GOOD parameter convergence (< 5 meV)\n")
            elif e_range < 10.0:
                f.write("  ⚠ FAIR parameter convergence (5-10 meV)\n")
            else:
                f.write("  ✗ POOR parameter convergence (> 10 meV)\n")
            f.write("\n")

            # Detailed results table for this basis
            f.write(f"Detailed Results for {basis} Basis:\n")
            f.write("-" * 90 + "\n")
            f.write(
                f"{'ES (Ry)':<12} {'SN':<10} {'Energy (eV)':<16} {'ΔE (meV)':<12} "
                f"{'Max F':<10} {'Max σ':<10} {'Time (s)':<10}\n"  # noqa: RUF001
            )
            f.write("-" * 90 + "\n")

            # Sort by energy for clearer presentation
            sort_idx = np.argsort(basis_energies)
            for idx in sort_idx:
                es = basis_shifts[idx]
                sn = basis_norms[idx]
                e = basis_energies[idx]
                de = (e - e_min) * 1000
                f_max = basis_forces[idx]
                s_max = basis_stresses[idx]
                t = basis_times[idx]

                time_str = f"{t:.1f}" if t > 0 else "N/A"
                marker = " ★" if idx == optimal_idx else ""
                f.write(
                    f"{es:<12.6f} {sn:<10.4f} {e:<16.8f} {de:<12.4f} "
                    f"{f_max:<10.6f} {s_max:<10.6f} {time_str:<10}{marker}\n"
                )

            f.write("-" * 90 + "\n")
            f.write("★ = Optimal parameters for this basis size\n")
            f.write("\n")

        # Final recommendations
        f.write("=" * 90 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 90 + "\n\n")

        # Find converged basis (within 5 meV of global minimum)
        for basis in unique_basis:
            mask = basis_sizes == basis
            if np.sum(mask) > 0:
                min_e = np.min(energies[mask])
                e_diff = (min_e - global_min_e) * 1000
                if e_diff < 5.0:
                    f.write(f"✓ Recommended basis: {basis}\n")
                    f.write(f"  Energy difference from best: {e_diff:.4f} meV\n")

                    # Get optimal parameters for this basis
                    basis_mask = basis_sizes == basis
                    basis_energies = energies[basis_mask]
                    basis_shifts = energy_shifts[basis_mask]
                    basis_norms = split_norms[basis_mask]
                    opt_idx = np.argmin(basis_energies)

                    f.write(
                        f"  Optimal PAO.EnergyShift: {basis_shifts[opt_idx]:.6f} Ry\n"
                    )
                    f.write(f"  Optimal PAO.SplitNorm:   {basis_norms[opt_idx]:.4f}\n")
                    break
        f.write("\n")

        f.write("Balance between accuracy and cost:\n")
        f.write("  → For production: Use smallest basis converged within 5 meV\n")
        f.write("  → For high accuracy: Use largest tested basis\n")
        f.write("  → Consider computational cost scaling with system size\n")
        f.write("\n")

        # Add standard footer
        from atomate2.siesta.utils.text_output import get_standard_footer

        f.write(
            get_standard_footer(
                width=90,
                additional_info={
                    "Analysis type": "Complete basis convergence study",
                    "Basis sizes tested": str(len(unique_basis)),
                    "Total calculations": str(len(energies)),
                },
            )
        )

    logger.info(f"Complete basis summary written to {output_file}")
    if console:
        console.print(f"[green]Complete summary written to: {output_file}[/green]")

    return {"summary": output_file}
