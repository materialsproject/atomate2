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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from jobflow import Flow, job

from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.jobs.core import StaticMaker

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Maker
    from pymatgen.core import Molecule, Structure

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
class BasisSizeConvergenceFlowMaker(BaseSiestaFlowMaker):
    """
    Maker to test convergence with different basis set sizes.

    This workflow runs SCF calculations with different basis sizes (SZ, DZ, DZP,
    TZP, etc.) using fixed PAO.EnergyShift and PAO.SplitNorm parameters. It helps
    determine the minimum basis size needed for converged results.

    Parameters
    ----------
    name : str
        Name of the workflow
    basis_sizes : list[str]
        List of basis sizes to test (e.g., ["SZ", "DZ", "DZP", "TZP"])
    energy_shift : float
        PAO.EnergyShift value (Ry) - fixed for all basis sizes
    split_norm : float
        PAO.SplitNorm value - fixed for all basis sizes
    kpts : list[int] | None
        K-points grid [nk1, nk2, nk3]
    static_maker : StaticMaker | None
        Maker for static calculations (if None, uses default StaticMaker)

    Examples
    --------
    >>> from pymatgen.core import Structure
    >>> structure = Structure.from_file("POSCAR")
    >>> maker = BasisSizeConvergenceFlowMaker(
    ...     basis_sizes=["SZ", "DZ", "DZP", "DZDP", "TZP"],
    ...     energy_shift=0.01,
    ...     split_norm=0.15,
    ...     kpts=[4, 4, 4],
    ... )
    >>> flow = maker.make(structure)
    """

    name: str = "Basis Size Convergence"
    basis_sizes: list[str] = None
    energy_shift: float = 0.01
    split_norm: float = 0.15
    kpts: list[int] | None = None
    static_maker: StaticMaker | None = None

    def __post_init__(self) -> None:
        """Set default basis sizes if not provided."""
        if self.basis_sizes is None:
            self.basis_sizes = ["SZ", "DZ", "DZP", "TZP"]

        if self.static_maker is None:
            self.static_maker = StaticMaker()

        # Propagate dry_run, use_custodian, tier, manager_config to child makers
        super().__post_init__()

    def make(
        self, structure: Structure | Molecule, prev_dir: str | Path | None = None
    ) -> Flow:
        """
        Create a flow to test basis size convergence.

        Parameters
        ----------
        structure : Structure | Molecule
            Structure to run calculations on
        prev_dir : str | Path | None
            Previous directory for continuation jobs

        Returns
        -------
        Flow
            Jobflow Flow with all basis size jobs and analysis
        """
        logger.info(
            f"BasisSizeConvergenceFlowMaker.make() for "
            f"{len(self.basis_sizes)} basis sizes"
        )

        jobs = []
        job_metadata = []

        # Calculate total number of jobs
        total_jobs = len(self.basis_sizes)

        # Create a static job for each basis size
        for job_counter, basis_size in enumerate(self.basis_sizes, start=1):
            # Update user parameters with basis settings
            user_params = {
                "PAO.BasisSize": basis_size,
                "PAO.EnergyShift": f"{self.energy_shift} Ry",
                "PAO.SplitNorm": self.split_norm,
            }

            if self.kpts is not None:
                user_params["a2s_kpts"] = self.kpts

            # Create static maker with updated parameters
            from atomate2.siesta.powerups import update_user_siesta_settings

            maker = cast(
                "Maker",
                update_user_siesta_settings(self.static_maker, user_params),
            )

            # Generate job with counter
            job = maker.make(structure, prev_dir=prev_dir)
            job.name = f"{self.name}_{basis_size}_[{job_counter}_of_{total_jobs}]"
            jobs.append(job)

            # Store metadata
            job_metadata.append(
                {
                    "uuid": job.uuid,
                    "name": job.name,
                    "basis_size": basis_size,
                    "energy_shift": self.energy_shift,
                    "split_norm": self.split_norm,
                }
            )

        # Collect results
        collect_job = collect_basis_size_data(
            [job.output for job in jobs], job_metadata
        )
        jobs.append(collect_job)

        # Plot convergence
        plot_job = plot_basis_size_convergence(
            collect_job.output, output_file="basis_size_convergence.png"
        )
        jobs.append(plot_job)

        # Write summary
        summary_job = write_basis_size_summary(
            collect_job.output, output_file="basis_size_summary.txt"
        )
        jobs.append(summary_job)

        # Create flow
        return Flow(jobs, output=collect_job.output, name=self.name)


@job
def collect_basis_size_data(
    job_outputs: list[Any], job_metadata: list[dict]
) -> dict[str, Any]:
    """
    Collect energy, forces, stress, and timing from basis size tests.

    Parameters
    ----------
    job_outputs : list[Any]
        List of job outputs (SiestaTaskDoc)
    job_metadata : list[dict]
        List of dictionaries with job info (uuid, name, basis_size, etc.)

    Returns
    -------
    dict[str, Any]
        Dictionary with basis_sizes, energies, forces, stresses, timings
    """
    import numpy as np

    logger.info("Collecting basis size convergence data")
    logger.info(
        f"Received {len(job_outputs)} job outputs and "
        f"{len(job_metadata)} metadata entries"
    )
    logger.info(f"First output type: {type(job_outputs[0]) if job_outputs else 'None'}")

    data: dict[str, list[Any]] = {
        "basis_sizes": [],
        "energy_shifts": [],
        "split_norms": [],
        "energies": [],
        "max_forces": [],
        "max_stresses": [],
        "stress_tensors": [],
        "names": [],
        "run_times": [],
    }

    for i, (job_info, output) in enumerate(
        zip(job_metadata, job_outputs, strict=False)
    ):
        basis_size = job_info["basis_size"]
        energy_shift = job_info["energy_shift"]
        split_norm = job_info["split_norm"]
        job_name = job_info["name"]

        logger.info(f"Processing job {i + 1}/{len(job_metadata)}: {job_name}")
        logger.info(f"  Output type: {type(output)}")
        logger.info(f"  Output is None: {output is None}")
        if output is not None:
            logger.info(f"  Has 'energy' attr: {hasattr(output, 'energy')}")
            logger.info(f"  Output dir(): {dir(output)[:10]}...")  # First 10 attributes

        try:
            # Extract energy from output
            energy = (
                output.output.energy
                if hasattr(output, "output") and output.output
                else None
            )
            if energy is None:
                logger.warning(f"No energy found for {job_name}")
                continue

            # Get maximum force
            max_force = 0.0
            forces = (
                output.output.forces
                if hasattr(output, "output") and output.output
                else None
            )
            if forces is not None:
                forces_array = np.array(forces)
                max_force = np.max(np.linalg.norm(forces_array, axis=1))

            # Get stress tensor
            max_stress = 0.0
            stress_tensor = None
            stress = (
                output.output.stress
                if hasattr(output, "output") and output.output
                else None
            )
            if stress is not None:
                stress_tensor = np.array(stress)
                max_stress = np.max(np.abs(stress_tensor))

            # Get timing (if available)
            run_time = 0.0
            if hasattr(output, "run_stats") and output.run_stats is not None:
                if hasattr(output.run_stats, "elapsed_time"):
                    run_time = output.run_stats.elapsed_time
                elif hasattr(output.run_stats, "wall_time"):
                    run_time = output.run_stats.wall_time

            data["basis_sizes"].append(basis_size)
            data["energy_shifts"].append(energy_shift)
            data["split_norms"].append(split_norm)
            data["energies"].append(energy)
            data["max_forces"].append(max_force)
            data["max_stresses"].append(max_stress)
            data["stress_tensors"].append(
                stress_tensor if stress_tensor is not None else [0] * 6
            )
            data["names"].append(job_name)
            data["run_times"].append(run_time)

            logger.debug(
                f"{job_name}: E={energy:.6f} eV, max_F={max_force:.6f} eV/Å, "
                f"max_σ={max_stress:.4f} GPa, t={run_time:.1f}s"  # noqa: RUF001
            )

        except (KeyError, TypeError, ValueError, AttributeError):
            logger.exception(f"Error processing job {job_name}")
            import traceback

            logger.exception(traceback.format_exc())
            logger.exception(
                f"Output type: {type(output)}, has energy: {hasattr(output, 'energy')}"
            )
            continue

    logger.info(f"Collected data for {len(data['energies'])} basis sizes")
    return data


@job
def plot_basis_size_convergence(
    basis_data: dict[str, Any], output_file: str = "basis_size_convergence.png"
) -> dict[str, str]:
    """
    Plot basis size convergence (energy, forces, stress, timing).

    Parameters
    ----------
    basis_data : dict[str, Any]
        Dictionary with basis sizes and properties
    output_file : str
        Output filename for plot

    Returns
    -------
    dict[str, str]
        Dictionary with path to plot file
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if console:
        console.print("[green]Plotting basis size convergence[/green]")

    basis_sizes = basis_data["basis_sizes"]
    energies = np.array(basis_data["energies"])
    max_forces = np.array(basis_data["max_forces"])
    max_stresses = np.array(basis_data["max_stresses"])
    run_times = np.array(basis_data.get("run_times", [0] * len(energies)))

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
        return {"plot": output_file}

    # Define basis size ordering
    basis_order = ["SZ", "DZ", "DZP", "SZP", "DZDP", "TZ", "TZP", "TZDP", "QZ", "QZP"]

    # Sort data by basis size
    indices = [basis_sizes.index(bs) for bs in basis_order if bs in basis_sizes]

    # If any basis sizes not in standard order, append them
    for i, _bs in enumerate(basis_sizes):
        if i not in indices:
            indices.append(i)

    sorted_basis = [basis_sizes[i] for i in indices]
    sorted_energies = energies[indices]
    sorted_forces = max_forces[indices]
    sorted_stresses = max_stresses[indices]
    sorted_times = run_times[indices]

    # Create figure with 4 panels (or 5 if timing available)
    n_panels = 5 if np.any(sorted_times > 0) else 4
    _fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.5 * n_panels))

    # Panel 1: Absolute energies
    ax1 = axes[0]
    x_pos = np.arange(len(sorted_basis))
    ax1.plot(x_pos, sorted_energies, "o-", linewidth=2, markersize=10, color="darkblue")
    ax1.set_ylabel("Energy (eV)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Total Energy Convergence vs Basis Size", fontsize=13, fontweight="bold"
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sorted_basis, fontsize=11)
    ax1.grid(True, alpha=0.3)  # noqa: FBT003

    # Panel 2: Energy differences
    ax2 = axes[1]
    e_min = np.min(sorted_energies)
    e_diff = (sorted_energies - e_min) * 1000  # meV
    ax2.plot(x_pos, e_diff, "s-", linewidth=2, markersize=10, color="darkgreen")
    ax2.axhline(
        y=1, color="green", linestyle="--", alpha=0.5, linewidth=2, label="1 meV"
    )
    ax2.axhline(
        y=5, color="orange", linestyle="--", alpha=0.5, linewidth=2, label="5 meV"
    )
    ax2.axhline(
        y=10, color="red", linestyle="--", alpha=0.5, linewidth=2, label="10 meV"
    )
    ax2.set_ylabel("ΔE (meV)", fontsize=12, fontweight="bold")
    ax2.set_title("Energy Difference from Minimum", fontsize=13, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sorted_basis, fontsize=11)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)  # noqa: FBT003

    # Panel 3: Forces
    ax3 = axes[2]
    ax3.plot(x_pos, sorted_forces, "^-", linewidth=2, markersize=10, color="darkred")
    ax3.axhline(
        y=0.01, color="green", linestyle="--", alpha=0.5, linewidth=2, label="0.01 eV/Å"
    )
    ax3.set_ylabel("Max Force (eV/Å)", fontsize=12, fontweight="bold")
    ax3.set_title("Maximum Force Component", fontsize=13, fontweight="bold")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(sorted_basis, fontsize=11)
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(True, alpha=0.3)  # noqa: FBT003

    # Panel 4: Stresses
    ax4 = axes[3]
    ax4.plot(x_pos, sorted_stresses, "D-", linewidth=2, markersize=10, color="purple")
    ax4.axhline(
        y=0.1, color="green", linestyle="--", alpha=0.5, linewidth=2, label="0.1 GPa"
    )
    ax4.set_ylabel("Max Stress (GPa)", fontsize=12, fontweight="bold")
    ax4.set_title("Maximum Stress Component", fontsize=13, fontweight="bold")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(sorted_basis, fontsize=11)
    ax4.legend(loc="upper right", fontsize=10)
    ax4.grid(True, alpha=0.3)  # noqa: FBT003

    # Panel 5: Timing (if available)
    if n_panels == 5:
        ax5 = axes[4]
        valid_times = sorted_times > 0
        if np.any(valid_times):
            ax5.plot(
                x_pos[valid_times],
                sorted_times[valid_times],
                "o-",
                linewidth=2,
                markersize=10,
                color="darkorange",
            )
            ax5.set_ylabel("Wall Time (s)", fontsize=12, fontweight="bold")
            ax5.set_title(
                "Computational Time vs Basis Size", fontsize=13, fontweight="bold"
            )
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(sorted_basis, fontsize=11)
            ax5.grid(True, alpha=0.3)  # noqa: FBT003

    plt.suptitle("SIESTA Basis Size Convergence Study", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Basis size convergence plot saved to {output_file}")
    if console:
        console.print(f"[green]Basis size plot saved to: {output_file}[/green]")

    return {"plot": output_file}


@job
def write_basis_size_summary(
    basis_data: dict[str, Any], output_file: str = "basis_size_summary.txt"
) -> dict[str, str]:
    """
    Write summary of basis size convergence results.

    Parameters
    ----------
    basis_data : dict[str, Any]
        Dictionary with basis sizes and properties
    output_file : str
        Output filename for summary

    Returns
    -------
    dict[str, str]
        Dictionary with path to summary file
    """
    import numpy as np

    basis_sizes = basis_data["basis_sizes"]
    energies = np.array(basis_data["energies"])
    max_forces = np.array(basis_data["max_forces"])
    max_stresses = np.array(basis_data["max_stresses"])
    run_times = np.array(basis_data.get("run_times", [0] * len(energies)))

    # Check if we have data
    if len(energies) == 0:
        logger.error("No energy data collected - cannot write summary")
        with open(output_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("BASIS SIZE CONVERGENCE STUDY\n")
            f.write("=" * 80 + "\n\n")
            f.write("ERROR: No convergence data was collected.\n\n")
            f.write("This usually means:\n")
            f.write("1. The SCF calculations failed\n")
            f.write(
                "2. The job outputs were not properly passed to the collection "
                "function\n"
            )
            f.write("3. The output format was not recognized\n\n")
            f.write("Check the individual job directories for calculation outputs.\n")
        return {"summary": output_file}

    # Sort by basis size
    basis_order = ["SZ", "DZ", "DZP", "SZP", "DZDP", "TZ", "TZP", "TZDP", "QZ", "QZP"]
    indices = [basis_sizes.index(bs) for bs in basis_order if bs in basis_sizes]
    for i, _bs in enumerate(basis_sizes):
        if i not in indices:
            indices.append(i)

    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BASIS SIZE CONVERGENCE STUDY\n")
        f.write("=" * 80 + "\n\n")

        f.write("CONVERGENCE RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Basis':<10} {'Energy (eV)':<16} {'ΔE (meV)':<12} {'Max Force':<12} "
            f"{'Max Stress':<12} {'Time (s)':<12}\n"
        )
        f.write(
            f"{'Size':<10} {'':<16} {'vs min':<12} {'(eV/Å)':<12} "
            f"{'(GPa)':<12} {'wall':<12}\n"
        )
        f.write("-" * 80 + "\n")

        e_min = np.min(energies)
        for idx in indices:
            bs = basis_sizes[idx]
            e = energies[idx]
            de = (e - e_min) * 1000
            f_max = max_forces[idx]
            s_max = max_stresses[idx]
            t = run_times[idx]

            time_str = f"{t:.1f}" if t > 0 else "N/A"
            f.write(
                f"{bs:<10} {e:<16.8f} {de:<12.4f} {f_max:<12.6f} "
                f"{s_max:<12.6f} {time_str:<12}\n"
            )

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("CONVERGENCE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # Find optimal basis
        optimal_idx = np.argmin(energies)
        f.write("Lowest Energy Configuration:\n")
        f.write(f"  Basis Size = {basis_sizes[optimal_idx]}\n")
        f.write(f"  Energy     = {energies[optimal_idx]:.8f} eV\n")
        f.write(f"  Max Force  = {max_forces[optimal_idx]:.6f} eV/Å\n")
        f.write(f"  Max Stress = {max_stresses[optimal_idx]:.6f} GPa\n\n")

        # Energy range
        e_range = (energies.max() - energies.min()) * 1000
        f.write("Energy Spread:\n")
        f.write(f"  Total range: {e_range:.4f} meV\n\n")

        # Convergence recommendation
        f.write("Convergence Assessment:\n")
        if e_range < 1.0:
            f.write("  ✓ EXCELLENT: All basis sizes converged within 1 meV\n")
        elif e_range < 5.0:
            f.write("  ✓ GOOD: Converged within 5 meV\n")
        elif e_range < 10.0:
            f.write("  ⚠ FAIR: Variation 5-10 meV - consider larger basis\n")
        else:
            f.write("  ✗ POOR: Variation > 10 meV - need larger basis\n")
        f.write("\n")

        # Timing analysis
        valid_times = run_times[run_times > 0]
        if len(valid_times) > 0:
            f.write("Computational Performance:\n")
            f.write(f"  Total wall time:   {np.sum(valid_times):.1f} s\n")
            f.write(f"  Average time:      {np.mean(valid_times):.1f} s\n")
            f.write(f"  Fastest calc:      {np.min(valid_times):.1f} s\n")
            f.write(f"  Slowest calc:      {np.max(valid_times):.1f} s\n\n")

        # Recommendations
        f.write("Recommendations:\n")
        if e_range < 5.0:
            # Find smallest basis within 5 meV
            for idx in indices:
                e_diff = (energies[idx] - e_min) * 1000
                if e_diff < 5.0:
                    f.write(
                        f"  → Use {basis_sizes[idx]} basis (converged within 5 meV)\n"
                    )
                    break
        else:
            f.write(f"  → Use larger basis (current range: {e_range:.2f} meV)\n")
            f.write("  → Consider testing DZDP, TZP, or TZDP\n")
        f.write("\n")

        # Add standard footer
        from atomate2.siesta.utils.text_output import get_standard_footer

        f.write(
            get_standard_footer(
                width=80,
                additional_info={
                    "Analysis type": "Basis size convergence study",
                    "Number of basis sizes": str(len(basis_sizes)),
                },
            )
        )

    logger.info(f"Basis size summary written to {output_file}")
    if console:
        console.print(f"[green]Summary written to: {output_file}[/green]")

    return {"summary": output_file}


# =============================================================================
# Complete Basis Convergence Workflow (Size + Parameters)
# =============================================================================
