"""
Basis convergence workflows for SIESTA.

This module provides workflows for testing basis set quality and convergence:

1. DifferentBasisSCF / DifferentBasisSCFAdvance:
   - Tests different nominal basis sizes (SZ, DZ, DZP, TZP, etc.)
   - Helps determine overall basis quality needed

2. EOSBasisConvergenceFlowMaker:
   - Tests EOS parameters (V₀, B₀) with different basis sets
   - Validates basis convergence for structural/bulk properties

3. BasisParametersConvergenceMaker:
   - Optimizes PAO.EnergyShift and PAO.SplitNorm parameters
   - Fine-tunes basis generation for optimal accuracy/cost balance
   - Provides detailed 4-panel convergence plots and analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from jobflow.core.flow import Flow
from jobflow.core.job import job
from pymatgen.analysis.eos import EOS

from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.powerups import update_user_siesta_settings
from atomate2.siesta.utils.common import console, print_docstring_in_box
from atomate2.siesta.utils.verbosity import VerbosityLevel

if TYPE_CHECKING:
    from pymatgen.core import Molecule, Structure

logger = logging.getLogger(__name__)


@job
def print_energies(
    flow_results: dict[str, Any], job_metadata: list[dict]
) -> dict[str, float]:
    """
    Retrieve and print the total energies from each job in the Flow's results using job.output.

    Args:
        flow_results (Dict[str, Any]): The results dictionary returned by jobflow's run_locally.
        job_metadata (list[dict]): List of dictionaries containing job names and UUIDs.
        verbosity (VerbosityLevel): Verbosity level for console output. Defaults to INFO.

    Returns
    -------
        Dict[str, float]: A dictionary mapping job names to their total energies (in eV).
    """
    from atomate2.siesta.flows.basis.core import DifferentBasisSCFAdvanceFlowMaker

    verbosity = DifferentBasisSCFAdvanceFlowMaker.CONSOLE_VERBOSITY
    energies = {}
    for job_info in job_metadata:
        job_uuid = job_info["uuid"]
        job_name = job_info["name"]
        try:
            if verbosity.value >= VerbosityLevel.DEBUG.value:
                console.print(f"[green]The job in flow name {job_name=}[/green]")
                console.print(f"[green]The job in flow uuid:{job_uuid=}[/green]")
            result = flow_results[job_uuid]
            output = result.output  # Access SiestaTaskDoc
            energy = output.energy
            energies[job_name] = energy
            if verbosity.value >= VerbosityLevel.DEBUG.value:
                console.print(f"[blue]{job_name=}: {result=}[/blue]")
                console.print(f"[blue]{job_name=}: {output=}[/blue]")
                console.print(f"[blue]{job_name=}: {energy=}[/blue]")
            if job_uuid not in flow_results:
                if verbosity.value >= VerbosityLevel.WARNING.value:
                    console.print(
                        f"[yellow]No results found for job {job_name} (UUID: {job_uuid})[/yellow]"
                    )
                continue
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            if verbosity.value >= VerbosityLevel.WARNING.value:
                console.print(
                    f"[red]Error processing job {job_name} (UUID: {job_uuid}): {e}[/red]"
                )
            continue

    if not energies:
        if verbosity.value >= VerbosityLevel.WARNING.value:
            console.print("[red]No energies retrieved from Flow results[/red]")
    elif verbosity.value >= VerbosityLevel.WARNING.value:
        console.print(
            f"[green]Energies retrieved from Flow results {energies=}[/green]"
        )

    return energies


@job
def print_energies_old(flow: Flow, flow_results: dict[str, Any]) -> dict[str, float]:
    """
    Retrieve and print the total energies from each job in the Flow's results using job.output.

    Args:
        flow (Flow): The Flow object containing the SCF jobs.
        flow_results (Dict[str, Any]): The results dictionary returned by jobflow's run_locally.
        verbosity (VerbosityLevel): Verbosity level for console output. Defaults to DEBUG.

    Returns
    -------
        Dict[str, float]: A dictionary mapping job names to their total energies (in eV).
    """
    from atomate2.siesta.flows.basis.core import DifferentBasisSCFAdvanceFlowMaker

    verbosity = DifferentBasisSCFAdvanceFlowMaker.CONSOLE_VERBOSITY
    if verbosity.value >= VerbosityLevel.INFO.value:
        console.print("[green]Retrieving & plotting energies from Flow results[/green]")

    # if verbosity.value >= VerbosityLevel.INFO.value:
    #     console.print(f"[green]The Flow is :{flow=}[/green]")
    #     console.print(f"[green]The Flow results is:{flow_results=}[/green]")

    energies = {}
    for flow_job in flow.jobs:
        job_uuid = flow_job.uuid
        job_name = flow_job.name
        try:
            if verbosity.value >= VerbosityLevel.DEBUG.value:
                console.print(f"[green]The job in flow name {job_name=}[/green]")
                console.print(f"[green]The job in flow uuid:{job_uuid=}[/green]")

                result = flow_results[job_uuid]
                input = result.input
                output = result.output  # Access SiestaTaskDoc
                energy = output.energy
                energies[job_name] = energy
                if verbosity.value >= VerbosityLevel.DEBUG.value:
                    console.print(f"[blue]{job_name=}: {result=}[/blue]")
                    console.print(f"[blue]{job_name=}: {input=}[/blue]")
                    console.print(f"[blue]{job_name=}: {output=}[/blue]")
                    console.print(f"[blue]{job_name=}: {energy=}[/blue]")

            # Check if job results exist in flow_results
            if job_uuid not in flow_results:
                if verbosity.value >= VerbosityLevel.WARNING.value:
                    console.print(
                        f"[yellow]No results found for job {job_name} (UUID: {job_uuid})[/yellow]"
                    )
                continue

        except (KeyError, TypeError, ValueError, AttributeError) as e:
            if verbosity.value >= VerbosityLevel.WARNING.value:
                console.print(
                    f"[red]Error processing job {job_name} (UUID: {job_uuid}): {e}[/red]"
                )
            continue

    if not energies:
        if verbosity.value >= VerbosityLevel.WARNING.value:
            console.print("[red]No energies retrieved from Flow results[/red]")
    elif verbosity.value >= VerbosityLevel.WARNING.value:
        console.print(
            f"[green]Energies retrieved from Flow results {energies=}[/green]"
        )

    return energies


@job
def plot_energies(
    energies: dict[str, float], verbosity: VerbosityLevel = VerbosityLevel.INFO
):
    """
    Plot total energies vs. basis size.

    Args:
        energies (Dict[str, float]): Dictionary mapping job names to total energies (in eV).
        verbosity (VerbosityLevel): Verbosity level for console output. Defaults to INFO.
    """
    import matplotlib.pyplot as plt

    if verbosity.value >= VerbosityLevel.INFO.value:
        console.print("[green]Plotting energies vs. basis size[/green]")
    basis_sizes = [name.split("-")[-1] for name in energies]
    energy_values = list(energies.values())
    plt.figure(figsize=(10, 6))
    plt.plot(basis_sizes, energy_values, "o-")
    plt.xlabel("Basis Size")
    plt.ylabel("Total Energy (eV)")
    plt.title("Total Energy vs. Basis Size")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("total_energy_vs_basis_size.png")
    # plt.show()


@dataclass
class EOSBasisConvergenceFlowMaker(BaseSiestaFlowMaker):
    """
    Workflow for testing EOS parameters with different basis sets.

    This workflow:
    1. Runs EOS calculations with different basis sets in parallel
    2. Collects equilibrium volume (V₀), energy (E₀), and bulk modulus (B₀)
    3. Plots the convergence of these parameters with basis set quality
    4. Generates a summary comparing all basis sets

    This helps determine the optimal basis set for accurate bulk properties.

    Example:
        >>> from atomate2.siesta.flows.basis import EOSBasisConvergenceFlowMaker
        >>> from pymatgen.core import Structure
        >>> structure = Structure.from_file("structure.cif")
        >>> maker = EOSBasisConvergenceFlowMaker(basis_sets=["SZ", "DZ", "DZP", "TZP"])
        >>> flow = maker.make(structure)
    """

    CONSOLE_VERBOSITY: VerbosityLevel = VerbosityLevel.INFO
    name: str = "EOS Basis Convergence"
    basis_sets: list[str] = field(
        default_factory=lambda: ["SZ", "DZ", "DZP", "DZDP", "TZP", "TZDP"]
    )
    linear_strain: tuple[float, float] = (-0.05, 0.05)
    number_of_frames: int = 6

    # Dry-run support
    dry_run: bool = False
    dry_run_output_dir: str = "dry_run_output"
    dry_run_format: str = "cif"

    def make(
        self, structure: Structure | Molecule, prev_dir: str | Path | None = None
    ) -> Flow:
        """
        Create EOS convergence flow for different basis sets.

        Args:
            structure: Structure to calculate
            prev_dir: Previous directory (optional)

        Returns
        -------
            Flow with EOS jobs for each basis set
        """
        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        from atomate2.siesta.flows.eos import SiestaEosFlowMaker

        jobs = []
        job_metadata = []

        # Define basis parameters
        basis_params = {
            "SZ": {"PAO.EnergyShift": "0.02 Ry", "PAO.SplitNorm": 0.15},
            "DZ": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
            "DZP": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
            "DZDP": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
            "TZ": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
            "TZP": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
            "TZDP": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
        }

        for basis in self.basis_sets:
            # Create EOS maker
            eos_maker = SiestaEosFlowMaker(
                name=f"EOS-{basis}",
                linear_strain=self.linear_strain,
                number_of_frames=self.number_of_frames,
                dry_run=self.dry_run,
                dry_run_output_dir=self.dry_run_output_dir,
                dry_run_format=self.dry_run_format,
            )

            # Create the EOS flow for this basis
            eos_flow = eos_maker.make(structure, prev_dir=prev_dir)

            # Update all jobs in the EOS flow with the specific basis settings
            siesta_updates = {
                "PAO.BasisSize": basis,
                "PAO.BasisType": "split",
                **basis_params.get(
                    basis, {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.15}
                ),
            }

            # Update the flow with basis-specific parameters
            eos_flow = update_user_siesta_settings(eos_flow, siesta_updates)

            jobs.append(eos_flow)
            job_metadata.append(
                {"uuid": eos_flow.uuid, "name": f"EOS-{basis}", "basis_set": basis}
            )

        # Create a flow from the parallel EOS jobs
        eos_flows = Flow(
            jobs,
            name="EOS Calculations - Different Basis Sets",
            output={job.uuid: job.output for job in jobs},
        )

        # Create collection job
        collect_job = collect_eos_basis_data(
            flow_results=eos_flows.output,
            job_metadata=job_metadata,
        )
        collect_job.name = f"{self.name}-collect"

        # Create unified output job that generates all plots and summaries in one directory
        unified_output_job = generate_eos_basis_outputs(
            flow_results=eos_flows.output,
            job_metadata=job_metadata,
            basis_data=collect_job.output,
        )
        unified_output_job.name = f"{self.name}-outputs"

        # Combine into final flow
        return Flow(
            [eos_flows, collect_job, unified_output_job],
            output=unified_output_job.output,
            name=self.name,
        )


# =============================================================================
# Basis Parameters (PAO.EnergyShift & PAO.SplitNorm) Convergence Workflow
# =============================================================================


def _extract_siesta_timing(siesta_out_path):
    """Extract wall time from siesta.out file (handles both plain and gzipped)."""
    import gzip
    from pathlib import Path

    try:
        # Check if file is gzipped
        if str(siesta_out_path).endswith(".gz"):
            opener = gzip.open
            mode = "rt"  # text mode for gzip
        else:
            opener = open
            mode = "r"

        with opener(siesta_out_path, mode) as f:
            for line in f:
                # Match multiple timing formats:
                # 1. "timer: Elapsed wall time (sec)"  (old format)
                # 2. "Elapsed wall time (sec)"          (old format variant)
                # 3. "timer: Total elapsed wall-clock time (sec)" (new format in siesta.times)
                if (
                    "timer: Elapsed wall time (sec)" in line
                    or "Elapsed wall time (sec)" in line
                    or "Total elapsed wall-clock time (sec)" in line
                ):
                    parts = line.split("=")
                    if len(parts) == 2:
                        time_val = float(parts[1].strip())
                        logger.debug(
                            f"Extracted timing {time_val:.2f}s from {Path(siesta_out_path).name}"
                        )
                        return time_val
    except Exception as e:
        logger.debug(f"Could not extract timing from {siesta_out_path}: {e}")
    return 0.0


@job
def collect_basis_params_data(
    flow_results: dict[str, Any], job_metadata: list[dict]
) -> dict[str, Any]:
    """
    Collect energy, forces, stress, and timing from basis parameter convergence tests.

    Args:
        flow_results: Results dictionary from jobflow's run_locally
        job_metadata: List of dictionaries with job info (uuid, name, energy_shift, split_norm)

    Returns
    -------
        Dictionary with energy_shifts, split_norms, energies, forces, stresses, and timings
    """
    logger.info("Collecting basis parameter convergence data")

    # First, scan all job directories for timing information
    from pathlib import Path

    timing_by_dir = {}  # Map directory path -> timing
    timing_by_params = {}  # Map (energy_shift, split_norm) -> timing

    cwd = Path.cwd()
    search_dirs = [cwd, cwd.parent] if cwd.name.startswith("job_") else [cwd]

    logger.info(f"Searching for job directories in: {[str(d) for d in search_dirs]}")

    for search_dir in search_dirs:
        for job_dir in Path(search_dir).glob("job_*"):
            # Check for both plain and gzipped siesta.out
            siesta_out = job_dir / "siesta.out"
            siesta_out_gz = job_dir / "siesta.out.gz"

            if siesta_out_gz.exists():
                siesta_out = siesta_out_gz
            elif not siesta_out.exists():
                continue

            wall_time = _extract_siesta_timing(siesta_out)

            # If timing not found in siesta.out, check siesta_compressed/siesta.times.gz
            # (this happens when custodian runs and compresses output files)
            if wall_time == 0:
                siesta_times_gz = job_dir / "siesta_compressed" / "siesta.times.gz"
                if siesta_times_gz.exists():
                    wall_time = _extract_siesta_timing(siesta_times_gz)
                    logger.debug(
                        f"Found timing in compressed siesta.times.gz for {job_dir.name}"
                    )

            if wall_time > 0:
                # Store by absolute path
                abs_path = str(job_dir.resolve())
                timing_by_dir[abs_path] = wall_time
                timing_by_dir[str(job_dir)] = wall_time  # Also store relative
                timing_by_dir[job_dir.name] = wall_time  # Also store just name
                logger.info(f"Found timing {wall_time:.2f}s in {job_dir.name}")

    logger.info(
        f"Found {len(set(timing_by_dir.values()))} job directories with timing information"
    )

    # Also try to build a mapping by inspecting job directories for their parameters
    for job_dir_path in list(timing_by_dir.keys()):
        if not isinstance(job_dir_path, str) or not job_dir_path.startswith("job_"):
            continue
        job_dir = Path(job_dir_path) if "/" in job_dir_path else cwd / job_dir_path
        if not job_dir.exists():
            continue

        # Try to read parameters from siesta.fdf (handle both plain and gzipped)
        fdf_file = job_dir / "siesta.fdf"
        fdf_file_gz = job_dir / "siesta.fdf.gz"

        if fdf_file_gz.exists():
            fdf_file = fdf_file_gz

        if fdf_file.exists():
            try:
                import gzip

                es, sn = None, None

                # Choose opener based on file extension
                if str(fdf_file).endswith(".gz"):
                    opener = gzip.open  # type: ignore[assignment]
                    mode = "rt"
                else:
                    opener = open  # type: ignore[assignment]
                    mode = "r"

                with opener(fdf_file, mode) as f:
                    for line in f:
                        if "PAO.EnergyShift" in line and "PAO.SplitNormH" not in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                es = float(parts[1])
                        if "PAO.SplitNorm" in line and "PAO.SplitNormH" not in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                sn = float(parts[1])
                if es is not None and sn is not None:
                    timing_by_params[(es, sn)] = timing_by_dir[job_dir_path]
                    logger.info(
                        f"Mapped {job_dir.name} -> ES={es}, SN={sn}, t={timing_by_dir[job_dir_path]:.2f}s"
                    )
            except Exception as e:
                logger.debug(f"Could not read parameters from {fdf_file}: {e}")

    logger.info(f"Successfully mapped {len(timing_by_params)} jobs by parameters")

    data: dict[str, Any] = {
        "energy_shifts": [],
        "split_norms": [],
        "energies": [],
        "max_forces": [],
        "max_stresses": [],
        "stress_tensors": [],
        "param_labels": [],
        "names": [],
        "run_times": [],  # Calculation times in seconds
    }

    for job_info in job_metadata:
        job_uuid = job_info["uuid"]
        job_name = job_info["name"]
        energy_shift = job_info["energy_shift"]
        split_norm = job_info["split_norm"]

        try:
            if job_uuid not in flow_results:
                logger.warning(
                    f"No results found for job {job_name} (UUID: {job_uuid})"
                )
                continue

            result = flow_results[job_uuid]
            output = result.output  # SiestaTaskDoc

            energy = output.energy

            # Get maximum force magnitude
            max_force = 0.0
            if hasattr(output, "forces") and output.forces is not None:
                import numpy as np

                forces_array = np.array(output.forces)
                max_force = np.max(np.linalg.norm(forces_array, axis=1))

            # Get stress tensor and maximum stress component
            max_stress = 0.0
            stress_tensor = None
            if hasattr(output, "stress") and output.stress is not None:
                import numpy as np

                stress_tensor = np.array(output.stress)
                # Max absolute value of stress components (GPa)
                max_stress = np.max(np.abs(stress_tensor))

            # Get timing information - try multiple strategies
            run_time = 0.0

            # Strategy 1: Match by parameters (most reliable)
            param_key = (energy_shift, split_norm)
            if param_key in timing_by_params:
                run_time = timing_by_params[param_key]
                logger.debug(
                    f"Matched timing {run_time:.2f}s for {job_name} by parameters ES={energy_shift}, SN={split_norm}"
                )

            # Strategy 2: Try to get directory from output object
            if run_time == 0.0:
                calc_dir = None
                if hasattr(output, "dir_name") and output.dir_name:
                    calc_dir = output.dir_name
                elif hasattr(result, "dir_name") and result.dir_name:
                    calc_dir = result.dir_name

                if calc_dir:
                    # Try multiple path formats
                    for path_variant in [
                        str(calc_dir),
                        str(Path(calc_dir).resolve()),
                        Path(calc_dir).name,
                    ]:
                        if path_variant in timing_by_dir:
                            run_time = timing_by_dir[path_variant]
                            logger.debug(
                                f"Matched timing {run_time:.2f}s for {job_name} from directory {path_variant}"
                            )
                            break

                    # If still not found, try to extract directly (check both .gz and plain)
                    if run_time == 0.0:
                        siesta_out = Path(calc_dir) / "siesta.out"
                        siesta_out_gz = Path(calc_dir) / "siesta.out.gz"

                        if siesta_out_gz.exists():
                            run_time = _extract_siesta_timing(siesta_out_gz)
                            if run_time > 0:
                                logger.debug(
                                    f"Extracted timing {run_time:.2f}s directly from {siesta_out_gz}"
                                )
                        elif siesta_out.exists():
                            run_time = _extract_siesta_timing(siesta_out)
                            if run_time > 0:
                                logger.debug(
                                    f"Extracted timing {run_time:.2f}s directly from {siesta_out}"
                                )

            # Strategy 3: If parameters don't match exactly, try with tolerance
            if run_time == 0.0:
                import numpy as np

                for (es, sn), time_val in timing_by_params.items():
                    if np.isclose(es, energy_shift, rtol=1e-6) and np.isclose(
                        sn, split_norm, rtol=1e-6
                    ):
                        run_time = time_val
                        logger.debug(
                            f"Matched timing {run_time:.2f}s for {job_name} by approximate parameters"
                        )
                        break

            # Warn if no timing found
            if run_time == 0.0:
                logger.warning(
                    f"Could not find timing for {job_name} (ES={energy_shift}, SN={split_norm})"
                )
                logger.warning(
                    f"  Available parameter mappings: {list(timing_by_params.keys())[:5]}"
                )

            data["names"].append(job_name)
            data["energy_shifts"].append(energy_shift)
            data["split_norms"].append(split_norm)
            data["energies"].append(energy)
            data["max_forces"].append(max_force)
            data["max_stresses"].append(max_stress)
            data["stress_tensors"].append(
                stress_tensor if stress_tensor is not None else [0] * 6
            )
            data["param_labels"].append(f"ES={energy_shift:.3f},SN={split_norm:.2f}")
            data["run_times"].append(run_time)

            logger.debug(
                f"{job_name}: E={energy:.6f} eV, ES={energy_shift}, SN={split_norm}, "
                f"max_F={max_force:.6f} eV/Å, max_σ={max_stress:.4f} GPa, t={run_time:.1f}s"  # noqa: RUF001
            )

        except (KeyError, TypeError, ValueError, AttributeError) as e:
            logger.exception(f"Error processing job {job_name}: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            continue

    if not data["energies"]:
        logger.warning("No basis parameter data retrieved")
    else:
        logger.info(
            f"Successfully collected {len(data['energies'])} results for basis parameter analysis"
        )

    return data


@job
def plot_basis_params_convergence(
    basis_params_data: dict[str, Any], output_file: str = "basis_params_convergence.png"
) -> dict[str, str]:
    """
    Plot energy, forces, and timing convergence vs PAO.EnergyShift and PAO.SplitNorm.

    Creates a combined multi-panel plot and individual plots:
    1. Energy vs EnergyShift for each SplitNorm
    2. Energy vs SplitNorm for each EnergyShift
    3. 2D heatmap of energies
    4. Basis quality map with forces and stresses
    5. Timing vs EnergyShift (if timing data available)
    6. Timing vs SplitNorm (if timing data available)
    7. 2D timing heatmap (if timing data available)
    8. Efficiency plot: Energy vs Time (if timing data available)

    Args:
        basis_params_data: Dictionary with energies, parameters, and timing
        output_file: Output filename for combined plot

    Returns
    -------
        Dictionary with paths to all saved plot files
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm

    console.print("[green]Plotting basis parameter convergence[/green]")

    # Check if we have data
    if not basis_params_data["energies"]:
        error_msg = "No basis parameter data available for plotting"
        logger.error(error_msg)
        console.print(f"[red]{error_msg}[/red]")
        raise ValueError(error_msg)

    energy_shifts = np.array(basis_params_data["energy_shifts"])
    split_norms = np.array(basis_params_data["split_norms"])
    energies = np.array(basis_params_data["energies"])
    max_forces = np.array(basis_params_data["max_forces"])

    logger.info(f"Plotting data for {len(energies)} parameter combinations")

    # Create figure with 4 subplots in 2x2 grid
    plt.figure(figsize=(16, 12))

    # Get unique values
    unique_shifts = np.unique(energy_shifts)
    unique_norms = np.unique(split_norms)

    # Plot 1: Energy vs EnergyShift (for each SplitNorm)
    ax1 = plt.subplot(2, 2, 1)
    colors = cm.viridis(np.linspace(0, 1, len(unique_norms)))

    for i, norm in enumerate(unique_norms):
        mask = split_norms == norm
        shifts_filtered = energy_shifts[mask]
        energies_filtered = energies[mask]

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
        "Energy Convergence vs PAO.EnergyShift", fontsize=13, fontweight="bold"
    )
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)  # noqa: FBT003

    # Plot 2: Energy vs SplitNorm (for each EnergyShift)
    ax2 = plt.subplot(2, 2, 2)
    colors = cm.plasma(np.linspace(0, 1, len(unique_shifts)))

    for i, shift in enumerate(unique_shifts):
        mask = energy_shifts == shift
        norms_filtered = split_norms[mask]
        energies_filtered = energies[mask]

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
    ax2.set_title("Energy Convergence vs PAO.SplitNorm", fontsize=13, fontweight="bold")
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)  # noqa: FBT003

    # Plot 3: 2D Energy Heatmap
    ax3 = plt.subplot(2, 2, 3)

    # Create grid for heatmap
    shift_grid, norm_grid = np.meshgrid(unique_shifts, unique_norms)
    energy_grid = np.zeros_like(shift_grid)

    for i in range(len(energies)):
        shift_idx = np.where(unique_shifts == energy_shifts[i])[0][0]
        norm_idx = np.where(unique_norms == split_norms[i])[0][0]
        energy_grid[norm_idx, shift_idx] = energies[i]

    # Normalize to show relative energies (difference from minimum)
    energy_diff = (energy_grid - np.min(energies)) * 1000  # Convert to meV

    im = ax3.contourf(shift_grid, norm_grid, energy_diff, levels=20, cmap="RdYlGn_r")
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
        energy_shifts,
        split_norms,
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
    ax3.set_title("Energy Landscape (2D Heatmap)", fontsize=13, fontweight="bold")

    # Plot 4: Force and Stress convergence map
    ax4 = plt.subplot(2, 2, 4)

    max_stresses = np.array(basis_params_data.get("max_stresses", [0] * len(energies)))

    # Calculate energy spread for each parameter combination
    energy_relative = (energies - np.min(energies)) * 1000  # meV

    # Create scatter plot: EnergyShift vs SplitNorm, colored by energy diff
    scatter = ax4.scatter(
        energy_shifts,
        split_norms,
        c=energy_relative,
        s=200,
        cmap="RdYlGn_r",
        edgecolors="black",
        linewidths=1.5,
    )

    # Add text labels showing max force and max stress
    for i in range(len(energies)):
        # Display force (top) and stress (bottom)
        label_text = f"F:{max_forces[i]:.2f}\nσ:{max_stresses[i]:.2f}"  # noqa: RUF001
        ax4.annotate(
            label_text,
            xy=(energy_shifts[i], split_norms[i]),
            xytext=(0, 8),  # 8 points up from the point
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

    cbar2 = plt.colorbar(scatter, ax=ax4)
    cbar2.set_label("ΔE from minimum (meV)", fontsize=11, fontweight="bold")

    ax4.set_xlabel("PAO.EnergyShift (Ry)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("PAO.SplitNorm", fontsize=12, fontweight="bold")
    ax4.set_title(
        "Basis Quality Map\n(F: max force eV/Å, σ: max stress GPa)",  # noqa: RUF001
        fontsize=13,
        fontweight="bold",
    )
    ax4.grid(True, alpha=0.3)  # noqa: FBT003

    # Add convergence zones
    ax4.axhline(y=0.15, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax4.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax4.axvline(x=0.01, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax4.axvline(x=0.005, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Add annotations for typical zones
    ax4.text(0.025, 0.10, "SZ zone", fontsize=9, style="italic", alpha=0.6)
    ax4.text(0.015, 0.175, "DZ zone", fontsize=9, style="italic", alpha=0.6)
    ax4.text(0.007, 0.275, "TZ zone", fontsize=9, style="italic", alpha=0.6)

    plt.suptitle(
        "SIESTA Basis Parameters Convergence Study\nPAO.EnergyShift & PAO.SplitNorm",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Combined convergence plot saved to {output_file}")
    console.print(f"[green]Combined convergence plot saved to: {output_file}[/green]")

    # Save individual plots
    from pathlib import Path

    base_name = Path(output_file).stem
    output_dir = Path(output_file).parent

    individual_files = {}
    individual_files["combined"] = output_file

    # Individual Plot 1: Energy vs EnergyShift
    _fig1, ax1 = plt.subplots(figsize=(10, 7))
    colors = cm.viridis(np.linspace(0, 1, len(unique_norms)))
    for i, norm in enumerate(unique_norms):
        mask = split_norms == norm
        shifts_filtered = energy_shifts[mask]
        energies_filtered = energies[mask]
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
        "Energy Convergence vs PAO.EnergyShift", fontsize=13, fontweight="bold"
    )
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)  # noqa: FBT003
    file1 = str(output_dir / f"{base_name}_energy_vs_shift.png")
    plt.savefig(file1, dpi=150, bbox_inches="tight")
    plt.close()
    individual_files["energy_vs_shift"] = file1

    # Individual Plot 2: Energy vs SplitNorm
    _fig2, ax2 = plt.subplots(figsize=(10, 7))
    colors = cm.plasma(np.linspace(0, 1, len(unique_shifts)))
    for i, shift in enumerate(unique_shifts):
        mask = energy_shifts == shift
        norms_filtered = split_norms[mask]
        energies_filtered = energies[mask]
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
    ax2.set_title("Energy Convergence vs PAO.SplitNorm", fontsize=13, fontweight="bold")
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)  # noqa: FBT003
    file2 = str(output_dir / f"{base_name}_energy_vs_norm.png")
    plt.savefig(file2, dpi=150, bbox_inches="tight")
    plt.close()
    individual_files["energy_vs_norm"] = file2

    # Individual Plot 3: 2D Energy Heatmap
    _fig3, ax3 = plt.subplots(figsize=(10, 8))
    shift_grid, norm_grid = np.meshgrid(unique_shifts, unique_norms)
    energy_grid = np.zeros_like(shift_grid)
    for i in range(len(energies)):
        shift_idx = np.where(unique_shifts == energy_shifts[i])[0][0]
        norm_idx = np.where(unique_norms == split_norms[i])[0][0]
        energy_grid[norm_idx, shift_idx] = energies[i]
    energy_diff = (energy_grid - np.min(energies)) * 1000
    im = ax3.contourf(shift_grid, norm_grid, energy_diff, levels=20, cmap="RdYlGn_r")
    ax3.contour(
        shift_grid,
        norm_grid,
        energy_diff,
        levels=10,
        colors="black",
        linewidths=0.5,
        alpha=0.3,
    )
    ax3.scatter(
        energy_shifts,
        split_norms,
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
    ax3.set_title("Energy Landscape (2D Heatmap)", fontsize=13, fontweight="bold")
    file3 = str(output_dir / f"{base_name}_heatmap.png")
    plt.savefig(file3, dpi=150, bbox_inches="tight")
    plt.close()
    individual_files["heatmap"] = file3

    # Individual Plot 4: Quality Map
    _fig4, ax4 = plt.subplots(figsize=(10, 8))
    max_stresses = np.array(basis_params_data.get("max_stresses", [0] * len(energies)))
    energy_relative = (energies - np.min(energies)) * 1000
    scatter = ax4.scatter(
        energy_shifts,
        split_norms,
        c=energy_relative,
        s=200,
        cmap="RdYlGn_r",
        edgecolors="black",
        linewidths=1.5,
    )
    for i in range(len(energies)):
        label_text = f"F:{max_forces[i]:.2f}\nσ:{max_stresses[i]:.2f}"  # noqa: RUF001
        ax4.annotate(
            label_text,
            xy=(energy_shifts[i], split_norms[i]),
            xytext=(0, 8),  # 8 points up from the point
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )
    cbar2 = plt.colorbar(scatter, ax=ax4)
    cbar2.set_label("ΔE from minimum (meV)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("PAO.EnergyShift (Ry)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("PAO.SplitNorm", fontsize=12, fontweight="bold")
    ax4.set_title(
        "Basis Quality Map\n(F: max force eV/Å, σ: max stress GPa)",  # noqa: RUF001
        fontsize=13,
        fontweight="bold",
    )
    ax4.grid(True, alpha=0.3)  # noqa: FBT003
    ax4.axhline(y=0.15, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax4.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax4.axvline(x=0.01, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax4.axvline(x=0.005, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax4.text(0.025, 0.10, "SZ zone", fontsize=9, style="italic", alpha=0.6)
    ax4.text(0.015, 0.175, "DZ zone", fontsize=9, style="italic", alpha=0.6)
    ax4.text(0.007, 0.275, "TZ zone", fontsize=9, style="italic", alpha=0.6)
    file4 = str(output_dir / f"{base_name}_quality_map.png")
    plt.savefig(file4, dpi=150, bbox_inches="tight")
    plt.close()
    individual_files["quality_map"] = file4

    # Add timing plots if timing data is available
    run_times = np.array(basis_params_data.get("run_times", []))
    valid_times = run_times[run_times > 0]

    if len(valid_times) > 0:
        logger.info("Creating timing analysis plots")

        # Individual Plot 5: Timing vs EnergyShift
        _fig5, ax5 = plt.subplots(figsize=(10, 7))
        colors = cm.viridis(np.linspace(0, 1, len(unique_norms)))
        for i, norm in enumerate(unique_norms):
            mask = split_norms == norm
            shifts_filtered = energy_shifts[mask]
            times_filtered = run_times[mask]
            # Filter out zero times
            valid_mask = times_filtered > 0
            if np.sum(valid_mask) > 0:
                sort_idx = np.argsort(shifts_filtered[valid_mask])
                ax5.plot(
                    shifts_filtered[valid_mask][sort_idx],
                    times_filtered[valid_mask][sort_idx],
                    "o-",
                    color=colors[i],
                    linewidth=2,
                    markersize=8,
                    label=f"SplitNorm={norm:.2f}",
                )
        ax5.set_xlabel("PAO.EnergyShift (Ry)", fontsize=12, fontweight="bold")
        ax5.set_ylabel("Wall Time (s)", fontsize=12, fontweight="bold")
        ax5.set_title(
            "Computational Time vs PAO.EnergyShift\n(Lower ES → Larger basis → Slower)",
            fontsize=13,
            fontweight="bold",
        )
        ax5.legend(loc="best", fontsize=10)
        ax5.grid(True, alpha=0.3)  # noqa: FBT003
        file5 = str(output_dir / f"{base_name}_timing_vs_shift.png")
        plt.savefig(file5, dpi=150, bbox_inches="tight")
        plt.close()
        individual_files["timing_vs_shift"] = file5

        # Individual Plot 6: Timing vs SplitNorm
        _fig6, ax6 = plt.subplots(figsize=(10, 7))
        colors = cm.plasma(np.linspace(0, 1, len(unique_shifts)))
        for i, shift in enumerate(unique_shifts):
            mask = energy_shifts == shift
            norms_filtered = split_norms[mask]
            times_filtered = run_times[mask]
            # Filter out zero times
            valid_mask = times_filtered > 0
            if np.sum(valid_mask) > 0:
                sort_idx = np.argsort(norms_filtered[valid_mask])
                ax6.plot(
                    norms_filtered[valid_mask][sort_idx],
                    times_filtered[valid_mask][sort_idx],
                    "s-",
                    color=colors[i],
                    linewidth=2,
                    markersize=8,
                    label=f"EnergyShift={shift:.3f}",
                )
        ax6.set_xlabel("PAO.SplitNorm", fontsize=12, fontweight="bold")
        ax6.set_ylabel("Wall Time (s)", fontsize=12, fontweight="bold")
        ax6.set_title(
            "Computational Time vs PAO.SplitNorm\n(Higher SN → More split orbitals → Slower)",
            fontsize=13,
            fontweight="bold",
        )
        ax6.legend(loc="best", fontsize=10)
        ax6.grid(True, alpha=0.3)  # noqa: FBT003
        file6 = str(output_dir / f"{base_name}_timing_vs_norm.png")
        plt.savefig(file6, dpi=150, bbox_inches="tight")
        plt.close()
        individual_files["timing_vs_norm"] = file6

        # Individual Plot 7: 2D Timing Heatmap
        _fig7, ax7 = plt.subplots(figsize=(10, 8))
        time_grid = np.zeros_like(shift_grid)
        for i in range(len(run_times)):
            if run_times[i] > 0:
                shift_idx = np.where(unique_shifts == energy_shifts[i])[0][0]
                norm_idx = np.where(unique_norms == split_norms[i])[0][0]
                time_grid[norm_idx, shift_idx] = run_times[i]

        # Replace zeros with NaN for better visualization
        time_grid[time_grid == 0] = np.nan

        contour = ax7.contourf(
            shift_grid, norm_grid, time_grid, levels=15, cmap="YlOrRd"
        )
        cbar = plt.colorbar(contour, ax=ax7)
        cbar.set_label("Wall Time (s)", fontsize=11, fontweight="bold")

        # Add data points
        for i in range(len(run_times)):
            if run_times[i] > 0:
                ax7.plot(energy_shifts[i], split_norms[i], "ko", markersize=6)
                ax7.text(
                    energy_shifts[i],
                    split_norms[i],
                    f"{run_times[i]:.1f}",
                    fontsize=9,
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        ax7.set_xlabel("PAO.EnergyShift (Ry)", fontsize=12, fontweight="bold")
        ax7.set_ylabel("PAO.SplitNorm", fontsize=12, fontweight="bold")
        ax7.set_title(
            "Computational Time Landscape\n(Wall time in seconds)",
            fontsize=13,
            fontweight="bold",
        )
        ax7.grid(True, alpha=0.3)  # noqa: FBT003
        file7 = str(output_dir / f"{base_name}_timing_heatmap.png")
        plt.savefig(file7, dpi=150, bbox_inches="tight")
        plt.close()
        individual_files["timing_heatmap"] = file7

        # Individual Plot 8: Efficiency plot (Energy vs Time)
        _fig8, ax8 = plt.subplots(figsize=(10, 7))
        # Normalize energies to meV relative to minimum
        e_min = np.min(energies)
        energies_rel = (energies - e_min) * 1000  # meV

        # Only plot points with valid timing
        valid_mask = run_times > 0
        scatter = ax8.scatter(
            run_times[valid_mask],
            energies_rel[valid_mask],
            c=energy_shifts[valid_mask],
            s=100,
            cmap="viridis",
            edgecolors="black",
            linewidth=1.5,
            alpha=0.7,
        )
        cbar = plt.colorbar(scatter, ax=ax8)
        cbar.set_label("PAO.EnergyShift (Ry)", fontsize=11, fontweight="bold")

        # Annotate points with SplitNorm values
        for i in range(len(run_times)):
            if run_times[i] > 0:
                ax8.annotate(
                    f"{split_norms[i]:.2f}",
                    xy=(run_times[i], energies_rel[i]),
                    xytext=(5, 5),  # 5 points right and up from the point
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.7,
                )

        ax8.set_xlabel("Wall Time (s)", fontsize=12, fontweight="bold")
        ax8.set_ylabel("Energy above minimum (meV)", fontsize=12, fontweight="bold")
        ax8.set_title(
            "Computational Efficiency\n(Lower-left = fast & accurate)",
            fontsize=13,
            fontweight="bold",
        )
        ax8.grid(True, alpha=0.3)  # noqa: FBT003

        # Add "optimal zone" annotation
        ax8.axhline(
            y=5,
            color="green",
            linestyle="--",
            alpha=0.5,
            linewidth=2,
            label="ΔE < 5 meV (good)",
        )
        ax8.axhline(
            y=10,
            color="orange",
            linestyle="--",
            alpha=0.5,
            linewidth=2,
            label="ΔE < 10 meV (fair)",
        )
        ax8.legend(loc="upper right", fontsize=10)

        file8 = str(output_dir / f"{base_name}_efficiency.png")
        plt.savefig(file8, dpi=150, bbox_inches="tight")
        plt.close()
        individual_files["efficiency"] = file8

        logger.info(f"Timing analysis plots added ({4} files)")
        console.print("[green]Timing analysis plots saved (4 files)[/green]")

    logger.info(f"Individual plots saved: {list(individual_files.values())}")
    console.print(
        f"[green]Individual plots saved ({len(individual_files) - 1} files)[/green]"
    )

    return individual_files


@job
def plot_basis_functions(
    flow_results: dict[str, Any],
    job_metadata: list[dict],
    output_file: str = "basis_functions_visualization.png",
) -> dict[str, str]:
    """
    Plot actual PAO basis functions from ion.xml files (inspired by plot_siesta_basis.py).

    This creates a combined visualization and individual plots of the numerical atomic
    orbitals generated with different PAO.EnergyShift and PAO.SplitNorm parameters.

    Args:
        flow_results: Results dictionary from jobflow's run_locally
        job_metadata: List of dictionaries with job info
        output_file: Output filename for combined plot

    Returns
    -------
        Dictionary with paths to all saved plot files
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    console.print("[green]Creating basis function visualization[/green]")

    # Try to find ion.xml files from the jobs
    basis_functions_data = {}

    for job_info in job_metadata[:6]:  # Limit to first 6 for readability
        energy_shift = job_info["energy_shift"]
        split_norm = job_info["split_norm"]

        try:
            # Try to find job directory
            # This is a simplified approach - in practice, need to traverse job folders
            logger.debug(f"Looking for ion.xml files for job {job_info['name']}")

            # Store metadata for later if we find files
            label = f"ES={energy_shift:.3f}, SN={split_norm:.2f}"
            basis_functions_data[label] = {
                "energy_shift": energy_shift,
                "split_norm": split_norm,
                "found": False,
            }

        except Exception as e:
            logger.debug(f"Could not process job {job_info['name']}: {e}")
            continue

    # Create a  conceptual visualization showing basis quality
    _fig = plt.figure(figsize=(16, 10))

    # Plot 1: Schematic showing how EnergyShift affects orbital extent
    ax1 = plt.subplot(2, 2, 1)

    # Get unique EnergyShift values
    unique_shifts = sorted(
        set([d["energy_shift"] for d in basis_functions_data.values()])
    )

    # Create schematic orbitals
    r = np.linspace(0, 10, 200)  # radial coordinate in bohr
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_shifts)))

    for i, es in enumerate(unique_shifts):
        # Simplified PAO model: confined Gaussian-like function
        # r_c ∝ 1/sqrt(EnergyShift) (larger confinement → smaller radius)
        r_cutoff = 5.0 / np.sqrt(es * 10)  # Approximate cutoff radius
        pao = np.exp(-(r**2) / (2 * r_cutoff**2)) * np.exp(-es * r / r_cutoff)
        pao[r > r_cutoff] = 0  # Hard cutoff

        ax1.plot(
            r,
            pao,
            linewidth=2.5,
            color=colors[i],
            label=f"ES={es:.3f} Ry (r_c≈{r_cutoff:.1f} bohr)",
        )
        # Mark cutoff
        ax1.axvline(x=r_cutoff, color=colors[i], linestyle="--", alpha=0.3, linewidth=1)

    ax1.set_xlabel("r (bohr)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("φ(r) [arbitrary units]", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Effect of PAO.EnergyShift on Orbital Extent\n(Schematic)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)  # noqa: FBT003
    ax1.set_xlim(0, 10)

    # Plot 2: Schematic showing split-valence orbitals
    ax2 = plt.subplot(2, 2, 2)

    r = np.linspace(0, 8, 200)
    es_demo = 0.010  # Demo EnergyShift
    r_c = 5.0

    # Primary orbital (no splitting)
    phi_1 = np.exp(-(r**2) / (2 * r_c**2)) * np.exp(-es_demo * r / r_c)
    phi_1[r > r_c] = 0

    # Split-valence orbital (r * phi_1, renormalized)
    phi_2 = r * phi_1
    phi_2 = phi_2 / np.max(np.abs(phi_2)) * 0.7  # Normalize for display

    ax2.plot(r, phi_1, "b-", linewidth=2.5, label="Primary orbital φ₁(r)")
    ax2.plot(r, phi_2, "r--", linewidth=2.5, label="Split orbital φ₂(r) ∝ r·φ₁(r)")
    ax2.fill_between(r, 0, phi_1, alpha=0.2, color="blue")
    ax2.fill_between(r, 0, phi_2, alpha=0.2, color="red")

    ax2.set_xlabel("r (bohr)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("φ(r) [arbitrary units]", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Split-Valence Orbitals (PAO.SplitNorm effect)\n(Schematic)",
        fontsize=13,
        fontweight="bold",
    )
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)  # noqa: FBT003
    ax2.set_xlim(0, 8)

    # Plot 3: Basis quality indicator - orbital overlap
    ax3 = plt.subplot(2, 2, 3)

    # Show how different EnergyShift values affect overlap
    r = np.linspace(0, 15, 300)
    bond_length = 4.0  # Typical bond length in bohr

    for i, es in enumerate(unique_shifts):
        r_cutoff = 5.0 / np.sqrt(es * 10)

        # Two orbitals separated by bond_length
        phi_a = np.exp(-((r) ** 2) / (2 * r_cutoff**2)) * np.exp(-es * r / r_cutoff)
        phi_b = np.exp(-((r - bond_length) ** 2) / (2 * r_cutoff**2)) * np.exp(
            -es * (r - bond_length) / r_cutoff
        )

        phi_a[r > r_cutoff] = 0
        phi_b[(r - bond_length) > r_cutoff] = 0

        # Overlap
        overlap = phi_a * phi_b
        # np.trapz was renamed to np.trapezoid in numpy 2.0
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        overlap_integral = trapezoid(overlap, r)

        ax3.fill_between(
            r,
            0,
            overlap,
            alpha=0.4,
            color=colors[i],
            label=f"ES={es:.3f} (overlap integral ∝ {overlap_integral:.2f})",
        )

    ax3.set_xlabel("r (bohr)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Orbital Overlap", fontsize=12, fontweight="bold")
    ax3.set_title(
        f"Basis Quality: Orbital Overlap\n(atoms separated by {bond_length} bohr)",
        fontsize=13,
        fontweight="bold",
    )
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)  # noqa: FBT003
    ax3.set_xlim(0, 15)

    # Plot 4: Summary diagram
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("off")

    summary_text = []
    summary_text.append("PAO BASIS FUNCTION CHARACTERISTICS")
    summary_text.append("=" * 50)
    summary_text.append("")
    summary_text.append("PAO.EnergyShift Effect:")
    summary_text.append("  • Lower ES → More extended orbitals")
    summary_text.append("  • More extended → Better overlap")
    summary_text.append("  • Better overlap → Better bonding description")
    summary_text.append("  • Trade-off: Larger basis = slower calculation")
    summary_text.append("")
    summary_text.append("PAO.SplitNorm Effect:")
    summary_text.append("  • Higher SN → More split-valence orbitals")
    summary_text.append("  • Split orbitals = double-zeta, triple-zeta...")
    summary_text.append("  • Better polarization → Better charge transfer")
    summary_text.append("  • Essential for chemical bonding accuracy")
    summary_text.append("")
    summary_text.append("Tested Parameter Combinations:")

    # Add tested combinations
    for label, data in sorted(
        basis_functions_data.items(),
        key=lambda x: (x[1]["energy_shift"], x[1]["split_norm"]),
    ):
        summary_text.append(f"  • {label}")

    summary_text.append("")
    summary_text.append("Visualization Notes:")
    summary_text.append("  • Top-left: Orbital confinement vs EnergyShift")
    summary_text.append("  • Top-right: Split-valence orbital generation")
    summary_text.append("  • Bottom-left: Bonding overlap integral")
    summary_text.append("  • Lower ES → Extended tails → Better chemistry")

    ax4.text(
        0.05,
        0.95,
        "\n".join(summary_text),
        transform=ax4.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.suptitle(
        "SIESTA Numerical Atomic Orbitals (PAO) Basis Functions\n"
        "Schematic Visualization of Parameter Effects",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Combined basis function visualization saved to {output_file}")
    console.print(
        f"[green]Combined basis visualization saved to: {output_file}[/green]"
    )

    # Save individual schematic plots
    base_name = Path(output_file).stem
    output_dir = Path(output_file).parent

    individual_files = {}
    individual_files["combined"] = output_file

    # Individual Plot 1: Orbital Extent vs EnergyShift
    _fig1, ax1 = plt.subplots(figsize=(10, 7))
    unique_shifts = sorted(
        set([d["energy_shift"] for d in basis_functions_data.values()])
    )
    radial = np.linspace(0, 10, 200)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_shifts)))
    for i, es in enumerate(unique_shifts):
        r_cutoff = 5.0 / np.sqrt(es * 10)
        pao = np.exp(-(radial**2) / (2 * r_cutoff**2)) * np.exp(-es * radial / r_cutoff)
        pao[radial > r_cutoff] = 0
        ax1.plot(
            radial,
            pao,
            linewidth=2.5,
            color=colors[i],
            label=f"ES={es:.3f} Ry (r_c≈{r_cutoff:.1f} bohr)",
        )
        ax1.axvline(x=r_cutoff, color=colors[i], linestyle="--", alpha=0.3, linewidth=1)
    ax1.set_xlabel("r (bohr)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("φ(r) [arbitrary units]", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Effect of PAO.EnergyShift on Orbital Extent\n(Schematic)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)  # noqa: FBT003
    ax1.set_xlim(0, 10)
    file1 = str(output_dir / f"{base_name}_orbital_extent.png")
    plt.savefig(file1, dpi=150, bbox_inches="tight")
    plt.close()
    individual_files["orbital_extent"] = file1

    # Individual Plot 2: Split-valence orbitals
    _fig2, ax2 = plt.subplots(figsize=(10, 7))
    r = np.linspace(0, 8, 200)
    es_demo = 0.010
    unique_norms = sorted(
        set([d["split_norm"] for d in basis_functions_data.values()])
    )[:3]
    colors2 = plt.cm.plasma(np.linspace(0, 1, len(unique_norms)))
    for i, sn in enumerate(unique_norms):
        r_c1 = 4.0 / np.sqrt(es_demo * 10)
        r_c2 = r_c1 * 0.7
        pao_1 = np.exp(-(r**2) / (2 * r_c1**2))
        pao_1[r > r_c1] = 0
        pao_2 = np.exp(-(r**2) / (2 * r_c2**2)) * (1 - np.exp(-(r**2) / (2 * r_c1**2)))
        pao_2[r > r_c2] = 0
        pao_2 *= sn / 0.25
        ax2.plot(r, pao_1, linewidth=2.5, color=colors2[i], label=f"SN={sn:.2f}: 1st ζ")
        ax2.plot(
            r,
            pao_2,
            linewidth=2.5,
            color=colors2[i],
            linestyle="--",
            label=f"SN={sn:.2f}: 2nd ζ (split)",
            alpha=0.7,
        )
    ax2.set_xlabel("r (bohr)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("φ(r) [arbitrary units]", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Effect of PAO.SplitNorm on Split-Valence Orbitals\n(Schematic)",
        fontsize=13,
        fontweight="bold",
    )
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)  # noqa: FBT003
    ax2.set_xlim(0, 8)
    file2 = str(output_dir / f"{base_name}_split_valence.png")
    plt.savefig(file2, dpi=150, bbox_inches="tight")
    plt.close()
    individual_files["split_valence"] = file2

    # Individual Plot 3: Bonding overlap integral
    _fig3, ax3 = plt.subplots(figsize=(10, 7))
    radial_dist = np.linspace(0, 12, 200)
    bond_distance = 5.0
    colors3 = plt.cm.viridis(np.linspace(0, 1, len(unique_shifts)))
    for i, es in enumerate(unique_shifts):
        r_cutoff = 5.0 / np.sqrt(es * 10)
        pao_A = np.exp(-(radial_dist**2) / (2 * r_cutoff**2)) * np.exp(  # noqa: N806
            -es * radial_dist / r_cutoff
        )
        pao_A[radial_dist > r_cutoff] = 0
        pao_B = np.exp(  # noqa: N806
            -((radial_dist - bond_distance) ** 2) / (2 * r_cutoff**2)
        ) * np.exp(-es * np.abs(radial_dist - bond_distance) / r_cutoff)
        pao_B[np.abs(radial_dist - bond_distance) > r_cutoff] = 0
        overlap = pao_A * pao_B
        ax3.fill_between(
            radial_dist, overlap, alpha=0.4, color=colors3[i], label=f"ES={es:.3f} Ry"
        )
        ax3.plot(radial_dist, pao_A, linewidth=1.5, color=colors3[i], alpha=0.6)
        ax3.plot(radial_dist, pao_B, linewidth=1.5, color=colors3[i], alpha=0.6)
    ax3.axvline(x=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    ax3.axvline(x=bond_distance, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    ax3.set_xlabel("r (bohr)", fontsize=12, fontweight="bold")
    ax3.set_ylabel(
        "Orbital amplitude [arbitrary units]", fontsize=12, fontweight="bold"
    )
    ax3.set_title(
        "Bonding Overlap Integral (Shaded Area)\n(Schematic)",
        fontsize=13,
        fontweight="bold",
    )
    ax3.legend(loc="best", fontsize=10)
    ax3.grid(True, alpha=0.3)  # noqa: FBT003
    file3 = str(output_dir / f"{base_name}_bonding_overlap.png")
    plt.savefig(file3, dpi=150, bbox_inches="tight")
    plt.close()
    individual_files["bonding_overlap"] = file3

    logger.info(f"Individual schematic plots saved: {list(individual_files.values())}")
    console.print(
        f"[green]Individual schematic plots saved ({len(individual_files) - 1} files)[/green]"
    )

    return individual_files


@job
def plot_real_basis_functions(
    flow_results: dict[str, Any],
    job_metadata: list[dict],
    output_file: str = "basis_functions_real.png",
) -> dict[str, str]:
    """
    Plot actual PAO basis functions from SIESTA ion.xml files.

    Reads the ion.xml files generated by SIESTA and plots the actual
    numerical atomic orbitals for different PAO.EnergyShift and PAO.SplitNorm
    parameter combinations.

    Creates both a combined multi-panel plot and individual plots per angular momentum.
    Legends are placed below plots for better readability with many entries.

    Inspired by plot_siesta_basis.py but adapted for convergence workflow.

    Args:
        flow_results: Results dictionary from jobflow's run_locally
        job_metadata: List of dictionaries with job info
        output_file: Output filename for combined plot

    Returns
    -------
        Dictionary with paths to all saved plot files
    """
    import glob
    import xml.etree.ElementTree as ET
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    console.print("[green]Plotting real PAO basis functions from ion.xml files[/green]")

    # Structure to store basis function data per element
    # basis_data[element][label] = {'energy_shift': ..., 'split_norm': ..., 'orbitals': [...]}
    basis_data: dict[str, Any] = {}

    # Try to find and read ion.xml files from job directories
    logger.info("Searching for ion.xml files in job directories...")

    for job_info in job_metadata[:9]:  # Limit to first 9 for readability
        job_uuid = job_info["uuid"]
        job_info["name"]
        energy_shift = job_info["energy_shift"]
        split_norm = job_info["split_norm"]

        # Try to get the job directory from flow_results
        job_ion_files = []

        if job_uuid in flow_results:
            result = flow_results[job_uuid]

            # Try to get directory from result object
            job_dir = None
            if hasattr(result, "dir_name"):
                job_dir = result.dir_name
            elif hasattr(result, "output") and hasattr(result.output, "dir_name"):
                job_dir = result.output.dir_name

            # If we have a directory, look for ALL ion.xml files there
            if job_dir:
                import os

                possible_files = [
                    f"{job_dir}/*.ion.xml",  # Uncompressed in main dir
                    f"{job_dir}/*.ion.xml.gz",  # Compressed in main dir
                    f"{job_dir}/siesta_compressed/*.ion.xml.gz",  # Compressed in subfolder
                    f"{job_dir}/*/*.ion.xml",  # One level deep
                ]
                for pattern in possible_files:
                    files = glob.glob(pattern)
                    if files:
                        job_ion_files.extend(files)  # Get ALL ion files for this job
                        logger.info(
                            f"Found {len(files)} ion files with pattern {pattern}"
                        )
                        break

        # Fallback: search from parent directory if no direct match
        if not job_ion_files:
            import os

            cwd = os.getcwd()
            parent_dir = (
                os.path.dirname(cwd) if "job_" in os.path.basename(cwd) else cwd
            )

            possible_patterns = [
                f"{parent_dir}/job_*/*.ion.xml",  # Direct in SCF job dir
                f"{parent_dir}/job_*/*.ion.xml.gz",  # Compressed in SCF job dir
                f"{parent_dir}/job_*/siesta_compressed/*.ion.xml.gz",  # Compressed in subfolder
                f"{parent_dir}/job_*/*/*.ion.xml",  # One level deep
                "job_*/*.ion.xml",  # Search from current dir
                "job_*/*.ion.xml.gz",  # Compressed from current dir
                "job_*/siesta_compressed/*.ion.xml.gz",  # Compressed in subfolder from current dir
            ]

            for pattern in possible_patterns:
                found = glob.glob(pattern)
                if found:
                    logger.debug(f"Pattern {pattern} found {len(found)} files")
                    job_ion_files.extend(found)
                    break

        # Read ALL ion.xml files for this job (one per element)
        for job_ion_file in job_ion_files:
            try:
                logger.info(
                    f"Reading {job_ion_file} for ES={energy_shift}, SN={split_norm}"
                )

                # Handle gzipped files
                import gzip

                if job_ion_file.endswith(".gz"):
                    with gzip.open(job_ion_file, "rt") as f:
                        tree = ET.parse(f)
                else:
                    tree = ET.parse(job_ion_file)
                root = tree.getroot()

                # Get element symbol for THIS file
                symbol_elem = root.find("symbol")
                if symbol_elem is None:
                    logger.warning(f"No element symbol found in {job_ion_file}")
                    continue

                element_symbol = symbol_elem.text.strip()
                logger.info(f"Found element: {element_symbol}")

                # Initialize element dictionary if needed
                if element_symbol not in basis_data:
                    basis_data[element_symbol] = {}

                # Extract basis functions
                paos = root.find("paos")
                if paos is not None:
                    label = f"ES={energy_shift:.3f}, SN={split_norm:.2f}"
                    basis_data[element_symbol][label] = {
                        "energy_shift": energy_shift,
                        "split_norm": split_norm,
                        "orbitals": [],
                    }

                    for orbital in paos.findall("orbital"):
                        l = int(orbital.get("l").strip())  # noqa: E741  # Angular momentum quantum number
                        n = int(orbital.get("n").strip())
                        z = int(orbital.get("z").strip())

                        radfunc = orbital.find("radfunc")
                        if radfunc is not None:
                            int(radfunc.find("npts").text.strip())
                            float(radfunc.find("delta").text.strip())
                            cutoff = float(radfunc.find("cutoff").text.strip())

                            data_elem = radfunc.find("data")
                            if data_elem is not None:
                                lines = data_elem.text.strip().split("\n")
                                r_vals = []
                                phi_vals = []

                                for line in lines:
                                    parts = line.strip().split()
                                    if len(parts) >= 2:
                                        r_vals.append(float(parts[0]))
                                        phi_vals.append(float(parts[1]))

                                r = np.array(r_vals)
                                phi = np.array(phi_vals)

                                basis_data[element_symbol][label]["orbitals"].append(
                                    {
                                        "n": n,
                                        "l": l,
                                        "z": z,
                                        "r": r,
                                        "phi": phi,
                                        "cutoff": cutoff,
                                    }
                                )

                                logger.debug(
                                    f"  {element_symbol}: Read orbital {n}{['s', 'p', 'd', 'f', 'g'][l]}ζ{z}: "
                                    f"{len(r)} points, r_c={cutoff:.3f} bohr"
                                )

            except Exception as e:
                logger.warning(f"Could not read {job_ion_file}: {e}")
                continue

    # If no ion files found, create informative plot
    if not basis_data:
        logger.warning("No ion.xml files found. Creating informative message plot.")
        _fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.axis("off")

        message = [
            "REAL BASIS FUNCTION PLOT",
            "=" * 60,
            "",
            "⚠ No ion.xml files found in job directories.",
            "",
            "This plot shows actual PAO basis functions generated by SIESTA.",
            "To see real basis functions:",
            "",
            "1. Ensure SIESTA calculations completed successfully",
            "2. Check that ion.xml files were generated",
            "3. Typical locations:",
            "   - job_*/output/*.ion.xml",
            "   - job_*/siesta_calc/*/*.ion.xml",
            "",
            "For now, refer to the schematic visualization:",
            "   → basis_functions_visualization.png",
            "",
            "The schematic plot shows the conceptual behavior of",
            "PAO basis functions with different parameters.",
        ]

        ax.text(
            0.5,
            0.5,
            "\n".join(message),
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Info plot saved to {output_file}")
        console.print(
            f"[yellow]Info plot saved to: {output_file} (no ion.xml files found)[/yellow]"
        )
        return {"info": output_file}

    # Create plots for ALL elements
    logger.info(f"Creating plots for {len(basis_data)} elements")

    # Organize by angular momentum
    l_labels = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}

    # Get list of all elements
    all_elements = sorted(basis_data.keys())

    # Find all unique l values across ALL elements
    all_l_values_set = set()
    for element in all_elements:
        for label_data in basis_data[element].values():
            for orb in label_data["orbitals"]:
                all_l_values_set.add(orb["l"])

    all_l_values = sorted(list(all_l_values_set))

    # Create combined plot with subplots for each element × l combination  # noqa: RUF003
    # Calculate grid size
    n_plots = len(all_elements) * len(all_l_values)
    n_cols = min(3, max(len(all_l_values), 2))
    n_rows = (n_plots + n_cols - 1) // n_cols

    plt.figure(figsize=(6 * n_cols, 5 * n_rows))

    plot_idx = 1
    for element in all_elements:
        element_data = basis_data[element]
        colors = plt.cm.tab10(np.linspace(0, 1, len(element_data)))

        for l in all_l_values:  # noqa: E741  # Angular momentum quantum number
            # Check if this element has orbitals with this l
            has_l = any(
                any(orb["l"] == l for orb in data["orbitals"])
                for data in element_data.values()
            )

            if not has_l:
                continue

            ax = plt.subplot(n_rows, n_cols, plot_idx)
            plot_idx += 1

            color_idx = 0
            for label, data in sorted(
                element_data.items(),
                key=lambda x: (x[1]["energy_shift"], x[1]["split_norm"]),
            ):
                es = data["energy_shift"]
                sn = data["split_norm"]

                # Plot all orbitals with this l
                for orb in data["orbitals"]:
                    if orb["l"] == l:
                        n = orb["n"]
                        z = orb["z"]
                        r = orb["r"]
                        phi = orb["phi"]
                        cutoff = orb["cutoff"]

                        # Create label with cutoff radius
                        orb_label = f"{n}{l_labels.get(l, f'l={l}')}ζ{z} (ES={es:.3f}, SN={sn:.2f}, r_c={cutoff:.2f})"

                        ax.plot(
                            r,
                            phi,
                            linewidth=2,
                            color=colors[color_idx],
                            label=orb_label,
                        )
                        # Mark cutoff
                        ax.axvline(
                            x=cutoff,
                            color=colors[color_idx],
                            linestyle="--",
                            alpha=0.3,
                            linewidth=1,
                        )

                color_idx += 1

            ax.set_xlabel("r (bohr)", fontsize=11, fontweight="bold")
            ax.set_ylabel("φ(r)", fontsize=11, fontweight="bold")
            ax.set_title(
                f"{element} - {l_labels.get(l, f'l={l}')}-type",
                fontsize=12,
                fontweight="bold",
            )

            # Place legend inside the plot (upper right to avoid data overlap)
            ax.legend(
                fontsize=6,
                loc="upper right",
                ncol=1,
                frameon=True,
                fancybox=True,
                framealpha=0.9,
            )
            ax.grid(True, alpha=0.3)  # noqa: FBT003
            ax.set_xlim(left=0)

    # Overall title - include basis size if available from metadata
    basis_size = None
    if job_metadata and "basis_size" in job_metadata[0]:
        basis_sizes_in_meta = set(
            m.get("basis_size") for m in job_metadata if m.get("basis_size")
        )
        if len(basis_sizes_in_meta) == 1:
            basis_size = list(basis_sizes_in_meta)[0]

    title = f"SIESTA PAO Basis Functions: {', '.join(all_elements)}"
    if basis_size:
        title += f" ({basis_size} Basis)"
    title += "\nReal basis functions from ion.xml files"
    plt.suptitle(title, fontsize=14, fontweight="bold")

    # Standard tight layout - legends are now inside plots
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Combined real basis function plot saved to {output_file}")
    console.print(f"[green]Combined real basis plot saved to: {output_file}[/green]")

    # Save individual plots for each element × l combination  # noqa: RUF003
    base_name = Path(output_file).stem
    output_dir = Path(output_file).parent

    individual_files = {}
    individual_files["combined"] = output_file

    # Create individual plots for each element and l value
    for element in all_elements:
        element_data = basis_data[element]
        colors = plt.cm.tab10(np.linspace(0, 1, len(element_data)))

        for l in all_l_values:  # noqa: E741  # Angular momentum quantum number
            # Check if this element has orbitals with this l
            has_l = any(
                any(orb["l"] == l for orb in data["orbitals"])
                for data in element_data.values()
            )

            if not has_l:
                continue

            _fig_ind, ax_ind = plt.subplots(figsize=(10, 8))

            color_idx = 0
            legend_entries = []

            for label, data in sorted(
                element_data.items(),
                key=lambda x: (x[1]["energy_shift"], x[1]["split_norm"]),
            ):
                es = data["energy_shift"]
                sn = data["split_norm"]

                # Plot all orbitals with this l
                for orb in data["orbitals"]:
                    if orb["l"] == l:
                        n = orb["n"]
                        z = orb["z"]
                        r = orb["r"]
                        phi = orb["phi"]
                        cutoff = orb["cutoff"]

                        # Create label with cutoff radius
                        orb_label = f"{n}{l_labels.get(l, f'l={l}')}ζ{z} (ES={es:.3f}, SN={sn:.2f}, r_c={cutoff:.2f})"

                        (line,) = ax_ind.plot(
                            r,
                            phi,
                            linewidth=2.5,
                            color=colors[color_idx],
                            label=orb_label,
                        )
                        legend_entries.append(line)

                        # Mark cutoff
                        ax_ind.axvline(
                            x=cutoff,
                            color=colors[color_idx],
                            linestyle="--",
                            alpha=0.3,
                            linewidth=1,
                        )

                color_idx += 1

            ax_ind.set_xlabel("r (bohr)", fontsize=13, fontweight="bold")
            ax_ind.set_ylabel("φ(r)", fontsize=13, fontweight="bold")
            ax_ind.set_title(
                f"{element} - {l_labels.get(l, f'l={l}')}-type Orbitals\n"
                "PAO Basis Functions from ion.xml",
                fontsize=14,
                fontweight="bold",
            )

            # Place legend inside the plot (upper right)
            ax_ind.legend(
                handles=legend_entries,
                fontsize=8,
                loc="upper right",
                ncol=1,
                frameon=True,
                fancybox=True,
                framealpha=0.9,
            )

            ax_ind.grid(True, alpha=0.3)  # noqa: FBT003
            ax_ind.set_xlim(left=0)

            # Save individual plot with element name
            l_name = l_labels.get(l, f"l{l}")
            file_ind = str(output_dir / f"{base_name}_{element}_{l_name}_orbitals.png")
            plt.tight_layout()
            plt.savefig(file_ind, dpi=150, bbox_inches="tight")
            plt.close()

            individual_files[f"{element}_{l_name}_orbitals"] = file_ind

    logger.info(f"Individual real basis plots saved: {list(individual_files.values())}")
    console.print(
        f"[green]Individual real basis plots saved ({len(individual_files) - 1} files)[/green]"
    )

    return individual_files


@job
def write_basis_params_summary(
    basis_params_data: dict[str, Any], output_file: str = "basis_params_summary.txt"
) -> str:
    """
    Write summary of basis parameter convergence results.

    Args:
        basis_params_data: Dictionary with energies and parameters
        output_file: Output filename for summary

    Returns
    -------
        Path to saved summary file
    """
    import numpy as np

    # Check if we have data
    if not basis_params_data["energies"]:
        error_msg = "No basis parameter data available for summary"
        logger.error(error_msg)
        with open(output_file, "w") as f:
            f.write("=" * 90 + "\n")
            f.write("BASIS PARAMETERS CONVERGENCE STUDY\n")
            f.write("PAO.EnergyShift & PAO.SplitNorm\n")
            f.write("=" * 90 + "\n\n")
            f.write("ERROR: No data collected. Check calculation logs.\n")
            f.write("=" * 90 + "\n")
        return output_file

    energy_shifts = np.array(basis_params_data["energy_shifts"])
    split_norms = np.array(basis_params_data["split_norms"])
    energies = np.array(basis_params_data["energies"])
    max_forces = np.array(basis_params_data["max_forces"])
    _names = basis_params_data["names"]

    with open(output_file, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("BASIS PARAMETERS CONVERGENCE STUDY\n")
        f.write("PAO.EnergyShift & PAO.SplitNorm Optimization\n")
        f.write("=" * 90 + "\n\n")

        logger.info(f"Writing summary for {len(energies)} parameter combinations")

        # Calculate relative energies
        e_min = np.min(energies)
        e_relative = (energies - e_min) * 1000  # meV

        # Main results table
        f.write("CONVERGENCE RESULTS:\n")
        f.write("-" * 120 + "\n")
        f.write(
            f"{'EnergyShift':<14} {'SplitNorm':<12} {'Energy (eV)':<16} "
            f"{'ΔE (meV)':<12} {'Max Force':<12} {'Max Stress':<12} {'Time (s)':<12}\n"
        )
        f.write(
            f"{'(Ry)':<14} {'':<12} {'':<16} {'vs min':<12} {'(eV/Å)':<12} {'(GPa)':<12} {'wall':<12}\n"
        )
        f.write("-" * 120 + "\n")

        # Get stress and timing data
        max_stresses = np.array(
            basis_params_data.get("max_stresses", [0.0] * len(energies))
        )
        run_times = np.array(basis_params_data.get("run_times", [0.0] * len(energies)))

        # Sort by energy
        sort_idx = np.argsort(energies)

        for idx in sort_idx:
            time_str = f"{run_times[idx]:.1f}" if run_times[idx] > 0 else "N/A"
            f.write(
                f"{energy_shifts[idx]:<14.6f} {split_norms[idx]:<12.4f} "
                f"{energies[idx]:<16.8f} {e_relative[idx]:<12.4f} "
                f"{max_forces[idx]:<12.6f} {max_stresses[idx]:<12.6f} {time_str:<12}\n"
            )

        # Convergence analysis
        f.write("\n" + "=" * 90 + "\n")
        f.write("CONVERGENCE ANALYSIS\n")
        f.write("=" * 90 + "\n\n")

        # Find optimal parameters
        optimal_idx = np.argmin(energies)
        f.write("Lowest Energy Configuration:\n")
        f.write(f"  PAO.EnergyShift = {energy_shifts[optimal_idx]:.6f} Ry\n")
        f.write(f"  PAO.SplitNorm   = {split_norms[optimal_idx]:.4f}\n")
        f.write(f"  Energy          = {energies[optimal_idx]:.8f} eV\n")
        f.write(f"  Max Force       = {max_forces[optimal_idx]:.6f} eV/Å\n")
        f.write(f"  Max Stress      = {max_stresses[optimal_idx]:.6f} GPa\n\n")

        # Energy range analysis
        e_range = np.max(energies) - np.min(energies)
        f.write("Energy Spread:\n")
        f.write(f"  Total range: {e_range * 1000:.4f} meV ({e_range:.6f} eV)\n")
        f.write(f"  Maximum ΔE:  {np.max(e_relative):.4f} meV\n\n")

        # Force and stress analysis
        f.write("Force and Stress Analysis:\n")
        f.write(
            f"  Max force range:  {np.min(max_forces):.6f} - {np.max(max_forces):.6f} eV/Å\n"
        )
        f.write(
            f"  Max stress range: {np.min(max_stresses):.6f} - {np.max(max_stresses):.6f} GPa\n"
        )
        f.write(f"  Average max force:  {np.mean(max_forces):.6f} eV/Å\n")
        f.write(f"  Average max stress: {np.mean(max_stresses):.6f} GPa\n\n")

        # Get unique parameter values (needed for analysis below)
        unique_shifts = np.unique(energy_shifts)
        unique_norms = np.unique(split_norms)

        # Timing analysis
        valid_times = run_times[run_times > 0]
        if len(valid_times) > 0:
            f.write("Computational Performance:\n")
            f.write(
                f"  Total wall time:   {np.sum(valid_times):.1f} s ({np.sum(valid_times) / 60:.1f} min)\n"
            )
            f.write(f"  Average time:      {np.mean(valid_times):.1f} s\n")
            f.write(f"  Fastest calc:      {np.min(valid_times):.1f} s\n")
            f.write(f"  Slowest calc:      {np.max(valid_times):.1f} s\n")
            f.write(f"  Std deviation:     {np.std(valid_times):.1f} s\n")

            # Timing vs basis size correlation
            time_variation = (
                (np.max(valid_times) - np.min(valid_times)) / np.mean(valid_times) * 100
            )
            f.write(f"  Timing variation:  {time_variation:.1f}%\n")

            # Timing efficiency: lower EnergyShift means larger basis, usually slower
            if len(unique_shifts) > 1 and len(valid_times) == len(run_times):
                f.write("\n  Timing vs EnergyShift (larger basis → slower):\n")
                for shift in sorted(unique_shifts):
                    mask = energy_shifts == shift
                    shift_times = run_times[mask]
                    valid_shift_times = shift_times[shift_times > 0]
                    if len(valid_shift_times) > 0:
                        f.write(
                            f"    ES={shift:.6f} Ry: avg={np.mean(valid_shift_times):.1f} s "
                            f"(range: {np.min(valid_shift_times):.1f}-{np.max(valid_shift_times):.1f} s)\n"
                        )
            f.write("\n")
        else:
            f.write("Computational Performance:\n")
            f.write("  Timing data not available\n\n")

        # Analyze trends with EnergyShift
        if len(unique_shifts) > 1:
            f.write("EnergyShift Sensitivity:\n")
            avg_energies_by_shift = []
            for shift in unique_shifts:
                mask = energy_shifts == shift
                avg_e = np.mean(energies[mask])
                std_e = np.std(energies[mask])
                avg_energies_by_shift.append(avg_e)
                f.write(
                    f"  ES={shift:.6f}: E_avg={avg_e:.6f} eV (σ={std_e * 1000:.2f} meV)\n"  # noqa: RUF001
                )

            shift_range = (
                np.max(avg_energies_by_shift) - np.min(avg_energies_by_shift)
            ) * 1000
            f.write(f"  Energy range across EnergyShifts: {shift_range:.4f} meV\n\n")

        # Analyze trends with SplitNorm
        if len(unique_norms) > 1:
            f.write("SplitNorm Sensitivity:\n")
            avg_energies_by_norm = []
            for norm in unique_norms:
                mask = split_norms == norm
                avg_e = np.mean(energies[mask])
                std_e = np.std(energies[mask])
                avg_energies_by_norm.append(avg_e)
                f.write(
                    f"  SN={norm:.4f}: E_avg={avg_e:.6f} eV (σ={std_e * 1000:.2f} meV)\n"  # noqa: RUF001
                )

            norm_range = (
                np.max(avg_energies_by_norm) - np.min(avg_energies_by_norm)
            ) * 1000
            f.write(f"  Energy range across SplitNorms: {norm_range:.4f} meV\n\n")

        # Recommendations
        f.write("=" * 90 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 90 + "\n\n")

        f.write("Standard SIESTA Basis Parameter Values:\n")
        f.write("  SZ/DZ basis:  EnergyShift = 0.010-0.020 Ry, SplitNorm = 0.15-0.20\n")
        f.write("  TZ basis:     EnergyShift = 0.005-0.010 Ry, SplitNorm = 0.20-0.25\n")
        f.write(
            "  High accuracy: EnergyShift = 0.001-0.005 Ry, SplitNorm = 0.25-0.30\n\n"
        )

        f.write("Parameter Effects:\n")
        f.write("  PAO.EnergyShift:\n")
        f.write("    - Controls basis confinement energy\n")
        f.write("    - Lower values → more extended orbitals → larger basis\n")
        f.write("    - Typical range: 0.001-0.050 Ry\n\n")

        f.write("  PAO.SplitNorm:\n")
        f.write("    - Controls split-valence orbital generation\n")
        f.write("    - Higher values → more split orbitals → better polarization\n")
        f.write("    - Typical range: 0.10-0.30\n\n")

        # Convergence criteria
        f.write("Convergence Criteria:\n")
        if e_range * 1000 < 1.0:
            f.write("  ✓ EXCELLENT: Energy variation < 1 meV - well converged!\n")
        elif e_range * 1000 < 5.0:
            f.write("  ✓ GOOD: Energy variation < 5 meV - adequate convergence\n")
        elif e_range * 1000 < 10.0:
            f.write(
                "  ⚠ FAIR: Energy variation 5-10 meV - consider finer parameter scan\n"
            )
        else:
            f.write("  ✗ POOR: Energy variation > 10 meV - parameters not converged!\n")
            f.write("     → Expand parameter range or use finer sampling\n")

        f.write("\n" + "=" * 90 + "\n")
        f.write("UNDERSTANDING REAL BASIS FUNCTIONS (from ion.xml files)\n")
        f.write("=" * 90 + "\n\n")

        f.write(
            "The workflow generates plots of actual PAO basis functions extracted from SIESTA\n"
        )
        f.write(
            "ion.xml output files. These show the real radial functions φ(r) used in the\n"
        )
        f.write("calculation, not schematic representations.\n\n")

        f.write("Plot Legend Information:\n")
        f.write("  Each orbital is labeled: nℓζz (ES=X.XXX, SN=X.XX, r_c=X.XX)\n")  # noqa: RUF001
        f.write("  where:\n")
        f.write("    n     = Principal quantum number (shell)\n")
        f.write("    ℓ     = Angular momentum (s=0, p=1, d=2, f=3, g=4)\n")  # noqa: RUF001
        f.write("    ζ     = Zeta (1=first radial function, 2=split-valence, etc.)\n")
        f.write("    z     = Zeta index from SIESTA\n")
        f.write("    ES    = PAO.EnergyShift value (Ry)\n")
        f.write("    SN    = PAO.SplitNorm value\n")
        f.write("    r_c   = Cutoff radius (bohr) - where φ(r) → 0\n\n")

        f.write("Physical Interpretation:\n\n")

        f.write("1. Cutoff Radius (r_c):\n")
        f.write("   • Defines strict localization: φ(r>r_c) = 0 exactly\n")
        f.write("   • Lower EnergyShift → larger r_c (more extended orbitals)\n")
        f.write("   • Relationship: r_c ∝ 1/√(EnergyShift)\n")
        f.write("   • Extended orbitals improve overlap → better bonding description\n")
        f.write("   • But larger r_c → more neighbors → slower calculations\n\n")

        f.write("2. Orbital Shape φ(r):\n")
        f.write(
            "   • First zeta (ζ1): Main radial function, similar to atomic orbital\n"
        )
        f.write(
            "   • Second zeta (ζ2): Split-valence, generated by multiplying ζ1 by r\n"
        )
        f.write("   • Higher zetas provide flexibility for bonding/antibonding\n")
        f.write("   • Normalization: ∫|φ(r)|² r² dr = 1 within cutoff sphere\n\n")

        f.write("3. Multiple Radial Functions (Split-Valence):\n")
        f.write("   • Generated when SplitNorm threshold is exceeded\n")
        f.write("   • Allows different orbital sizes for same ℓ channel\n")  # noqa: RUF001
        f.write("   • Essential for polarization in chemical bonding\n")
        f.write("   • ζ2 orbitals typically have smaller r_c than ζ1\n\n")

        f.write("4. Angular Momentum Channels:\n")
        f.write("   • s-orbitals: Spherically symmetric, single radial function\n")
        f.write("   • p-orbitals: Directional (px, py, pz), same radial part\n")
        f.write(
            "   • d-orbitals: Five degenerate orbitals, important for polarization\n"
        )
        f.write(
            "   • Higher ℓ channels added based on basis type (SZ, DZ, DZP, TZ, etc.)\n\n"  # noqa: RUF001
        )

        f.write("Material-Specific Considerations:\n\n")

        f.write("• Anions (O²⁻, Cl⁻, etc.):\n")
        f.write("  - Require very extended basis (low EnergyShift: 0.001-0.010 Ry)\n")
        f.write("  - High polarizability → need high SplitNorm (0.20-0.30)\n")
        f.write("  - Observe: ζ2 p-orbitals particularly important\n\n")

        f.write("• Cations (Mg²⁺, Na⁺, etc.):\n")
        f.write("  - More compact → moderate EnergyShift (0.010-0.020 Ry)\n")
        f.write("  - Lower SplitNorm sufficient (0.15-0.20)\n")
        f.write("  - Smaller r_c values expected\n\n")

        f.write("• Transition Metals:\n")
        f.write("  - d-orbitals crucial → need good split-valence\n")
        f.write("  - May require polarization orbitals (f for d-elements)\n")
        f.write("  - Check d-orbital r_c values carefully\n\n")

        f.write("Troubleshooting:\n\n")

        f.write("• If no basis function plots generated:\n")
        f.write("  - Check that SIESTA calculations completed successfully\n")
        f.write("  - Verify ion.xml files exist in job directories\n")
        f.write("  - Check SIESTA output for basis generation errors\n\n")

        f.write("• If orbitals seem too localized (small r_c):\n")
        f.write("  - Lower PAO.EnergyShift (try 0.001, 0.003, 0.005 Ry)\n")
        f.write("  - This is especially important for anions and polarizable atoms\n\n")

        f.write("• If insufficient split-valence orbitals:\n")
        f.write("  - Increase PAO.SplitNorm (try 0.25, 0.28, 0.30)\n")
        f.write("  - Check that PAO.BasisType = 'split' is set\n\n")

        f.write("• If basis too large/slow:\n")
        f.write("  - Increase PAO.EnergyShift to reduce r_c\n")
        f.write("  - Balance: accuracy vs computational cost\n")
        f.write("  - Monitor total number of orbitals in SIESTA output\n\n")

        f.write("=" * 90 + "\n")
        f.write("THEORETICAL BACKGROUND AND REFERENCES\n")
        f.write("=" * 90 + "\n\n")

        f.write("PAO Generation Method:\n\n")

        f.write("SIESTA generates Numerical Atomic Orbitals by solving:\n")
        f.write("  [H_atom + V_conf(r)] φ_nℓζ(r) = ε_nℓζ φ_nℓζ(r)\n\n")  # noqa: RUF001

        f.write("Where:\n")
        f.write("  • H_atom: Atomic Hamiltonian for isolated atom\n")
        f.write("  • V_conf(r): Confinement potential at cutoff radius r_c\n")
        f.write("  • ε_nℓζ: Pseudo-eigenvalue (shifted by EnergyShift)\n\n")  # noqa: RUF001

        f.write("The cutoff radius r_c is determined by:\n")
        f.write("  V_conf(r_c) = EnergyShift\n\n")

        f.write("Split-valence orbitals generated via:\n")
        f.write("  φ_nℓζ2(r) ∝ r × φ_nℓζ1(r)  (normalized and confined)\n\n")  # noqa: RUF001

        f.write("Key Publications:\n\n")

        f.write("1. Soler et al., J. Phys.: Condens. Matter 14, 2745 (2002)\n")
        f.write("   'The SIESTA method for ab initio order-N materials simulation'\n")
        f.write("   DOI: 10.1088/0953-8984/14/11/302\n")
        f.write("   • Foundational SIESTA paper\n")
        f.write("   • Describes PAO generation algorithm\n")
        f.write("   • Linear-scaling methodology\n\n")

        f.write("2. Artacho et al., Phys. Status Solidi B 215, 809 (1999)\n")
        f.write("   'The SIESTA method; developments and applicability'\n")
        f.write(
            "   DOI: 10.1002/(SICI)1521-3951(199909)215:1<809::AID-PSSB809>3.0.CO;2-0\n"
        )
        f.write("   • Early SIESTA development\n")
        f.write("   • Numerical atomic orbitals discussion\n\n")

        f.write("3. Junquera et al., Phys. Rev. B 64, 235111 (2001)\n")
        f.write("   'Numerical atomic orbitals for linear-scaling calculations'\n")
        f.write("   DOI: 10.1103/PhysRevB.64.235111\n")
        f.write("   • Detailed PAO optimization strategies\n")
        f.write("   • Basis set convergence studies\n")
        f.write("   • Energy shift and confinement effects\n\n")

        f.write("4. Anglada et al., Phys. Rev. B 66, 205101 (2002)\n")
        f.write(
            "   'Systematic generation of finite-range atomic basis sets for linear-scaling'\n"
        )
        f.write("   DOI: 10.1103/PhysRevB.66.205101\n")
        f.write("   • Automatic basis set generation\n")
        f.write("   • PAO.SplitNorm methodology\n")
        f.write("   • Optimization procedures\n\n")

        f.write("5. García-Gil et al., Phys. Rev. B 79, 075441 (2009)\n")
        f.write("   'Optimal strictly localized basis sets for noble metal surfaces'\n")
        f.write("   DOI: 10.1103/PhysRevB.79.075441\n")
        f.write("   • Basis optimization for metals\n")
        f.write("   • Extended state description with localized orbitals\n\n")

        f.write("6. Cuadrado et al., J. Phys.: Condens. Matter 24, 086005 (2012)\n")
        f.write(
            "   'Approach to an optimal basis set for density-functional calculations'\n"
        )
        f.write("   DOI: 10.1088/0953-8984/24/8/086005\n")
        f.write("   • Systematic basis optimization\n")
        f.write("   • Energy vs basis size trade-offs\n\n")

        f.write("Additional Resources:\n\n")

        f.write("• SIESTA Documentation: https://docs.siesta-project.org/\n")
        f.write("  - PAO.BasisType, PAO.EnergyShift, PAO.SplitNorm keywords\n")
        f.write("  - Basis set generation guide\n\n")

        f.write("• Basis Set Recommendations:\n")
        f.write("  - Anglada et al. PRB 66, 205101 (2002) - Table II\n")
        f.write("  - García-Gil et al. PRB 79, 075441 (2009) - Table I\n")
        f.write("  - SIESTA tutorials: siesta-project.org/tutorials\n\n")

        # Add standard footer
        from atomate2.siesta.utils.text_output import get_standard_footer

        f.write(
            get_standard_footer(
                width=90,
                additional_info={
                    "Analysis type": "Basis parameters convergence",
                    "Energy shifts tested": str(len(unique_shifts)),
                    "Split norms tested": str(len(unique_norms)),
                },
            )
        )

    logger.info(f"Summary written to {output_file}")
    console.print(f"[green]Summary written to: {output_file}[/green]")

    return output_file


# =============================================================================
# EOS Basis Convergence Analysis Functions
# =============================================================================


@job
def collect_eos_basis_data(
    flow_results: dict[str, Any], job_metadata: list[dict]
) -> dict[str, Any]:
    """
    Collect EOS data from multiple basis sets.

    Args:
        flow_results: Results dictionary from jobflow's run_locally
        job_metadata: List of dictionaries containing job names, UUIDs, and basis info

    Returns
    -------
        Dictionary with basis sets and their EOS parameters (V0, E0, B0)
    """
    logger.info("Collecting EOS basis convergence data")
    logger.info(f"Job metadata entries: {len(job_metadata)}")
    logger.info(f"Flow results type: {type(flow_results)}")
    logger.info(
        f"Flow results keys: {list(flow_results.keys()) if isinstance(flow_results, dict) else 'Not a dict'}"
    )

    basis_data: dict[str, Any] = {
        "basis_sets": [],
        "V0": [],  # Equilibrium volume
        "E0": [],  # Equilibrium energy
        "B0": [],  # Bulk modulus
        "lattice_a": [],  # Lattice parameter a (from V₀^(1/3) assuming cubic)
        "lattice_b": [],  # Lattice parameter b (assuming cubic: b=a)
        "lattice_c": [],  # Lattice parameter c (assuming cubic: c=a)
        "lattice_alpha": [],  # Angle α (assuming 90° for cubic)  # noqa: RUF003
        "lattice_beta": [],  # Angle β (assuming 90° for cubic)
        "lattice_gamma": [],  # Angle γ (assuming 90° for cubic)  # noqa: RUF003
        "run_time": [],  # Calculation time in seconds
        "names": [],
    }

    for job_info in job_metadata:
        job_uuid = job_info["uuid"]
        job_name = job_info["name"]
        basis_set = job_info.get("basis_set", "Unknown")

        logger.info(f"Processing {job_name} (basis: {basis_set}, UUID: {job_uuid})")

        try:
            if job_uuid not in flow_results:
                logger.warning(
                    f"No results found for job {job_name} (UUID: {job_uuid})"
                )
                logger.warning(f"Available UUIDs: {list(flow_results.keys())[:5]}...")
                continue

            result = flow_results[job_uuid]
            logger.info(f"Result type: {type(result)}")

            # EOS flow returns dict with structure: result["relax"]["EOS"]["murnaghan"]
            # where "murnaghan" (or other model) contains v0, e0, b0_GPa
            if isinstance(result, dict) and "relax" in result:
                # DEBUG: Check what keys are in result["relax"]
                logger.info(f"result['relax'] keys: {list(result['relax'].keys())}")
                if "EOS" in result["relax"]:
                    # Use murnaghan EOS by default (most common)
                    eos_models = result["relax"]["EOS"]
                    if "murnaghan" in eos_models:
                        eos_fit = eos_models["murnaghan"]
                        v0 = eos_fit["v0"]

                        # Extract equilibrium lattice parameters by scaling reference structure to V₀
                        # The EOS applies isotropic strain, so lattice parameters scale as V^(1/3)
                        structures = result["relax"].get("structure", [])
                        volumes = result["relax"].get("volume", [])
                        a, b, c = None, None, None
                        alpha, beta, gamma = 90.0, 90.0, 90.0

                        if structures and volumes:
                            import numpy as np

                            # Find the structure closest to v0 as reference
                            vol_array = np.array(volumes)
                            closest_idx = np.argmin(np.abs(vol_array - v0))
                            ref_structure = structures[closest_idx]
                            v_ref = volumes[closest_idx]

                            try:
                                ref_lattice = ref_structure.lattice
                                # Scale factor: a₀/a_ref = (V₀/V_ref)^(1/3)
                                scale = (v0 / v_ref) ** (1.0 / 3.0)

                                # Scale lattice parameters to equilibrium volume
                                a = ref_lattice.a * scale
                                b = ref_lattice.b * scale
                                c = ref_lattice.c * scale
                                # Angles are preserved under isotropic scaling
                                alpha, beta, gamma = (
                                    ref_lattice.alpha,
                                    ref_lattice.beta,
                                    ref_lattice.gamma,
                                )

                                logger.info(
                                    f"Scaled lattice from V_ref={v_ref:.3f} Å³ to V₀={v0:.3f} Å³ "
                                    f"(scale={scale:.6f}): a={a:.4f}, b={b:.4f}, c={c:.4f} Å"
                                )
                            except AttributeError:
                                logger.warning(
                                    f"Could not extract lattice from structure for {basis_set}, using cubic approximation"
                                )
                                a = v0 ** (1.0 / 3.0)
                                b, c = a, a
                                alpha, beta, gamma = 90.0, 90.0, 90.0
                        else:
                            # Fallback: calculate assuming cubic if no structures available
                            logger.warning(
                                f"No structures available for {basis_set}, using cubic approximation"
                            )
                            a = v0 ** (1.0 / 3.0)
                            b, c = a, a
                            alpha, beta, gamma = 90.0, 90.0, 90.0

                        # Extract timing data from relax results
                        run_time = None
                        if "run_time" in result["relax"]:
                            timing_list = result["relax"]["run_time"]
                            logger.info(
                                f"Found run_time in result['relax']: {timing_list}"
                            )
                            # Sum all run times for this EOS workflow
                            valid_times = [
                                t
                                for t in timing_list
                                if t is not None and isinstance(t, (int, float))
                            ]
                            if valid_times:
                                run_time = sum(valid_times)
                                logger.info(f"Calculated total run_time: {run_time}s")
                            else:
                                logger.warning(
                                    f"No valid times in timing_list for {basis_set}"
                                )
                        else:
                            logger.warning(
                                f"No run_time key in result['relax'] for {basis_set}"
                            )

                        basis_data["basis_sets"].append(basis_set)
                        basis_data["V0"].append(v0)  # Equilibrium volume
                        basis_data["E0"].append(eos_fit["e0"])  # Equilibrium energy
                        basis_data["B0"].append(eos_fit["b0 GPa"])  # Bulk modulus (GPa)
                        basis_data["lattice_a"].append(a)
                        basis_data["lattice_b"].append(b)
                        basis_data["lattice_c"].append(c)
                        basis_data["lattice_alpha"].append(alpha)
                        basis_data["lattice_beta"].append(beta)
                        basis_data["lattice_gamma"].append(gamma)
                        basis_data["run_time"].append(run_time)
                        basis_data["names"].append(job_name)

                        logger.info(
                            f"✓ Collected EOS data for {basis_set}: V0={v0:.3f}, a={a:.6f} Å, B0={eos_fit['b0 GPa']:.1f} GPa, t={run_time:.1f}s"
                            if run_time
                            else f"✓ Collected EOS data for {basis_set}: V0={v0:.3f}, a={a:.6f} Å, B0={eos_fit['b0 GPa']:.1f} GPa"
                        )
                    else:
                        logger.warning(f"No murnaghan EOS fit found for {job_name}")
                        logger.warning(
                            f"Available EOS models: {list(eos_models.keys())}"
                        )
                else:
                    logger.warning(f"No EOS fits found in relax results for {job_name}")
            else:
                logger.warning(f"Unexpected result structure for {job_name}")
                logger.warning(
                    f"Result type: {type(result)}, keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}"
                )

        except (KeyError, TypeError, ValueError, AttributeError) as e:
            logger.exception(f"Error processing job {job_name} (UUID: {job_uuid}): {e}")
            import traceback

            logger.exception(traceback.format_exc())
            continue

    if not basis_data["basis_sets"]:
        logger.error("No EOS data collected from any job!")
    else:
        logger.info(
            f"Successfully collected EOS data for {len(basis_data['basis_sets'])} basis sets"
        )
        console.print(
            f"[green]Collected EOS data for basis sets: {', '.join(basis_data['basis_sets'])}[/green]"
        )

    return basis_data


@job
def write_eos_basis_summary(basis_data: dict[str, Any]) -> str:
    """
    Write summary of EOS parameters vs basis set to file.

    Args:
        basis_data: Dictionary with basis_sets, V0, E0, B0 lists

    Returns
    -------
        Path to the summary file
    """
    output_file = "eos_basis_summary.txt"

    logger.info(f"Writing EOS basis summary to {output_file}")

    with open(output_file, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("EOS BASIS CONVERGENCE SUMMARY\n")
        f.write("=" * 90 + "\n\n")

        f.write("Basis Set Convergence for Equation of State Parameters\n")
        f.write("-" * 90 + "\n\n")

        if not basis_data["basis_sets"]:
            f.write("❌ No data collected\n")
            return output_file

        # Check if we have timing data
        logger.info(f"basis_data keys: {list(basis_data.keys())}")
        if "run_time" in basis_data:
            logger.info(f"basis_data['run_time']: {basis_data['run_time']}")
        has_timing = "run_time" in basis_data and any(
            t is not None for t in basis_data["run_time"]
        )

        # Write table header
        if has_timing:
            f.write(
                f"{'Basis':<12} {'V0 (Ų)':<15} {'E0 (eV)':<15} {'B0 (GPa)':<15} {'Time (s)':<12}\n"
            )
            f.write("-" * 90 + "\n")
        else:
            f.write(f"{'Basis':<12} {'V0 (Ų)':<15} {'E0 (eV)':<15} {'B0 (GPa)':<15}\n")
            f.write("-" * 90 + "\n")

        # Write data for each basis set
        for i, basis in enumerate(basis_data["basis_sets"]):
            v0 = basis_data["V0"][i]
            e0 = basis_data["E0"][i]
            b0 = basis_data["B0"][i]

            if has_timing:
                run_time = basis_data["run_time"][i]
                time_str = f"{run_time:.1f}" if run_time is not None else "N/A"
                f.write(
                    f"{basis:<12} {v0:<15.4f} {e0:<15.6f} {b0:<15.2f} {time_str:<12}\n"
                )
            else:
                f.write(f"{basis:<12} {v0:<15.4f} {e0:<15.6f} {b0:<15.2f}\n")

        f.write("\n")
        f.write("=" * 90 + "\n\n")

        # Calculate convergence metrics
        if len(basis_data["basis_sets"]) > 1:
            v0_range = max(basis_data["V0"]) - min(basis_data["V0"])
            b0_range = max(basis_data["B0"]) - min(basis_data["B0"])

            f.write("Convergence Analysis:\n")
            f.write(
                f"  V0 variation: {v0_range:.4f} Ų ({v0_range / max(basis_data['V0']) * 100:.2f}%)\n"
            )
            f.write(
                f"  B0 variation: {b0_range:.2f} GPa ({b0_range / max(basis_data['B0']) * 100:.2f}%)\n"
            )
            f.write("\n")

            # Recommendations
            f.write("Recommendations:\n")
            if (
                v0_range / max(basis_data["V0"]) < 0.01
                and b0_range / max(basis_data["B0"]) < 0.05
            ):
                f.write("  ✓ EOS parameters well converged with respect to basis set\n")
                f.write(
                    f"  ✓ Recommended basis: {basis_data['basis_sets'][-2]} (good balance)\n"
                )
            else:
                f.write("  ⚠ Significant basis set dependence observed\n")
                f.write(
                    f"  ⚠ Consider using: {basis_data['basis_sets'][-1]} for production\n"
                )

        f.write("\n")

        # Add standard footer
        from atomate2.siesta.utils.text_output import get_standard_footer

        f.write(
            get_standard_footer(
                width=90,
                additional_info={
                    "Analysis type": "EOS basis convergence",
                    "Basis sets tested": str(len(basis_data["basis_sets"])),
                },
            )
        )

    logger.info(f"Summary written to {output_file}")
    console.print(f"[green]EOS basis summary written to: {output_file}[/green]")

    return output_file


@job
def plot_eos_basis_comparison(basis_data: dict[str, Any]) -> str:
    """
    Create comparison plot of EOS parameters vs basis set.

    Args:
        basis_data: Dictionary with basis_sets, V0, E0, B0 lists

    Returns
    -------
        Path to the plot file
    """
    import matplotlib.pyplot as plt

    output_file = "eos_basis_comparison.png"

    logger.info(f"Creating EOS basis comparison plot: {output_file}")

    if not basis_data["basis_sets"]:
        logger.warning("No data to plot")
        return output_file

    basis_sets = basis_data["basis_sets"]
    v0_values = basis_data["V0"]
    e0_values = basis_data["E0"]
    b0_values = basis_data["B0"]

    # Check if we have timing data
    has_timing = "run_time" in basis_data and any(
        t is not None for t in basis_data["run_time"]
    )

    # Create figure with 3 or 4 subplots depending on timing availability
    if has_timing:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    else:
        _fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot V0
    axes[0].plot(range(len(basis_sets)), v0_values, "o-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Basis Set", fontsize=12)
    axes[0].set_ylabel("V₀ (Ų)", fontsize=12)
    axes[0].set_title("Equilibrium Volume", fontsize=14, fontweight="bold")
    axes[0].set_xticks(range(len(basis_sets)))
    axes[0].set_xticklabels(basis_sets, rotation=45)
    axes[0].grid(True, alpha=0.3)  # noqa: FBT003

    # Plot E0
    axes[1].plot(
        range(len(basis_sets)), e0_values, "o-", color="red", linewidth=2, markersize=8
    )
    axes[1].set_xlabel("Basis Set", fontsize=12)
    axes[1].set_ylabel("E₀ (eV)", fontsize=12)
    axes[1].set_title("Equilibrium Energy", fontsize=14, fontweight="bold")
    axes[1].set_xticks(range(len(basis_sets)))
    axes[1].set_xticklabels(basis_sets, rotation=45)
    axes[1].grid(True, alpha=0.3)  # noqa: FBT003

    # Plot B0
    axes[2].plot(
        range(len(basis_sets)),
        b0_values,
        "o-",
        color="green",
        linewidth=2,
        markersize=8,
    )
    axes[2].set_xlabel("Basis Set", fontsize=12)
    axes[2].set_ylabel("B₀ (GPa)", fontsize=12)
    axes[2].set_title("Bulk Modulus", fontsize=14, fontweight="bold")
    axes[2].set_xticks(range(len(basis_sets)))
    axes[2].set_xticklabels(basis_sets, rotation=45)
    axes[2].grid(True, alpha=0.3)  # noqa: FBT003

    # Plot timing if available
    if has_timing:
        run_times = basis_data["run_time"]
        # Filter out None values for plotting
        valid_indices = [i for i, t in enumerate(run_times) if t is not None]
        valid_times = [run_times[i] for i in valid_indices]
        valid_basis = [basis_sets[i] for i in valid_indices]

        axes[3].plot(
            range(len(valid_basis)),
            valid_times,
            "o-",
            color="purple",
            linewidth=2,
            markersize=8,
        )
        axes[3].set_xlabel("Basis Set", fontsize=12)
        axes[3].set_ylabel("Time (s)", fontsize=12)
        axes[3].set_title("Calculation Time", fontsize=14, fontweight="bold")
        axes[3].set_xticks(range(len(valid_basis)))
        axes[3].set_xticklabels(valid_basis, rotation=45)
        axes[3].grid(True, alpha=0.3)  # noqa: FBT003

    plt.suptitle("EOS Parameters vs Basis Set", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Plot saved to {output_file}")
    console.print(f"[green]EOS basis comparison plot saved to: {output_file}[/green]")

    return output_file


@job
def plot_eos_overlay(flow_results: dict[str, Any], job_metadata: list[dict]) -> str:
    """
    Create overlay plot of all EOS curves on same axes for easy comparison.

    Shows both the calculated data points and fitted EOS curves for each
    basis set, allowing direct visual comparison of how basis quality
    affects the equation of state.

    Args:
        flow_results: Results dictionary from jobflow
        job_metadata: List with job info and basis sets

    Returns
    -------
        Path to overlay plot file
    """
    import matplotlib.pyplot as plt
    import numpy as np

    output_file = "eos_overlay.png"

    logger.info(f"Creating EOS overlay plot: {output_file}")

    _fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(job_metadata)))

    plotted_count = 0
    for i, job_info in enumerate(job_metadata):
        job_uuid = job_info["uuid"]
        basis_set = job_info.get("basis_set", "Unknown")

        try:
            if job_uuid not in flow_results:
                continue

            result = flow_results[job_uuid]

            # EOS flow returns dict with structure: result["relax"]["volume"], result["relax"]["energy"]
            # and result["relax"]["EOS"]["murnaghan"] for fitted parameters
            if isinstance(result, dict) and "relax" in result:
                relax_data = result["relax"]

                if "volume" in relax_data and "energy" in relax_data:
                    # Get data points
                    volumes = np.array(relax_data["volume"])
                    energies = np.array(relax_data["energy"])

                    # Get fitted parameters from murnaghan EOS
                    if "EOS" in relax_data and "murnaghan" in relax_data["EOS"]:
                        eos_fit = relax_data["EOS"]["murnaghan"]
                        v0 = eos_fit["v0"]
                        b0 = eos_fit["b0 GPa"]

                        # Generate smooth fitted curve using pymatgen EOS
                        from pymatgen.analysis.eos import EOS

                        eos = EOS(eos_name="murnaghan")
                        eos_obj = eos.fit(volumes, energies)
                        v_fit = np.linspace(volumes.min(), volumes.max(), 100)
                        e_fit = eos_obj.func(v_fit)

                        # Plot fitted curve
                        ax.plot(
                            v_fit,
                            e_fit,
                            "-",
                            color=colors[i],
                            linewidth=2.5,
                            alpha=0.8,
                            label=f"{basis_set}: V₀={v0:.2f} Ų, B₀={b0:.1f} GPa",
                        )

                        # Plot data points
                        ax.plot(
                            volumes,
                            energies,
                            "o",
                            color=colors[i],
                            markersize=8,
                            markeredgecolor="white",
                            markeredgewidth=1.5,
                        )

                        plotted_count += 1
                    else:
                        logger.warning(f"No murnaghan EOS fit found for {basis_set}")
                else:
                    logger.warning(f"No volume/energy data for {basis_set}")
            else:
                logger.warning(f"Unexpected result structure for {basis_set}")

        except (KeyError, AttributeError) as e:
            logger.warning(f"Could not plot EOS for {basis_set}: {e}")
            continue

    if plotted_count == 0:
        logger.warning("No EOS data could be plotted")
        plt.close()
        return output_file

    ax.set_xlabel("Volume (Ų)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Energy (eV)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Equation of State - Basis Set Comparison",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Enhanced legend
    ax.legend(
        fontsize=11,
        framealpha=0.95,
        edgecolor="gray",
        loc="best",
        ncol=1 if plotted_count <= 4 else 2,
    )

    ax.grid(True, alpha=0.3, linestyle="--")  # noqa: FBT003
    ax.tick_params(labelsize=12)

    # Add text box with info
    textstr = (
        f"Comparison of {plotted_count} basis sets\nCircles: DFT data | Lines: EOS fit"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Overlay plot saved to {output_file}")
    console.print(f"[green]EOS overlay plot saved to: {output_file}[/green]")

    return output_file


@job
def generate_eos_basis_outputs(
    flow_results: dict[str, Any],
    job_metadata: list[dict],
    basis_data: dict[str, Any],
    output_dir: str | Path = ".",
) -> dict[str, Any]:
    """
    Generate unified EOS basis convergence outputs in one directory.

    Creates all plots and summaries with proper naming:
    - Individual EOS plots: eos_fit_SZ.png, eos_fit_DZ.png, etc.
    - Individual summaries: eos_summary_SZ.txt, eos_summary_DZ.txt, etc.
    - Comparison plot: eos_basis_comparison.png
    - Overlay plot: eos_overlay_all_basis.png
    - Summary: eos_basis_summary.txt

    Args:
        flow_results: Results dictionary from jobflow
        job_metadata: List with job info and basis sets
        basis_data: Collected basis convergence data
        output_dir: Directory to save all outputs (default: current directory)

    Returns
    -------
        Dictionary with paths to all generated files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating unified EOS basis outputs in {output_path}")
    console.print(f"[cyan]Collecting EOS basis outputs into: {output_path}[/cyan]")

    output_files: dict[str, Any] = {
        "individual_plots": [],
        "individual_summaries": [],
        "comparison_plot": None,
        "overlay_plot": None,
        "basis_summary": None,
    }

    # For each basis set, extract and save its individual EOS plots/summaries
    for job_info in job_metadata:
        job_uuid = job_info["uuid"]
        basis_set = job_info.get("basis_set", "Unknown")

        if job_uuid not in flow_results:
            logger.warning(f"No results for {basis_set}")
            continue

        result = flow_results[job_uuid]

        # Check if this EOS flow has output files
        if isinstance(result, dict) and "relax" in result:
            relax_data = result["relax"]

            # Generate individual EOS plot for this basis
            if "volume" in relax_data and "energy" in relax_data:
                volumes = np.array(relax_data["volume"])
                energies = np.array(relax_data["energy"])

                # Create EOS fit plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(
                    volumes,
                    energies,
                    "o",
                    markersize=10,
                    label="Calculated",
                    color="blue",
                )

                # Add fitted curve if available
                if "EOS" in relax_data and "murnaghan" in relax_data["EOS"]:
                    eos_fit = relax_data["EOS"]["murnaghan"]
                    if "exception" not in eos_fit:
                        # Generate smooth curve
                        vol_fit = np.linspace(
                            volumes.min() * 0.95, volumes.max() * 1.05, 100
                        )
                        try:
                            eos_obj = EOS(eos_name="murnaghan").fit(volumes, energies)
                            energy_fit = eos_obj.func(vol_fit)
                            ax.plot(
                                vol_fit,
                                energy_fit,
                                "-",
                                linewidth=2,
                                label="Murnaghan fit",
                                color="red",
                            )

                            # Mark equilibrium point
                            v0 = eos_fit.get("v0")
                            e0 = eos_fit.get("e0")
                            if v0 and e0:
                                ax.plot(
                                    v0,
                                    e0,
                                    "X",
                                    markersize=15,
                                    color="green",
                                    label=f"V₀={v0:.2f} ų",
                                    zorder=10,
                                )
                        except Exception as e:
                            logger.warning(f"Could not plot fit for {basis_set}: {e}")

                ax.set_xlabel("Volume (ų)", fontsize=14, fontweight="bold")
                ax.set_ylabel("Energy (eV)", fontsize=14, fontweight="bold")
                ax.set_title(f"EOS - {basis_set} Basis", fontsize=16, fontweight="bold")
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)  # noqa: FBT003

                plot_file = output_path / f"eos_fit_{basis_set}.png"
                plt.tight_layout()
                plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                plt.close()

                output_files["individual_plots"].append(str(plot_file))
                logger.info(f"Saved EOS plot for {basis_set}: {plot_file}")

            # Generate individual summary for this basis
            if "EOS" in relax_data and "murnaghan" in relax_data["EOS"]:
                eos_fit = relax_data["EOS"]["murnaghan"]
                if "exception" not in eos_fit:
                    summary_file = output_path / f"eos_summary_{basis_set}.txt"

                    with open(summary_file, "w") as f:
                        f.write("=" * 80 + "\n")
                        f.write(f"EOS RESULTS - {basis_set} BASIS\n")
                        f.write("=" * 80 + "\n\n")
                        f.write("Murnaghan Equation of State Fit:\n")
                        f.write("-" * 80 + "\n")
                        f.write(
                            f"  V₀ (equilibrium volume):   {eos_fit.get('v0', 'N/A'):.6f} ų\n"
                        )
                        f.write(
                            f"  E₀ (equilibrium energy):   {eos_fit.get('e0', 'N/A'):.6f} eV\n"
                        )
                        f.write(
                            f"  B₀ (bulk modulus):         {eos_fit.get('b0 GPa', 'N/A'):.4f} GPa\n"
                        )
                        f.write(
                            f"  B₁ (pressure derivative):  {eos_fit.get('b1', 'N/A')}\n"
                        )
                        f.write("\n" + "=" * 80 + "\n")

                    output_files["individual_summaries"].append(str(summary_file))
                    logger.info(f"Saved EOS summary for {basis_set}: {summary_file}")

    # Generate comparison plot
    comparison_plot = output_path / "eos_basis_comparison.png"
    if basis_data and basis_data.get("basis_sets"):
        basis_sets = basis_data["basis_sets"]
        v0_values = basis_data["V0"]
        e0_values = basis_data["E0"]
        b0_values = basis_data["B0"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # V0 plot
        axes[0].plot(
            range(len(basis_sets)),
            v0_values,
            "o-",
            linewidth=2,
            markersize=8,
            color="blue",
        )
        axes[0].set_xlabel("Basis Set", fontsize=12)
        axes[0].set_ylabel("V₀ (ų)", fontsize=12)
        axes[0].set_title("Equilibrium Volume", fontsize=14, fontweight="bold")
        axes[0].set_xticks(range(len(basis_sets)))
        axes[0].set_xticklabels(basis_sets, rotation=45)
        axes[0].grid(True, alpha=0.3)  # noqa: FBT003

        # E0 plot
        axes[1].plot(
            range(len(basis_sets)),
            e0_values,
            "o-",
            color="red",
            linewidth=2,
            markersize=8,
        )
        axes[1].set_xlabel("Basis Set", fontsize=12)
        axes[1].set_ylabel("E₀ (eV)", fontsize=12)
        axes[1].set_title("Equilibrium Energy", fontsize=14, fontweight="bold")
        axes[1].set_xticks(range(len(basis_sets)))
        axes[1].set_xticklabels(basis_sets, rotation=45)
        axes[1].grid(True, alpha=0.3)  # noqa: FBT003

        # B0 plot
        axes[2].plot(
            range(len(basis_sets)),
            b0_values,
            "o-",
            color="green",
            linewidth=2,
            markersize=8,
        )
        axes[2].set_xlabel("Basis Set", fontsize=12)
        axes[2].set_ylabel("B₀ (GPa)", fontsize=12)
        axes[2].set_title("Bulk Modulus", fontsize=14, fontweight="bold")
        axes[2].set_xticks(range(len(basis_sets)))
        axes[2].set_xticklabels(basis_sets, rotation=45)
        axes[2].grid(True, alpha=0.3)  # noqa: FBT003

        plt.suptitle("EOS Parameters vs Basis Set", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(comparison_plot, dpi=300, bbox_inches="tight")
        plt.close()

        output_files["comparison_plot"] = str(comparison_plot)
        logger.info(f"Saved comparison plot: {comparison_plot}")

    # Generate overlay plot
    overlay_plot = output_path / "eos_overlay_all_basis.png"
    _fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(job_metadata)))

    for i, job_info in enumerate(job_metadata):
        job_uuid = job_info["uuid"]
        basis_set = job_info.get("basis_set", "Unknown")

        if job_uuid not in flow_results:
            continue

        result = flow_results[job_uuid]
        if isinstance(result, dict) and "relax" in result:
            relax_data = result["relax"]
            if "volume" in relax_data and "energy" in relax_data:
                volumes = np.array(relax_data["volume"])
                energies = np.array(relax_data["energy"])

                # Plot data points
                ax.scatter(
                    volumes,
                    energies,
                    s=80,
                    c=[colors[i]],
                    edgecolors="black",
                    linewidths=1,
                    zorder=5,
                    label=f"{basis_set} (data)",
                )

                # Plot fit curve
                if "EOS" in relax_data and "murnaghan" in relax_data["EOS"]:
                    eos_fit = relax_data["EOS"]["murnaghan"]
                    if "exception" not in eos_fit:
                        vol_fit = np.linspace(
                            volumes.min() * 0.95, volumes.max() * 1.05, 100
                        )
                        try:
                            eos_obj = EOS(eos_name="murnaghan").fit(volumes, energies)
                            energy_fit = eos_obj.func(vol_fit)
                            ax.plot(
                                vol_fit,
                                energy_fit,
                                color=colors[i],
                                linewidth=2.5,
                                linestyle="-",
                                alpha=0.7,
                                label=f"{basis_set} (fit)",
                            )
                        except Exception:
                            pass

    ax.set_xlabel("Volume (ų)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Energy (eV)", fontsize=14, fontweight="bold")
    ax.set_title(
        "EOS Comparison: All Basis Sets Overlaid", fontsize=16, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, linestyle="--")  # noqa: FBT003
    ax.legend(loc="best", fontsize=10, ncol=2, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(overlay_plot, dpi=300, bbox_inches="tight")
    plt.close()

    output_files["overlay_plot"] = str(overlay_plot)
    logger.info(f"Saved overlay plot: {overlay_plot}")

    # Generate comprehensive basis summary
    summary_file = output_path / "eos_basis_summary.txt"
    if basis_data and basis_data.get("basis_sets"):
        with open(summary_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("EOS PARAMETERS FOR DIFFERENT BASIS SETS\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Tested {len(basis_data['basis_sets'])} basis sets\n\n")

            # Main EOS parameters table with lattice parameters
            # Check if all structures have equilibrium structure data
            a_values = basis_data["lattice_a"]
            b_values = basis_data["lattice_b"]
            is_cubic_approx = all(
                np.isclose(a_values[i], b_values[i], rtol=0.001)
                and np.isclose(a_values[i], basis_data["lattice_c"][i], rtol=0.001)
                for i in range(len(a_values))
            )

            f.write("EQUILIBRIUM PROPERTIES:\n")
            if is_cubic_approx:
                f.write("Note: Cubic system detected (a ≈ b ≈ c)\n")
            f.write("-" * 120 + "\n")
            f.write(
                f"{'Basis Set':<12} {'V₀ (ų)':<12} {'E₀ (eV)':<15} {'B₀ (GPa)':<12} "
                f"{'a (Å)':<10} {'b (Å)':<10} {'c (Å)':<10} {'α (°)':<8} {'β (°)':<8} {'γ (°)':<8}\n"  # noqa: RUF001
            )
            f.write("-" * 120 + "\n")

            for i, basis in enumerate(basis_data["basis_sets"]):
                v0 = basis_data["V0"][i]
                e0 = basis_data["E0"][i]
                b0 = basis_data["B0"][i]
                a = basis_data["lattice_a"][i]
                b = basis_data["lattice_b"][i]
                c = basis_data["lattice_c"][i]
                alpha = basis_data["lattice_alpha"][i]
                beta = basis_data["lattice_beta"][i]
                gamma = basis_data["lattice_gamma"][i]
                f.write(
                    f"{basis:<12} {v0:<12.6f} {e0:<15.6f} {b0:<12.4f} "
                    f"{a:<10.6f} {b:<10.6f} {c:<10.6f} {alpha:<8.2f} {beta:<8.2f} {gamma:<8.2f}\n"
                )

            f.write("\n")

            # Convergence analysis
            if len(basis_data["basis_sets"]) > 1:
                f.write("=" * 80 + "\n")
                f.write("CONVERGENCE ANALYSIS\n")
                f.write("=" * 80 + "\n\n")

                # Volume and bulk modulus convergence
                v0_values = basis_data["V0"]
                b0_values = basis_data["B0"]
                e0_values = basis_data["E0"]
                a_values = basis_data["lattice_a"]
                b_values = basis_data["lattice_b"]
                c_values = basis_data["lattice_c"]

                v0_range = max(v0_values) - min(v0_values)
                b0_range = max(b0_values) - min(b0_values)
                e0_range = (max(e0_values) - min(e0_values)) * 1000  # meV
                a_range = max(a_values) - min(a_values)

                v0_percent = (v0_range / np.mean(v0_values)) * 100
                b0_percent = (b0_range / np.mean(b0_values)) * 100
                a_percent = (a_range / np.mean(a_values)) * 100

                f.write("Bulk Properties Variation:\n")
                f.write(f"  V₀ range: {v0_range:.6f} ų ({v0_percent:.2f}%)\n")
                f.write(f"  E₀ range: {e0_range:.4f} meV\n")
                f.write(f"  B₀ range: {b0_range:.4f} GPa ({b0_percent:.2f}%)\n\n")

                f.write("Lattice Parameters Variation:\n")
                f.write(f"  a range: {a_range:.6f} Å ({a_percent:.2f}%)\n")

                # Check if system is cubic
                if is_cubic_approx:
                    f.write("  (Cubic system: b and c ranges identical to a)\n")
                else:
                    b_range = max(b_values) - min(b_values)
                    c_range = max(c_values) - min(c_values)
                    b_percent = (
                        (b_range / np.mean(b_values)) * 100
                        if np.mean(b_values) > 0
                        else 0
                    )
                    c_percent = (
                        (c_range / np.mean(c_values)) * 100
                        if np.mean(c_values) > 0
                        else 0
                    )
                    f.write(f"  b range: {b_range:.6f} Å ({b_percent:.2f}%)\n")
                    f.write(f"  c range: {c_range:.6f} Å ({c_percent:.2f}%)\n")
                f.write("\n")

                # Convergence assessment
                f.write("Convergence Assessment:\n")
                if v0_percent < 0.5 and b0_percent < 2.0:
                    f.write("  ✓ EXCELLENT: EOS parameters well converged\n")
                    f.write(
                        f"  ✓ Recommended: {basis_data['basis_sets'][-2]} (good balance)\n"
                    )
                elif v0_percent < 1.0 and b0_percent < 5.0:
                    f.write("  ✓ GOOD: Adequate convergence for most purposes\n")
                    f.write(
                        f"  → Consider: {basis_data['basis_sets'][-1]} for high accuracy\n"
                    )
                else:
                    f.write("  ⚠ FAIR: Significant basis set dependence\n")
                    f.write(
                        f"  → Use: {basis_data['basis_sets'][-1]} for production calculations\n"
                    )
                    f.write("  → Consider testing larger basis (TZP, TZDP)\n")

                f.write("\nConvergence Criteria:\n")
                f.write("  Excellent: V₀ < 0.5%, B₀ < 2%\n")
                f.write("  Good:      V₀ < 1.0%, B₀ < 5%\n")
                f.write("  Fair:      V₀ < 2.0%, B₀ < 10%\n")
                f.write("  Poor:      > 2% variation (need larger basis)\n\n")

            # Energy vs Basis comparison
            f.write("=" * 80 + "\n")
            f.write("ENERGY DIFFERENCES\n")
            f.write("=" * 80 + "\n\n")

            e_min = min(basis_data["E0"])
            f.write("Energy relative to lowest (most negative):\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Basis Set':<12} {'E₀ (eV)':<15} {'ΔE (meV)':<12} {'Status'}\n")
            f.write("-" * 80 + "\n")

            for i, basis in enumerate(basis_data["basis_sets"]):
                e0 = basis_data["E0"][i]
                de = (e0 - e_min) * 1000  # meV
                status = "LOWEST ✓" if abs(de) < 0.1 else ""
                f.write(f"{basis:<12} {e0:<15.6f} {de:<12.4f}  {status}\n")

            f.write("\n")

            # Recommendations
            f.write("=" * 80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")

            f.write("Basis Set Selection Guide:\n\n")

            f.write("• For Quick Testing / Screening:\n")
            f.write(
                f"  → {basis_data['basis_sets'][0]} or {basis_data['basis_sets'][1]}\n"
            )
            f.write("  → Fast but may lack accuracy\n\n")

            f.write("• For Production Calculations:\n")
            if len(basis_data["basis_sets"]) > 2:
                f.write(f"  → {basis_data['basis_sets'][-2]} (recommended balance)\n")
            f.write(f"  → {basis_data['basis_sets'][-1]} (highest tested accuracy)\n\n")

            f.write("• For High-Precision Work:\n")
            f.write("  → Use largest converged basis from this study\n")
            f.write("  → Consider testing even larger basis (QZP) if not converged\n\n")

            f.write("General Guidelines:\n")
            f.write("  • Check convergence: parameters stable with basis increase?\n")
            f.write("  • Balance accuracy vs computational cost\n")
            f.write("  • Larger systems → can use smaller basis per atom\n")
            f.write("  • Metals often need larger basis than insulators\n\n")

            # Output files reference
            f.write("=" * 80 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("=" * 80 + "\n\n")

            f.write("Individual Basis Results:\n")
            for basis in basis_data["basis_sets"]:
                f.write(f"  • eos_fit_{basis}.png - EOS curve and fit\n")
                f.write(f"  • eos_summary_{basis}.txt - Detailed parameters\n")

            f.write("\nComparison Plots:\n")
            f.write("  • eos_basis_comparison.png - V₀, E₀, B₀ vs basis\n")
            f.write("  • eos_overlay_all_basis.png - All EOS curves overlaid\n\n")

            # Add standard footer
            from atomate2.siesta.utils.text_output import get_standard_footer

            f.write(
                get_standard_footer(
                    width=80,
                    additional_info={
                        "Analysis type": "EOS basis convergence",
                        "Basis sets tested": str(len(basis_data["basis_sets"])),
                    },
                )
            )

        output_files["basis_summary"] = str(summary_file)
        logger.info(f"Saved comprehensive basis summary: {summary_file}")

    console.print("[green]✓ All EOS basis outputs generated successfully![/green]")
    console.print(f"[cyan]  Output directory: {output_path}[/cyan]")
    console.print(
        f"[cyan]  Individual plots: {len(output_files['individual_plots'])}[/cyan]"
    )
    console.print(
        f"[cyan]  Individual summaries: {len(output_files['individual_summaries'])}[/cyan]"
    )

    return output_files
