"""
Basis convergence workflows for SIESTA.

This module provides workflows for testing basis set quality and convergence:

1. DifferentBasisSCF / DifferentBasisSCFAdvanceFlowMaker:
   - Tests different nominal basis sizes (SZ, DZ, DZP, TZP, etc.)
   - Helps determine overall basis quality needed

2. EOSBasisConvergenceFlowMaker:
   - Tests EOS parameters (V₀, B₀) with different basis sets
   - Validates basis convergence for structural/bulk properties

3. BasisParametersConvergenceFlowMaker:
   - Optimizes PAO.EnergyShift and PAO.SplitNorm parameters
   - Fine-tunes basis generation for optimal accuracy/cost balance
   - Provides detailed 4-panel convergence plots and analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jobflow.core.flow import Flow
from jobflow.core.job import job

from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.powerups import update_user_siesta_settings
from atomate2.siesta.utils.common import console, print_docstring_in_box
from atomate2.siesta.utils.verbosity import VerbosityLevel

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Molecule, Structure

    from atomate2.siesta.jobs.base import BaseSiestaMaker

logger = logging.getLogger(__name__)


@job
def print_energies(
    flow_results: dict[str, Any], job_metadata: list[dict]
) -> dict[str, float]:
    """
    Retrieve and print the total energies from each job in the Flow's results.

    Uses job.output to access the energies.

    Args:
        flow_results (Dict[str, Any]): The results dictionary returned by
            jobflow's run_locally.
        job_metadata (list[dict]): List of dictionaries containing job names
            and UUIDs.
        verbosity (VerbosityLevel): Verbosity level for console output.
            Defaults to INFO.

    Returns
    -------
        Dict[str, float]: A dictionary mapping job names to their total
        energies (in eV).
    """
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
                        f"[yellow]No results found for job {job_name} "
                        f"(UUID: {job_uuid})[/yellow]"
                    )
                continue
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            if verbosity.value >= VerbosityLevel.WARNING.value:
                console.print(
                    f"[red]Error processing job {job_name} "
                    f"(UUID: {job_uuid}): {e}[/red]"
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
    Retrieve and print the total energies from each job in the Flow's results.

    Uses job.output to access the energies.

    Args:
        flow (Flow): The Flow object containing the SCF jobs.
        flow_results (Dict[str, Any]): The results dictionary returned by
            jobflow's run_locally.
        verbosity (VerbosityLevel): Verbosity level for console output.
            Defaults to DEBUG.

    Returns
    -------
        Dict[str, float]: A dictionary mapping job names to their total
        energies (in eV).
    """
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
                input = result.input  # noqa: A001
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
                        f"[yellow]No results found for job {job_name} "
                        f"(UUID: {job_uuid})[/yellow]"
                    )
                continue

        except (KeyError, TypeError, ValueError, AttributeError) as e:
            if verbosity.value >= VerbosityLevel.WARNING.value:
                console.print(
                    f"[red]Error processing job {job_name} "
                    f"(UUID: {job_uuid}): {e}[/red]"
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
) -> None:
    """
    Plot total energies vs. basis size.

    Args:
        energies (Dict[str, float]): Dictionary mapping job names to total
            energies (in eV).
        verbosity (VerbosityLevel): Verbosity level for console output.
            Defaults to INFO.
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
class DifferentBasisSCFAdvanceFlowMaker(BaseSiestaFlowMaker):
    """
    A Flow (maker) to run SCF calculations with different basis sizes.

    Create a complete Flow:

    1. Runs SCF jobs for different basis sizes in parallel.
    2. Gathers the results to print energies.
    3. Plots the energies vs. basis size.
    """

    CONSOLE_VERBOSITY: VerbosityLevel = (
        VerbosityLevel.INFO
    )  # Default to show info messages
    name: str = "Different basis scf"
    static_maker: BaseSiestaMaker = field(default_factory=StaticMaker)

    def make(
        self,
        structure: Structure | Molecule,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """Create a Flow with SCF jobs for different basis sizes."""
        # Get the docstring from the class
        doc_to_print = self.__doc__

        # Get the class name to use as a title
        class_name = self.__class__.__name__

        # Call the function to print it
        print_docstring_in_box(doc_to_print, title=class_name)
        # ======================================================================
        # Step 1: Create the initial parallel SCF calculation jobs
        # (This part is the same as your original code)
        # ======================================================================
        jobs = []
        # Dictionary to store job metadata
        job_metadata = []
        basis_params = {
            # Single-Zeta
            "SZ": {"PAO.EnergyShift": "0.02 Ry", "PAO.SplitNorm": 0.15},
            "MINIMAL": {"PAO.EnergyShift": "0.02 Ry", "PAO.SplitNorm": 0.15},
            "SZP": {"PAO.EnergyShift": "0.02 Ry", "PAO.SplitNorm": 0.15},
            "SZSP": {"PAO.EnergyShift": "0.02 Ry", "PAO.SplitNorm": 0.15},
            "SZ1P": {"PAO.EnergyShift": "0.02 Ry", "PAO.SplitNorm": 0.15},
            "SZP1": {"PAO.EnergyShift": "0.02 Ry", "PAO.SplitNorm": 0.15},
            # Double-Zeta
            "DZ": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
            "DZP": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
            "DZSP": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
            "DZP1": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
            "DZ1P": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
            "STANDARD": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
            "DZDP": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
            "DZP2": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
            "DZ2P": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
            # Triple-Zeta
            "TZ": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
            "TZP": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
            "TZSP": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
            "TZP1": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
            "TZ1P": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
            "TZDP": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
            "TZP2": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
            "TZ2P": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
            "TZTP": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
            "TZP3": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
            "TZ3P": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
        }
        allowed_basis_size = [
            "SZ",
            "MINIMAL",
            "SZP",
            "SZSP",
            "SZ1P",
            "SZP1",
            "DZ",
            "DZP",
            "DZSP",
            "DZP1",
            "DZ1P",
            "STANDARD",
            "DZDP",
            "DZP2",
            "DZ2P",
            "TZ",
            "TZP",
            "TZSP",
            "TZP1",
            "TZ1P",
            "TZDP",
            "TZP2",
            "TZ2P",
            "TZTP",
            "TZP3",
            "TZ3P",
        ]
        # Dictionary to store job outputs for dependency tracking
        for basis in allowed_basis_size:
            # print(f"DEBUG: {basis=}")
            maker = self.static_maker.scf()

            # Propagate custodian settings
            self.propagate_custodian_to_maker(maker)

            siesta_updates = {
                "PAO.BasisSize": basis,
                "PAO.BasisType": "split",
                **basis_params.get(
                    basis, {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.15}
                ),
            }
            maker = update_user_siesta_settings(
                maker, siesta_updates, class_filter=StaticMaker
            )
            maker = update_user_siesta_settings(
                maker, {"a2s_kpts": [3, 3, 3]}, class_filter=StaticMaker
            )
            scf_maker_basis_job = maker.make(structure, prev_dir=None)
            scf_maker_basis_job.name += f"-{basis}"
            job_metadata.append(
                {"uuid": scf_maker_basis_job.uuid, "name": scf_maker_basis_job.name}
            )
            jobs.append(scf_maker_basis_job)

        #  Create a flow from the parallel SCF jobs
        # scf_flow = Flow(jobs, name="Parallel SCF Calculations",
        #   output={job_.uuid: job_.output for job_ in jobs})
        scf_flow = Flow(
            jobs,
            name="Parallel SCF Calculations",
            output={job.uuid: job.output for job in jobs},
        )

        # ======================================================================
        # Step 2: Create the post-processing jobs and chain them
        # (This is the new, integrated logic)
        # ======================================================================

        # Create the job that will print the energies after the SCF flow is done.
        # It takes the output of the scf_flow (`scf_flow.output`) as input.

        # print_job = print_energies_old(
        #     flow=scf_flow,
        #     flow_results=scf_flow.output,

        # )

        print_job = print_energies(
            flow_results=scf_flow.output,
            job_metadata=job_metadata,
        )

        print_job.name = "Print_Energies"

        # # Create the job that will plot the energies.
        # # It takes the output of the print_job (`print_job.output`) as input.

        plot_job = plot_energies(energies=print_job.output)
        plot_job.name = "Plot_Energies"

        # # ======================================================================
        # # Step 3: Combine all components into a single, final Flow
        # # ======================================================================

        # # The final flow contains the initial SCF flow and the two subsequent jobs.
        # # Jobflow automatically resolves the dependency chain.
        # We can expose the energies dictionary as the final output of the
        # entire workflow.
        return Flow(
            [scf_flow, print_job, plot_job],
            name=self.name,
            output=print_job.output,
        )


@job
def collect_eos_basis_data(
    flow_results: dict[str, Any], job_metadata: list[dict]
) -> dict[str, Any]:
    """
    Collect EOS results from different basis set calculations.

    Args:
        flow_results: Results dictionary from jobflow's run_locally
        job_metadata: List of dictionaries containing job names and UUIDs

    Returns
    -------
        Dictionary with basis sets and their EOS parameters
    """
    logger.info("Collecting EOS data for different basis sets")
    data: dict[str, list] = {
        "basis_sets": [],
        "v0": [],
        "e0": [],
        "b0": [],
        "b1": [],
        "lattice_a": [],
        "lattice_b": [],
        "lattice_c": [],
        "lattice_alpha": [],
        "lattice_beta": [],
        "lattice_gamma": [],
        "names": [],
    }

    for job_info in job_metadata:
        job_uuid = job_info["uuid"]
        job_name = job_info["name"]
        basis_set = job_info["basis"]

        try:
            if job_uuid not in flow_results:
                logger.warning(
                    f"No results found for job {job_name} (UUID: {job_uuid})"
                )
                continue

            # flow_results[job_uuid] is already the output dictionary from the EOS flow
            output = flow_results[job_uuid]

            logger.debug(
                f"Processing {job_name}, output keys: "
                f"{output.keys() if isinstance(output, dict) else 'not a dict'}"
            )

            # Extract EOS fit results (use Birch-Murnaghan as default)
            if (
                isinstance(output, dict)
                and "relax" in output
                and "EOS" in output["relax"]
            ):
                eos_data = output["relax"]["EOS"]

                # Try to get Birch-Murnaghan results first, fallback to first available
                bm_data = None
                if (
                    "birch_murnaghan" in eos_data
                    and "exception" not in eos_data["birch_murnaghan"]
                ):
                    bm_data = eos_data["birch_murnaghan"]
                else:
                    # Use first successful fit
                    for model_data in eos_data.values():
                        if "exception" not in model_data:
                            bm_data = model_data
                            break

                if bm_data:
                    data["names"].append(job_name)
                    data["basis_sets"].append(basis_set)
                    data["v0"].append(bm_data.get("v0", None))
                    data["e0"].append(bm_data.get("e0", None))
                    data["b0"].append(bm_data.get("b0 GPa", None))
                    data["b1"].append(bm_data.get("b1", None))

                    # Extract equilibrium lattice parameters by scaling
                    # reference structure to V₀. The EOS applies isotropic
                    # strain, so lattice parameters scale as V^(1/3)
                    v0 = bm_data.get("v0")
                    structures = output["relax"].get("structure", [])
                    volumes = output["relax"].get("volume", [])

                    if structures and volumes and v0:
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
                            data["lattice_a"].append(ref_lattice.a * scale)
                            data["lattice_b"].append(ref_lattice.b * scale)
                            data["lattice_c"].append(ref_lattice.c * scale)
                            # Angles are preserved under isotropic scaling
                            data["lattice_alpha"].append(ref_lattice.alpha)
                            data["lattice_beta"].append(ref_lattice.beta)
                            data["lattice_gamma"].append(ref_lattice.gamma)

                            logger.info(
                                f"Scaled lattice from V_ref={v_ref:.3f} Å³ "
                                f"to V₀={v0:.3f} Å³ "
                                f"(scale={scale:.6f})"
                            )
                        except AttributeError:
                            logger.warning(
                                f"Could not extract lattice from structure "
                                f"for {job_name}, using cubic approximation"
                            )
                            a_cubic = v0 ** (1.0 / 3.0)
                            data["lattice_a"].append(a_cubic)
                            data["lattice_b"].append(a_cubic)
                            data["lattice_c"].append(a_cubic)
                            data["lattice_alpha"].append(90.0)
                            data["lattice_beta"].append(90.0)
                            data["lattice_gamma"].append(90.0)
                    else:
                        # Fallback: calculate from volume assuming cubic
                        logger.warning(
                            f"No structures available for {job_name}, "
                            f"using cubic approximation"
                        )
                        if v0:
                            a_cubic = v0 ** (1.0 / 3.0)
                            data["lattice_a"].append(a_cubic)
                            data["lattice_b"].append(a_cubic)
                            data["lattice_c"].append(a_cubic)
                            data["lattice_alpha"].append(90.0)
                            data["lattice_beta"].append(90.0)
                            data["lattice_gamma"].append(90.0)
                        else:
                            data["lattice_a"].append(None)
                            data["lattice_b"].append(None)
                            data["lattice_c"].append(None)
                            data["lattice_alpha"].append(None)
                            data["lattice_beta"].append(None)
                            data["lattice_gamma"].append(None)

                    logger.debug(
                        f"{job_name}: V0={bm_data.get('v0')} Å³, "
                        f"B0={bm_data.get('b0 GPa')} GPa, "
                        f"a={data['lattice_a'][-1]:.4f} Å"
                    )
                else:
                    logger.warning(f"No successful EOS fit found for {job_name}")
            else:
                logger.warning(
                    f"Expected 'relax/EOS' structure not found in output for {job_name}"
                )

        except (KeyError, TypeError, ValueError, AttributeError):
            logger.exception(f"Error processing job {job_name}")
            import traceback

            logger.debug(traceback.format_exc())
            continue

    if not data["basis_sets"]:
        logger.warning("No EOS data retrieved for basis set comparison")
    else:
        logger.info(
            f"Successfully collected {len(data['basis_sets'])} "
            f"EOS results for basis comparison"
        )

    return data


@job
def plot_eos_basis_comparison(
    basis_data: dict[str, Any], output_file: str = "eos_basis_comparison.png"
) -> str:
    """
    Plot EOS parameters vs basis set quality.

    Args:
        basis_data: Dictionary with basis sets and EOS parameters
        output_file: Output filename for plot

    Returns
    -------
        Path to saved plot file
    """
    import matplotlib.pyplot as plt
    import numpy as np

    console.print("[green]Plotting EOS parameters vs basis set[/green]")

    basis_sets = basis_data["basis_sets"]

    # Check if we have data
    if not basis_sets:
        error_msg = "No basis set data available for plotting"
        logger.error(error_msg)
        console.print(f"[red]{error_msg}[/red]")
        raise ValueError(error_msg)

    v0 = np.array(basis_data["v0"])
    e0 = np.array(basis_data["e0"])
    b0 = np.array(basis_data["b0"])

    logger.info(f"Plotting data for {len(basis_sets)} basis sets")

    # Create figure with subplots
    _fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Equilibrium Volume
    ax1 = axes[0]
    ax1.plot(range(len(basis_sets)), v0, "o-", linewidth=2, markersize=8, color="blue")
    ax1.set_ylabel("V₀ (Ų)", fontsize=12)
    ax1.set_title("EOS Parameters vs Basis Set", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)  # noqa: FBT003
    ax1.set_xticks(range(len(basis_sets)))
    ax1.set_xticklabels([])

    # Plot 2: Equilibrium Energy
    ax2 = axes[1]
    ax2.plot(range(len(basis_sets)), e0, "s-", linewidth=2, markersize=8, color="green")
    ax2.set_ylabel("E₀ (eV)", fontsize=12)
    ax2.grid(True, alpha=0.3)  # noqa: FBT003
    ax2.set_xticks(range(len(basis_sets)))
    ax2.set_xticklabels([])

    # Plot 3: Bulk Modulus
    ax3 = axes[2]
    ax3.plot(range(len(basis_sets)), b0, "^-", linewidth=2, markersize=8, color="red")
    ax3.set_ylabel("B₀ (GPa)", fontsize=12)
    ax3.set_xlabel("Basis Set", fontsize=12)
    ax3.grid(True, alpha=0.3)  # noqa: FBT003
    ax3.set_xticks(range(len(basis_sets)))
    ax3.set_xticklabels(basis_sets, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved to {output_file}")

    console.print(f"[green]Plot saved to: {output_file}[/green]")

    return output_file


@job
def plot_eos_overlay(
    flow_results: dict[str, Any],
    job_metadata: list[dict],
    output_file: str = "eos_overlay_all_basis.png",
) -> str:
    """
    Plot all EOS curves overlaid on a single plot for comparison.

    Args:
        flow_results: Results dictionary from jobflow's run_locally
        job_metadata: List of dictionaries containing job names and UUIDs
        output_file: Output filename for plot

    Returns
    -------
        Path to saved plot file
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pymatgen.analysis.eos import EOS

    console.print("[green]Creating overlay plot of all EOS fits[/green]")

    # Colors and line styles for different basis sets
    colors = ["blue", "red", "green", "purple", "orange", "brown", "pink", "gray"]
    markers = ["o", "s", "^", "v", "D", "p", "*", "h"]

    _fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    plotted_count = 0

    for idx, job_info in enumerate(job_metadata):
        job_uuid = job_info["uuid"]
        job_name = job_info["name"]
        basis_set = job_info["basis"]

        try:
            if job_uuid not in flow_results:
                logger.warning(f"No results found for {job_name}")
                continue

            output = flow_results[job_uuid]

            # Extract volume and energy data
            if isinstance(output, dict) and "relax" in output:
                relax_data = output["relax"]

                if "volume" in relax_data and "energy" in relax_data:
                    volumes = np.array(relax_data["volume"])
                    energies = np.array(relax_data["energy"])

                    # Get color and marker for this basis
                    color = colors[idx % len(colors)]
                    marker = markers[idx % len(markers)]

                    # Plot raw data points
                    ax.scatter(
                        volumes,
                        energies,
                        s=80,
                        c=color,
                        marker=marker,
                        edgecolors="black",
                        linewidths=1,
                        zorder=5,
                        label=f"{basis_set} (data)",
                    )

                    # Try to plot the fit curve if available
                    if "EOS" in relax_data:
                        eos_fits = relax_data["EOS"]

                        # Try Birch-Murnaghan first
                        fit_data = None
                        if (
                            "birch_murnaghan" in eos_fits
                            and "exception" not in eos_fits["birch_murnaghan"]
                        ):
                            fit_data = eos_fits["birch_murnaghan"]
                        else:
                            # Use first successful fit
                            for model_data in eos_fits.values():
                                if "exception" not in model_data:
                                    fit_data = model_data
                                    break

                        if fit_data:
                            # Create smooth curve for the fit
                            vol_fit = np.linspace(
                                volumes.min() * 0.95, volumes.max() * 1.05, 100
                            )

                            try:
                                # Refit to get smooth curve
                                eos_obj = EOS(eos_name="birch_murnaghan").fit(
                                    volumes, energies
                                )
                                energy_fit = eos_obj.func(vol_fit)

                                # Plot fit curve
                                ax.plot(
                                    vol_fit,
                                    energy_fit,
                                    color=color,
                                    linewidth=2.5,
                                    linestyle="-",
                                    alpha=0.7,
                                    label=f"{basis_set} (B-M fit)",
                                )

                                # Mark equilibrium point
                                v0 = fit_data.get("v0")
                                e0 = fit_data.get("e0")
                                if v0 and e0:
                                    ax.plot(
                                        v0,
                                        e0,
                                        "X",
                                        color=color,
                                        markersize=12,
                                        markeredgecolor="black",
                                        markeredgewidth=1.5,
                                        zorder=10,
                                    )

                                plotted_count += 1

                            except Exception as e:  # noqa: BLE001
                                logger.warning(
                                    f"Could not plot fit for {basis_set}: {e}"
                                )

        except Exception:
            logger.exception(f"Error plotting {job_name}")
            continue

    if plotted_count == 0:
        error_msg = "No EOS data could be plotted"
        logger.error(error_msg)
        console.print(f"[red]{error_msg}[/red]")
        # Create empty plot with error message
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=16,
            color="red",
        )
    else:
        logger.info(f"Successfully plotted {plotted_count} basis sets")

    # Formatting
    ax.set_xlabel("Volume (Ų)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Energy (eV)", fontsize=14, fontweight="bold")
    ax.set_title(
        "EOS Comparison: All Basis Sets Overlaid", fontsize=16, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, linestyle="--")  # noqa: FBT003
    ax.legend(loc="best", fontsize=10, ncol=2, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Overlay plot saved to {output_file}")
    console.print(f"[green]Overlay plot saved to: {output_file}[/green]")

    return output_file


@job
def write_eos_basis_summary(
    basis_data: dict[str, Any], output_file: str = "eos_basis_summary.txt"
) -> str:
    """
    Write summary of EOS results for different basis sets.

    Args:
        basis_data: Dictionary with basis sets and EOS parameters
        output_file: Output filename for summary

    Returns
    -------
        Path to saved summary file
    """
    # Check if we have data
    if not basis_data["basis_sets"]:
        error_msg = "No basis set data available for summary"
        logger.error(error_msg)
        with open(output_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("EOS PARAMETERS FOR DIFFERENT BASIS SETS\n")
            f.write("=" * 80 + "\n\n")
            f.write("ERROR: No data collected. Check calculation logs.\n")
            f.write("=" * 80 + "\n")
        return output_file

    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("EOS PARAMETERS FOR DIFFERENT BASIS SETS\n")
        f.write("=" * 80 + "\n\n")

        logger.info(f"Writing summary for {len(basis_data['basis_sets'])} basis sets")

        # Main EOS parameters table
        f.write("EQUILIBRIUM PROPERTIES:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Basis Set':<12} {'V₀ (Ų)':<12} {'E₀ (eV)':<15} "
            f"{'B₀ (GPa)':<12} {'B₁':<10}\n"
        )
        f.write("-" * 80 + "\n")

        for i, basis in enumerate(basis_data["basis_sets"]):
            v0 = basis_data["v0"][i]
            e0 = basis_data["e0"][i]
            b0 = basis_data["b0"][i]
            b1 = basis_data["b1"][i] if basis_data["b1"][i] is not None else "N/A"

            f.write(f"{basis:<12} {v0:<12.6f} {e0:<15.6f} {b0:<12.4f} {b1:<10}\n")

        # Lattice parameters table
        f.write("\n" + "-" * 80 + "\n")
        f.write("EQUILIBRIUM LATTICE PARAMETERS (from V₀):\n")
        f.write(
            "Note: Lattice constants calculated assuming cubic symmetry (a³ = V₀)\n"
        )
        f.write(
            "      For accurate lattice parameters, check the relaxed structure files\n"
        )
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Basis Set':<12} {'a (Å)':<12} {'b (Å)':<12} {'c (Å)':<12} "
            f"{'α (°)':<8} {'β (°)':<8} {'γ (°)':<8}\n"  # noqa: RUF001
        )
        f.write("-" * 80 + "\n")

        for i, basis in enumerate(basis_data["basis_sets"]):
            a = basis_data["lattice_a"][i]
            b = basis_data["lattice_b"][i]
            c = basis_data["lattice_c"][i]
            alpha = basis_data["lattice_alpha"][i]
            beta = basis_data["lattice_beta"][i]
            gamma = basis_data["lattice_gamma"][i]

            if a is not None:
                f.write(
                    f"{basis:<12} {a:<12.6f} {b:<12.6f} {c:<12.6f} "
                    f"{alpha:<8.2f} {beta:<8.2f} {gamma:<8.2f}\n"
                )
            else:
                f.write(
                    f"{basis:<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} "
                    f"{'N/A':<8} {'N/A':<8} {'N/A':<8}\n"
                )

        # Calculate convergence statistics
        if len(basis_data["basis_sets"]) > 1:
            import numpy as np

            f.write("\n" + "=" * 80 + "\n")
            f.write("CONVERGENCE ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            # Volume and bulk modulus convergence
            v0_range = np.max(basis_data["v0"]) - np.min(basis_data["v0"])
            b0_range = np.max(basis_data["b0"]) - np.min(basis_data["b0"])

            f.write("Bulk Properties:\n")
            f.write(
                f"  V₀ range: {v0_range:.6f} Ų "
                f"({v0_range / np.mean(basis_data['v0']) * 100:.2f}%)\n"
            )
            f.write(
                f"  B₀ range: {b0_range:.4f} GPa "
                f"({b0_range / np.mean(basis_data['b0']) * 100:.2f}%)\n"
            )

            # Lattice constant convergence
            if basis_data["lattice_a"][0] is not None:
                a_values = [a for a in basis_data["lattice_a"] if a is not None]
                if len(a_values) > 1:
                    a_range = np.max(a_values) - np.min(a_values)
                    a_mean = np.mean(a_values)
                    f.write("\nLattice Parameters:\n")
                    f.write(
                        f"  a range: {a_range:.6f} Å ({a_range / a_mean * 100:.2f}%)\n"
                    )

                    # For cubic systems (or when a=b=c), this is the lattice constant
                    if all(
                        abs(basis_data["lattice_a"][i] - basis_data["lattice_b"][i])
                        < 1e-6
                        for i in range(len(basis_data["lattice_a"]))
                        if basis_data["lattice_a"][i] is not None
                    ):
                        f.write("  (Cubic system detected: a = b = c)\n")
                        f.write(
                            f"  Lattice constant 'a' converges from "
                            f"{min(a_values):.6f} Å to {max(a_values):.6f} Å\n"
                        )

            f.write(
                "\nRecommendation: Choose basis set where parameters have converged\n"
            )
            f.write("(minimal change with further increase in basis quality)\n")
            f.write("\nConvergence Criteria:\n")
            f.write("  Excellent: < 0.5% variation\n")
            f.write("  Good:      < 1.0% variation\n")
            f.write("  Fair:      < 2.0% variation\n")
            f.write("  Poor:      > 2.0% variation (need higher basis quality)\n")

        f.write("\n")

        # Add standard footer
        from atomate2.siesta.utils.text_output import get_standard_footer

        f.write(
            get_standard_footer(
                width=80,
                additional_info={
                    "Analysis type": "EOS parameters for different basis sets",
                    "Number of basis sets": str(len(basis_data["basis_sets"])),
                },
            )
        )

    logger.info(f"Summary written to {output_file}")
    return output_file
