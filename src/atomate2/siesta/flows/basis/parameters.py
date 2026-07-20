"""
Basis convergence workflows for SIESTA.

This module provides workflows for testing basis set quality and convergence:

1. DifferentBasisSCF / DifferentBasisSCFAdvance:
   - Tests different nominal basis sizes (SZ, DZ, DZP, TZP, etc.)
   - Helps determine overall basis quality needed

2. EOSBasisConvergenceMaker:
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
from typing import TYPE_CHECKING, Any, cast

from jobflow.core.flow import Flow
from jobflow.core.job import job

from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.flows.basis.eos import (
    collect_basis_params_data,
    plot_basis_functions,
    plot_basis_params_convergence,
    plot_real_basis_functions,
    write_basis_params_summary,
)
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.powerups import update_user_siesta_settings
from atomate2.siesta.utils.common import console, print_docstring_in_box
from atomate2.siesta.utils.verbosity import VerbosityLevel

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Maker
    from pymatgen.core import Molecule, Structure

    from atomate2.siesta.jobs.base import BaseSiestaMaker

logger = logging.getLogger(__name__)


@job
def print_energies(
    flow_results: dict[str, Any], job_metadata: list[dict]
) -> dict[str, float]:
    """
    Retrieve and print the total energies from each job in the Flow's results.

    Energies are accessed via ``job.output``.

    Args:
        flow_results (Dict[str, Any]): The results dictionary returned by
            jobflow's run_locally.
        job_metadata (list[dict]): List of dictionaries containing job names and UUIDs.
        verbosity (VerbosityLevel): Verbosity level for console output.
            Defaults to INFO.

    Returns
    -------
        Dict[str, float]: A dictionary mapping job names to their total
        energies (in eV).
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

    Energies are accessed via ``job.output``.

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
class BasisParametersConvergenceFlowMaker(BaseSiestaFlowMaker):
    """
    Workflow for testing PAO.EnergyShift and PAO.SplitNorm convergence.

    This workflow systematically tests different combinations of PAO.EnergyShift
    and PAO.SplitNorm parameters to find optimal basis generation settings.

    Workflow steps:

    1. Runs SCF calculations with different (EnergyShift, SplitNorm) combinations
    2. Collects total energies and forces for each combination
    3. Creates convergence plots:

       - Energy vs EnergyShift (for each SplitNorm)
       - Energy vs SplitNorm (for each EnergyShift)
       - 2D energy landscape heatmap
       - Basis quality map with force indicators

    4. Generates detailed summary with recommendations

    These parameters control SIESTA's numerical atomic orbital (PAO) generation:

    - PAO.EnergyShift: Confinement energy for basis orbitals (Ry)
      Lower values → more extended orbitals → larger, more accurate basis
    - PAO.SplitNorm: Threshold for split-valence orbital generation
      Higher values → more split orbitals → better description of bonding

    Example:
        >>> from atomate2.siesta.flows.basis import BasisParametersConvergenceFlowMaker
        >>> from pymatgen.core import Structure
        >>> structure = Structure.from_file("structure.cif")
        >>> maker = BasisParametersConvergenceFlowMaker(
        ...     energy_shifts=[0.005, 0.01, 0.015, 0.02],
        ...     split_norms=[0.15, 0.20, 0.25],
        ...     basis_size="DZP",
        ... )
        >>> flow = maker.make(structure)
    """

    CONSOLE_VERBOSITY: VerbosityLevel = VerbosityLevel.INFO
    name: str = "Basis Parameters Convergence"

    # Parameter ranges to test
    energy_shifts: list[float] = field(
        default_factory=lambda: [0.001, 0.005, 0.01, 0.015, 0.02, 0.03]
    )
    split_norms: list[float] = field(
        default_factory=lambda: [0.10, 0.15, 0.20, 0.25, 0.30]
    )

    # Fixed basis size for testing
    basis_size: str = "DZP"

    # K-points for calculations
    kpts: list[int] = field(default_factory=lambda: [4, 4, 4])

    # Static maker for SCF calculations
    static_maker: BaseSiestaMaker = field(default_factory=StaticMaker)

    def make(
        self, structure: Structure | Molecule, prev_dir: str | Path | None = None
    ) -> Flow:
        """
        Create basis parameters convergence flow.

        Args:
            structure: Structure to calculate
            prev_dir: Previous directory (optional)

        Returns
        -------
            Flow with SCF jobs for each parameter combination
        """
        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        jobs = []
        job_metadata = []

        # Calculate total number of jobs
        total_jobs = len(self.energy_shifts) * len(self.split_norms)
        job_counter = 0

        # Test all combinations of energy_shift and split_norm
        for energy_shift in self.energy_shifts:
            for split_norm in self.split_norms:
                job_counter += 1

                # Use the provided static maker
                maker = (
                    self.static_maker.scf()
                    if hasattr(self.static_maker, "scf")
                    else self.static_maker
                )

                # Propagate custodian settings (scf() creates new maker that
                # loses settings)
                self.propagate_custodian_to_maker(maker)

                # Configure basis parameters
                siesta_updates = {
                    "PAO.BasisSize": self.basis_size,
                    "PAO.BasisType": "split",
                    "PAO.EnergyShift": f"{energy_shift} Ry",
                    "PAO.SplitNorm": split_norm,
                    "a2s_kpts": self.kpts,
                }

                maker = cast(
                    "Maker",
                    update_user_siesta_settings(
                        maker, siesta_updates, class_filter=StaticMaker
                    ),
                )

                # Create job with counter
                scf_job = maker.make(structure, prev_dir=None)
                job_label = f"ES{energy_shift:.3f} SN{split_norm:.2f}".replace(".", "p")
                scf_job.name = (
                    f"{self.name}_{job_label}_[{job_counter}_of_{total_jobs}]"
                )

                job_metadata.append(
                    {
                        "uuid": scf_job.uuid,
                        "name": scf_job.name,
                        "energy_shift": energy_shift,
                        "split_norm": split_norm,
                    }
                )

                jobs.append(scf_job)

        logger.info(f"Created {len(jobs)} jobs for basis parameter convergence")

        # Create flow from parallel SCF jobs
        scf_flow = Flow(
            jobs,
            name="Basis Parameters Scan",
            output={job.uuid: job.output for job in jobs},
        )

        # Create data collection job
        collect_job = collect_basis_params_data(
            flow_results=scf_flow.output,
            job_metadata=job_metadata,
        )
        collect_job.name = f"{self.name}-collect"

        # Create summary writing job
        summary_job = write_basis_params_summary(basis_params_data=collect_job.output)
        summary_job.name = f"{self.name}-summary"

        # Create convergence plot
        plot_job = plot_basis_params_convergence(basis_params_data=collect_job.output)
        plot_job.name = f"{self.name}-plot"

        # Create basis function visualization (schematic - inspired by
        # plot_siesta_basis.py)
        basis_viz_job = plot_basis_functions(
            flow_results=scf_flow.output,
            job_metadata=job_metadata,
        )
        basis_viz_job.name = f"{self.name}-basis-viz"

        # Create real basis function plot from ion.xml files (like plot_siesta_basis.py)
        real_basis_job = plot_real_basis_functions(
            flow_results=scf_flow.output,
            job_metadata=job_metadata,
        )
        real_basis_job.name = f"{self.name}-real-basis"

        # Combine into final flow
        return Flow(
            [
                scf_flow,
                collect_job,
                summary_job,
                plot_job,
                basis_viz_job,
                real_basis_job,
            ],
            output=collect_job.output,
            name=self.name,
        )
