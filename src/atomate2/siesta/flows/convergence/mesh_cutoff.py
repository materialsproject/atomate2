"""Mesh cutoff convergence workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from jobflow.core.flow import Flow
from pymatgen.core import Molecule, Structure

from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.flows.convergence.utils import (
    collect_convergence_data,
    plot_convergence,
)
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.powerups import update_user_siesta_settings
from atomate2.siesta.utils.common import print_docstring_in_box
from atomate2.siesta.utils.verbosity import VerbosityLevel

if TYPE_CHECKING:
    from atomate2.siesta.jobs.base import BaseSiestaMaker


@dataclass
class MeshCutoffConvergenceFlowMaker(BaseSiestaFlowMaker):
    """
    Workflow for converging mesh cutoff parameter.

    This workflow:
    1. Runs static calculations with different mesh cutoff values in parallel
    2. Collects the total energies from each calculation
    3. Plots convergence (absolute energies and energy differences)

    The mesh cutoff determines the fineness of the real-space grid used in SIESTA.
    Typical convergence criterion: energy difference < 1-5 meV.

    Inherits from BaseSiestaFlowMaker, so dry_run=True automatically propagates
    to the static_maker.

    Parameters
    ----------
    name : str
        Name of the workflow.
    static_maker : BaseSiestaMaker
        Maker for static calculations.
    mesh_cutoffs : list[float]
        List of mesh cutoff values to test (in Ry).
    dry_run : bool
        If True, skip SIESTA calculations and only save structures (inherited).
    dry_run_output_dir : str
        Directory to save dry-run structures (inherited).
    dry_run_format : str
        Output format for dry-run structures (inherited).

    Example:
        >>> from atomate2.siesta.flows.convergence import MeshCutoffConvergenceFlowMaker
        >>> from pymatgen.core import Structure
        >>> structure = Structure.from_file("structure.cif")
        >>> maker = MeshCutoffConvergenceFlowMaker(
        ...     mesh_cutoffs=[100, 150, 200, 250, 300, 350, 400]
        ... )
        >>> flow = maker.make(structure)
    """

    CONSOLE_VERBOSITY: VerbosityLevel = VerbosityLevel.INFO
    name: str = "Mesh Cutoff Convergence"
    static_maker: BaseSiestaMaker = field(default_factory=lambda: StaticMaker())
    mesh_cutoffs: list[float] = field(
        default_factory=lambda: [100, 150, 200, 250, 300, 350, 400, 450, 500]
    )  # in Ry

    def make(
        self, structure: Structure | Molecule, prev_dir: str | Path | None = None
    ) -> Flow:
        """
        Create convergence flow for mesh cutoff.

        Args:
            structure: Structure to calculate
            prev_dir: Previous directory (optional)

        Returns
        -------
            Flow with mesh cutoff convergence jobs
        """
        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        jobs = []
        job_metadata = []

        for mesh_cutoff in self.mesh_cutoffs:
            # Create static job with specific mesh cutoff
            static_job = self.static_maker.make(structure, prev_dir=prev_dir)

            # Update mesh cutoff parameter (value in Ry)
            static_job = update_user_siesta_settings(
                static_job, {"Mesh.Cutoff": f"{mesh_cutoff} Ry"}
            )

            # Set job name
            static_job.name = f"{self.name}-{mesh_cutoff}Ry"

            jobs.append(static_job)
            job_metadata.append({"name": static_job.name, "uuid": static_job.uuid})

        # Create a flow from the parallel jobs
        calc_flow = Flow(
            jobs,
            name="Mesh Cutoff Calculations",
            output={job.uuid: job.output for job in jobs},
        )

        # Create collection job
        collect_job = collect_convergence_data(
            flow_results=calc_flow.output,
            job_metadata=job_metadata,
            parameter_name="mesh_cutoff",
        )
        collect_job.name = f"{self.name}-collect"

        # Create plotting job
        plot_job = plot_convergence(
            convergence_data=collect_job.output,
            parameter_name="mesh_cutoff",
            verbosity=self.CONSOLE_VERBOSITY,
        )
        plot_job.name = f"{self.name}-plot"

        # Combine into final flow
        return Flow([calc_flow, collect_job, plot_job], output=collect_job.output)
