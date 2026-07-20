"""K-points convergence workflow."""

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
class KpointsConvergenceFlowMaker(BaseSiestaFlowMaker):
    """
    Workflow for converging k-points sampling.

    This workflow:
    1. Runs static calculations with different k-point grids in parallel
    2. Collects the total energies from each calculation
    3. Plots convergence (absolute energies and energy differences)

    K-points determine the sampling of the Brillouin zone.
    Typical convergence criterion: energy difference < 1-5 meV.

    Inherits from BaseSiestaFlowMaker, so dry_run=True automatically propagates
    to the static_maker.

    Parameters
    ----------
    name : str
        Name of the workflow.
    static_maker : BaseSiestaMaker
        Maker for static calculations.
    kpoints_list : list[list[int]]
        List of k-point grids to test (e.g., [[2,2,2], [4,4,4]]).
    dry_run : bool
        If True, skip SIESTA calculations and only save structures (inherited).
    dry_run_output_dir : str
        Directory to save dry-run structures (inherited).
    dry_run_format : str
        Output format for dry-run structures (inherited).

    Example:
        >>> from atomate2.siesta.flows.convergence import KpointsConvergenceFlowMaker
        >>> from pymatgen.core import Structure
        >>> structure = Structure.from_file("structure.cif")
        >>> maker = KpointsConvergenceFlowMaker(
        ...     kpoints_list=[[2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8], [10, 10, 10]]
        ... )
        >>> flow = maker.make(structure)
    """

    CONSOLE_VERBOSITY: VerbosityLevel = VerbosityLevel.INFO
    name: str = "K-points Convergence"
    static_maker: BaseSiestaMaker = field(default_factory=lambda: StaticMaker())
    kpoints_list: list[list[int]] = field(
        default_factory=lambda: [
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
            [8, 8, 8],
            [10, 10, 10],
        ]
    )

    def make(
        self, structure: Structure | Molecule, prev_dir: str | Path | None = None
    ) -> Flow:
        """
        Create convergence flow for k-points.

        Args:
            structure: Structure to calculate
            prev_dir: Previous directory (optional)

        Returns
        -------
            Flow with k-points convergence jobs
        """
        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        jobs = []
        job_metadata = []

        for kpoints in self.kpoints_list:
            # Create static job with specific k-points
            static_job = self.static_maker.make(structure, prev_dir=prev_dir)

            # Update kpoints parameter
            static_job = update_user_siesta_settings(static_job, {"a2s_kpts": kpoints})

            # Set job name
            kpts_str = "x".join(map(str, kpoints))
            static_job.name = f"{self.name}-{kpts_str}"

            jobs.append(static_job)
            job_metadata.append({"name": static_job.name, "uuid": static_job.uuid})

        # Create a flow from the parallel jobs
        calc_flow = Flow(
            jobs,
            name="K-points Calculations",
            output={job.uuid: job.output for job in jobs},
        )

        # Create collection job
        collect_job = collect_convergence_data(
            flow_results=calc_flow.output,
            job_metadata=job_metadata,
            parameter_name="kpoints",
        )
        collect_job.name = f"{self.name}-collect"

        # Create plotting job
        plot_job = plot_convergence(
            convergence_data=collect_job.output,
            parameter_name="kpoints",
            verbosity=self.CONSOLE_VERBOSITY,
        )
        plot_job.name = f"{self.name}-plot"

        # Combine into final flow
        return Flow([calc_flow, collect_job, plot_job], output=collect_job.output)
