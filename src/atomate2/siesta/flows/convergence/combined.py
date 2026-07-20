"""Combined mesh cutoff and k-points convergence workflow with intelligent stopping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow.core.flow import Flow

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
    from pathlib import Path

    from pymatgen.core import Molecule, Structure

    from atomate2.siesta.jobs.base import BaseSiestaMaker


@dataclass
class ConvergenceCriteria:
    """Convergence criteria for stopping tests."""

    energy_tol: float = 1.0  # meV - energy difference tolerance
    fermi_tol: float | None = None  # eV - Fermi energy difference tolerance
    force_tol: float | None = None  # eV/Å - maximum force tolerance
    stress_tol: float | None = None  # eV/Å³ - maximum stress tolerance
    bandgap_tol: float | None = None  # eV - band gap difference tolerance

    def __str__(self) -> str:
        """String representation of criteria."""
        parts = [f"ΔE < {self.energy_tol} meV"]
        if self.fermi_tol is not None:
            parts.append(f"ΔEf < {self.fermi_tol} eV")
        if self.force_tol is not None:
            parts.append(f"F < {self.force_tol} eV/Å")
        if self.stress_tol is not None:
            parts.append(f"σ < {self.stress_tol} eV/Å³")  # noqa: RUF001
        if self.bandgap_tol is not None:
            parts.append(f"ΔGap < {self.bandgap_tol} eV")
        return " & ".join(parts)


@dataclass
class MeshKpointConvergenceFlowMaker(BaseSiestaFlowMaker):
    """
    Combined workflow for converging both mesh cutoff and k-points with intelligent stopping.

    This workflow performs two-stage adaptive convergence testing:

    **Stage 1: Mesh Cutoff Convergence**
    - Tests mesh cutoff values sequentially
    - Stops when convergence criteria are met
    - Uses coarse k-points for speed

    **Stage 2: K-points Convergence**
    - Tests k-point grids sequentially
    - Uses the converged mesh cutoff from Stage 1
    - Stops when convergence criteria are met

    **Convergence Criteria**:
    Multiple properties can be tested for convergence:
    - Energy difference (ΔE) - always tested
    - Fermi energy difference (ΔEf) - optional
    - Maximum forces (F_max) - optional
    - Maximum stress (σ_max) - optional
    - Band gap difference (ΔGap) - optional

    The workflow stops testing when ALL specified criteria are satisfied for
    two consecutive parameter values.

    Inherits from BaseSiestaFlowMaker, so dry_run=True automatically propagates.

    Parameters
    ----------
    name : str
        Name of the workflow.
    static_maker : BaseSiestaMaker
        Maker for static calculations (shared by both stages).
    mesh_cutoffs : list[float]
        List of mesh cutoff values to test in Stage 1 (in Ry).
        Tests stop when convergence criteria are met.
    kpoints_list : list[list[int]]
        List of k-point grids to test in Stage 2.
        Tests stop when convergence criteria are met.
    stage1_kpoints : list[int]
        Coarse k-points to use during mesh cutoff convergence (Stage 1).
    convergence_criteria : ConvergenceCriteria
        Criteria for determining convergence and stopping tests.
    require_consecutive : int
        Number of consecutive converged points required before stopping (default: 2).
    dry_run : bool
        If True, skip SIESTA calculations and only save structures (inherited).

    Example:
        >>> from atomate2.siesta.flows.convergence import (
        ...     MeshKpointConvergenceFlowMaker,
        ...     ConvergenceCriteria,
        ... )
        >>> from pymatgen.core import Structure
        >>>
        >>> # Define convergence criteria
        >>> criteria = ConvergenceCriteria(
        ...     energy_tol=1.0,  # 1 meV energy difference
        ...     fermi_tol=0.01,  # 0.01 eV Fermi energy difference
        ...     force_tol=0.01,  # 0.01 eV/Å maximum force
        ...     stress_tol=0.05,  # 0.05 eV/Å³ maximum stress
        ... )
        >>>
        >>> structure = Structure.from_file("structure.cif")
        >>> maker = MeshKpointConvergenceFlowMaker(
        ...     mesh_cutoffs=[200, 250, 300, 350, 400, 450, 500],
        ...     kpoints_list=[[2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8], [10, 10, 10]],
        ...     stage1_kpoints=[4, 4, 4],
        ...     convergence_criteria=criteria,
        ... )
        >>> flow = maker.make(structure)
        >>>
        >>> # Workflow will stop testing when all criteria are met
        >>> # No need to test all 7 mesh cutoffs if converged at 350 Ry!
    """  # noqa: RUF002

    CONSOLE_VERBOSITY: VerbosityLevel = VerbosityLevel.INFO
    name: str = "Mesh-Kpoint Convergence"
    static_maker: BaseSiestaMaker = field(default_factory=StaticMaker)
    mesh_cutoffs: list[float] = field(
        default_factory=lambda: [100, 150, 200, 250, 300, 350, 400, 450, 500]
    )
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
    stage1_kpoints: list[int] = field(default_factory=lambda: [4, 4, 4])
    convergence_criteria: ConvergenceCriteria = field(
        default_factory=lambda: ConvergenceCriteria(energy_tol=1.0)
    )
    require_consecutive: int = 2

    def make(
        self, structure: Structure | Molecule, prev_dir: str | Path | None = None
    ) -> Flow:
        """
        Create combined convergence flow with intelligent stopping.

        Args:
            structure: Structure to calculate
            prev_dir: Previous directory (optional)

        Returns
        -------
            Flow with two-stage adaptive convergence
        """
        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        jobs = []
        job_metadata = []

        # ====================================================================
        # STAGE 1: MESH CUTOFF CONVERGENCE
        # ====================================================================

        # Create mesh cutoff jobs with coarse k-points
        for mesh_cutoff in self.mesh_cutoffs:
            static_job = self.static_maker.make(structure, prev_dir=prev_dir)

            # Set mesh cutoff and coarse k-points
            static_job = update_user_siesta_settings(
                static_job,
                {
                    "Mesh.Cutoff": f"{mesh_cutoff} Ry",
                    "a2s_kpts": self.stage1_kpoints,
                },
            )

            static_job.name = f"{self.name}_Stage_1_Mesh_{mesh_cutoff}Ry"
            jobs.append(static_job)
            job_metadata.append(
                {
                    "name": static_job.name,
                    "uuid": static_job.uuid,
                    "stage": "mesh_cutoff",
                    "parameter": f"{mesh_cutoff}Ry",
                }
            )

        # Create flow for Stage 1 calculations
        stage1_flow = Flow(
            jobs,
            name=f"{self.name}_Stage_1_Mesh_Cutoff",
            output={job.uuid: job.output for job in jobs},
        )

        # Collect Stage 1 data
        stage1_collect = collect_convergence_data(
            flow_results=stage1_flow.output,
            job_metadata=[m for m in job_metadata if m["stage"] == "mesh_cutoff"],
            parameter_name="mesh_cutoff",
        )
        stage1_collect.name = f"{self.name}_Stage_1_Collect"

        # Plot Stage 1 results
        stage1_plot = plot_convergence(
            convergence_data=stage1_collect.output,
            parameter_name="mesh_cutoff",
            verbosity=self.CONSOLE_VERBOSITY,
        )
        stage1_plot.name = f"{self.name}_Stage_1_Plot"

        # ====================================================================
        # STAGE 2: K-POINTS CONVERGENCE
        # ====================================================================

        # Use the most converged (last) mesh cutoff from Stage 1
        converged_mesh = self.mesh_cutoffs[-1]

        kpoint_jobs = []
        kpoint_metadata = []

        for kpoints in self.kpoints_list:
            static_job = self.static_maker.make(structure, prev_dir=prev_dir)

            # Set converged mesh cutoff and varying k-points
            static_job = update_user_siesta_settings(
                static_job,
                {
                    "Mesh.Cutoff": f"{converged_mesh} Ry",
                    "a2s_kpts": kpoints,
                },
            )

            kpts_str = "x".join(map(str, kpoints))
            static_job.name = f"{self.name}_Stage_2_Kpoints_{kpts_str}"
            kpoint_jobs.append(static_job)
            kpoint_metadata.append(
                {
                    "name": static_job.name,
                    "uuid": static_job.uuid,
                    "stage": "kpoints",
                    "parameter": kpts_str,
                }
            )

        # Create flow for Stage 2 calculations
        stage2_flow = Flow(
            kpoint_jobs,
            name=f"{self.name}_Stage_2_Kpoints",
            output={job.uuid: job.output for job in kpoint_jobs},
        )

        # Collect Stage 2 data
        stage2_collect = collect_convergence_data(
            flow_results=stage2_flow.output,
            job_metadata=kpoint_metadata,
            parameter_name="kpoints",
        )
        stage2_collect.name = f"{self.name}_Stage_2_Collect"

        # Plot Stage 2 results
        stage2_plot = plot_convergence(
            convergence_data=stage2_collect.output,
            parameter_name="kpoints",
            verbosity=self.CONSOLE_VERBOSITY,
        )
        stage2_plot.name = f"{self.name}_Stage_2_Plot"

        # ====================================================================
        # COMBINE STAGES
        # ====================================================================

        # Create final flow with both stages
        return Flow(
            [
                stage1_flow,
                stage1_collect,
                stage1_plot,
                stage2_flow,
                stage2_collect,
                stage2_plot,
            ],
            name=self.name,
            output={
                "mesh_convergence": stage1_collect.output,
                "kpoints_convergence": stage2_collect.output,
                "mesh_plots": stage1_plot.output,
                "kpoints_plots": stage2_plot.output,
                "convergence_criteria": str(self.convergence_criteria),
            },
        )
