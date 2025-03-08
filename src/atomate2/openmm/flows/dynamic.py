"""Dynamic flows for OpenMM module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from emmet.core.openmm import Calculation, OpenMMInterchange, OpenMMTaskDocument
from jobflow import CURRENT_JOB, Flow, Job, Maker, Response, job
from scipy.signal import savgol_filter

from atomate2.openmm.flows.core import _get_calcs_reversed, collect_outputs
from atomate2.openmm.jobs.base import BaseOpenMMMaker, openmm_job

if TYPE_CHECKING:
    from collections.abc import Callable

    from openff.interchange import Interchange

    from atomate2.openmm.flows.core import OpenMMFlowMaker


def _get_final_jobs(input_jobs: list[Job] | Flow) -> list[Job]:
    """Unwrap nested jobs from a dynamic flow."""
    jobs = input_jobs.jobs if isinstance(input_jobs, Flow) else input_jobs
    if not jobs:
        return []

    # check if the last job is a flow with .maker.jobs
    last = jobs[-1]
    if (
        hasattr(last, "maker")
        and hasattr(last.maker, "jobs")
        and isinstance(last.maker.jobs, list)
    ):
        # recursively explore .maker.jobs
        return _get_final_jobs(last.maker.jobs)
    return jobs


@openmm_job
def default_should_continue(
    task_docs: list[OpenMMTaskDocument],
    stage_index: int,
    max_stages: int,
    physical_property: str = "potential_energy",
    target: float | None = None,
    threshold: float = 1e-3,
    burn_in_ratio: float = 0.2,
) -> Response:
    """Decide dynamic flow logic (True).

    This serves as a template for any bespoke "should_continue" functions
    written by the user. By default, simulation logic depends on stability
    of potential energy as a function of time, dU_dt.
    """
    task_doc = task_docs[-1]

    # get key physical parameters from calculation list
    potential_energy: list[float] = []
    density: list[float] = []
    for doc in task_docs:
        potential_energy.extend(doc.calcs_reversed[0].output.potential_energy)
        density.extend(doc.calcs_reversed[0].output.density)
    dt = doc.calcs_reversed[0].input.state_interval

    if physical_property == "density":
        values = np.array(density)
    elif physical_property == "potential_energy":
        values = np.array(potential_energy)

    # toss out first X% of values, default 20%
    burn_in = int(burn_in_ratio * len(values))
    values = values[burn_in:]
    window_length = max(5, burn_in + 1) if burn_in % 2 == 0 else max(5, burn_in)

    avg = np.mean(values)
    dvalue_dt = savgol_filter(
        values / avg, window_length, polyorder=3, deriv=1, delta=dt
    )
    decay_rate = np.max(np.abs(dvalue_dt))
    job = CURRENT_JOB.job

    if target:
        delta = np.abs((avg - target) / target)
        should_continue = not delta < threshold
        job.append_name(
            f" [Stage {stage_index}, delta={delta:.3e}"
            f"-> should_continue={should_continue}]"
        )
    elif stage_index > max_stages or decay_rate < threshold:  # max_stages exceeded
        should_continue = False
    else:  # decay_rate not stable
        should_continue = True

    job.append_name(
        f" [Stage {stage_index}, decay_rate={decay_rate:.3e}"
        f"-> should_continue={should_continue}]"
    )

    task_doc.should_continue = should_continue
    return Response(output=task_doc)


@dataclass
class DynamicOpenMMFlowMaker(Maker):
    """Run a dynamic equlibration or production simulation.

    Create a dynamic flow out of an existing OpenMM simulation
    job or a linear sequence of linked jobs, i.e., an OpenMM
    flow.

    Attributes
    ----------
    name : str
        The name of the dynamic OpenMM job or flow. Default is the name
        of the inherited maker name with "dynamic" prepended.
    tags : list[str]
        Tags to apply to the final job. Will only be applied if collect_jobs is True.
    maker: Union[BaseOpenMMMaker, OpenMMFlowMaker]
        A single (either job or flow) maker to make dynamic.
    max_stages: int
        Maximum number of stages to run consecutively before terminating
        dynamic flow logic.
    collect_outputs : bool
        If True, a final job is added that collects all jobs into a single
        task document.
    should_continue: Callable
        A general function for evaluating properties in `calcs_reversed`
        to determine simulation flow logic (i.e., termination, pausing,
        or continuing).
    jobs: list[BaseOpenMMMaker | OpenMMFlowMaker]
        A running list of jobs in simulation flow.
    job_uuids: list
        A running list of job uuids in simulation flow.
    calcs_reversed: list[Calculation]
        A running list of Calculations in simulation flow.
    """

    name: str = field(default=None)
    tags: list[str] = field(default_factory=list)
    maker: BaseOpenMMMaker | OpenMMFlowMaker = field(
        default_factory=lambda: BaseOpenMMMaker()
    )
    max_stages: int = field(default=5)
    collect_outputs: bool = True
    (
        list[OpenMMTaskDocument],
        int,
        int,
        str,
        float | None,
        float,
        float,
    )
    should_continue: Callable[
        [list[OpenMMTaskDocument], int, int, str, float | None, float, str], Response
    ] = field(default_factory=lambda: default_should_continue)

    jobs: list = field(default_factory=list)
    job_uuids: list = field(default_factory=list)
    calcs_reversed: list[Calculation] = field(default_factory=list)
    stage_task_type: str = "collect"

    def __post_init__(self) -> None:
        """Post init formatting of arguments."""
        if self.name is None:
            self.name = f"dynamic {self.maker.name}"

    def make(
        self,
        interchange: Interchange | OpenMMInterchange | str,
        prev_dir: str | None = None,
    ) -> Flow:
        """Run the dynamic simulation using the provided Interchange object.

        Parameters
        ----------
        interchange : Interchange
            The Interchange object containing the system
            to simulate.
        prev_task : Optional[ClassicalMDTaskDocument]
            The directory of the previous task.

        Returns
        -------
        Flow
            A Flow object containing the OpenMM jobs for the simulation.
        """
        # Run initial stage job
        stage_job_0 = self.maker.make(
            interchange=interchange,
            prev_dir=prev_dir,
        )
        self.jobs.append(stage_job_0)

        # collect the uuids and calcs for the final collect job
        if isinstance(stage_job_0, Flow):
            self.job_uuids.extend(stage_job_0.job_uuids)
        else:
            self.job_uuids.append(stage_job_0.uuid)
        self.calcs_reversed.append(_get_calcs_reversed(stage_job_0))

        # Determine stage job control logic
        control_stage_0 = self.should_continue(
            task_docs=[stage_job_0.output],
            stage_index=0,
            max_stages=self.max_stages,
        )
        self.jobs.append(control_stage_0)

        stage_n = self.dynamic_flow(
            prev_stage_index=0,
            prev_docs=[control_stage_0.output],
        )
        self.jobs.append(stage_n)
        return Flow([stage_job_0, control_stage_0, stage_n], output=stage_n.output)

    @job
    def dynamic_flow(
        self,
        prev_stage_index: int,
        prev_docs: list[OpenMMTaskDocument],
    ) -> Response | None:
        """Run stage n and dynamically decide to continue or terminate flow."""
        prev_stage = prev_docs[-1]

        # stage control logic: (a) begin, (b) continue, (c) terminate, (d) pause
        if (
            prev_stage_index >= self.max_stages or not prev_stage.should_continue
        ):  # pause
            if self.collect_outputs:
                collect_job = collect_outputs(
                    prev_stage.dir_name,
                    tags=self.tags or None,
                    job_uuids=self.job_uuids,
                    calcs_reversed=self.calcs_reversed,
                    task_type=self.stage_task_type,
                )
                return Response(replace=collect_job, output=collect_job.output)
            return Response(output=prev_stage)

        stage_index = prev_stage_index + 1

        stage_job_n = self.maker.make(
            interchange=prev_stage.interchange,
            prev_dir=prev_stage.dir_name,
        )
        self.jobs.append(stage_job_n)

        # collect the uuids and calcs for the final collect job
        if isinstance(stage_job_n, Flow):
            self.job_uuids.extend(stage_job_n.job_uuids)
        else:
            self.job_uuids.append(stage_job_n.uuid)
        self.calcs_reversed.append(_get_calcs_reversed(stage_job_n))

        control_stage_n = self.should_continue(
            task_docs=[*prev_docs, stage_job_n.output],
            stage_index=stage_index,
            max_stages=self.max_stages,
        )
        self.jobs.append(control_stage_n)

        next_stage_n = self.dynamic_flow(
            prev_stage_index=stage_index,
            prev_docs=[*prev_docs, control_stage_n.output],
        )
        self.jobs.append(next_stage_n)
        stage_n_flow = Flow([stage_job_n, control_stage_n, next_stage_n])

        return Response(replace=stage_n_flow, output=next_stage_n.output)
