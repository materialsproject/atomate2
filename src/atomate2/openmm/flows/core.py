"""Core flows for OpenMM module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from emmet.core.openmm import Calculation, OpenMMTaskDocument
from jobflow import Flow, Job, Response

from atomate2.openff.core import openff_job
from atomate2.openff.utils import create_list_summing_to
from atomate2.openmm.jobs.core import NVTMaker, TempChangeMaker

if TYPE_CHECKING:
    from emmet.core.openff import ClassicalMDTaskDocument
    from openff.interchange import Interchange

    from atomate2.openmm.jobs.base import BaseOpenMMMaker


def _get_calcs_reversed(job: Job | Flow) -> list[Calculation | list]:
    """Unwrap a nested list of calcs from jobs or flows."""
    if isinstance(job, Flow):
        return [_get_calcs_reversed(sub_job) for sub_job in job.jobs]
    return job.output.calcs_reversed


def _flatten_calcs(nested_calcs: list) -> list[Calculation]:
    """Flattening nested calcs."""
    flattened = []
    for item in nested_calcs:
        if isinstance(item, list):
            flattened.extend(_flatten_calcs(item))
        else:
            flattened.append(item)
    return flattened


@dataclass
class OpenMMFlowMaker:
    """Run a production simulation.

    This flexible flow links together any flows of OpenMM jobs in
    a linear sequence

    Attributes
    ----------
    name : str
        The name of the production job. Default is "production".
    tags : list[str]
        Tags to apply to the final job. Will only be applied if collect_jobs is True.
    makers: list[BaseOpenMMMaker]
        A list of makers to string together.
    collect_jobs : bool
        If True, a final job is added that collects all jobs into a single
        task document.
    """

    name: str = "flexible"
    tags: list[str] = field(default_factory=list)
    makers: list[BaseOpenMMMaker | OpenMMFlowMaker] = field(default_factory=list)
    collect_jobs: bool = True

    def make(
        self,
        interchange: Interchange | bytes,
        prev_task: ClassicalMDTaskDocument | None = None,
    ) -> Flow:
        """Run the production simulation using the provided Interchange object.

        Parameters
        ----------
        interchange : Interchange
            The Interchange object containing the system
            to simulate.
        prev_task : Optional[ClassicalMDTaskDocument]
            The previous task to use as
            a starting point for the production simulation.
        output_dir : Optional[Union[str, Path]]
            The directory to write reporter files to.

        Returns
        -------
        Flow
            A Flow object containing the OpenMM jobs for the simulation.
        """
        if len(self.makers) == 0:
            raise ValueError("At least one maker must be included")

        jobs: list = []
        job_uuids: list = []
        calcs_reversed = []
        for maker in self.makers:
            job = maker.make(
                interchange=interchange,
                prev_task=prev_task,
            )
            interchange = job.output.interchange
            prev_task = job.output
            jobs.append(job)

            if isinstance(job, Flow):
                job_uuids.extend(job.job_uuids)
            else:
                job_uuids.append(job.uuid)

            calcs_reversed.append(_get_calcs_reversed(job))

        if self.collect_jobs:

            @openff_job
            def organize_flow_output(**kwargs) -> Response:
                if "calcs_reversed" in kwargs:
                    # this must be done here because we cannot unwrap the calcs
                    # when they are an output reference
                    kwargs["calcs_reversed"] = _flatten_calcs(kwargs["calcs_reversed"])
                    kwargs["calcs_reversed"].reverse()

                task_doc = OpenMMTaskDocument(**kwargs)

                with open(Path(task_doc.dir_name) / "taskdoc.json", "w") as file:
                    file.write(task_doc.json())

                return Response(output=task_doc)

            final_collect = organize_flow_output(
                tags=self.tags or None,
                dir_name=prev_task.dir_name,
                state=prev_task.state,
                job_uuids=job_uuids,
                calcs_reversed=calcs_reversed,
                interchange=prev_task.interchange,
                molecule_specs=prev_task.molecule_specs,
                force_field=prev_task.force_field,
                task_type="collect",
                last_updated=prev_task.last_updated,
            )
            jobs.append(final_collect)

            return Flow(
                jobs,
                output=final_collect.output,
            )
        return Flow(
            jobs,
            output=prev_task,
        )

    @classmethod
    def anneal_flow(
        cls,
        name: str = "anneal",
        tags: list[str] | None = None,
        anneal_temp: int = 400,
        final_temp: int = 298,
        n_steps: int | tuple[int, int, int] = 1500000,
        temp_steps: int | tuple[int, int, int] | None = None,
        job_names: tuple[str, str, str] = ("raise temp", "hold temp", "lower temp"),
        **kwargs,
    ) -> OpenMMFlowMaker:
        """Create an AnnealMaker from the specified temperatures, steps, and job names.

        Parameters
        ----------
        name : str, optional
            The name of the annealing job. Default is "anneal".
        tags : list[str], optional
            Tags to apply to the final job.
        anneal_temp : int, optional
            The annealing temperature. Default is 400.
        final_temp : int, optional
            The final temperature after annealing. Default is 298.
        n_steps : int or Tuple[int, int, int], optional
            The number of steps for each stage of annealing.
            If an integer is provided, it will be divided into three equal parts.
            If a tuple of three integers is provided, each value represents the
            steps for the corresponding stage. Default is 1500000.
        temp_steps : int or Tuple[int, int, int], optional
            The number of temperature steps for raising and
            lowering the temperature. If an integer is provided, it will be used
            for both stages. If a tuple of three integers is provided, each value
            represents the temperature steps for the corresponding stage.
            Default is None and all jobs will automatically determine temp_steps.
        job_names : Tuple[str, str, str], optional
            The names for the jobs in each stage of annealing.
            Default is ("raise temp", "hold temp", "lower temp").
        **kwargs
            Additional keyword arguments to be passed to the job makers.

        Returns
        -------
        AnnealMaker
            An AnnealMaker instance with the specified parameters.
        """
        if isinstance(n_steps, int):
            n_steps = tuple(create_list_summing_to(n_steps, 3))
        if isinstance(temp_steps, int) or temp_steps is None:
            temp_steps = (temp_steps, temp_steps, temp_steps)

        raise_temp_maker = TempChangeMaker(
            n_steps=n_steps[0],
            name=job_names[0],
            temperature=anneal_temp,
            temp_steps=temp_steps[0],
            **kwargs,
        )
        nvt_maker = NVTMaker(
            n_steps=n_steps[1], name=job_names[1], temperature=anneal_temp, **kwargs
        )
        lower_temp_maker = TempChangeMaker(
            n_steps=n_steps[2],
            name=job_names[2],
            temperature=final_temp,
            temp_steps=temp_steps[2],
            **kwargs,
        )
        return cls(
            name=name,
            tags=tags,
            makers=[raise_temp_maker, nvt_maker, lower_temp_maker],
            collect_jobs=False,
        )
