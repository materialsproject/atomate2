"""Core flows for OpenMM module."""

from __future__ import annotations

import json
import numpy as np
from scipy.signal import savgol_filter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from emmet.core.openmm import Calculation, OpenMMInterchange, OpenMMTaskDocument
from jobflow import Flow, Job, Maker, Response, job
from monty.json import MontyDecoder, MontyEncoder

from atomate2.openmm.jobs.base import BaseOpenMMMaker, openmm_job
from atomate2.openmm.jobs.core import NVTMaker, TempChangeMaker
from atomate2.openmm.utils import create_list_summing_to

if TYPE_CHECKING:
    from openff.interchange import Interchange

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


@openmm_job
def collect_outputs(
    prev_dir: str,
    tags: list[str] | None,
    job_uuids: list[str],
    calcs_reversed: list[Calculation | list],
    task_type: str,
) -> Response:
    """Reformat the output of the OpenMMFlowMaker into a OpenMMTaskDocument."""
    with open(Path(prev_dir) / "taskdoc.json") as file:
        task_dict = json.load(file, cls=MontyDecoder)
        task_doc = OpenMMTaskDocument.model_validate(task_dict)

    # this must be done here because we cannot unwrap the calcs
    # when they are an output reference
    calcs = _flatten_calcs(calcs_reversed)
    calcs.reverse()
    task_doc.calcs_reversed = calcs
    task_doc.tags = tags
    task_doc.job_uuids = job_uuids
    task_doc.task_type = task_type

    with open(Path(task_doc.dir_name) / "taskdoc.json", "w") as file:
        json.dump(task_doc.model_dump(), file, cls=MontyEncoder)

    return Response(output=task_doc)

@openmm_job
def default_should_continue(
    task_docs: list[OpenMMTaskDocument], 
    stage_index: int,
    max_stages: int,
    physical_property: str = "potential_energy", 
    target: float | None = None,
    threshold: float = 1e-3,
) -> Response:
    """Default dynamic flow logic for deciding to continue flow (True)
        
    This serves as a template for any bespoke "should_continue" functions
    written by the user. By default, simulation logic depends on stability 
    of potential energy as a function of time, dU_dt. 
    """
    print(f"%%%%% Running should_continue for stage index {stage_index} %%%%%")
    task_doc = task_docs[-1]
    setattr(task_doc, "stage_index", stage_index)
    print(f"task_doc.stage_index = {stage_index}")
    print(f"there are {len(task_docs)} task_docs")

    potential_energy: list[float] = []
    density: list[float] = []
    for doc in task_docs:
        potential_energy.extend(doc.calcs_reversed[0].output.potential_energy)
        density.extend(doc.calcs_reversed[0].output.density)
    dt = doc.calcs_reversed[0].input.state_interval
    print(f"total number of energies: {len(potential_energy)}")
    
    # toss out first 20% of values
    burn_in = int(0.2 * len(potential_energy))
    if physical_property == "density":
        values = np.array(density[burn_in:])
    elif physical_property == "potential_energy":
        values = np.array(potential_energy[burn_in:])
             
    if int(0.2 * len(values)) % 2 == 0:
        window_length = max(5, int(0.2 * len(values)) + 1)
    else:
        window_length = max(5, int(0.2 * len(values)))
    print(f"window_length={window_length}")
    dValue_dt = savgol_filter(values, window_length, polyorder=3, deriv=1, delta=dt)
    decay_rate = np.max(np.abs(dValue_dt))
    avg = np.mean(values)
    std = np.std(values)
    fluctuations = std / np.abs(avg)
    
    print(f"checking {physical_property}")
    print(f"Mean of last 80% energies: {avg:.5f}")
    print(f"Std of last 80% energies: {std:.5f}")
    print(f"Fluctuations: {fluctuations:.5f}")
    print(f"N^(1/2) ~ {len(values)**0.5}")
    print(f"decay_rate: {decay_rate:.5f}")
    if target: 
        delta = np.abs( (avg - target) / target )
        if delta < threshold:
            print(f"{physical_property} within {threshold} threshold")
            should_continue = False
            print(f"should_continue={should_continue}")
        else:
            print(f"{physical_property} error at {delta*100} percent")
            should_continue = True
            print(f"should_continue={should_continue}")
    elif stage_index > max_stages:
        print(f"MAX STAGES REACHED!!")
        should_continue=False
        print(f"should_continue={should_continue}")
    elif decay_rate < threshold:
        print(f"EQUILIBRIUM REACHED!!")
        should_continue=False
        print(f"should_continue={should_continue}")
    else:
        print(f"decay_rate = {decay_rate}")
        should_continue=True
        print(f"should_continue={should_continue}")
    
    if stage_index < max_stages:
        should_continue=True 
    else:
        should_continue=False
    setattr(task_doc, "should_continue", should_continue)
    return Response(output=task_doc)

@dataclass
class OpenMMFlowMaker(Maker):
    """Run a production simulation.

    This flexible flow links together any flows of OpenMM jobs in
    a linear sequence.

    Attributes
    ----------
    name : str
        The name of the production job. Default is "production".
    tags : list[str]
        Tags to apply to the final job. Will only be applied if collect_jobs is True.
    makers: Union[BaseOpenMMMaker, OpenMMFlowMaker]
        A list of makers to string together.
    collect_outputs : bool
        If True, a final job is added that collects all jobs into a single
        task document.
    """

    name: str = "flexible"
    tags: list[str] = field(default_factory=list)
    makers: list[BaseOpenMMMaker | OpenMMFlowMaker] = field(default_factory=list)
    collect_outputs: bool = True
    final_task_type: str = "collect"

    def make(
        self,
        interchange: Interchange | OpenMMInterchange | str,
        prev_dir: str | None = None,
    ) -> Flow:
        """Run the production simulation using the provided Interchange object.

        Parameters
        ----------
        interchange : Interchange
            The Interchange object containing the system
            to simulate.
        prev_task : Optional[ClassicalMDTaskDocument]
            The directory of the previous task.
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
                prev_dir=prev_dir,
            )
            interchange = job.output.interchange
            prev_dir = job.output.dir_name
            jobs.append(job)

            # collect the uuids and calcs for the final collect job
            if isinstance(job, Flow):
                job_uuids.extend(job.job_uuids)
            else:
                job_uuids.append(job.uuid)
            calcs_reversed.append(_get_calcs_reversed(job))

        if self.collect_outputs:
            collect_job = collect_outputs(
                prev_dir,
                tags=self.tags or None,
                job_uuids=job_uuids,
                calcs_reversed=calcs_reversed,
                task_type=self.final_task_type,
            )
            jobs.append(collect_job)

            return Flow(
                jobs,
                output=collect_job.output,
            )
        return Flow(
            jobs,
            output=job.output,
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
            collect_outputs=False,
        )


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
        of the inherited maker name with "dynamic" preprended.
    maker: Union[BaseOpenMMMaker, OpenMMFlowMaker]
        A list of makers to string together.
    max_stages: int
        Maximum number of stages to run consecutively before terminating
        dynamic flow logic. 
    collect_outputs : bool
        If True, a final job is added that collects all jobs into a single
        task document.
    should_continue: ...
        A function ... Default is _default_should_continue...#TODO
    """
    
    name: str = field(default=None)
    maker: BaseOpenMMMaker | OpenMMFlowMaker = field(
        default_factory=lambda: BaseOpenMMMaker()
    )
    max_stages: int = field(default=5)
    collect_outputs: bool = True
    should_continue: Callable[[list[OpenMMTaskDocument], int, int], Response] = field(
        default_factory=lambda: default_should_continue
    )   
    stage_task_type: str = "collect" # or default to OpenMMFlowMaker final_task_type?
    
    jobs: list = field(default_factory=list)
    job_uuids: list = field(default_factory=list)
    calcs_reversed: list[Calculation] = field(default_factory=list)
 
    def __post_init__(self) -> None:
        """Post init formatting of arguments."""
        if self.name is None: 
            self.name = f"dynamic {self.maker.name}"

    def make(
        self,
        interchange: Interchange | OpenMMInterchange | str,
        prev_dir: str | None = None,
    ) -> Flow:
        
        # Run initial stage job
        stage_job_0 = self.maker.make(
            interchange=interchange,
            prev_dir=prev_dir,
        )
        self.jobs.append(stage_job_0)
        print(f"stage_job_0->{stage_job_0.name}")
        
        # collect the uuids and calcs for the final collect job
        if isinstance(stage_job_0, Flow):
            self.job_uuids.extend(stage_job_0.job_uuids)
        else:
            self.job_uuids.append(stage_job_0.uuid)
        self.calcs_reversed.append(_get_calcs_reversed(stage_job_0))
        print(f"appended stage {self.jobs[-1]}")
        print(f"appended job_uuid {self.job_uuids[-1]}")
        print(f"appended calc {self.calcs_reversed[-1]}")
        
        # Determine stage job control logic
        control_stage_0 = self.should_continue(
            task_docs=[stage_job_0.output], 
            stage_index=0,
            max_stages=self.max_stages, 
        )
        self.jobs.append(control_stage_0)
        print(f"control_stage_0->{control_stage_0}")
        
        # stage_0 = Flow([stage_job_0, control_stage_0])
    
        stage_n = self.dynamic_flow(
            prev_stage_index=0,
            prev_docs=[control_stage_0.output],
        )
        self.jobs.append(stage_n)
        
        if self.collect_outputs:
            collect_job = collect_outputs(
                stage_n.output.dir_name,
                tags=None, #self.tags or None,
                job_uuids=self.job_uuids,
                calcs_reversed=self.calcs_reversed,
                task_type=self.stage_task_type,
            )
            # jobs.append(collect_job)
            stage_n_flow = Flow([stage_job_0,control_stage_0,stage_n,collect_job])
        else:
            stage_n_flow = Flow([stage_job_0,control_stage_0,stage_n])
 
        return stage_n_flow

        #return Response(output=stage_n.output)
    
    @job
    def dynamic_flow(
        self,
        prev_stage_index: int,
        prev_docs: list[OpenMMTaskDocument],
    ) -> Response | None: 
        """Run stage n and dynamically decide to continue or terminate flow."""
        
        print(f"previous stage index: {prev_stage_index}")
        prev_stage = prev_docs[-1] 
        
        # stage control logic: (a) begin, (b) continue, (c) terminate, (d) pause
        if not prev_stage.should_continue: # terminate
            print(f"Equilibration complete!")
            return Response(output=prev_stage)
        if prev_stage_index > self.max_stages: # pause
            print(f"Reached max stages with equilibration incomplete")
            print(f"prev_stage.output should_continue={prev_stage.should_continue}")
            #setattr(prev_stage.output, "should_continue", False)
            #setattr(prev_stage.output, "stage_index", stage_index)
            return Response(output=prev_stage)

        stage_index = prev_stage_index + 1
 
        stage_job_n = self.maker.make(
            interchange=prev_stage.interchange, 
            prev_dir=prev_stage.dir_name,
        )
        self.jobs.append(stage_job_n)
        print(f"stage_job_n->{stage_job_n.name}")
        
        # collect the uuids and calcs for the final collect job
        if isinstance(stage_job_n, Flow):
            self.job_uuids.extend(stage_job_n.job_uuids)
        else:
            self.job_uuids.append(stage_job_n.uuid)
        self.calcs_reversed.append(_get_calcs_reversed(stage_job_n))
        print(f"appended stage {self.jobs[-1]}")
        print(f"appended job_uuid {self.job_uuids[-1]}")
        print(f"appended calc {self.calcs_reversed[-1]}")
        print(f"stage_job_n->{stage_job_n}")
         
        control_stage_n = self.should_continue(
            task_docs=prev_docs+[stage_job_n.output], 
            stage_index=stage_index,
            max_stages=self.max_stages, 
        )
        self.jobs.append(control_stage_n)
        print(f"control_stage_n->{control_stage_n}")
         
        next_stage_n = self.dynamic_flow(
            prev_stage_index=stage_index,
            prev_docs=prev_docs + [control_stage_n.output],
        )
        self.jobs.append(next_stage_n)
        stage_n_flow = Flow([stage_job_n,control_stage_n, next_stage_n])
 
        return Response(replace=stage_n_flow, output=next_stage_n.output)
                
                 
