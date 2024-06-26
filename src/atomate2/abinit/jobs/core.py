"""Core jobs for running ABINIT calculations."""

from __future__ import annotations

import logging
import json
import numpy as np
from scipy.integrate import simpson
from pathlib import Path
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from abipy.flowtk.events import (
    AbinitCriticalWarning,
    NscfConvergenceWarning,
    RelaxConvergenceWarning,
    ScfConvergenceWarning,
)
from jobflow import Job, job, Maker, Response, Flow

from atomate2.abinit.jobs.base import BaseAbinitMaker
from atomate2.abinit.powerups import update_user_abinit_settings, update_user_kpoints_settings
from pymatgen.io.abinit.abiobjects import KSampling
from atomate2.abinit.schemas.task import AbinitTaskDoc, ConvergenceSummary 
from atomate2.abinit.sets.core import (
    LineNonSCFSetGenerator,
    NonSCFSetGenerator,
    NonScfWfqInputGenerator,
    RelaxSetGenerator,
    StaticSetGenerator,
    UniformNonSCFSetGenerator,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core.structure import Structure

    from atomate2.abinit.sets.base import AbinitInputGenerator
    from atomate2.abinit.utils.history import JobHistory

logger = logging.getLogger(__name__)
CONVERGENCE_FILE_NAME = "convergence.json" 

__all__ = ["StaticMaker", "NonSCFMaker", "RelaxMaker", "ConvergenceMaker"]



@dataclass
class StaticMaker(BaseAbinitMaker):
    """Maker to create ABINIT scf jobs.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "scf"
    name: str = "Scf calculation"
    input_set_generator: AbinitInputGenerator = field(
        default_factory=StaticSetGenerator
    )

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        ScfConvergenceWarning,
    )


@dataclass
class LineNonSCFMaker(BaseAbinitMaker):
    """Maker to create a jobs with a non-scf ABINIT calculation along a line.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "nscf_line"
    name: str = "Line non-Scf calculation"
    input_set_generator: AbinitInputGenerator = field(
        default_factory=LineNonSCFSetGenerator
    )

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        NscfConvergenceWarning,
    )


@dataclass
class UniformNonSCFMaker(BaseAbinitMaker):
    """Maker to create a jobs with a non-scf ABINIT calculation along a line.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "nscf_uniform"
    name: str = "Uniform non-Scf calculation"
    input_set_generator: AbinitInputGenerator = field(
        default_factory=UniformNonSCFSetGenerator
    )

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        NscfConvergenceWarning,
    )


@dataclass
class NonSCFMaker(BaseAbinitMaker):
    """Maker to create non SCF calculations."""

    calc_type: str = "nscf"
    name: str = "non-Scf calculation"

    input_set_generator: AbinitInputGenerator = field(
        default_factory=NonSCFSetGenerator
    )

    # Non dataclass variables:
    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        NscfConvergenceWarning,
    )

    @job
    def make(
        self,
        structure: Structure | None = None,
        prev_outputs: str | list[str] | None = None,
        restart_from: str | list[str] | None = None,
        history: JobHistory | None = None,
        mode: str = "uniform",
    ) -> Job:
        """
        Run a non-scf ABINIT job.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        mode : str
            Type of band structure calculation. Options are:
            - "line": Full band structure along symmetry lines.
            - "uniform": Uniform mesh band structure.
        """
        self.input_set_generator.mode = mode

        return super().make.original(
            self,
            structure=structure,
            prev_outputs=prev_outputs,
            restart_from=restart_from,
            history=history,
        )


@dataclass
class NonSCFWfqMaker(NonSCFMaker):
    """Maker to create non SCF calculations for the WFQ."""

    calc_type: str = "nscf_wfq"
    name: str = "non-Scf calculation"

    input_set_generator: AbinitInputGenerator = field(
        default_factory=NonScfWfqInputGenerator
    )

    # Non dataclass variables:
    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        NscfConvergenceWarning,
    )


@dataclass
class RelaxMaker(BaseAbinitMaker):
    """Maker to create relaxation calculations."""

    calc_type: str = "relax"
    input_set_generator: AbinitInputGenerator = field(default_factory=RelaxSetGenerator)
    name: str = "Relaxation calculation"

    # non-dataclass variables
    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        RelaxConvergenceWarning,
    )

    @classmethod
    def ionic_relaxation(cls, *args, **kwargs) -> Job:
        """Create an ionic relaxation maker."""
        # TODO: add the possibility to tune the RelaxInputGenerator options
        #  in this class method.
        return cls(
            input_set_generator=RelaxSetGenerator(*args, relax_cell=False, **kwargs),
            name=cls.name + " (ions only)",
        )

    @classmethod
    def full_relaxation(cls, *args, **kwargs) -> Job:
        """Create a full relaxation maker."""
        # TODO: add the possibility to tune the RelaxInputGenerator options
        #  in this class method.
        return cls(
            input_set_generator=RelaxSetGenerator(*args, relax_cell=True, **kwargs)
        )
@dataclass
class ConvergenceMaker(Maker):
    """A job that performs convergence run for a given number of steps. Stops either
    when all steps are done, or when the convergence criterion is reached, that is when
    the absolute difference between the subsequent values of the convergence field is
    less than a given epsilon.

    Parameters
    ----------
    name : str
        A name for the job
    maker: .BaseAbinitMaker
        A maker for the run
    criterion_name: str
        A name for the convergence criterion. Must be in the run results
    epsilon: float
        A difference in criterion value for subsequent runs
    convergence_field: str
        An input parameter that changes to achieve convergence
    convergence_steps: list | tuple
        An iterable of the possible values for the convergence field.
        If the iterable is depleted and the convergence is not reached,
        that the job is failed
    """

    name: str = "Convergence job"
    maker: BaseAbinitMaker = field(default_factory=BaseAbinitMaker)
    criterion_name: str = "energy_per_atom"
    epsilon: float = 0.001
    convergence_field: str = field(default_factory=str)
    convergence_steps: list = field(default_factory=list)

    def __post_init__(self):
        self.last_idx = len(self.convergence_steps)

    def make(
        self, 
        structure: Structure,
        prev_outputs: str | list[str] | Path = None
        ):
        """A top-level flow controlling convergence iteration

        Parameters
        ----------
            atoms : MSONableAtoms
                a structure to run a job
        """
        convergence_job = self.convergence_iteration(structure, prev_outputs=prev_outputs)
        return Flow([convergence_job], output=convergence_job.output)

    @job
    def convergence_iteration(
        self,
        structure: Structure,
        prev_dir: str | Path = None,
        prev_outputs: str | list[str] | Path = None,
    ) -> Response:
        """
        Runs several jobs with changing inputs consecutively to investigate
        convergence in the results

        Parameters
        ----------
        structure : Structure
            The structure to run the job for
        prev_dir: str | None
            An Abinit calculation directory in which previous run contents are stored

        Returns
        -------
        The output response for the job
        """

        # getting the calculation index
        idx = 0
        converged = False
        num_prev_outputs=len(prev_outputs)
        if prev_dir is not None:
            prev_dir = prev_dir.split(":")[-1]
            convergence_file = Path(prev_dir) / CONVERGENCE_FILE_NAME
            idx += 1
            if convergence_file.exists():
                with open(convergence_file) as f:
                    data = json.load(f)
                    idx = data["idx"] + 1
                    # check for convergence
                    converged = data["converged"]

        if idx < self.last_idx and not converged:
            # finding next jobs
            if self.convergence_field=="kppa":
                next_base_job = self.maker.make(
                        structure,
                        prev_outputs=prev_outputs, 
                        kppa=self.convergence_steps[idx])
                print(idx,self.convergence_steps[idx])   
            else:
                base_job = self.maker.make(
                        structure, 
                        prev_outputs=prev_outputs)
                next_base_job = update_user_abinit_settings(
                        flow=base_job, 
                        abinit_updates={
                        self.convergence_field: self.convergence_steps[idx]})
            next_base_job.append_name(append_str=f" {idx}")
            update_file_job = self.update_convergence_file(
                prev_dir=prev_dir,
                job_dir=next_base_job.output.dir_name,
                output=next_base_job.output)
            prev_outputs=prev_outputs[:num_prev_outputs]
            next_job = self.convergence_iteration(
                structure, 
                prev_dir=next_base_job.output.dir_name, 
                prev_outputs=prev_outputs)
            replace_flow = Flow(
                [next_base_job, update_file_job, next_job], output=next_base_job.output)
            return Response(detour=replace_flow, output=replace_flow.output)
        else:
            task_doc = AbinitTaskDoc.from_directory(prev_dir)
            summary = ConvergenceSummary.from_abinit_calc_doc(task_doc)
            return summary

    @job(name="Writing a convergence file")
    def update_convergence_file(
        self, prev_dir: str | Path, job_dir: str | Path, output
    ):
        """Write a convergence file

        Parameters
        ----------
        TO DO: fill out
        """
        idx = 0
        if prev_dir is not None:
            prev_dir = prev_dir.split(":")[-1]
            convergence_file = Path(prev_dir) / CONVERGENCE_FILE_NAME
            if convergence_file.exists():
                with open(convergence_file) as f:
                    convergence_data = json.load(f)
                    idx = convergence_data["idx"] + 1
        else:
            idx = 0
            convergence_data = {
                "criterion_name": self.criterion_name,
                "asked_epsilon": self.epsilon,   
                "criterion_values": [],
                "convergence_field_name": self.convergence_field,
                "convergence_field_values": [],
                "converged": False,
            }
        convergence_data["convergence_field_values"].append(self.convergence_steps[idx])
        convergence_data["criterion_values"].append(
            getattr(output.output, self.criterion_name) 
        )
        convergence_data["idx"] = idx
        if len(convergence_data["criterion_values"]) > 1:
            # checking for convergence
            if type(convergence_data["criterion_values"][-1]) is list:
                old_data=np.array(convergence_data["criterion_values"][-2])
                new_data=np.array(convergence_data["criterion_values"][-1])
                mesh0=old_data[0]
                mesh=new_data[0]
                values0=old_data[1]
                values=new_data[1]
                values1=np.interp(mesh0, mesh, values)
                delta=abs(values1-values0)
                delarea=simpson(delta) 
                area=simpson(values0)
                print(delarea/area) 
                convergence_data["converged"] = bool(delarea/area < self.epsilon)
            if type(convergence_data["criterion_values"][-1]) is float:
                convergence_data["converged"] = bool(
                    abs(
                         convergence_data["criterion_values"][-1]
                        - convergence_data["criterion_values"][-2]
                    )
                    < self.epsilon
                )
        job_dir = job_dir.split(":")[-1]
        convergence_file = Path(job_dir) / CONVERGENCE_FILE_NAME
        with open(convergence_file, "w") as f:
            json.dump(convergence_data, f)

    @job(name="Getting the results")
    def get_results(self, prev_dir: Path | str) -> Dict[str, Any]:
        """Get the results for a calculation from a given directory

        Parameters
        ----------
        prev_dir: Path or str
            The calculation directory to get the results for

        Results
        -------
        The results dictionary loaded from the JSON file
        """
        convergence_file = Path(prev_dir) / CONVERGENCE_FILE_NAME
        with open(convergence_file) as f:
            return json.load(f)


