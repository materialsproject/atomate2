"""Define a Flow to test the convergence of a calculation.

Checks to see if the calculations are converged with respect to a particular parameter.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Maker, Response, job
from pymatgen.core import Molecule, Structure

from atomate2.aims.jobs.base import BaseAimsMaker
from atomate2.aims.jobs.convergence import (
    CONVERGENCE_FILE_NAME,
    update_convergence_file,
)
from atomate2.aims.schemas.task import AimsTaskDoc, ConvergenceSummary
from atomate2.aims.utils.msonable_atoms import MSONableAtoms


@dataclass
class ConvergenceMaker(Maker):
    """Defines a convergence workflow with a maximum number of steps.

    A job that performs convergence run for a given number of steps. Stops either
    when all steps are done, or when the convergence criterion is reached, that is when
    the absolute difference between the subsequent values of the convergence field is
    less than a given epsilon.

    Parameters
    ----------
    name : str
        A name for the job
    maker: .BaseAimsMaker
        A maker for the run
    criterion_name: str
        A name for the convergence criterion. Must be in the run results
    epsilon: float
        A difference in criterion value for subsequent runs
    convergence_field: str
        An input parameter that changes to achieve convergence
    convergence_steps: Iterable
        An iterable of the possible values for the convergence field.
        If the iterable is depleted and the convergence is not reached,
        that the job is failed
    last_idx: int
        The index of the last allowed convergence step (max number of steps)
    """

    name: str = "convergence"
    maker: BaseAimsMaker = field(default_factory=BaseAimsMaker)
    criterion_name: str = "energy_per_atom"
    epsilon: float = 0.001
    convergence_field: str = field(default_factory=str)
    convergence_steps: list = field(default_factory=list)
    last_idx: int = None

    def __post_init__(self):
        """Set the value of the last index."""
        self.last_idx = len(self.convergence_steps)

    @job
    def make(
        self,
        structure: MSONableAtoms | Structure | Molecule,
        prev_dir: str | Path | None = None,
    ):
        """Create a top-level flow controlling convergence iteration.

        Parameters
        ----------
        structure : .MSONableAtoms or Structure or Molecule
            a structure to run a job
        prev_dir: str or Path or None
            An FHI-aims calculation directory in which previous run contents are stored
        """
        if isinstance(structure, (Structure, Molecule)):
            atoms = MSONableAtoms.from_pymatgen(structure)
        else:
            atoms = structure.copy()

        # getting the calculation index
        idx = 0
        converged = False
        if prev_dir is not None:
            prev_dir_no_host = str(prev_dir).split(":")[-1]
            convergence_file = Path(prev_dir_no_host) / CONVERGENCE_FILE_NAME
            idx += 1
            if convergence_file.exists():
                with open(convergence_file) as f:
                    data = json.load(f)
                    idx = data["idx"] + 1
                    # check for convergence
                    converged = data["converged"]
        else:
            prev_dir_no_host = None

        if idx < self.last_idx and not converged:
            # finding next jobs
            next_base_job = self.maker.make(atoms, prev_dir=prev_dir_no_host)
            next_base_job.update_maker_kwargs(
                {
                    "_set": {
                        f"input_set_generator->user_parameters->"
                        f"{self.convergence_field}": self.convergence_steps[idx]
                    }
                },
                dict_mod=True,
            )
            next_base_job.append_name(append_str=f" {idx}")

            update_file_job = update_convergence_file(
                prev_dir=prev_dir_no_host,
                job_dir=next_base_job.output.dir_name,
                criterion_name=self.criterion_name,
                epsilon=self.epsilon,
                convergence_field=self.convergence_field,
                convergence_steps=self.convergence_steps,
                output=next_base_job.output,
            )

            next_job = self.make(
                structure,
                prev_dir=next_base_job.output.dir_name,
            )

            replace_flow = Flow(
                [next_base_job, update_file_job, next_job], output=next_base_job.output
            )
            return Response(detour=replace_flow, output=replace_flow.output)

        task_doc = AimsTaskDoc.from_directory(prev_dir_no_host)
        return ConvergenceSummary.from_aims_calc_doc(task_doc)
