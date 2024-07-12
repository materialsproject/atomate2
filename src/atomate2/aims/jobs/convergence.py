"""Defines the base FHI-aims convergence jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from jobflow import Flow, Maker, Response, job

from atomate2.aims.jobs.base import BaseAimsMaker
from atomate2.aims.schemas.task import ConvergenceSummary

if TYPE_CHECKING:
    from pymatgen.core import Molecule, Structure

CONVERGENCE_FILE_NAME = "convergence.json"  # make it a constant?


@dataclass
class ConvergenceMaker(Maker):
    """Defines a convergence workflow with a maximum number of steps.

    A job that performs convergence run for a given number of steps. Stops either
    when all steps are done, or when the convergence criterion is reached, that is when
    the absolute difference between the subsequent values of the convergence field is
    less than a given epsilon.

    Parameters
    ----------
    convergence_field: str
        An input parameter that changes to achieve convergence
    convergence_steps: Iterable
        An iterable of the possible values for the convergence field.
        If the iterable is depleted and the convergence is not reached,
        then the job is failed
    name : str
        A name for the job
    maker: .BaseAimsMaker
        A maker for the run
    criterion_name: str
        A name for the convergence criterion. Must be in the run results
    epsilon: float
        A difference in criterion value for subsequent runs
    """

    convergence_field: str
    convergence_steps: list | tuple
    name: str = "convergence"
    maker: BaseAimsMaker = field(default_factory=BaseAimsMaker)
    criterion_name: str = "energy_per_atom"
    epsilon: float = 0.001

    @job
    def make(
        self,
        structure: Structure | Molecule,
        prev_dir: str | Path | None = None,
        convergence_data: dict | None = None,
        prev_output_value: float | None = None,
    ) -> ConvergenceSummary:
        """Create a top-level flow controlling convergence iteration.

        Parameters
        ----------
        structure : Structure or Molecule
            a structure to run a job
        prev_dir : str or Path or None
            An FHI-aims calculation directory in which previous run contents are stored
        convergence_data : dict or None
            The convergence information to date.
        prev_output_value : float or None
            The output value being converged from the previous aims calculation.
        """
        # getting the calculation index
        idx = 0
        converged = False
        if convergence_data is not None:
            idx = convergence_data["idx"]
            convergence_data["convergence_field_values"].append(
                self.convergence_steps[idx]
            )
            convergence_data["criterion_values"].append(prev_output_value)
            if len(convergence_data["criterion_values"]) > 1:
                # checking for convergence
                converged = (
                    abs(prev_output_value - convergence_data["criterion_values"][-2])
                    < self.epsilon
                )
            idx += 1
        else:
            convergence_data = {
                "criterion_name": self.criterion_name,
                "criterion_values": [],
                "convergence_field_name": self.convergence_field,
                "convergence_field_values": [],
                "epsilon": self.epsilon,
            }
        convergence_data.update(idx=idx, converged=converged)

        if prev_dir is not None:
            split_prev_dir = str(prev_dir).split(":")[-1]
            convergence_file = Path(split_prev_dir) / CONVERGENCE_FILE_NAME
            with open(convergence_file, "w") as file:
                json.dump(convergence_data, file)

        if idx < len(self.convergence_steps) and not converged:
            # finding next jobs
            next_base_job = self.maker.make(structure, prev_dir=prev_dir)
            next_base_job.update_maker_kwargs(
                {
                    "_set": {
                        f"input_set_generator->user_params->"
                        f"{self.convergence_field}": self.convergence_steps[idx]
                    }
                },
                dict_mod=True,
            )
            next_base_job.append_name(append_str=f" {idx}")

            next_job = self.make(
                structure,
                prev_dir=next_base_job.output.dir_name,
                convergence_data=convergence_data,
                prev_output_value=getattr(
                    next_base_job.output.output, self.criterion_name
                ),
            )

            replace_flow = Flow([next_base_job, next_job], output=next_base_job.output)
            return Response(replace=replace_flow)

        return ConvergenceSummary.from_data(structure, convergence_data)
