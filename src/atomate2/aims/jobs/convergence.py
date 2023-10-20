"""Defines the base FHI-aims convergence jobs."""

from __future__ import annotations

import json
from pathlib import Path

from jobflow import job

CONVERGENCE_FILE_NAME = "convergence.json"  # make it a constant?


@job(name="Writing a convergence file")
def update_convergence_file(
    prev_dir: str | Path,
    job_dir: str | Path,
    criterion_name: str,
    epsilon: float,
    convergence_field: str,
    convergence_steps: list,
    output,
):
    """Write a convergence file.

    Parameters
    ----------
    prev_dir: str or Path
        The previous calculation directory
    job_dir: str or Path
        The current calculation directory
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
    output: .ConvergenceSummary
        The current output of the convergence flow
    """
    idx = 0
    if prev_dir is not None:
        prev_dir_no_host = str(prev_dir).split(":")[-1]
        convergence_file = Path(prev_dir_no_host) / CONVERGENCE_FILE_NAME
        if convergence_file.exists():
            with open(convergence_file) as f:
                convergence_data = json.load(f)
                idx = convergence_data["idx"] + 1
    else:
        idx = 0
        convergence_data = {
            "criterion_name": criterion_name,
            "criterion_values": [],
            "convergence_field_name": convergence_field,
            "convergence_field_values": [],
            "converged": False,
        }
    convergence_data["convergence_field_values"].append(convergence_steps[idx])
    convergence_data["criterion_values"].append(getattr(output.output, criterion_name))
    convergence_data["idx"] = idx

    if len(convergence_data["criterion_values"]) > 1:
        # checking for convergence
        convergence_data["converged"] = (
            abs(
                convergence_data["criterion_values"][-1]
                - convergence_data["criterion_values"][-2]
            )
            < epsilon
        )

    split_job_dir = str(job_dir).split(":")[-1]
    convergence_file = Path(split_job_dir) / CONVERGENCE_FILE_NAME
    with open(convergence_file, "w") as f:
        json.dump(convergence_data, f)
