"""Defines the base FHI-aims convergence jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from jobflow import Flow, Response, job
from pymatgen.core import Molecule, Structure

from atomate2.aims.schemas.task import AimsTaskDoc, ConvergenceSummary
from atomate2.aims.utils.msonable_atoms import MSONableAtoms

if TYPE_CHECKING:
    from atomate2.aims.jobs.base import BaseAimsMaker

CONVERGENCE_FILE_NAME = "convergence.json"  # make it a constant?


@job
def convergence_iteration(
    structure: MSONableAtoms | Structure | Molecule,
    last_idx: int,
    static_maker: BaseAimsMaker,
    criterion_name: str,
    epsilon: float,
    convergence_field: str,
    convergence_steps: list,
    prev_dir: str | Path = None,
) -> Response | ConvergenceSummary:
    """Run several jobs with changing inputs to test convergence.

    Parameters
    ----------
    structure : .MSONableAtoms or Structure or Molecule
        The structure to run the job for
    static_maker: .BaseAimsMaker
        A maker for the static calculations
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
    prev_dir: str or None
        An FHI-aims calculation directory in which previous run contents are stored

    Returns
    -------
    The output response for the job
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

    if idx < last_idx and not converged:
        # finding next jobs
        next_base_job = static_maker.make(atoms, prev_dir=prev_dir_no_host)
        next_base_job.update_maker_kwargs(
            {
                "_set": {
                    f"input_set_generator->user_parameters->"
                    f"{convergence_field}": convergence_steps[idx]
                }
            },
            dict_mod=True,
        )
        next_base_job.append_name(append_str=f" {idx}")

        update_file_job = update_convergence_file(
            prev_dir=prev_dir_no_host,
            job_dir=next_base_job.output.dir_name,
            criterion_name=criterion_name,
            epsilon=epsilon,
            convergence_field=convergence_field,
            convergence_steps=convergence_steps,
            output=next_base_job.output,
        )

        next_job = convergence_iteration(
            atoms,
            last_idx,
            static_maker,
            criterion_name,
            epsilon,
            convergence_field,
            convergence_steps,
            prev_dir=next_base_job.output.dir_name,
        )

        replace_flow = Flow(
            [next_base_job, update_file_job, next_job], output=next_base_job.output
        )
        return Response(detour=replace_flow, output=replace_flow.output)
    task_doc = AimsTaskDoc.from_directory(prev_dir_no_host)
    return ConvergenceSummary.from_aims_calc_doc(task_doc)


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
