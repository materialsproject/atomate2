"""Define common testing utils used in atomate2."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jobflow import Flow, Job, Response


def get_job_uuid_name_map(job_flow_resp: Job | Flow | Response) -> dict[str, str]:
    """
    Get all job UUIDs and map them to the job name.

    Useful for running complex flows locally / testing in CI, where one often
    wants the output of a job with a specific name.

    Parameters
    ----------
    job_flow_resp : jobflow Job, Flow, or Response

    Returns
    -------
    dict mapping string UUIDs to string names.
    """
    uuid_to_name: dict[str, str] = {}

    def recursive_get_job_names(
        flow_like: Job | Flow, uuid_to_name: dict[str, str]
    ) -> None:
        if flow_jobs := getattr(flow_like, "jobs", None):
            for job in flow_jobs:
                recursive_get_job_names(job, uuid_to_name)
        elif replacement := getattr(flow_like, "replace", None):
            recursive_get_job_names(replacement, uuid_to_name)
        else:
            uuid_to_name[flow_like.uuid] = flow_like.name

    recursive_get_job_names(job_flow_resp, uuid_to_name)
    return uuid_to_name
