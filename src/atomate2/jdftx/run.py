"""Functions to run JDFTx."""

from __future__ import annotations

import os
from typing import Any

from custodian.jdftx.jobs import JDFTxJob
from jobflow.utils import ValueEnum
from atomate2 import SETTINGS
from atomate2.jdftx.schemas.task import JDFTxStatus, TaskDoc
from atomate2.jdftx.sets.base import FILE_NAMES


class JobType(ValueEnum):
    """Type of JDFTx job."""

    NORMAL = "normal"
    # Only running through Custodian now, can add DIRECT method later.


def get_jdftx_cmd():
    return SETTINGS.JDFTX_CMD


def run_jdftx(
    job_type: JobType | str = JobType.NORMAL,
    jdftx_cmd: str = None,
    jdftx_job_kwargs: dict[str, Any] = None,
) -> None:
    jdftx_job_kwargs = jdftx_job_kwargs or {}
    if jdftx_cmd is None:
        jdftx_cmd = get_jdftx_cmd()

    if job_type == JobType.NORMAL:
        job = JDFTxJob(
            jdftx_cmd, 
            input_file=FILE_NAMES["in"],
            output_file=FILE_NAMES["out"],
            **jdftx_job_kwargs)

    job.run()


# need to call job = run_jdftx() to run calc


def should_stop_children(
    task_document: TaskDoc,
) -> bool:
    """
    Parse JDFTx TaskDoc and decide whether to stop child processes.
    If JDFTx failed, stop child processes.
    """
    if task_document.state == JDFTxStatus.SUCCESS:
        return False
    return True