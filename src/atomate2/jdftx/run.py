"""Functions to run JDFTx."""

from __future__ import annotations

from typing import Any

from jobflow.utils import ValueEnum
from custodian.jdftx.jobs import JDFTxJob
from atomate2.jdftx.schemas.task import TaskDoc, JDFTxStatus

class JobType(ValueEnum):
    """
    Type of JDFTx job

    """

    NORMAL = "normal"
    # Only running through Custodian now, can add DIRECT method later.


def run_jdftx(
    job_type: JobType | str = JobType.NORMAL,
    jdftx_cmd: str = "docker run -t --rm -v $PWD:/root/research jdftx jdftx",
    jdftx_job_kwargs: dict[str, Any] = None,
) -> None:
    jdftx_job_kwargs = jdftx_job_kwargs or {}

    if job_type == JobType.NORMAL:
        job = JDFTxJob(jdftx_cmd, **jdftx_job_kwargs)
    
    job.run()

#need to call job = run_jdftx() to run calc

def should_stop_children(
    task_document: TaskDoc,
) -> bool:
    """
    Parse JDFTx TaskDoc and decide whether to stop child processes.
    If JDFTx failed, stop child processes.
    """
    if task_document.state == JDFTxStatus.SUCCESS:
        return False
    else:
        return True