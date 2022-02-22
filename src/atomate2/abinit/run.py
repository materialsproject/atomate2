"""Functions to run ABINIT."""

from __future__ import annotations

import logging
import subprocess
from jobflow.utils import ValueEnum
from custodian.custodian import Job


__all__ = ["run_abinit"]


logger = logging.getLogger(__name__)


class JobType(ValueEnum):
    """
    Type of ABINIT job.

    - ``DIRECT``: Run ABINIT without using custodian.
    - ``NORMAL``: Normal custodian :obj:`.AbinitJob`.
    """

    DIRECT = "direct"
    NORMAL = "normal"


class AbinitJob(Job):
    def setup(self):
        pass

    def run(self):
        pass

    def postprocess(self):
        pass


def run_abinit(
    job_type: JobType | str = JobType.DIRECT,
    abinit_cmd: str = 'abinit',
    mpirun_cmd: str = None,

):
    """
    Run ABINIT.
    """

    cmd = [abinit_cmd] if mpirun_cmd is None else [mpirun_cmd, abinit_cmd]

    if job_type == JobType.DIRECT:
        logger.info(f"Running command: {' '.join(cmd)}")
        return_code = subprocess.call(cmd, shell=True)
        logger.info(f"{' '.join(cmd)} finished running with returncode: {return_code}")
        return

    elif job_type == JobType.NORMAL:
        raise NotImplementedError()
    else:
        raise ValueError(f"Unsupported job type: {job_type}")
