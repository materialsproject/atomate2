"""
Functions to run LOBSTER.

"""

from __future__ import annotations

import logging
import shlex
import subprocess
from os.path import expandvars
from typing import Any, Sequence

from custodian import Custodian
from custodian.custodian import Validator
from custodian.lobster.handlers import EnoughBandsValidator, LobsterFilesValidator
from custodian.lobster.jobs import LobsterJob
from jobflow.utils import ValueEnum

from atomate2 import SETTINGS

__all__ = [
    "JobType",
    "run_lobster",
]

_DEFAULT_VALIDATORS = (LobsterFilesValidator(), EnoughBandsValidator())
_DEFAULT_HANDLERS = ()

logger = logging.getLogger(__name__)


class JobType(ValueEnum):
    """
    Type of Lobster job.

    - ``DIRECT``: Run Lobster without custodian.
    - ``NORMAL``: Normal custodian :obj:`.LobsterJob`.
    """

    DIRECT = "direct"
    NORMAL = "normal"


def run_lobster(
    job_type: JobType | str = JobType.NORMAL,
    lobster_cmd: str = SETTINGS.LOBSTER_CMD,
    max_errors: int = SETTINGS.LOBSTER_CUSTODIAN_MAX_ERRORS,
    scratch_dir: str = SETTINGS.CUSTODIAN_SCRATCH_DIR,
    validators: Sequence[Validator] = _DEFAULT_VALIDATORS,
    lobster_job_kwargs: dict[str, Any] = None,
    custodian_kwargs: dict[str, Any] = None,
):
    """
    Run Lobster.
    Supports running Lobster with or without custodian (see :obj:`JobType`).

    Parameters
    ----------
    job_type : str or .JobType
         The job type.
    lobster_cmd : str
        Command to run lobster.
    max_errors : int
        Maximum number of errors.
    scratch_dir : str or Path
        Scratch directory.
    validators : list of .Validator
        The validators handlers used by custodian.
    lobster_job_kwargs : dict
        Keyword arguments that are passed to :obj:`.LosterJob`.
    custodian_kwargs : dict
         Keyword arguments that are passed to :obj:`.Custodian`.
    """
    lobster_job_kwargs = {} if lobster_job_kwargs is None else lobster_job_kwargs
    custodian_kwargs = {} if custodian_kwargs is None else custodian_kwargs

    lobster_cmd = expandvars(lobster_cmd)
    split_lobster_cmd = shlex.split(lobster_cmd)

    if job_type == JobType.DIRECT:
        logger.info(f"Running command: {lobster_cmd}")
        return_code = subprocess.call(lobster_cmd, shell=True)
        logger.info(f"{lobster_cmd} finished running with returncode: {return_code}")
        return

    elif job_type == JobType.NORMAL:
        jobs = [LobsterJob(split_lobster_cmd, **lobster_job_kwargs)]
    else:
        raise ValueError(f"Unsupported job type: {job_type}")

    handlers: list = []

    c = Custodian(
        handlers,
        jobs,
        validators=validators,
        max_errors=max_errors,
        scratch_dir=scratch_dir,
        **custodian_kwargs,
    )

    logger.info("Running Lobster using custodian.")
    c.run()
