"""Functions to run CP2K."""

from __future__ import annotations

import logging
import shlex
import subprocess
from os.path import expandvars
from typing import TYPE_CHECKING, Any

from custodian import Custodian
from custodian.cp2k.handlers import (
    AbortHandler,
    DivergingScfErrorHandler,
    FrozenJobErrorHandler,
    NumericalPrecisionHandler,
    StdErrHandler,
    UnconvergedRelaxationErrorHandler,
    UnconvergedScfErrorHandler,
    WalltimeHandler,
)
from custodian.cp2k.jobs import Cp2kJob
from custodian.cp2k.validators import Cp2kOutputValidator
from jobflow.utils import ValueEnum

from atomate2 import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from custodian.custodian import ErrorHandler, Validator

    from atomate2.cp2k.schemas.task import TaskDocument


_DEFAULT_HANDLERS = (
    StdErrHandler(),
    UnconvergedScfErrorHandler(),
    DivergingScfErrorHandler(),
    FrozenJobErrorHandler(),
    AbortHandler(),
    NumericalPrecisionHandler(),
    UnconvergedRelaxationErrorHandler(),
    WalltimeHandler(),
)
_DEFAULT_VALIDATORS = (Cp2kOutputValidator(),)

logger = logging.getLogger(__name__)


class JobType(ValueEnum):
    """
    Type of CP2K job.

    - ``NORMAL``: Normal custodian :obj:`.Cp2kJob`.
    """

    DIRECT = "direct"
    NORMAL = "normal"


def run_cp2k(
    job_type: JobType | str = JobType.NORMAL,
    cp2k_cmd: str = SETTINGS.CP2K_CMD,
    max_errors: int = SETTINGS.CP2K_CUSTODIAN_MAX_ERRORS,
    scratch_dir: str = SETTINGS.CUSTODIAN_SCRATCH_DIR,
    handlers: Sequence[ErrorHandler] = _DEFAULT_HANDLERS,
    validators: Sequence[Validator] = _DEFAULT_VALIDATORS,
    cp2k_job_kwargs: dict[str, Any] = None,
    custodian_kwargs: dict[str, Any] = None,
) -> None:
    """
    Run CP2K.

    Supports running CP2K with or without custodian (see :obj:`JobType`).

    Parameters
    ----------
    job_type : str or .JobType
        The job type.
    cp2k_cmd : str
        The command used to run cp2k.
    max_errors : int
        The maximum number of errors allowed by custodian.
    scratch_dir : str
        The scratch directory used by custodian.
    handlers : list of .ErrorHandler
        The error handlers used by custodian.
    validators : list of .Validator
        The validators handlers used by custodian.
    wall_time : int
        The maximum wall time. If set, a WallTimeHandler will be added to the list
        of handlers.
    cp2k_job_kwargs : dict
        Keyword arguments that are passed to :obj:`.Cp2kJob`.
    custodian_kwargs : dict
        Keyword arguments that are passed to :obj:`.Custodian`.
    """
    cp2k_job_kwargs = cp2k_job_kwargs or {}
    custodian_kwargs = custodian_kwargs or {}

    cp2k_cmd = expandvars(cp2k_cmd)
    split_cp2k_cmd = shlex.split(cp2k_cmd)

    if job_type == JobType.DIRECT:
        logger.info(f"Running command: {cp2k_cmd}")
        return_code = subprocess.call(cp2k_cmd, shell=True)  # noqa: S602
        logger.info(f"{cp2k_cmd} finished running with returncode: {return_code}")
        return
    if job_type == JobType.NORMAL:
        jobs = [Cp2kJob(split_cp2k_cmd, **cp2k_job_kwargs)]
    else:
        raise ValueError(f"Unsupported {job_type=}")

    custodian = Custodian(
        handlers,
        jobs,
        validators=validators,
        max_errors=max_errors,
        scratch_dir=scratch_dir,
        gzipped_output=SETTINGS.CUSTODIAN_GZIPPED_OUTPUT,
        **custodian_kwargs,
    )

    logger.info("Running CP2K using custodian.")
    custodian.run()


def should_stop_children(
    task_document: TaskDocument,
    handle_unsuccessful: bool | str = SETTINGS.CP2K_HANDLE_UNSUCCESSFUL,
) -> bool:
    """
    Parse CP2K outputs and decide whether child jobs should continue.

    Parameters
    ----------
    task_document : .TaskDocument
        A CP2K task document.
    handle_unsuccessful : bool or str
        This is a three-way toggle on what to do if your job looks OK, but is actually
        unconverged (either electronic or ionic):

        - `True`: Mark job as completed, but stop children.
        - `False`: Do nothing, continue with workflow as normal.
        - `"error"`: Throw an error.

    Returns
    -------
    bool
        Whether to stop child jobs.
    """
    if task_document.state == "successful":
        return False

    if isinstance(handle_unsuccessful, bool):
        return handle_unsuccessful

    if handle_unsuccessful == "error":
        raise RuntimeError(
            "Job was not successful (perhaps your job did not converge within the "
            "limit of electronic/ionic iterations)!"
        )

    raise RuntimeError(f"Unknown option for {handle_unsuccessful=}")
