"""
Functions to run VASP.

Todo
----
- Implement vasp_ncl and auto_ncl in custodian.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from os.path import expandvars
from typing import TYPE_CHECKING, Any

from custodian import Custodian
from custodian.vasp.handlers import (
    FrozenJobErrorHandler,
    IncorrectSmearingHandler,
    KspacingMetalHandler,
    LargeSigmaHandler,
    MeshSymmetryErrorHandler,
    NonConvergingErrorHandler,
    PositiveEnergyErrorHandler,
    PotimErrorHandler,
    StdErrHandler,
    UnconvergedErrorHandler,
    VaspErrorHandler,
    WalltimeHandler,
)
from custodian.vasp.jobs import VaspJob
from custodian.vasp.validators import VaspFilesValidator, VasprunXMLValidator
from jobflow.utils import ValueEnum

from atomate2 import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from custodian.custodian import ErrorHandler, Validator
    from emmet.core.tasks import TaskDoc


DEFAULT_HANDLERS = (
    VaspErrorHandler(),
    MeshSymmetryErrorHandler(),
    UnconvergedErrorHandler(),
    NonConvergingErrorHandler(),
    PotimErrorHandler(),
    PositiveEnergyErrorHandler(),
    FrozenJobErrorHandler(),
    StdErrHandler(),
    LargeSigmaHandler(),
    IncorrectSmearingHandler(),
    KspacingMetalHandler(),
)
_DEFAULT_VALIDATORS = (VasprunXMLValidator(), VaspFilesValidator())

logger = logging.getLogger(__name__)


class JobType(ValueEnum):
    """
    Type of VASP job.

    - ``DIRECT``: Run VASP without using custodian.
    - ``NORMAL``: Normal custodian :obj:`.VaspJob`.
    - ``DOUBLE_RELAXATION``: Custodian double relaxation run from
      :obj:`.VaspJob.double_relaxation_run`.
    - ``METAGGA_OPT``: Custodian meta-GGA optimization run from
      :obj:`.VaspJob.metagga_opt_run`.
    - ``FULL_OPT``: Custodian full optimization run from
      :obj:`.VaspJob.full_opt_run`.
    """

    DIRECT = "direct"
    NORMAL = "normal"
    DOUBLE_RELAXATION = "double relaxation"
    METAGGA_OPT = "metagga opt"
    FULL_OPT = "full opt"


def run_vasp(
    job_type: JobType | str = JobType.NORMAL,
    vasp_cmd: str = SETTINGS.VASP_CMD,
    vasp_gamma_cmd: str = SETTINGS.VASP_GAMMA_CMD,
    max_errors: int = SETTINGS.VASP_CUSTODIAN_MAX_ERRORS,
    scratch_dir: str = SETTINGS.CUSTODIAN_SCRATCH_DIR,
    handlers: Sequence[ErrorHandler] = DEFAULT_HANDLERS,
    validators: Sequence[Validator] = _DEFAULT_VALIDATORS,
    wall_time: int | None = None,
    vasp_job_kwargs: dict[str, Any] = None,
    custodian_kwargs: dict[str, Any] = None,
) -> None:
    """
    Run VASP.

    Supports running VASP with or without custodian (see :obj:`JobType`).

    Parameters
    ----------
    job_type : str or .JobType
        The job type.
    vasp_cmd : str
        The command used to run the standard version of vasp.
    vasp_gamma_cmd : str
        The command used to run the gamma version of vasp.
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
    vasp_job_kwargs : dict
        Keyword arguments that are passed to :obj:`.VaspJob`.
    custodian_kwargs : dict
        Keyword arguments that are passed to :obj:`.Custodian`.
    """
    vasp_job_kwargs = vasp_job_kwargs or {}
    custodian_kwargs = custodian_kwargs or {}

    vasp_cmd = expandvars(vasp_cmd)
    vasp_gamma_cmd = expandvars(vasp_gamma_cmd)
    split_vasp_cmd = shlex.split(vasp_cmd)
    split_vasp_gamma_cmd = shlex.split(vasp_gamma_cmd)

    vasp_job_kwargs.setdefault("auto_npar", False)

    vasp_job_kwargs.update(gamma_vasp_cmd=split_vasp_gamma_cmd)

    if job_type == JobType.DIRECT:
        logger.info(f"Running command: {vasp_cmd}")
        return_code = subprocess.call(vasp_cmd, shell=True)  # noqa: S602
        logger.info(f"{vasp_cmd} finished running with returncode: {return_code}")
        return

    if job_type == JobType.NORMAL:
        jobs = [VaspJob(split_vasp_cmd, **vasp_job_kwargs)]
    elif job_type == JobType.DOUBLE_RELAXATION:
        jobs = VaspJob.double_relaxation_run(split_vasp_cmd, **vasp_job_kwargs)
    elif job_type == JobType.METAGGA_OPT:
        jobs = VaspJob.metagga_opt_run(split_vasp_cmd, **vasp_job_kwargs)
    elif job_type == JobType.FULL_OPT:
        jobs = VaspJob.full_opt_run(split_vasp_cmd, **vasp_job_kwargs)
    else:
        raise ValueError(f"Unsupported {job_type=}")

    if wall_time is not None:
        handlers = [*handlers, WalltimeHandler(wall_time=wall_time)]

    custodian_manager = Custodian(
        handlers,
        jobs,
        validators=validators,
        max_errors=max_errors,
        scratch_dir=scratch_dir,
        **custodian_kwargs,
    )

    logger.info("Running VASP using custodian.")
    custodian_manager.run()


def should_stop_children(
    task_document: TaskDoc,
    handle_unsuccessful: bool | str = SETTINGS.VASP_HANDLE_UNSUCCESSFUL,
) -> bool:
    """
    Parse VASP outputs and decide whether child jobs should continue.

    Parameters
    ----------
    task_document : .TaskDoc
        A VASP task document.
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
