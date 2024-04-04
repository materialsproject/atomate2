"""Functions to run QChem in atomate 2."""

from __future__ import annotations

import logging
import shlex
from os.path import expandvars
from typing import TYPE_CHECKING, Any

from custodian import Custodian
from custodian.qchem.handlers import QChemErrorHandler
from custodian.qchem.jobs import QCJob
from jobflow.utils import ValueEnum

from atomate2 import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from custodian.custodian import ErrorHandler
    from emmet.core.qc_tasks import TaskDoc


_DEFAULT_HANDLERS = (QChemErrorHandler,)

logger = logging.getLogger(__name__)


class JobType(ValueEnum):
    """
    Type of QChem job.

        - ``DIRECT``: Run QChem without using custodian.
        - ``NORMAL``: Normal custodian :obj:`.QCJob`.
    """

    DIRECT = "direct"
    NORMAL = "normal"


def run_qchem(
    job_type: JobType | str = JobType.NORMAL,
    qchem_cmd: str = SETTINGS.QCHEM_CMD,
    max_errors: int = SETTINGS.QCHEM_CUSTODIAN_MAX_ERRORS,
    scratch_dir: str = SETTINGS.CUSTODIAN_SCRATCH_DIR,
    handlers: Sequence[ErrorHandler] = _DEFAULT_HANDLERS,
    # wall_time: int | None = None,
    qchem_job_kwargs: dict[str, Any] = None,
    custodian_kwargs: dict[str, Any] = None,
) -> None:
    """
    Run QChem.

    Supports running QChem with or without custodian (see :obj:`JobType`).

    Parameters
    ----------
    job_type : str or .JobType
        The job type.
    qchem_cmd : str
        The command used to run the standard version of QChem.
    max_errors : int
        The maximum number of errors allowed by custodian.
    scratch_dir : str
        The scratch directory used by custodian.
    handlers : list of .ErrorHandler
        The error handlers used by custodian.
    wall_time : int
        The maximum wall time. If set, a WallTimeHandler will be added to the list
        of handlers.
    qchem_job_kwargs : dict
        Keyword arguments that are passed to :obj:`.QCJob`.
    custodian_kwargs : dict
        Keyword arguments that are passed to :obj:`.Custodian`.
    """
    qchem_job_kwargs = {} if qchem_job_kwargs is None else qchem_job_kwargs
    custodian_kwargs = {} if custodian_kwargs is None else custodian_kwargs

    qchem_cmd = expandvars(qchem_cmd)
    split_qchem_cmd = shlex.split(qchem_cmd)

    if job_type == JobType.NORMAL:
        jobs = [
            QCJob(
                split_qchem_cmd, max_cores=SETTINGS.QCHEM_MAX_CORES, **qchem_job_kwargs
            )
        ]
    else:
        raise ValueError(f"Unsupported job type: {job_type}")

    c = Custodian(
        handlers,
        jobs,
        max_errors=max_errors,
        scratch_dir=scratch_dir,
        **custodian_kwargs,
    )

    logger.info("Running QChem using custodian.")
    c.run()


def should_stop_children(
    task_document: TaskDoc,
    handle_unsuccessful: bool | str = SETTINGS.QCHEM_HANDLE_UNSUCCESSFUL,
) -> bool:
    """
    Parse QChem outputs and decide whether child jobs should continue.

    Parameters
    ----------
    task_document : .TaskDocument
        A QChem task document.
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
        if isinstance(handle_unsuccessful, bool):
            return handle_unsuccessful

        if handle_unsuccessful == "error":
            raise RuntimeError(
                "Job was successful but children jobs need to be stopped!"
            )
        return False

    if task_document.state == "unsuccessful":
        raise RuntimeError(
            "Job was not successful (perhaps your job did not converge within the "
            "limit of electronic/ionic iterations)!"
        )

    raise RuntimeError(f"Unknown option for defuse_unsuccessful: {handle_unsuccessful}")
