"""Functions to run ABINIT."""

from __future__ import annotations

import logging
import subprocess
import time

from abipy.flowtk.qutils import time2slurm

from atomate2 import SETTINGS
from atomate2.abinit.utils.common import (
    INPUT_FILE_NAME,
    LOG_FILE_NAME,
    MRGDDB_INPUT_FILE_NAME,
    MRGDV_INPUT_FILE_NAME,
    STDERR_FILE_NAME,
)

__all__ = ["run_abinit", "run_anaddb", "run_mrgddb", "run_mrgdv"]


SLEEP_TIME_STEP = 30


logger = logging.getLogger(__name__)


def run_abinit(
    abinit_cmd: str | None = None,
    mpirun_cmd: str | None = None,
    wall_time: int | None = None,
    start_time: float | None = None,
) -> None:
    """
    Run ABINIT calculation.

    Executes ABINIT with optional MPI support and wall time management.
    If a wall time is specified, the process will be monitored and
    terminated before exceeding the limit.

    Parameters
    ----------
    abinit_cmd : str or None
        ABINIT executable command. If None, uses SETTINGS.ABINIT_CMD.
        Default is None.
    mpirun_cmd : str or None
        MPI run command. If None, uses SETTINGS.ABINIT_MPIRUN_CMD.
        Default is None.
    wall_time : int or None
        Wall time limit in seconds. If provided, ABINIT will be given a
        timelimit and the process will be monitored. Default is None.
    start_time : float or None
        Start time timestamp. If None, uses current time. Default is None.

    Notes
    -----
    When wall_time is specified and exceeds 480 seconds, a 240-second buffer
    is subtracted to allow ABINIT to cleanly finish before the hard limit.
    This buffer time is currently hardcoded, but a future implementation could
    make it customizable through configuration variables or per-job settings.
    """
    abinit_cmd = abinit_cmd or SETTINGS.ABINIT_CMD
    mpirun_cmd = mpirun_cmd or SETTINGS.ABINIT_MPIRUN_CMD
    command = []
    if mpirun_cmd:
        command.extend(mpirun_cmd.split())
    command.append(abinit_cmd)
    start_time = start_time or time.time()

    if wall_time is not None:
        abinit_timelimit = wall_time
        if abinit_timelimit > 480:
            abinit_timelimit -= 240
        command.extend(["--timelimit", time2slurm(abinit_timelimit)])
        max_end_time = start_time + wall_time
    else:
        max_end_time = 0.0

    command.append(INPUT_FILE_NAME)

    with open(LOG_FILE_NAME, "w") as stdout, open(STDERR_FILE_NAME, "w") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)

        if wall_time is not None:
            while True:
                time.sleep(SLEEP_TIME_STEP)
                if process.poll() is not None:
                    break
                current_time = time.time()
                remaining_time = max_end_time - current_time
                if remaining_time < 5 * SLEEP_TIME_STEP:
                    process.terminate()

        process.wait()


def run_mrgddb(
    mrgddb_cmd: str | None = None,
    mpirun_cmd: str | None = None,
    start_time: float | None = None,
) -> None:
    """
    Run MRGDDB utility to merge DDB files.

    Executes the MRGDDB utility with optional MPI support. If MPI is used
    without explicit process count specification, defaults to single process.

    Parameters
    ----------
    mrgddb_cmd : str or None
        MRGDDB executable command. If None, uses SETTINGS.ABINIT_MRGDDB_CMD.
        Default is None.
    mpirun_cmd : str or None
        MPI run command. If None, uses SETTINGS.ABINIT_MPIRUN_CMD.
        Default is None.
    start_time : float or None
        Start time timestamp. If None, uses current time. Default is None.
    """
    mrgddb_cmd = mrgddb_cmd or SETTINGS.ABINIT_MRGDDB_CMD
    mpirun_cmd = mpirun_cmd or SETTINGS.ABINIT_MPIRUN_CMD
    command = []
    if mpirun_cmd:
        if not any(opt in mpirun_cmd.split() for opt in ("-c", "-n", "--n", "-np")):
            mpirun_cmd += " -n 1"
        command.extend(mpirun_cmd.split())
    command.extend([mrgddb_cmd, "--nostrict"])
    _start_time = start_time or time.time()

    with (
        open(MRGDDB_INPUT_FILE_NAME) as stdin,
        open(LOG_FILE_NAME, "w") as stdout,
        open(STDERR_FILE_NAME, "w") as stderr,
    ):
        process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=stderr)

        process.wait()


def run_mrgdv(
    mrgdv_cmd: str | None = None,
    mpirun_cmd: str | None = None,
    start_time: float | None = None,
) -> None:
    """
    Run MRGDV utility to merge potential derivative files.

    Executes the MRGDV utility with optional MPI support. If MPI is used
    without explicit process count specification, defaults to single process.

    Parameters
    ----------
    mrgdv_cmd : str or None
        MRGDV executable command. If None, uses SETTINGS.ABINIT_MRGDV_CMD.
        Default is None.
    mpirun_cmd : str or None
        MPI run command. If None, uses SETTINGS.ABINIT_MPIRUN_CMD.
        Default is None.
    start_time : float or None
        Start time timestamp. If None, uses current time. Default is None.
    """
    mrgdv_cmd = mrgdv_cmd or SETTINGS.ABINIT_MRGDV_CMD
    mpirun_cmd = mpirun_cmd or SETTINGS.ABINIT_MPIRUN_CMD
    command = []
    if mpirun_cmd:
        if not any(opt in mpirun_cmd.split() for opt in ("-c", "-n", "--n", "-np")):
            mpirun_cmd += " -n 1"
        command.extend(mpirun_cmd.split())
    command.append(mrgdv_cmd)
    _start_time = start_time or time.time()

    with (
        open(MRGDV_INPUT_FILE_NAME) as stdin,
        open(LOG_FILE_NAME, "w") as stdout,
        open(STDERR_FILE_NAME, "w") as stderr,
    ):
        process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=stderr)

        process.wait()


def run_anaddb(
    anaddb_cmd: str | None = None,
    mpirun_cmd: str | None = None,
    start_time: float | None = None,
) -> None:
    """
    Run ANADDB utility for post-processing derivative databases.

    Executes the ANADDB utility with optional MPI support to analyze
    derivative databases (DDB files) and compute various physical properties.

    Parameters
    ----------
    anaddb_cmd : str or None
        ANADDB executable command. If None, uses SETTINGS.ABINIT_ANADDB_CMD.
        Default is None.
    mpirun_cmd : str or None
        MPI run command. If None, uses SETTINGS.ABINIT_MPIRUN_CMD.
        Default is None.
    start_time : float or None
        Start time timestamp. If None, uses current time. Default is None.
    """
    anaddb_cmd = anaddb_cmd or SETTINGS.ABINIT_ANADDB_CMD
    mpirun_cmd = mpirun_cmd or SETTINGS.ABINIT_MPIRUN_CMD
    command = []
    if mpirun_cmd:
        command.extend(mpirun_cmd.split())
    command.extend([anaddb_cmd, "anaddb.in"])
    _start_time = start_time or time.time()

    with (
        open(LOG_FILE_NAME, "w") as stdout,
        open(STDERR_FILE_NAME, "w") as stderr,
    ):
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)

        process.wait()
