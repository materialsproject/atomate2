"""Functions to run ABINIT."""

from __future__ import annotations

import logging
import subprocess
import time

from abipy.flowtk.qutils import time2slurm

from atomate2 import SETTINGS
from atomate2.abinit.utils.common import (
    ANADDB_INPUT_FILE_NAME,
    INPUT_FILE_NAME,
    LOG_FILE_NAME,
    MRGDDB_INPUT_FILE_NAME,
    STDERR_FILE_NAME,
)

__all__ = ["run_abinit", "run_mrgddb", "run_anaddb"]


SLEEP_TIME_STEP = 30


logger = logging.getLogger(__name__)


def run_abinit(
    abinit_cmd: str = None,
    mpirun_cmd: str = None,
    wall_time: int = None,
    start_time: float = None,
) -> None:
    """Run ABINIT."""
    abinit_cmd = abinit_cmd or SETTINGS.ABINIT_CMD
    mpirun_cmd = mpirun_cmd or SETTINGS.ABINIT_MPIRUN_CMD
    command = []
    if mpirun_cmd:
        command.extend(mpirun_cmd.split())
    command.append(abinit_cmd)
    start_time = start_time or time.time()

    max_end_time = 0.0
    if wall_time is not None:
        abinit_timelimit = wall_time
        if abinit_timelimit > 480:
            # TODO: allow tuning this timelimit buffer for abinit,
            #  e.g. using a config variable or possibly per job
            abinit_timelimit -= 240
        command.extend(["--timelimit", time2slurm(abinit_timelimit)])
        max_end_time = start_time + wall_time

    command.append(INPUT_FILE_NAME)

    with open(LOG_FILE_NAME, "w") as stdout, open(STDERR_FILE_NAME, "w") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)  # noqa: S603

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
    mrgddb_cmd: str = None,
    mpirun_cmd: str = None,
    wall_time: int = None,
    start_time: float = None,
) -> None:
    """Run mrgddb."""
    mrgddb_cmd = mrgddb_cmd or SETTINGS.ABINIT_MRGDDB_CMD
    mpirun_cmd = mpirun_cmd or SETTINGS.ABINIT_MPIRUN_CMD
    command = []
    if mpirun_cmd:
        command.extend(mpirun_cmd.split())
        command.extend(["-n", "1"])
    command.extend([mrgddb_cmd, "--nostrict"])
    start_time = start_time or time.time()

    max_end_time = 0.0
    if wall_time is not None:
        mrgddb_timelimit = wall_time
        if mrgddb_timelimit > 480:
            # TODO: allow tuning this timelimit buffer for mrgddb,
            #  e.g. using a config variable or possibly per job
            mrgddb_timelimit -= 240
        command.extend(["--timelimit", time2slurm(mrgddb_timelimit)])
        max_end_time = start_time + wall_time

    with (
        open(MRGDDB_INPUT_FILE_NAME) as stdin,
        open(LOG_FILE_NAME, "w") as stdout,
        open(STDERR_FILE_NAME, "w") as stderr,
    ):
        process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=stderr)  # noqa: S603

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


def run_anaddb(
    anaddb_cmd: str = None,
    mpirun_cmd: str = None,
    wall_time: int = None,
    start_time: float = None,
) -> None:
    """Run anaddb."""
    anaddb_cmd = anaddb_cmd or SETTINGS.ABINIT_ANADDB_CMD
    mpirun_cmd = mpirun_cmd or SETTINGS.ABINIT_MPIRUN_CMD
    command = []
    if mpirun_cmd:
        command.extend(mpirun_cmd.split())
        command.extend(["-n", "1"])
    command.extend([anaddb_cmd, "anaddb.in"])
    start_time = start_time or time.time()

    max_end_time = 0.0
    if wall_time is not None:
        anaddb_timelimit = wall_time
        if anaddb_timelimit > 480:
            # TODO: allow tuning this timelimit buffer for anaddb,
            #  e.g. using a config variable or possibly per job
            anaddb_timelimit -= 240
        command.extend(["--timelimit", time2slurm(anaddb_timelimit)])
        max_end_time = start_time + wall_time

    with (
        open(ANADDB_INPUT_FILE_NAME) as stdin,
        open(LOG_FILE_NAME, "w") as stdout,
        open(STDERR_FILE_NAME, "w") as stderr,
    ):
        # process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=stderr)
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)  # noqa: S603

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
