"""Functions to run ABINIT."""

from __future__ import annotations

import logging
import subprocess
import time

from abipy.flowtk.qutils import time2slurm

from atomate2.abinit.utils.common import (
    INPUT_FILE_NAME,
    LOG_FILE_NAME,
    STDERR_FILE_NAME,
)

__all__ = ["run_abinit"]


SLEEP_TIME_STEP = 30


logger = logging.getLogger(__name__)


def run_abinit(
    abinit_cmd: str = "abinit",
    mpirun_cmd: str = None,
    wall_time: int = None,
    start_time: float = None,
):
    """Run ABINIT."""
    start_time = start_time or time.time()
    if mpirun_cmd is not None:
        command = [mpirun_cmd, abinit_cmd]
    else:
        command = [abinit_cmd]

    max_end_time = 0.0
    if wall_time is not None:
        mytimelimit = wall_time
        if mytimelimit > 240:
            mytimelimit -= 120
        command.extend(["--timelimit", time2slurm(mytimelimit)])
        max_end_time = start_time + wall_time

    command.append(INPUT_FILE_NAME)

    status = "completed"

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
                    status = "killed"

        process.wait()
    return status
