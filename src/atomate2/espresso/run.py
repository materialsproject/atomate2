"""Functions to run Quantum ESPRESSO."""

from __future__ import annotations

import logging
import subprocess
from os.path import expandvars

logger = logging.getLogger(__name__)


def run_espresso(command: str, input_file: str, output_file: str) -> None:
    """
    Run a Quantum ESPRESSO program in the current directory.

    Parameters
    ----------
    command : str
        The command that runs the program, e.g. "mpirun -n 16 pw.x".
    input_file : str
        The name of the input file.
    output_file : str
        The name of the file that the output is written to.
    """
    command = f"{expandvars(command)} -in {input_file} > {output_file}"
    logger.info(f"Running command: {command}")
    subprocess.run(command, shell=True, check=True)  # noqa: S602
