"""An SIESTA jobflow runner."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from os.path import expandvars
from typing import TYPE_CHECKING

# from ase.calculators.siesta import Siesta
# Had to bring the ASE siesta stuff for vibra
# from atomate2.siesta.sets.siesta import Siesta
from atomate2.siesta.sets.ase import Siesta

from ase.calculators.socketio import SocketIOCalculator
from monty.json import MontyDecoder
from pymatgen.io.ase import AseAtomsAdaptor

from atomate2.siesta import SETTINGS

if TYPE_CHECKING:
    from pymatgen.core import Molecule, Structure
    from atomate2.siesta.schemas.task import SiestaTaskDoc

logger = logging.getLogger(__name__)


def run_siesta(
    siesta_cmd: str = None,
) -> None:
    """
    Run SIESTA.

    Parameters
    ----------
    siesta_cmd : str
        The command used to run SIESTA (defaults to SETTINGS.SIESTA_CMD).
    """
    logger.info("run_siesta()")
    if siesta_cmd is None:
        siesta_cmd = SETTINGS.SIESTA_CMD

    siesta_cmd = expandvars(siesta_cmd)
    logger.info(f"Running command: {siesta_cmd}")
    # return_code = subprocess.call(["/bin/bash", "-c", siesta_cmd], env=os.environ)
    # Redirect stdout and stderr to /dev/null to suppress output
    # with open(os.devnull, 'w') as devnull:
    #     return_code = subprocess.call(
    #         ["/bin/bash", "-c", siesta_cmd],
    #         env=os.environ,
    #         capture_output=True,
    #         stdout=devnull,
    #         stderr=devnull
    #     )
    # Capture stdout and stderr without displaying
    result = subprocess.run(
        ["/bin/bash", "-c", siesta_cmd], env=os.environ, capture_output=True, text=True
    )
    # Optionally log the output for debugging
    if result.stdout:
        logger.debug(f"SIESTA stdout: {result.stdout}")
        logger.info(f"SIESTA stdout: {result.stdout}")
    if result.stderr:
        logger.debug(f"SIESTA stderr: {result.stderr}")
        logger.info(f"SIESTA stderr: {result.stderr}")
    logger.info(f"Finished running with return code: {result}")


def should_stop_children(
    task_document: SiestaTaskDoc,
    handle_unsuccessful: bool | str = True,
) -> bool:
    """
    Decide whether child jobs should continue.

    Parameters
    ----------
    task_document : .TaskDoc
        An SIESTA task document.
    handle_unsuccessful : bool or str
        This is a three-way toggle on what to do if your job looks OK, but is actually
        not converged (either electronic or ionic):

        - `True`: Mark job as completed, but stop children.
        - `False`: Do nothing, continue with workflow as normal.
        - `"error"`: Throw an error.

    Returns
    -------
    bool
        Whether to stop child jobs.
    """
    logger.info("should_stop_children()")
    if task_document.state == "successful":
        return False

    if isinstance(handle_unsuccessful, bool):
        return handle_unsuccessful

    if handle_unsuccessful == "error":
        raise RuntimeError("Job was not successful (not converged)!")

    raise RuntimeError(f"Unknown option for handle_unsuccessful: {handle_unsuccessful}")


def run_siesta_socket(
    structures_to_calculate: list[Structure | Molecule], siesta_cmd: str = None
) -> None:
    """
    Use the ASE interface to run SIESTA from the socket.

    Parameters
    ----------
    structures_to_calculate: list[Structure or Molecule]
        The list of structures to run scf calculations on
    aims_cmd: str
        The aims command to use (defaults to SETTINGS.SIESTA_CMD).
    """
    logger.info("run_siesta_socket()")
    ase_adaptor = AseAtomsAdaptor()
    atoms_to_calculate = [
        ase_adaptor.get_atoms(structure) for structure in structures_to_calculate
    ]

    with open("parameters.json") as param_file:
        parameters = json.load(param_file, cls=MontyDecoder)
    if siesta_cmd:
        parameters["siesta_command"] = siesta_cmd
    elif "siesta_command" not in parameters:
        parameters["siesta_command"] = SETTINGS.SIESTA_CMD

    calculator = Siesta(**parameters)
    port = parameters["use_pimd_wrapper"][1]
    atoms = atoms_to_calculate[0].copy()

    with SocketIOCalculator(calc=calculator, port=port) as calc:
        for atoms_calc in atoms_to_calculate:
            # Delete prior calculation results
            calc.results.clear()

            # Reset atoms information to the new cell
            atoms.info = atoms_calc.info
            atoms.cell = atoms_calc.cell
            atoms.positions = atoms_calc.positions

            calc.calculate(atoms, system_changes=["positions", "cell"])

        calc.close()


def run_vibra(vibra_cmd: str = None):
    """
    Function to run the VIBRA command.
    """
    logger.info("run_vibra()")
    try:
        if vibra_cmd is None:
            vibra_command = SETTINGS.VIBRA_CMD
        process = subprocess.run(vibra_command, shell=True, check=True)
        if process.returncode == 0:
            logger.info("VIBRA ran successfully!")
        else:
            logger.error("VIBRA returned a non-zero exit code.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running VIBRA: {e}")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")


def run_optical_input(optical_input_cmd: str = None):
    """
    Function to run the Optical_input command.
    """
    logger.info("run_optical_input()")
    try:
        if optical_input_cmd is None:
            optical_input_command = SETTINGS.OPTICAL_INPUT_CMD

        process = subprocess.run(optical_input_command, shell=True, check=True)
        if process.returncode == 0:
            logger.info("Optical_input ran successfully!")
        else:
            logger.error("Optical_input returned a non-zero exit code.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Optical_input: {e}")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")


def run_optical(optical_cmd: str = None):
    """
    Function to run the Optical_input command.
    """
    logger.info("run_optical()")
    try:
        if optical_cmd is None:
            optical_command = SETTINGS.OPTICAL_CMD

        process = subprocess.run(optical_command, shell=True, check=True)
        if process.returncode == 0:
            logger.info("Optical ran successfully!")
        else:
            logger.error("Optical returned a non-zero exit code.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Optical: {e}")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
