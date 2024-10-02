"""Utilities for working with the OPLS forcefield in OpenMM."""

from __future__ import annotations

import io
import re
import tempfile
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import openmm.unit as omm_unit
from emmet.core.openmm import OpenMMInterchange
from openmm import LangevinMiddleIntegrator, XmlSerializer
from openmm.app import PDBFile

if TYPE_CHECKING:
    from emmet.core.openmm import OpenMMTaskDocument
    from openff.interchange import Interchange


def download_opls_xml(
    names_smiles: dict[str, str], output_dir: str | Path, overwrite_files: bool = False
) -> None:
    """Download an OPLS-AA/M XML file from the LigParGen website using Selenium."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service as ChromeService
        from selenium.webdriver.common.by import By
        from webdriver_manager.chrome import ChromeDriverManager

    except ImportError:
        warnings.warn(
            "The `selenium` package is not installed. "
            "It's required to run the opls web scraper.",
            stacklevel=1,
        )

    # Initialize the Chrome driver
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

    for name, smiles in names_smiles.items():
        final_file = Path(output_dir) / f"{name}.xml"
        if final_file.exists() and not overwrite_files:
            continue
        try:
            # Specify the directory where you want to download files
            with tempfile.TemporaryDirectory() as tmpdir:
                download_dir = tmpdir

                # Set up Chrome options
                chrome_options = webdriver.ChromeOptions()
                prefs = {"download.default_directory": download_dir}
                chrome_options.add_experimental_option("prefs", prefs)

                # Initialize Chrome with the options
                driver = webdriver.Chrome(options=chrome_options)

                # Open the first webpage
                driver.get("https://zarbi.chem.yale.edu/ligpargen/")

                # Find the SMILES input box and enter the SMILES code
                smiles_input = driver.find_element(By.ID, "smiles")
                smiles_input.send_keys(smiles)

                # Find and click the "Submit Molecule" button
                submit_button = driver.find_element(
                    By.XPATH,
                    '//button[@type="submit" and contains(text(), "Submit Molecule")]',
                )
                submit_button.click()

                # Wait for the second page to load
                # time.sleep(2)  # Adjust this delay as needed based on the loading time

                # Find and click the "XML" button under Downloads and OpenMM
                xml_button = driver.find_element(
                    By.XPATH, '//input[@type="submit" and @value="XML"]'
                )
                xml_button.click()

                # Wait for the file to download
                time.sleep(0.3)  # Adjust as needed based on the download time

                file = next(Path(tmpdir).iterdir())
                # copy downloaded file to output_file using os
                Path(file).rename(final_file)

        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"{name} ({smiles}) failed to download because an error occurred: {e}",
                stacklevel=1,
            )

    driver.quit()


def create_list_summing_to(total_sum: int, n_pieces: int) -> list:
    """Create a NumPy array with n_pieces elements that sum up to total_sum.

    Divides total_sum by n_pieces to determine the base value for each element.
    Distributes the remainder evenly among the elements.

    Parameters
    ----------
    total_sum : int
        The desired sum of the array elements.
    n_pieces : int
        The number of elements in the array.

    Returns
    -------
    numpy.ndarray
        A 1D NumPy array with n_pieces elements summing up to total_sum.
    """
    div, mod = total_sum // n_pieces, total_sum % n_pieces
    return [div + 1] * mod + [div] * (n_pieces - mod)


def increment_name(file_name: str) -> str:
    """Increment the count in a file name."""
    # logic to increment count on file name
    re_match = re.search(r"(\d*)$", file_name)
    position = re_match.start(1)
    new_count = int(re_match.group(1) or 1) + 1
    return f"{file_name[:position]}{new_count}"


def task_reports(task: OpenMMTaskDocument, traj_or_state: str = "traj") -> bool:
    """Check if a task reports trajectories or states."""
    if not task.calcs_reversed:
        return False
    calc_input = task.calcs_reversed[0].input
    if traj_or_state == "traj":
        report_freq = calc_input.traj_interval
    elif traj_or_state == "state":
        report_freq = calc_input.state_interval
    else:
        raise ValueError("traj_or_state must be 'traj' or 'state'")
    return calc_input.n_steps >= report_freq


def openff_to_openmm_interchange(
    openff_interchange: Interchange,
) -> OpenMMInterchange:
    """Convert an OpenFF Interchange object to an OpenMM Interchange object."""
    integrator = LangevinMiddleIntegrator(
        300 * omm_unit.kelvin,
        10.0 / omm_unit.picoseconds,
        1.0 * omm_unit.femtoseconds,
    )
    sim = openff_interchange.to_openmm_simulation(integrator)
    state = sim.context.getState(
        getPositions=True,
        getVelocities=True,
        enforcePeriodicBox=True,
    )
    with io.StringIO() as buffer:
        PDBFile.writeFile(
            sim.topology,
            np.zeros(shape=(sim.topology.getNumAtoms(), 3)),
            file=buffer,
        )
        buffer.seek(0)
        pdb = buffer.read()

        return OpenMMInterchange(
            system=XmlSerializer.serialize(sim.system),
            state=XmlSerializer.serialize(state),
            topology=pdb,
        )
