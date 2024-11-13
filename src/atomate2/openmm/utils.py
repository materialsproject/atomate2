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
from openmm import LangevinMiddleIntegrator, State, XmlSerializer
from openmm.app import PDBFile, Simulation
from pymatgen.core.trajectory import Trajectory

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
                # time.sleep(2)  # Adjust this delay as needed based on loading time

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


class PymatgenTrajectoryReporter:
    """Reporter that creates a pymatgen Trajectory from an OpenMM simulation.

    Accumulates structures and velocities during the simulation and writes them to a
    Trajectory object when the reporter is deleted.
    """

    def __init__(
        self,
        file: str | Path,
        reportInterval: int,  # noqa: N803
        enforcePeriodicBox: bool | None = None,  # noqa: N803
    ) -> None:
        """Initialize the reporter.

        Parameters
        ----------
        file : str | Path
            The file to write the trajectory to
        reportInterval : int
            The interval (in time steps) at which to save frames
        enforcePeriodicBox : bool | None
            Whether to wrap coordinates to the periodic box. If None, determined from
            simulation settings.
        """
        self._file = file
        self._reportInterval = reportInterval
        self._enforcePeriodicBox = enforcePeriodicBox
        self._topology = None
        self._nextModel = 0

        # Storage for trajectory data
        self._positions: list[np.ndarray] = []
        self._velocities: list[np.ndarray] = []
        self._lattices: list[np.ndarray] = []
        self._frame_properties: list[dict] = []
        self._species: list[str] | None = None
        self._time_step: float | None = None

    def describeNextReport(  # noqa: N802
        self, simulation: Simulation
    ) -> tuple[int, bool, bool, bool, bool, bool]:
        """Get information about the next report this object will generate.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for

        Returns
        -------
        tuple[int, bool, bool, bool, bool, bool]
            A six element tuple. The first element is the number of steps until the
            next report. The remaining elements specify whether that report will
            require positions, velocities, forces, energies, and periodic box info.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return steps, True, True, False, True, self._enforcePeriodicBox

    def report(self, simulation: Simulation, state: State) -> None:
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        if self._nextModel == 0:
            self._topology = simulation.topology
            self._species = [
                atom.element.symbol for atom in simulation.topology.atoms()
            ]
            self._time_step = (
                simulation.integrator.getStepSize() * self._reportInterval
            ).value_in_unit(omm_unit.femtoseconds)

        # Get positions and velocities in Angstrom and Angstrom/fs
        positions = state.getPositions(asNumpy=True).value_in_unit(omm_unit.angstrom)
        velocities = state.getVelocities(asNumpy=True).value_in_unit(
            omm_unit.angstrom / omm_unit.femtosecond
        )
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
            omm_unit.angstrom
        )

        # Get energies in eV
        kinetic_energy = (
            state.getKineticEnergy() / omm_unit.AVOGADRO_CONSTANT_NA
        ).value_in_unit(omm_unit.ev)

        potential_energy = (
            state.getPotentialEnergy() / omm_unit.AVOGADRO_CONSTANT_NA
        ).value_in_unit(omm_unit.ev)

        self._positions.append(positions)
        self._velocities.append(velocities)
        self._lattices.append(box_vectors)
        self._frame_properties.append(
            {
                "kinetic_energy": kinetic_energy,
                "potential_energy": potential_energy,
                "total_energy": kinetic_energy + potential_energy,
            }
        )

        self._nextModel += 1

    def save(self) -> None:
        """Write accumulated trajectory data to a pymatgen Trajectory object."""
        if not self._positions:
            return

        velocities = [
            [tuple(site_vel) for site_vel in frame_vel]
            for frame_vel in self._velocities
        ]

        # Format site properties as list of dicts, one per frame
        site_properties = []
        n_frames = len(self._positions)
        site_properties = [{"velocities": velocities[i]} for i in range(n_frames)]

        # Create trajectory with positions and lattices
        trajectory = Trajectory(
            species=self._species,
            coords=self._positions,
            lattice=self._lattices,
            frame_properties=self._frame_properties,
            site_properties=site_properties,  # Now properly formatted as list of dicts
            time_step=self._time_step,
        )

        # Store trajectory as a class attribute so it can be accessed after deletion
        self.trajectory = trajectory

        # write out trajectory to a file
        with open(self._file, mode="w") as file:
            file.write(trajectory.to_json())
