"""
Schemas for SIESTA calculation objects, including task states, output parsing,
and the main calculation schema that handles SIESTA outputs.
"""

from __future__ import annotations


import logging
import os
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Optional
from typing import Union

from sisl.io.siesta import stdoutSileSiesta
from sisl.io.siesta import xvSileSiesta

from emmet.core.math import Matrix3D
from emmet.core.math import Vector3D
from jobflow.utils import ValueEnum
from pydantic import BaseModel
from pydantic import Field
from pymatgen.core import Structure
from typing_extensions import Self

from atomate2.siesta.files import read_directly_from_siesta_out
from atomate2.common.jobs.eos import extract_siesta_timing
from rich.console import Console

logger = logging.getLogger(__name__)


class TaskState(ValueEnum):
    """SIESTA calculation state."""

    SUCCESS = "successful"
    FAILED = "failed"
    UNCONVERGED = "unconverged"
    RUNNING = "running"


class SiestaObject(ValueEnum):
    """Types of SIESTA data objects."""

    DOS = "dos"  # Density of states
    BAND_STRUCTURE = "band_structure"  # Bandstructure
    ELECTRON_DENSITY = "electron_density"  # e_density
    WFN = "wfn"  # Wavefunction file
    TRAJECTORY = "trajectory"  # Atomic trajectory file


class CalculationOutput(BaseModel):
    """Document defining SIESTA calculation outputs.

    Parameters
    ----------
    energy: float
        The final total DFT energy for the calculation
    energy_per_atom: float
        The final DFT energy per atom for the calculation
    structure: Structure
        The final pymatgen Structure of the system
    efermi: float
        The Fermi level from the calculation in eV
    forces: List[Vector3D]
        Forces acting on each atom
    stress: Matrix3D
        The stress on the cell
    is_metal: bool
        Whether the system is metallic
    bandgap: float
        The band gap from the calculation in eV
    cbm: float
        The conduction band minimum in eV (if system is not metallic
    vbm: float
        The valence band maximum in eV (if system is not metallic)
    run_time: float
        Wall time for the calculation in seconds
    """

    total_energy: Optional[float] = Field(
        None, description="The final total DFT energy for the calculation"
    )

    structure: Union[Structure] = Field(
        None, description="The final structure from the calculation"
    )

    efermi: Optional[float] = Field(
        None, description="The Fermi level from the calculation in eV"
    )

    forces: Optional[list[Vector3D]] = Field(
        None, description="Forces acting on each atom"
    )
    stress: Optional[Matrix3D] = Field(None, description="The stress on the cell")

    bandgap: Optional[float] = Field(
        None, description="The band gap from the calculation in eV"
    )
    direct_bandgap: Optional[float] = Field(
        None, description="The direct band gap from the calculation in eV"
    )
    cbm: Optional[float] = Field(
        None,
        description="The conduction band minimum, or LUMO for molecules, in eV "
        "(if system is not metallic)",
    )
    vbm: Optional[float] = Field(
        None,
        description="The valence band maximum, or HOMO for molecules, in eV "
        "(if system is not metallic)",
    )
    run_time: Optional[float] = Field(
        None, description="Wall time for the calculation in seconds"
    )

    @classmethod
    def from_siesta_out(
        cls,
        siesta_output: stdoutSileSiesta,
        siesta_XV: xvSileSiesta,
        dir_name: Path | str | None = None,
    ) -> Self:
        """
        Create a SIESTA output document from SIESTA output files.

        Parameters
        ----------
        siesta_output : stdoutSileSiesta
            Parsed content of the siesta.out file.
        siesta_XV : xvSileSiesta
            Parsed content of the XV file for final structure geometry.
        dir_name : Path | str | None, optional
            Directory containing SIESTA output files. If provided, will attempt
            to read band gap information from siesta.EIG file.

        Returns
        -------
        Self : CalculationOutput
            The constructed SIESTA calculation output document.
        """
        logger.info("CalculationOutput.from_siesta_out()")
        try:
            sisl_structure = siesta_XV.read_geometry()  # final structure
        except Exception as e:
            logger.error(f"Cannot read final structure: {e}")
            # Return partial output with None structure for database compatibility
            return cls(
                structure=None,
                efermi=None,
                total_energy=None,
                forces=None,
                stress=None,
            )

        structure = sisl_structure.to.pymatgen()

        # In case no conduction bands were included
        try:
            electronic_output = siesta_output.read_energy()
        except ValueError:
            electronic_output = {}

        # Extract forces if available
        forces = None
        if siesta_output.read_force[-1](total=False) is not None:
            forces = siesta_output.read_force[-1](total=False)

        # Extract stress if available
        stress = None
        if siesta_output.read_stress(skip_final=False) is not None:
            stress = siesta_output.read_stress(skip_final=False)

        # Extract energy and Fermi level from electronic output
        total_energy = electronic_output.get("total") if electronic_output else None
        efermi = electronic_output.get("fermi") if electronic_output else None

        # Try to extract band gap information from .EIG file if directory is provided
        bandgap = None
        cbm = None
        vbm = None
        direct_bandgap = None

        if dir_name is not None:
            from atomate2.siesta.utils.eig_parser import get_band_gap_from_eig

            eig_file = Path(dir_name) / "siesta.EIG"
            if eig_file.exists():
                try:
                    gap_info = get_band_gap_from_eig(eig_file)
                    vbm = gap_info["vbm"]
                    cbm = gap_info["cbm"]
                    bandgap = gap_info["gap"]
                    # For now, assume direct_bandgap equals bandgap
                    # (proper direct gap calculation would require k-point analysis)
                    direct_bandgap = bandgap
                    logger.info(
                        f"Extracted band gap from EIG: VBM={vbm:.3f} eV, "
                        f"CBM={cbm:.3f} eV, gap={bandgap:.3f} eV"
                    )
                except Exception as e:
                    logger.warning(f"Could not parse band gap from {eig_file}: {e}")

        # Extract wall time from SIESTA output
        run_time = None
        if dir_name is not None:
            run_time = extract_siesta_timing(dir_name)
            if run_time is not None:
                logger.info(f"Extracted run time: {run_time:.1f} seconds")

        # Return populated CalculationOutput instance
        return cls(
            structure=structure,
            efermi=efermi,
            total_energy=total_energy,
            forces=forces,
            stress=stress,
            bandgap=bandgap,
            cbm=cbm,
            vbm=vbm,
            direct_bandgap=direct_bandgap,
            run_time=run_time,
        )


class Calculation(BaseModel):
    """Document defining full SIESTA calculation details.

    Parameters
    ----------
    dir_name: str
        The directory for this Siesta calculation
    siesta_version: str
        Siesta version used to perform the calculation
    has_siesta_completed: .TaskState
        Whether Siesta completed the calculation successfully
    output: .CalculationOutput
        The Siesta calculation output
    completed_at: str
        Timestamp for when the calculation was completed
    output_file_paths: Dict[str, str]
        Paths (relative to dir_name) of the Siesta output files
        associated with this calculation
    """

    dir_name: str = Field(None, description="The directory for this Siesta calculation")
    siesta_version: str = Field(
        None, description="Siesta version used to perform the calculation"
    )
    has_siesta_completed: TaskState = Field(
        None, description="Whether Siesta completed the calculation successfully"
    )
    output: CalculationOutput = Field(None, description="The Siesta calculation output")
    completed_at: str = Field(
        None, description="Timestamp for when the calculation was completed"
    )
    output_file_paths: Optional[dict[str, str]] = Field(
        None,
        description="Paths (relative to dir_name) of the Siesta output files "
        "associated with this calculation",
    )

    @classmethod
    def from_siesta_files(
        cls,
        dir_name: Path | str,
        # task_name: str,
        siesta_output_file: Path | str = "siesta.out",
        siesta_MESSAGES_file: Path | str = "MESSAGES",
        siesta_xv_file: Path | str = "siesta.XV",
    ):
        """Create an Siesta calculation document from a directory and file paths.

        Parameters
        ----------
        dir_name: Path or str
            The directory containing the calculation outputs.
        task_name: str
            The task name.
        Siesta_gsr_file: Path or str
            Path to the GSR output of Siesta job, relative to dir_name.
        Siesta_log_file: Path or str
            Path to the main log of Siesta job, relative to dir_name.
        Siesta_abort_file: Path or str
            Path to the main abort file of Siesta job, relative to dir_name.

        Returns
        -------
        .Calculation
            An Siesta calculation document.
        """
        logger.info("Calculation.from_siesta_files()")
        dir_name = Path(dir_name)
        siesta_output_file = dir_name / siesta_output_file
        siesta_xv_file = dir_name / siesta_xv_file
        siesta_MESSAGES_file = dir_name / siesta_MESSAGES_file

        siesta_output = stdoutSileSiesta(siesta_output_file)
        siesta_xv = xvSileSiesta(siesta_xv_file)

        completed_at = str(
            datetime.fromtimestamp(
                os.stat(siesta_MESSAGES_file).st_mtime, tz=timezone.utc
            )
        )

        output_doc = CalculationOutput.from_siesta_out(
            siesta_output, siesta_xv, dir_name=dir_name
        )

        # Check calculation state from MESSAGES file
        has_siesta_completed = check_siesta_messages(siesta_MESSAGES_file)

        # Reading Siesta Version:
        siesta_version = read_directly_from_siesta_out(
            siesta_output_file, what="Version"
        )["Version"]

        return cls(
            dir_name=str(dir_name),
            siesta_version=siesta_version,
            has_siesta_completed=has_siesta_completed,
            completed_at=completed_at,
            output=output_doc,
        )


def check_siesta_messages(messages_file):
    """
    Checks the status of a SIESTA run based on message logs from the MESSAGES file.

    Args:
        messages_file (str): Path to the `MESSAGES` file.

    Returns:
        TaskState: The overall status of the SIESTA run.
    """

    def read_messages_from_siesta(file_path, keywords):
        """
        Reads a file and extracts messages based on provided keywords.

        Args:
            file_path (str): Path to the SIESTA messages file.
            keywords (list): List of keywords to extract from the file.

        Returns:
            dict: Dictionary with keyword as key and list of matching lines as value.
        """
        logger.info("check_siesta_messages.read_messages_from_siesta()")
        extracted_messages = {key: [] for key in keywords}

        with open(file_path, "r") as file:
            for line in file:
                for keyword in keywords:
                    if keyword in line:
                        extracted_messages[keyword].append(line.strip())

        return extracted_messages

    keywords = ["INFO", "FATAL", "ABNORMAL_TERMINATION", "WARNING"]

    # Check the normal messages file
    messages = read_messages_from_siesta(messages_file, keywords)

    console = Console()

    # Check for specific SCF non-convergence fatal error
    if "FATAL" in messages:
        for fatal_msg in messages["FATAL"]:
            if "SCF_NOT_CONV" in fatal_msg:
                return TaskState.UNCONVERGED
            if "GGAXC: Unknown author CA" in fatal_msg:
                return TaskState.FAILED

    # Check for fatal errors (ensure the list is not empty)
    if messages.get("FATAL") or messages.get("ABNORMAL_TERMINATION"):
        if messages["FATAL"] or messages["ABNORMAL_TERMINATION"]:
            return TaskState.FAILED

    # Print only unique WARNING messages
    unique_warnings = list(dict.fromkeys(messages["WARNING"]))  # Remove duplicates
    if unique_warnings:
        for warning in unique_warnings:
            console.print(f"[bold yellow]{warning}[/bold yellow]")

    # Check for job completion
    if "INFO" in messages and any("Job completed" in msg for msg in messages["INFO"]):
        return TaskState.SUCCESS

    # If no errors or completion, assume job is still running
    return TaskState.RUNNING
