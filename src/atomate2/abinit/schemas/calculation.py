"""Schemas for Abinit calculation objects."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from collections.abc import Sequence

    # from abipy.flowtk.events import AbinitCriticalWarning

from abipy.electrons.gsr import GsrFile
from abipy.flowtk import events
from abipy.flowtk.utils import File

# from pymatgen.io.common import VolumetricData
from emmet.core.math import Matrix3D, Vector3D
from jobflow.utils import ValueEnum
from pydantic import BaseModel, Field
from pymatgen.core import Structure

from atomate2.abinit.utils.common import (
    LOG_FILE_NAME,
    MPIABORTFILE,
    get_event_report,
)

# STORE_VOLUMETRIC_DATA = ("total_density",)

logger = logging.getLogger(__name__)


class TaskState(ValueEnum):
    """Abinit calculation state."""

    SUCCESS = "successful"
    FAILED = "failed"
    UNCONVERGED = "unconverged"


class AbinitObject(ValueEnum):
    """Types of Abinit data objects."""

    DOS = "dos"
    BAND_STRUCTURE = "band_structure"
    ELECTRON_DENSITY = "electron_density"  # e_density
    WFN = "wfn"  # Wavefunction file
    TRAJECTORY = "trajectory"


class CalculationOutput(BaseModel):
    """Document defining Abinit calculation outputs.

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
    """

    energy: float = Field(
        None, description="The final total DFT energy for the calculation"
    )
    energy_per_atom: float = Field(
        None, description="The final DFT energy per atom for the calculation"
    )

    structure: Union[Structure] = Field(
        None, description="The final structure from the calculation"
    )

    efermi: float = Field(
        None, description="The Fermi level from the calculation in eV"
    )

    forces: Optional[list[Vector3D]] = Field(
        None, description="Forces acting on each atom"
    )
    stress: Optional[Matrix3D] = Field(None, description="The stress on the cell")
    is_metal: Optional[bool] = Field(None, description="Whether the system is metallic")
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

    @classmethod
    def from_abinit_gsr(
        cls,
        output: GsrFile,  # Must use auto_load kwarg when passed ; VT ???
        # store_trajectory: bool = False,
    ) -> CalculationOutput:
        """
        Create an Abinit output document from Abinit outputs.

        Parameters
        ----------
        output: .AbinitOutput
            An AbinitOutput object.
        # store_trajectory: bool
            # A flag setting to store output trajectory

        Returns
        -------
        The Abinit calculation output document.
        """
        structure = output.structure  # final structure by default for GSR

        # In case no conduction bands were included
        try:
            cbm = output.ebands.get_edge_state("cbm").eig
            bandgap = output.ebands.fundamental_gaps[
                0
            ].energy  # [0] for one spin channel only
            direct_bandgap = output.ebands.direct_gaps[0].energy
        except ValueError:
            cbm = None
            bandgap = None
            direct_bandgap = None

        electronic_output = {
            "efermi": float(output.ebands.fermie),
            "vbm": output.ebands.get_edge_state("vbm").eig,
            "cbm": cbm,
            "bandgap": bandgap,
            "direct_bandgap": direct_bandgap,
        }

        forces = None
        if output.cart_forces is not None:
            forces = output.cart_forces.tolist()

        stress = None
        if output.cart_stress_tensor is not None:
            stress = output.cart_stress_tensor.tolist()

        return cls(
            structure=structure,
            energy=output.energy,
            energy_per_atom=output.energy_per_atom,
            **electronic_output,
            forces=forces,
            stress=stress,
        )


class Calculation(BaseModel):
    """Full Abinit calculation inputs and outputs.

    Parameters
    ----------
    dir_name: str
        The directory for this Abinit calculation
    abinit_version: str
        Abinit version used to perform the calculation
    has_abinit_completed: .TaskState
        Whether Abinit completed the calculation successfully
    output: .CalculationOutput
        The Abinit calculation output
    completed_at: str
        Timestamp for when the calculation was completed
    output_file_paths: Dict[str, str]
        Paths (relative to dir_name) of the Abinit output files
        associated with this calculation
    """

    dir_name: str = Field(None, description="The directory for this Abinit calculation")
    abinit_version: str = Field(
        None, description="Abinit version used to perform the calculation"
    )
    has_abinit_completed: TaskState = Field(
        None, description="Whether Abinit completed the calculation successfully"
    )
    output: CalculationOutput = Field(None, description="The Abinit calculation output")
    completed_at: str = Field(
        None, description="Timestamp for when the calculation was completed"
    )
    event_report: events.EventReport = Field(
        None, description="Event report of this abinit job."
    )
    output_file_paths: Optional[dict[str, str]] = Field(
        None,
        description="Paths (relative to dir_name) of the Abinit output files "
        "associated with this calculation",
    )

    @classmethod
    def from_abinit_files(
        cls,
        dir_name: Path | str,
        task_name: str,
        abinit_gsr_file: Path | str = "out_GSR.nc",
        abinit_log_file: Path | str = LOG_FILE_NAME,
        abinit_abort_file: Path | str = MPIABORTFILE,
        critical_events: Sequence[AbinitCriticalWarning] = (),
        # volumetric_files: list[str] = None,
        parse_dos: str | bool = False,
        parse_bandstructure: str | bool = False,
        # store_trajectory: bool = False,
        store_scf: bool = False,
        # store_volumetric_data: Optional[Sequence[str]] = STORE_VOLUMETRIC_DATA,
    ) -> tuple[Calculation, dict[AbinitObject, dict]]:
        """
        Create an Abinit calculation document from a directory and file paths.

        Parameters
        ----------
        dir_name: Path or str
            The directory containing the calculation outputs.
        task_name: str
            The task name.
        abinit_log_file: Path or str
            Path to the main log of abinit job, relative to dir_name.
        #volumetric_files: List[str]
            #Path to volumetric (Cube) files, relative to dir_name.
        parse_dos: str or bool
            Whether to parse the DOS. Can be:

            - "auto": Only parse DOS if there are no ionic steps.
            - True: Always parse DOS.
            - False: Never parse DOS.

        parse_bandstructure: str or bool
            How to parse the bandstructure. Can be:

            - "auto": Parse the bandstructure with projections for NSCF calculations
              and decide automatically if it's line or uniform mode.
            - "line": Parse the bandstructure as a line mode calculation with
              projections
            - True: Parse the bandstructure as a uniform calculation with
              projections .
            - False: Parse the band structure without projects and just store
              vbm, cbm, band_gap, is_metal and efermi rather than the full
              band structure object.

        store_trajectory: bool
            Whether to store the ionic steps as a pmg trajectory object, which can be
            pushed, to a bson data store, instead of as a list od dicts. Useful for
            large trajectories.
        store_scf: bool
            Whether to store the SCF convergence data.
        #store_volumetric_data: Sequence[str] or None
            #Which volumetric files to store.

        Returns
        -------
        .Calculation
            An Abinit calculation document.
        """
        dir_name = Path(dir_name)
        abinit_gsr_file = dir_name / abinit_gsr_file
        abinit_log_file = dir_name / abinit_log_file
        abinit_abort_file = dir_name / abinit_abort_file

        # volumetric_files = [] if volumetric_files is None else volumetric_files
        abinit_gsr = GsrFile.from_file(abinit_gsr_file)

        completed_at = str(datetime.fromtimestamp(os.stat(abinit_log_file).st_mtime))

        # output_file_paths = _get_output_file_paths(volumetric_files)
        # abinit_objects: dict[AbinitObject, Any] = _get_volumetric_data(
        #    dir_name, output_file_paths, store_volumetric_data
        # )

        # dos = _parse_dos(parse_dos, abinit_output)
        # if dos is not None:
        #    abinit_objects[AbinitObject.DOS] = dos  # type: ignore

        # bandstructure = _parse_bandstructure(parse_bandstructure, abinit_output)
        # if bandstructure is not None:
        #    abinit_objects[AbinitObject.BANDSTRUCTURE] = bandstructure  # type: ignore

        output_doc = CalculationOutput.from_abinit_gsr(abinit_gsr)

        report = None
        has_abinit_completed = TaskState.FAILED
        # TODO: How to detect which status it has here ?
        #  UNCONVERGED would be for scf/nscf/relax when it's not yet converged
        #  FAILED should be for a job that failed for other reasons.
        #  What about a job that has been killed by the run_abinit (i.e. before
        #  Slurm or PBS kills it) ?

        try:
            report = get_event_report(
                ofile=File(abinit_log_file), mpiabort_file=File(abinit_abort_file)
            )
            if report.run_completed:
                has_abinit_completed = TaskState.SUCCESS
            # critical_events_report = report.filter_types(critical_events)
            # if critical_events_report:
            # for warning in critical_events_report.warnings:
            #    if type(warning) in AbiConvergenceWarning.all:
            # has_abinit_completed = TaskState.UNCONVERGED
            # break

        except Exception as exc:
            msg = f"{cls} exception while parsing event_report:\n{exc}"
            logger.critical(msg)

        # if store_trajectory:
        #    traj = _parse_trajectory(abinit_output=abinit_output)
        #    abinit_objects[AbinitObject.TRAJECTORY] = traj  # type: ignore

        return (
            cls(
                dir_name=str(dir_name),
                task_name=task_name,
                abinit_version=abinit_gsr.abinit_version,
                has_abinit_completed=has_abinit_completed,
                completed_at=completed_at,
                output=output_doc,
                event_report=report,
                # output_file_paths={
                # k.name.lower(): v for k, v in output_file_paths.items()
                # },
            ),
            None,  # abinit_objects,
        )
