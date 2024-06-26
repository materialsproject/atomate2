"""Schemas for Abinit calculation objects."""

from __future__ import annotations

import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    pass

from abipy.electrons.gsr import GsrFile
from abipy.electrons.scr import ScrFile
from abipy.electrons.gw import SigresFile
from abipy.electrons.bse import MdfFile
from abipy.flowtk import events
from abipy.flowtk.utils import File
from emmet.core.math import Matrix3D, Vector3D
from jobflow.utils import ValueEnum
from pydantic import BaseModel, Field
from pymatgen.core import Structure

from atomate2.abinit.utils.common import LOG_FILE_NAME, MPIABORTFILE, get_event_report

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
    emacro: abipy Function1D object
        Macroscopic dielectric function 
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
    direct_gap: Optional[float] = Field(
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
    emacro: Optional[list] = Field(
        None,
        description="Macroscopic dielectric function",
    )
    mbpt_sciss: Optional[float] = Field(
        None,
        description="Scissor shift (scalar) obtained from GW calculation",
    )
    bandlims: Optional[list] = Field(
        None,
        description="Minimum and Maximum energies of each bands",
    )

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_abinit_gsr(
        cls,
        output: GsrFile,  # Must use auto_load kwarg when passed
    ) -> CalculationOutput:
        """
        Create an Abinit output document from Abinit outputs.

        Parameters
        ----------
        output: .AbinitOutput
            An AbinitOutput object.

        Returns
        -------
        The Abinit calculation output document.
        """
        structure = output.structure  # final structure by default for GSR

        # In case no conduction bands were included
        try:
            vbm = output.ebands.get_edge_state("vbm").eig
            cbm = output.ebands.get_edge_state("cbm").eig
            bandgap = output.ebands.fundamental_gaps[
                0
            ].energy  # [0] for one spin channel only
            direct_gap = output.ebands.direct_gaps[0].energy
        except ValueError:
            cbm = None
            bandgap = None
            direct_gap = None

        bandlims=[]
        for spin in output.ebands.spins:
            for iband in range(output.ebands.nband):
                enemin=output.ebands.enemin(spin=spin,band=iband)-vbm
                enemax=output.ebands.enemax(spin=spin,band=iband)-vbm
                bandlims.append([spin, iband, enemin, enemax])

        electronic_output = {
            "efermi": float(output.ebands.fermie),
            "vbm": vbm,
            "cbm": cbm,
            "bandgap": bandgap,
            "direct_gap": direct_gap,
            "bandlims": bandlims, 
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
    @classmethod
    def from_abinit_scr(
        cls,
        output: ScrFile,  # Must use auto_load kwarg when passed
    ) -> CalculationOutput:
        """
        Create an Abinit output document from Abinit outputs.

        Parameters
        ----------
        output: .AbinitOutput
            An AbinitOutput object.

        Returns
        -------
        The Abinit calculation output document.
        """
        structure = output.structure  # final structure by default for GSR

        return cls(
            structure=structure,
            energy=0.0,
            energy_per_atom=0.0,
            efermi=0.0,
        )
    @classmethod
    def from_abinit_sig(
        cls,
        output: SigresFile,  # Must use auto_load kwarg when passed
    ) -> CalculationOutput:
        """
        Create an Abinit output document from Abinit outputs.

        Parameters
        ----------
        output: .AbinitOutput
            An AbinitOutput object.

        Returns
        -------
        The Abinit calculation output document.
        """
        structure = output.structure  # final structure by default for GSR

        # In case no conduction bands were included
        ivbm=output.ebands.get_edge_state("vbm")
        icbm=output.ebands.get_edge_state("cbm")
        vbm=output.get_qpcorr(ivbm.spin, ivbm.kpoint, ivbm.band).re_qpe
        cbm=output.get_qpcorr(icbm.spin, icbm.kpoint, icbm.band).re_qpe
        ks_gap=icbm.eig-ivbm.eig
        bandgap=cbm-vbm
        mbpt_sciss=bandgap-ks_gap   
        electronic_output = {
            "efermi": float(output.ebands.fermie),
            "vbm": vbm, 
            "cbm": cbm,
            "bandgap": bandgap,
            "mbpt_sciss": mbpt_sciss,
        }
        return cls(
            structure=structure,
            energy=0.0,
            energy_per_atom=0.0,
            **electronic_output,
        )
    @classmethod
    def from_abinit_mdf(
        cls,
        output: MdfFile,  # Must use auto_load kwarg when passed
    ) -> CalculationOutput:
        """
        Create an Abinit output document from Abinit outputs.

        Parameters
        ----------
        output: .AbinitOutput
            An AbinitOutput object.

        Returns
        -------
        The Abinit calculation output document.
        """
        structure = output.structure  # final structure by default for GSR
        gw_eps=output.get_mdf(mdf_type="exc").emacro_avg
        emacromesh=gw_eps.mesh
        emacrovalues=gw_eps.values.imag
        emacro=[emacromesh, emacrovalues]
        return cls(
            structure=structure,
            energy=0.0,
            energy_per_atom=0.0,
            efermi=0.0,
            emacro=emacro
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
        abinit_scr_file: Path | str = "out_SCR.nc",
        abinit_sig_file: Path | str = "out_SIGRES.nc",
        abinit_mdf_file: Path | str = "out_MDF.nc",
        abinit_log_file: Path | str = LOG_FILE_NAME,
        abinit_abort_file: Path | str = MPIABORTFILE,

    ) -> tuple[Calculation, dict[AbinitObject, dict]]:
        """
        Create an Abinit calculation document from a directory and file paths.

        Parameters
        ----------
        dir_name: Path or str
            The directory containing the calculation outputs.
        task_name: str
            The task name.
        abinit_gsr_file: Path or str
            Path to the GSR output of abinit job, relative to dir_name.
        abinit_log_file: Path or str
            Path to the main log of abinit job, relative to dir_name.
        abinit_abort_file: Path or str
            Path to the main abort file of abinit job, relative to dir_name.

        Returns
        -------
        .Calculation
            An Abinit calculation document.
        """
        dir_name = Path(dir_name)
        abinit_gsr_file = dir_name / abinit_gsr_file
        abinit_scr_file = dir_name / abinit_scr_file
        abinit_sig_file = dir_name / abinit_sig_file
        abinit_mdf_file = dir_name / abinit_mdf_file
        abinit_log_file = dir_name / abinit_log_file
        abinit_abort_file = dir_name / abinit_abort_file
        if os.path.isfile(abinit_gsr_file):
            abinit_gsr = GsrFile.from_file(abinit_gsr_file)
            output_doc = CalculationOutput.from_abinit_gsr(abinit_gsr)
            abinit_version = abinit_gsr.abinit_version
        elif os.path.isfile(abinit_scr_file):
            abinit_scr = ScrFile.from_file(abinit_scr_file)
            output_doc = CalculationOutput.from_abinit_scr(abinit_scr)
            abinit_version = abinit_scr.abinit_version
        elif os.path.isfile(abinit_sig_file):
            abinit_sig = SigresFile.from_file(abinit_sig_file)
            output_doc = CalculationOutput.from_abinit_sig(abinit_sig)
            abinit_version = abinit_sig.abinit_version
        elif os.path.isfile(abinit_mdf_file):
            abinit_mdf = MdfFile.from_file(abinit_mdf_file)
            output_doc = CalculationOutput.from_abinit_mdf(abinit_mdf)
            abinit_version = abinit_mdf.abinit_version
        else:
            print("No ouput file found.")

        completed_at = str(datetime.fromtimestamp(os.stat(abinit_log_file).st_mtime))


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

        except Exception as exc:
            msg = f"{cls} exception while parsing event_report:\n{exc}"
            logger.critical(msg)

        return (
            cls(
                dir_name=str(dir_name),
                task_name=task_name,
                abinit_version=abinit_version,
                has_abinit_completed=has_abinit_completed,
                completed_at=completed_at,
                output=output_doc,
                event_report=report,
            ),
            None,  # abinit_objects,
        )
