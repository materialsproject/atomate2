"""Schemas for Abinit calculation objects."""
from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from ase.spectrum.band_structure import BandStructure
from jobflow.utils import ValueEnum
from pydantic import BaseModel, Field
from pymatgen.core import Molecule, Structure
from pymatgen.core.trajectory import Trajectory
from pymatgen.electronic_structure.dos import Dos
# from pymatgen.io.abinit.outputs import AbinitOutput
from abipy.electrons.gsr import GsrFile
#from pymatgen.io.common import VolumetricData

if TYPE_CHECKING:
    from emmet.core.math import Matrix3D, Vector3D

#STORE_VOLUMETRIC_DATA = ("total_density",)

class TaskState(ValueEnum):
    """Abinit calculation state."""

    SUCCESS = "successful"
    FAILED = "failed"


# class AbinitObject(ValueEnum):
#     """Types of Abinit data objects."""

#     DOS = "dos"
#     BAND_STRUCTURE = "band_structure"
#     ELECTRON_DENSITY = "electron_density"  # e_density
#     WFN = "wfn"  # Wavefunction file
#     TRAJECTORY = "trajectory"


class CalculationOutput(BaseModel):
    """Document defining Abinit calculation outputs.

    Parameters
    ----------
    energy: float
        The final total DFT energy for the calculation
    energy_per_atom: float
        The final DFT energy per atom for the calculation
    structure: Structure or Molecule
        The final pymatgen Structure or Molecule of the system
    efermi: float
        The Fermi level from the calculation in eV
    forces: List[Vector3D]
        Forces acting on each atom
    all_forces: List[List[Vector3D]]
        Forces acting on each atom for each structure in the output file
    stress: Matrix3D
        The stress on the cell
    stresses: List[Matrix3D]
        The atomic virial stresses
    is_metal: bool
        Whether the system is metallic
    bandgap: float
        The band gap from the calculation in eV
    cbm: float
        The conduction band minimum in eV (if system is not metallic
    vbm: float
        The valence band maximum in eV (if system is not metallic)
    atomic_steps: list[Structure or Molecule]
        Structures for each ionic step"
    """

    energy: float = Field(
        None, description="The final total DFT energy for the calculation"
    )
    energy_per_atom: float = Field(
        None, description="The final DFT energy per atom for the calculation"
    )

    structure: Union[Structure, Molecule] = Field(
        None, description="The final structure from the calculation"
    )

    efermi: float = Field(
        None, description="The Fermi level from the calculation in eV"
    )

    forces: Optional[list[Vector3D]] = Field(
        None, description="Forces acting on each atom"
    )
    all_forces: Optional[list[list[Vector3D]]] = Field(
        None,
        description="Forces acting on each atom for each structure in the output file",
    )
    stress: Optional[Matrix3D] = Field(None, description="The stress on the cell")
    stresses: Optional[list[Matrix3D]] = Field(
        None, description="The atomic virial stresses"
    )

    is_metal: Optional[bool] = Field(None, description="Whether the system is metallic")
    bandgap: Optional[float] = Field(
        None, description="The band gap from the calculation in eV"
    )
    cbm: float = Field(
        None,
        description="The conduction band minimum, or LUMO for molecules, in eV "
        "(if system is not metallic)",
    )
    vbm: Optional[float] = Field(
        None,
        description="The valence band maximum, or HOMO for molecules, in eV "
        "(if system is not metallic)",
    )
    atomic_steps: list[Union[Structure, Molecule]] = Field(
        None, description="Structures for each ionic step"
    )

    @classmethod
    def from_abinit_output(
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
        structure = output.structure # final structure by default for GSR

        electronic_output = {
            "efermi": float(output.ebands.fermie),
            "vbm": output.ebands.get_edge_state("vbm").eig,
            "cbm": output.ebands.get_edge_state("cbm").eig,
            "bandgap": output.ebands.fundamental_gaps[0].energy,  # [0] for one spin channel only
            "direct_bandgap": output.ebands.direct_gaps[0].energy,
        }

        forces = None
        if output.cart_forces is not None:
            forces = output.cart_forces.tolist()

        stress = None
        if output.cart_stress_tensor is not None:
            stress = output.cart_stress_tensor.tolist()

        stresses = None
        # if output.stresses is not None:
        #     stresses = [ensure_stress_full(st).tolist() for st in output.stresses]

        all_forces = None
        # if not any(ff is None for ff in output.all_forces):
        #     all_forces = [f if (f is not None) else None for f in output.all_forces]

        return cls(
            structure=structure,
            energy=output.energy,
            energy_per_atom=output.energy_per_atom,
            **electronic_output,
            # atomic_steps=output.structures,
            forces=forces,
            stress=stress,
            stresses=stresses,
            all_forces=all_forces,
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

    dir_name: str = Field(
        None, description="The directory for this Abinit calculation"
    )
    abinit_version: str = Field(
        None, description="Abinit version used to perform the calculation"
    )
    has_abinit_completed: TaskState = Field(
        None, description="Whether Abinit completed the calculation successfully"
    )
    output: CalculationOutput = Field(
        None, description="The Abinit calculation output"
    )
    completed_at: str = Field(
        None, description="Timestamp for when the calculation was completed"
    )
    output_file_paths: dict[str, str] = Field(
        None,
        description="Paths (relative to dir_name) of the Abinit output files "
        "associated with this calculation",
    )

    @classmethod
    def from_abinit_files(
        cls,
        dir_name: Path | str,
        task_name: str,
        abinit_output_file: Path | str = "out_GSR.nc",
        #volumetric_files: list[str] = None,
        parse_dos: str | bool = False,
        parse_bandstructure: str | bool = False,
        store_trajectory: bool = False,
        store_scf: bool = False,
        #store_volumetric_data: Optional[Sequence[str]] = STORE_VOLUMETRIC_DATA,
    ) -> tuple[Calculation, dict[AbinitObject, dict]]:
        """
        Create an Abinit calculation document from a directory and file paths.

        Parameters
        ----------
        dir_name: Path or str
            The directory containing the calculation outputs.
        task_name: str
            The task name.
        abinit_output_file: Path or str
            Path to the main output of abinit job, relative to dir_name.
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
        abinit_output_file = dir_name / abinit_output_file

        #volumetric_files = [] if volumetric_files is None else volumetric_files
        # abinit_output = AbinitOutput.from_outfile(abinit_output_file)
        abinit_output = GsrFile.from_file(abinit_output_file)

        completed_at = str(datetime.fromtimestamp(os.stat(abinit_output_file).st_mtime))

        #output_file_paths = _get_output_file_paths(volumetric_files)
        #abinit_objects: dict[AbinitObject, Any] = _get_volumetric_data(
        #    dir_name, output_file_paths, store_volumetric_data
        #)

        #dos = _parse_dos(parse_dos, abinit_output)
        #if dos is not None:
        #    abinit_objects[AbinitObject.DOS] = dos  # type: ignore

        #bandstructure = _parse_bandstructure(parse_bandstructure, abinit_output)
        #if bandstructure is not None:
        #    abinit_objects[AbinitObject.BANDSTRUCTURE] = bandstructure  # type: ignore

        output_doc = CalculationOutput.from_abinit_output(abinit_output)

        # has_abinit_completed = (
        #     TaskState.SUCCESS if abinit_output.completed else TaskState.FAILED
        # )

        #if store_trajectory:
        #    traj = _parse_trajectory(abinit_output=abinit_output)
        #    abinit_objects[AbinitObject.TRAJECTORY] = traj  # type: ignore

        return (
            cls(
                dir_name=str(dir_name),
                task_name=task_name,
                # abinit_version=abinit_output.abinit_version,
                # has_abinit_completed=has_abinit_completed,
                completed_at=completed_at,
                output=output_doc,
                # output_file_paths={
                    # k.name.lower(): v for k, v in output_file_paths.items()
                # },
            ),
            # abinit_objects,
        )