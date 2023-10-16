"""Schemas for FHI-aims calculation objects."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ase.spectrum.band_structure import BandStructure
from ase.stress import voigt_6_to_full_3x3_stress
from emmet.core.math import Matrix3D, Vector3D
from jobflow.utils import ValueEnum
from pydantic import BaseModel, Field
from pymatgen.core import Molecule, Structure
from pymatgen.core.trajectory import Trajectory
from pymatgen.electronic_structure.dos import Dos
from pymatgen.io.common import VolumetricData

from atomate2.aims.io.aims_output import AimsOutput
from atomate2.aims.utils.msonable_atoms import MSONableAtoms

STORE_VOLUMETRIC_DATA = ("total_density",)


class Status(ValueEnum):
    """FHI-aims calculation state."""

    SUCCESS = "successful"
    FAILED = "failed"


class AimsObject(ValueEnum):
    """Types of FHI-aims data objects."""

    DOS = "dos"
    BAND_STRUCTURE = "band_structure"
    ELECTRON_DENSITY = "electron_density"  # e_density
    WFN = "wfn"  # Wavefunction file
    TRAJECTORY = "trajectory"


class CalculationInput(BaseModel):
    """Summary of inputs for an FHI-aims calculation.

    Parameters
    ----------
    atoms: .MSONableAtoms
        The input Atoms object
    structure: Structure or Molecule
        The final pymatgen Structure or Molecule of the atoms
    species_info: Dict
        Description of parameters used for each atomic species
    aims_input: Dict
        The aims input parameters used for this task
    """

    atoms: MSONableAtoms = Field(None, description="The input Atoms object")

    structure: Union[Structure, Molecule] = Field(
        None, description="The pymatgen structure of the input Atoms"
    )

    species_info: Dict[str, Any] = Field(
        None, description="Description of parameters used for each atomic species"
    )

    aims_input: Dict[str, Any] = Field(
        None, description="The aims input parameters used for this task"
    )

    @classmethod
    def from_aims_output(cls, output: AimsOutput):
        """Initialize from AimsOutput object."""
        return cls(
            atoms=output.initial_structure,
            structure=output.initial_structure.pymatgen,
            species_info=output.data.get("species_info", None),
            aims_input=output.input.as_dict(),
        )


class CalculationOutput(BaseModel):
    """Document defining FHI-aims calculation outputs.

    Parameters
    ----------
    energy: float
        The final total DFT energy for the calculation
    energy_per_atom: float
        The final DFT energy per atom for the calculation
    atoms: .MSONableAtoms
        The final structure from the calculation
    structure: Structure or Molecule
        The final pymatgen Structure or Molecule of the atoms
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
    atomic_steps: List[.MSONableAtoms]
        Structures for each ionic step"
    """

    energy: float = Field(
        None, description="The final total DFT energy for the calculation"
    )
    energy_per_atom: float = Field(
        None, description="The final DFT energy per atom for the calculation"
    )
    atoms: MSONableAtoms = Field(
        None, description="The final structure from the calculation"
    )
    structure: Union[Structure, Molecule] = Field(
        None, description="The pymatgen structure of the final Atoms"
    )
    efermi: float = Field(
        None, description="The Fermi level from the calculation in eV"
    )

    forces: List[Vector3D] = Field(None, description="Forces acting on each atom")
    all_forces: List[List[Vector3D]] = Field(
        None,
        description="Forces acting on each atom for each structure in the output file",
    )
    stress: Matrix3D = Field(None, description="The stress on the cell")
    stresses: Union[List[Matrix3D], None] = Field(
        None, description="The atomic virial stresses"
    )

    is_metal: bool = Field(None, description="Whether the system is metallic")
    bandgap: float = Field(None, description="The band gap from the calculation in eV")
    cbm: float = Field(
        None,
        description="The conduction band minimum in eV (if system is not metallic)",
    )
    vbm: float = Field(
        None, description="The valence band maximum in eV (if system is not metallic)"
    )
    atomic_steps: List[MSONableAtoms] = Field(
        None, description="Structures for each ionic step"
    )

    @classmethod
    def from_aims_output(
        cls,
        output: AimsOutput,  # Must use auto_load kwarg when passed
        store_trajectory: bool = False,
    ) -> "CalculationOutput":
        """
        Create an FHI-aims output document from FHI-aims outputs.

        Parameters
        ----------
        output: .AimsOutput
            An AimsOutput object.
        store_trajectory: bool
            A flag setting to store output trajectory

        Returns
        -------
        The FHI-aims calculation output document.
        """
        atoms = output.final_structure

        electronic_output = {
            "efermi": output.fermi_energy,
            "homo": output.homo,
            "lumo": output.lumo,
            "bandgap": output.band_gap,
            "direct_bandgap": output.direct_band_gap,
        }

        forces = None
        if output.forces is not None:
            forces = output.forces

        stress = None
        if output.stress is not None:
            stress = voigt_6_to_full_3x3_stress(output.stress).tolist()

        stresses = None
        if output.stresses is not None:
            print("\n\n", output.stresses, "\n\n")
            stresses = [
                voigt_6_to_full_3x3_stress(st).tolist() for st in output.stresses
            ]

        all_forces = None
        if not any(ff is None for ff in output.all_forces):
            all_forces = [f if (f is not None) else None for f in output.all_forces]

        return cls(
            atoms=atoms,
            structure=atoms.pymatgen,
            energy=output.final_energy,
            energy_per_atom=output.final_energy / len(atoms),
            **electronic_output,
            atomic_steps=output.structures,
            forces=forces,
            stress=stress,
            stresses=stresses,
            all_forces=all_forces,
        )


class Calculation(BaseModel):
    """Full FHI-aims calculation inputs and outputs.

    Parameters
    ----------
    dir_name: str
        The directory for this FHI-aims calculation
    aims_version: str
        FHI-aims version used to perform the calculation
    has_aims_completed: .Status
        Whether FHI-aims completed the calculation successfully
    output: .CalculationOutput
        The FHI-aims calculation output
    completed_at: str
        Timestamp for when the calculation was completed
    output_file_paths: Dict[str, str]
        Paths (relative to dir_name) of the FHI-aims output files
        associated with this calculation
    """

    dir_name: str = Field(
        None, description="The directory for this FHI-aims calculation"
    )
    aims_version: str = Field(
        None, description="FHI-aims version used to perform the calculation"
    )
    has_aims_completed: Status = Field(
        None, description="Whether FHI-aims completed the calculation successfully"
    )
    output: CalculationOutput = Field(
        None, description="The FHI-aims calculation output"
    )
    completed_at: str = Field(
        None, description="Timestamp for when the calculation was completed"
    )
    output_file_paths: Dict[str, str] = Field(
        None,
        description="Paths (relative to dir_name) of the FHI-aims output files "
        "associated with this calculation",
    )

    @classmethod
    def from_aims_files(
        cls,
        dir_name: Union[Path, str],
        task_name: str,
        aims_output_file: Union[Path, str] = "aims.out",
        volumetric_files: List[str] = None,
        parse_dos: Union[str, bool] = False,
        parse_bandstructure: Union[str, bool] = False,
        store_trajectory: bool = False,
        store_scf: bool = False,
        store_volumetric_data: Union[Sequence[str], None] = STORE_VOLUMETRIC_DATA,
    ) -> Tuple["Calculation", Dict[AimsObject, Dict]]:
        """
        Create an FHI-aims calculation document from a directory and file paths.

        Parameters
        ----------
        dir_name: Path or str
            The directory containing the calculation outputs.
        task_name: str
            The task name.
        aims_output_file: Path or str
            Path to the main output of aims job, relative to dir_name.
        volumetric_files: List[str]
            Path to volumetric (Cube) files, relative to dir_name.
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
        store_volumetric_data: Sequence[str] or None
            Which volumetric files to store.

        Returns
        -------
        .Calculation
            An FHI-aims calculation document.
        """
        dir_name = Path(dir_name)
        aims_output_file = dir_name / aims_output_file

        volumetric_files = [] if volumetric_files is None else volumetric_files
        aims_output = AimsOutput.from_outfile(aims_output_file)

        completed_at = str(datetime.fromtimestamp(os.stat(aims_output_file).st_mtime))

        output_file_paths = _get_output_file_paths(volumetric_files)
        aims_objects: Dict[AimsObject, Any] = _get_volumetric_data(
            dir_name, output_file_paths, store_volumetric_data
        )

        dos = _parse_dos(parse_dos, aims_output)
        if dos is not None:
            aims_objects[AimsObject.DOS] = dos  # type: ignore

        bandstructure = _parse_bandstructure(parse_bandstructure, aims_output)
        if bandstructure is not None:
            aims_objects[AimsObject.BANDSTRUCTURE] = bandstructure  # type: ignore

        output_doc = CalculationOutput.from_aims_output(aims_output)

        has_aims_completed = Status.SUCCESS if aims_output.completed else Status.FAILED

        if store_trajectory:
            traj = _parse_trajectory(aims_output=aims_output)
            aims_objects[AimsObject.TRAJECTORY] = traj  # type: ignore

        return (
            cls(
                dir_name=str(dir_name),
                task_name=task_name,
                aims_version=aims_output.aims_version,
                has_aims_completed=has_aims_completed,
                completed_at=completed_at,
                output=output_doc,
                output_file_paths={
                    k.name.lower(): v for k, v in output_file_paths.items()
                },
            ),
            aims_objects,
        )


def _get_output_file_paths(volumetric_files: List[str]) -> Dict[AimsObject, str]:
    """Get the output file paths for FHI-aims output files.

    Parameters
    ----------
    volumetric_files: List[str]
        A list of volumetric files associated with the calculation.

    Returns
    -------
    Dict[AimsObject, str]
        A mapping between the Aims object type and the file path.
    """
    output_file_paths = {}
    for aims_object in AimsObject:  # type: ignore
        for volumetric_file in volumetric_files:
            if aims_object.name in str(volumetric_file):
                output_file_paths[aims_object] = str(volumetric_file)
    return output_file_paths


def _get_volumetric_data(
    dir_name: Path,
    output_file_paths: Dict[AimsObject, str],
    store_volumetric_data: Union[Sequence[str], None],
) -> Dict[AimsObject, VolumetricData]:
    """
    Load volumetric data files from a directory.

    Parameters
    ----------
    dir_name: Path
        The directory containing the files.
    output_file_paths: Dict[.AimsObject, str]
        A dictionary mapping the data type to file path relative to dir_name.
    store_volumetric_data: Sequence[str] or None
        The volumetric data files to load.

    Returns
    -------
    Dict[AimsObject, VolumetricData]
        A dictionary mapping the FHI-aims object data type (`AimsObject.total_density`,
        `AimsObject.electron_density`, etc) to the volumetric data object.
    """
    if store_volumetric_data is None or len(store_volumetric_data) == 0:
        return {}

    volumetric_data = {}
    for file_type, file in output_file_paths.items():
        if file_type.name not in store_volumetric_data:
            continue
        try:
            volumetric_data[file_type] = VolumetricData.from_cube(
                (dir_name / file).as_posix()
            )
        except Exception as err:
            raise ValueError(f"Failed to parse {file_type} at {file}.") from err

    return volumetric_data


def _parse_dos(parse_dos: Union[str, bool], aims_output: AimsOutput) -> Optional[Dos]:
    """Parse DOS outputs from FHI-aims calculation.

    Parameters
    ----------
    parse_dos: str or bool
        Whether to parse the DOS. Can be:
        - "auto": Only parse DOS if there are no ionic steps.
        - True: Always parse DOS.
        - False: Never parse DOS.
    aims_output: .AimsOutput
        The output object for the calculation being parsed.

    Returns
    -------
    A Dos object if parse_dos is set accordingly.
    """
    if parse_dos == "auto":
        if len(aims_output.ionic_steps) == 0:
            return aims_output.complete_dos
        return None
    if parse_dos:
        return aims_output.complete_dos
    return None


def _parse_bandstructure(
    parse_bandstructure: Union[str, bool], aims_output: AimsOutput
) -> Optional[BandStructure]:
    """
    Get the band structure.

    Parameters
    ----------
    parse_bandstructure: str or bool
        Whether to parse. Does not support the auto/line distinction currently.
    aims_ouput: .AimsOutput
        The output object to parse

    Returns
    -------
    The bandstructure
    """
    if parse_bandstructure:
        return aims_output.band_structure
    return None


def _parse_trajectory(aims_output: AimsOutput) -> Optional[Trajectory]:
    """Grab a Trajectory object given an FHI-aims output object.

    Parameters
    ----------
    aims_ouput: .AimsOutput
        The output object to parse

    Returns
    -------
    The trajectory for the calculation
    """
    return aims_output.structures
