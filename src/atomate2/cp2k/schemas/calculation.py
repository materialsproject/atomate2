"""Core definitions of a CP2K calculation documents."""

import os
import logging
from pathlib import Path
from shutil import which
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from jobflow.utils import ValueEnum
from pydantic import BaseModel, Field, validator
from pydantic.datetime_parse import datetime
from pymatgen.command_line.bader_caller import BaderAnalysis
from pymatgen.core.structure import Structure
from pymatgen.core.trajectory import Trajectory
from pymatgen.electronic_structure.dos import Dos, CompleteDos
from pymatgen.electronic_structure.bandstructure import BandStructure

from pymatgen.io.cube import Cube 
from pymatgen.io.cp2k.outputs import Cp2kOutput, parse_energy_file
from pymatgen.core.units import Ha_to_eV

from atomate2 import SETTINGS
from atomate2.common.schemas.math import Matrix3D, Vector3D
from atomate2.cp2k.schemas.calc_types import (
    CalcType,
    RunType,
    TaskType,
    calc_type,
    run_type,
    task_type,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Status",
    "Cp2kObject",
    "CalculationInput",
    "CalculationOutput",
    "RunStatistics",
    "Calculation",
]

# Can be expanded if support for other volumetric files is added
__is_stored_in_Ha__ = [
    "v_hartree"
]


_BADER_EXE_EXISTS = True if (which("bader") or which("bader.exe")) else False


class Status(ValueEnum):
    """CP2K calculation state."""

    SUCCESS = "successful"
    FAILED = "failed"


# TODO: Naming convention. Should we just use the VASP names?
class Cp2kObject(ValueEnum):
    """Types of CP2K data objects."""

    DOS = "dos"
    BANDSTRUCTURE = "band_structure"
    E_DENSITY = "e_density" # e_density
    SPIN_DENSITY = "spin_density" # spin density
    V_HARTREE = "v_hartree" # elec. potential
    TRAJECTORY = "trajectory" # Trajectory
    WFN = "wfn" # Wavefunction file


class CalculationInput(BaseModel):
    """
    Summary of inputs for a CP2K calculation
    """

    structure: Structure = Field(None, description="The input structure object")

    atomic_kind_info: Dict = Field(None, description="Description of parameters used for each atomic kind")

    cp2k_input: Dict = Field(None, description="The cp2k input used for this task")

    dft: Dict = Field(None, description="Description of the DFT parameters used in the last calc of this task")

    cp2k_global: Dict = Field(None, description="CP2K global parameters used in the last calc of this task")

    @validator('atomic_kind_info')
    def remove_unnecessary(cls, atomic_kind_info):
        for k in atomic_kind_info:
            if 'total_pseudopotential_energy' in atomic_kind_info[k]:
                del atomic_kind_info[k]['total_pseudopotential_energy']
        return atomic_kind_info

    @validator('dft')
    def cleanup_dft(cls, dft):
        if any(v.upper() == 'UKS' for v in dft.values()):
            dft['UKS'] = True
        return dft

    @classmethod
    def from_cp2k_output(
        cls,
        output: Cp2kOutput
    ):

        return cls(
            structure=output.initial_structure,
            atomic_kind_info=output.data.get("atomic_kind_info", None),
            cp2k_input=output.input.as_dict(),
            dft=output.data.get("dft", None),
            cp2k_global=output.data.get("global", None)
        )


class RunStatistics(BaseModel):
    """Summary of the run statistics for a CP2K calculation."""

    total_time: float = Field(0, description="The total CPU time for this calculation")

    @classmethod
    def from_cp2k_output(cls, output: Cp2kOutput) -> "RunStatistics":
        """
        Create a run statistics document from an CP2K Output object.

        Parameters
        ----------
        output:
            Cp2kOutput object

        Returns
        -------
        RunStatistics
            The run statistics.
        """
        # rename these statistics
        run_stats = {}
        output.parse_timing()
        run_stats['total_time'] = output.timing["CP2K"]["total_time"]["maximum"]
        return cls(**run_stats)


class CalculationOutput(BaseModel):
    """Document defining CP2K calculation outputs."""

    energy: float = Field(
        None, description="The final total DFT energy for the calculation"
    )
    energy_per_atom: float = Field(
        None, description="The final DFT energy per atom for the calculation"
    )
    structure: Structure = Field(
        None, description="The final structure from the calculation"
    )
    efermi: float = Field(
        None, description="The Fermi level from the calculation in eV"
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
    ionic_steps: List[Dict[str, Any]] = Field(
        None, description="Energy, forces, and structure for each ionic step"
    )
    locpot: Dict[int, List[float]] = Field(
        None, description="Average of the local potential along the crystal axes"
    )
    run_stats: RunStatistics = Field(
        None, description="Summary of runtime statistics for this calculation"
    )

    @classmethod
    def from_cp2k_output(
        cls,
        output: Cp2kOutput, # Must use auto_load kwarg when passed
        v_hartree: Optional[Cube] = None,
        store_trajectory: bool = False
    ) -> "CalculationOutput":
        """
        Create a CP2K output document from CP2K outputs.

        Parameters
        ----------
        output
            A Cp2kOutput object.
        v_hartree
            A Cube object for the V_HARTREE data

        Returns
        -------
            The CP2K calculation output document.
        """
        
        v_hart_avg = None
        if v_hartree:
            v_hart_avg = {
                i: v_hartree.get_average_along_axis(i).tolist() for i in range(3)
            }

        structure = output.final_structure

        if output.band_structure:
            bandgap_info = output.band_structure.get_band_gap()
            electronic_output = dict(
                efermi=output.band_structure.efermi,
                vbm=output.band_structure.get_vbm()["energy"],
                cbm=output.band_structure.get_cbm()["energy"],
                bandgap=bandgap_info["energy"],
                is_gap_direct=bandgap_info["direct"],
                is_metal=output.band_structure.is_metal(),
                direct_gap=output.band_structure.get_direct_band_gap(),
                transition=bandgap_info["transition"],
            )
        else:
            logger.warning("Unable to parse bandstructure. Collecting band edge info as available")
            electronic_output = {
                "efermi": output.efermi,
                "vbm": output.vbm,
                "cbm": output.cbm,
                "bandgap": output.band_gap,
                "is_metal": output.is_metal,
            }

        return cls(
            structure=structure,
            energy=output.final_energy,
            energy_per_atom=output.final_energy / len(structure),
            **electronic_output, 
            ionic_steps=None if store_trajectory else output.ionic_steps,
            v_hartree=v_hart_avg,
            run_stats=RunStatistics.from_cp2k_output(output),
        )


class Calculation(BaseModel):
    """Full CP2K calculation inputs and outputs."""

    dir_name: str = Field(None, description="The directory for this CP2K calculation")
    cp2k_version: str = Field(
        None, description="CP2K version used to perform the calculation"
    )
    has_cp2k_completed: Status = Field(
        None, description="Whether CP2K completed the calculation successfully"
    )
    input: CalculationInput = Field(
        None, description="CP2K input settings for the calculation"
    )
    output: CalculationOutput = Field(None, description="The CP2K calculation output")
    completed_at: str = Field(
        None, description="Timestamp for when the calculation was completed"
    )
    task_name: str = Field(
        None, description="Name of task given by custodian (e.g., relax1, relax2)"
    )
    output_file_paths: Dict[str, str] = Field(
        None,
        description="Paths (relative to dir_name) of the CP2K output files "
        "associated with this calculation",
    )
    bader: Dict = Field(None, description="Output from the bader software")
    run_type: RunType = Field(
        None, description="Calculation run type (e.g., HF, HSE06, PBE)"
    )
    task_type: TaskType = Field(
        None, description="Calculation task type (e.g., Structure Optimization)."
    )
    calc_type: CalcType = Field(
        None, description="Return calculation type (run type + task_type)."
    )

    @classmethod
    def from_cp2k_files(
        cls,
        dir_name: Union[Path, str],
        task_name: str,
        cp2k_output_file: Union[Path, str] = "cp2k.out",
        volumetric_files: List[str] = None,
        parse_dos: Union[str, bool] = False,
        parse_bandstructure: Union[str, bool] = False,
        average_v_hartree: bool = True,
        run_bader: bool = (SETTINGS.CP2K_RUN_BADER and _BADER_EXE_EXISTS),
        strip_bandstructure_projections: bool = False,
        strip_dos_projections: bool = False,
        store_trajectory: bool = False,
        store_volumetric_data: Optional[
            Tuple[str]
        ] = SETTINGS.CP2K_STORE_VOLUMETRIC_DATA,
    ) -> Tuple["Calculation", Dict[Cp2kObject, Dict]]:
        """
        Create a CP2K calculation document from a directory and file paths.

        Parameters
        ----------
        dir_name
            The directory containing the calculation outputs.
        task_name
            The task name.
        cp2k_output_file
            Path to the main output of cp2k job, relative to dir_name.
        volumetric_files
            Path to volumetric (Cube) files, relative to dir_name.
        parse_dos
            Whether to parse the DOS. Can be:

            - "auto": Only parse DOS if there are no ionic steps.
            - True: Always parse DOS.
            - False: Never parse DOS.

        parse_bandstructure
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

        average_v_hartree
            Whether to store the average of the V_HARTREE along the crystal axes.
        run_bader
            Whether to run bader on the charge density.
        strip_dos_projections : bool
            Whether to strip the element and site projections from the density of states.
            This can help reduce the size of DOS objects in systems with many atoms.
        strip_bandstructure_projections : bool
            Whether to strip the element and site projections from the band structure.
            This can help reduce the size of DOS objects in systems with many atoms.
        store_volumetric_data
            Which volumetric files to store.

        Returns
        -------
        Calculation
            A CP2K calculation document.
        """
        dir_name = Path(dir_name)
        cp2k_output_file = dir_name / cp2k_output_file

        volumetric_files = [] if volumetric_files is None else volumetric_files
        cp2k_output = Cp2kOutput(cp2k_output_file, auto_load=True)
        completed_at = str(datetime.fromtimestamp(os.stat(cp2k_output_file).st_mtime))

        output_file_paths = _get_output_file_paths(volumetric_files)
        cp2k_objects: Dict[Cp2kObject, Any] = _get_volumetric_data(
            dir_name, output_file_paths, store_volumetric_data
        )

        dos = _parse_dos(parse_dos, cp2k_output)
        if dos is not None:
            if strip_dos_projections:
                dos = Dos(dos.efermi, dos.energies, dos.densities)
            cp2k_objects[Cp2kObject.DOS] = dos  # type: ignore

        bandstructure = _parse_bandstructure(parse_bandstructure, cp2k_output)
        if bandstructure is not None:
            if strip_bandstructure_projections:
                bandstructure.projections = {}
            cp2k_objects[Cp2kObject.BANDSTRUCTURE] = bandstructure  # type: ignore

        bader = None
        if run_bader and Cp2kObject.E_DENSITY in output_file_paths:
            suffix = "" if task_name == "standard" else f".{task_name}"
            ba = BaderAnalysis(cube_filename=Cp2kObject.E_DENSITY)
            #TODO vasp version calls bader_analysis_from_path but cp2k
            # cube files don't support that yet, do it manually
            bader = {
                "min_dist": [d["min_dist"] for d in ba.data],
                "charge": [d["charge"] for d in ba.data],
                "atomic_volume": [d["atomic_vol"] for d in ba.data],
                "vacuum_charge": ba.vacuum_charge,
                "vacuum_volume": ba.vacuum_volume,
                "reference_used": bool(ba.chgref_filename),
                "bader_version": ba.version,
            }

        v_hartree = None
        if average_v_hartree:
            if Cp2kObject.V_HARTREE in cp2k_objects:
                v_hartree = cp2k_objects[Cp2kObject.V_HARTREE]  # type: ignore
            elif Cp2kObject.V_HARTREE in output_file_paths:
                v_hartree_file = output_file_paths[Cp2kObject.V_HARTREE]  # type: ignore
                v_hartree = Cube(dir_name / v_hartree_file)

                #TODO Very important am converting from native Ha to eV for storing
                np.multiply(v_hartree.data, Ha_to_eV)

        input_doc = CalculationInput.from_cp2k_output(cp2k_output)
        output_doc = CalculationOutput.from_cp2k_output(
            cp2k_output, v_hartree=v_hartree 
        )

        has_cp2k_completed = Status.SUCCESS if cp2k_output.completed else Status.FAILED

        if store_trajectory:
            traj = _parse_trajectory(cp2k_output=cp2k_output)
            cp2k_objects[Cp2kObject.TRAJECTORY] = traj  # type: ignore

        return cls(
                dir_name=str(dir_name),
                task_name=task_name,
                cp2k_version=cp2k_output.cp2k_version,
                has_cp2k_completed=has_cp2k_completed,
                completed_at=completed_at,
                input=input_doc,
                output=output_doc,
                output_file_paths={
                    k.name.lower(): v for k, v in output_file_paths.items()
                },
                bader=bader,
                run_type=run_type(input_doc.dict()),
                task_type=task_type(input_doc.dict()),
                calc_type=calc_type(input_doc.dict()),
        ), cp2k_objects


def _get_output_file_paths(volumetric_files: List[str]) -> Dict[Cp2kObject, str]:
    """
    Get the output file paths for CP2K output files from the list of volumetric files.

    Parameters
    ----------
    volumetric_files
        A list of volumetric files associated with the calculation.

    Returns
    -------
    Dict[Cp2kObject, str]
        A mapping between the CP2K object type and the file path.
    """
    output_file_paths = {}
    for cp2k_object in Cp2kObject:  # type: ignore
        for volumetric_file in volumetric_files:
            if cp2k_object.name in str(volumetric_file):
                output_file_paths[cp2k_object] = str(volumetric_file)
    return output_file_paths


def _get_volumetric_data(
    dir_name: Path,
    output_file_paths: Dict[Cp2kObject, str],
    store_volumetric_data: Optional[Tuple[str]],
) -> Dict[Cp2kObject, Cube]:
    """
    Load volumetric data files from a directory.

    Parameters
    ----------
    dir_name
        The directory containing the files.
    output_file_paths
        A dictionary mapping the data type to file path relative to dir_name.
    store_volumetric_data
        The volumetric data files to load. E.g., `("v_hartree", "e_density", "spin_density")

    Returns
    -------
    Dict[Cp2kObject, VolumetricData]
        A dictionary mapping the CP2K object data type (`Cp2kObject.v_hartree`,
        `Cp2kObject.electron_density`, etc) to the volumetric data object.
    """

    if store_volumetric_data is None or len(store_volumetric_data) == 0:
        return {}

    volumetric_data = {}
    for file_type, file in output_file_paths.items():
        if file_type.name not in store_volumetric_data:
            pass

        try:
            # TODO This volumetric data may be in atomic units. i.e. 
            # Cp2k version of locpot stores in Ha, not eV, so must be converted 
            # somewhere
            volumetric_data[file_type] = Cube(dir_name / file)
        except Exception:
            raise ValueError(f"Failed to parse {file_type} at {file}.")
    
    for file_type in volumetric_data:
        if file_type.name in __is_stored_in_Ha__:
            volumetric_data[file_type].scale() # TODO write this method

    return volumetric_data

# TODO As written, this will only get the complete dos if it is available. 
# cp2k can only generate the complete DOS for gamma-point only calculations
# and it has to be requested (not default). Should this method grab overall
# dos / elemental project dos if the complete dos is not available, or stick
# to grabbing the complete dos?
def _parse_dos(parse_dos: str | bool, cp2k_output: Cp2kOutput) -> Optional[Dos]:
    """
            parse_dos
            Whether to parse the DOS. Can be:

            - "auto": Only parse DOS if there are no ionic steps.
            - True: Always parse DOS.
            - False: Never parse DOS.
    """
    if parse_dos == "auto":
        if len(cp2k_output.ionic_steps) == 0:
            return cp2k_output.complete_dos
        return None
    if parse_dos:
        return cp2k_output.complete_dos
    return None

def _parse_bandstructure(parse_bandstructure: str | bool, cp2k_output: Cp2kOutput) -> Optional[BandStructure]:
    """
    Get the band structure.

    Parameters
    ----------
         parse_bandstructure (bool): Whether to parse. Does not support the auto/line distinction currently.
    """
    if parse_bandstructure:
        return cp2k_output.band_structure
    return None

def _parse_trajectory(cp2k_output: Cp2kOutput) -> Optional[Trajectory]:
    """
    Grab a Trajectory object given a cp2k output object. 

    If an "ener" file is present containing MD results, it will be added as frame data to the traj object
    """
    ener = cp2k_output.filenames.get("ener")[-1] if cp2k_output.filenames.get("ener") else None
    data = parse_energy_file(ener) if ener else None
    constant_lattice = all(s.lattice == cp2k_output.initial_structure.lattice for s in cp2k_output.structures)
    return Trajectory.from_structures(cp2k_output.structures, constant_lattice=constant_lattice, frame_properties=data)