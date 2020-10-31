"""Core definitions of a VASP calculation documents."""
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from atomate2.common.models import AtomateModel
from atomate2.settings import settings
from emmet.core.vasp.calc_types import (
    RunType,
    CalcType,
    TaskType,
    calc_type,
    run_type,
    task_type,
)
from emmet.stubs import Lattice, Matrix3D, Structure, Vector3D
from pydantic import Field
from pydantic.datetime_parse import datetime
from pymatgen.command_line.bader_caller import bader_analysis_from_path
from pymatgen.io.vasp import BSVasprun, Chgcar, Locpot, Outcar, Vasprun, VolumetricData

logger = logging.getLogger(__name__)


class Status(Enum):
    """VASP calculation state."""

    SUCCESS = "successful"
    FAILED = "failed"


class VaspObject(Enum):
    """Types of VASP data objects."""

    BANDSTRUCTURE = "bandstructure"
    DOS = "dos"
    CHGCAR = "chg"
    AECCAR0 = "aec0"
    AECCAR1 = "aec1"
    AECCAR2 = "aec2"
    TRAJECTORY = "traj"
    ELFCAR = "elf"
    WAVECAR = "wave"
    LOCPOT = "locpot"
    OPTIC = "optic"
    PROCAR = "proj"


class PotcarSpec(AtomateModel):
    """Document defining a VASP POTCAR specification."""

    titel: str = Field(None, description="TITEL field from POTCAR header")
    hash: str = Field(None, description="md5 hash of POTCAR file")


class VaspInputDoc(AtomateModel):
    """Document defining VASP calculation inputs."""

    incar: Dict[str, Any] = Field(
        None, description="INCAR parameters for the calculation"
    )
    kpoints: Dict[str, Any] = Field(None, description="KPOINTS for the calculation")
    nkpoints: int = Field(None, description="Total number of k-points")
    potcar: List[str] = Field(None, description="POTCAR symbols in the calculation")
    potcar_spec: List[PotcarSpec] = Field(
        None, description="Title and hash of POTCAR files used in the calculation"
    )
    potcar_type: List[str] = Field(None, description="List of POTCAR functional types.")
    parameters: Dict = Field(None, description="Parameters from vasprun")
    lattice_rec: Lattice = Field(
        None, description="Reciprocal lattice of the structure"
    )
    structure: Structure = Field(
        None, description="Input structure for the calculation"
    )

    @classmethod
    def from_vasprun(cls, vasprun: Vasprun) -> "VaspInputDoc":
        """
        Create a VASP input document from a Vasprun object.

        Args:
            vasprun: A vasprun object.

        Returns:
            The input document.
        """
        kpoints_dict = vasprun.kpoints.as_dict()
        kpoints_dict["actual_kpoints"] = [
            {"abc": list(k), "weight": w}
            for k, w in zip(vasprun.actual_kpoints, vasprun.actual_kpoints_weights)
        ]
        return cls(
            structure=vasprun.initial_structure,
            incar=dict(vasprun.incar),
            kpoints=kpoints_dict,
            nkpoints=len(kpoints_dict["actual_kpoints"]),
            potcar=[s.split(" ")[0] for s in vasprun.potcar_symbols],
            potcar_spec=vasprun.potcar_spec,
            potcar_type=[s.split(" ")[0] for s in vasprun.potcar_symbols],
            parameters=dict(vasprun.parameters),
            lattice_rec=vasprun.initial_structure.lattice.reciprocal_lattice,
        )


class RunStatistics(AtomateModel):
    """Summary of the run statistics for a VASP calculation."""

    average_memory: float = Field(None, description="The average memory used in kb")
    max_memory: float = Field(None, description="The maximum memory used in kb")
    elapsed_time: float = Field(None, description="The real time elapsed in seconds")
    system_time: float = Field(None, description="The system CPU time in seconds")
    user_time: float = Field(
        None, description="The user CPU time spent by VASP in seconds"
    )
    total_time: float = Field(
        None, description="The total CPU time for this calculation"
    )
    cores: int = Field(None, description="The number of cores used by VASP")

    @classmethod
    def from_outcar(cls, outcar: Outcar) -> "RunStatistics":
        """
        Create a run statistics document from an Outcar object.

        Args:
            outcar: An Outcar object.

        Returns:
            The run statistics.
        """
        # rename these statistics
        mapping = {
            "Average memory used (kb)": "average_memory",
            "Maximum memory used (kb)": "max_memory",
            "Elapsed time (sec)": "elapsed_time",
            "System time (sec)": "system_time",
            "User time (sec)": "user_time",
            "Total CPU time used (sec)": "total_time",
            "cores": "cores",
        }
        return cls(**{v: outcar.run_stats.get(k, None) for k, v in mapping.items()})


class VaspOutputDoc(AtomateModel):
    """Document defining VASP calculation outputs."""

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
    is_gap_direct: bool = Field(None, description="Whether the band gap is direct")
    direct_gap: float = Field(
        None, description="Direct band gap in eV (if system is not metallic)"
    )
    transition: str = Field(
        None, description="Band gap transition given by CBM and VBM k-points"
    )
    epsilon_static: Matrix3D = Field(
        None, description="The high-frequency dielectric constant"
    )
    epsilon_static_wolfe: Matrix3D = Field(
        None,
        description="The high-frequency dielectric constant w/o local field effects",
    )
    epsilon_ionic: Matrix3D = Field(
        None, description="The ionic part of the dielectric constant"
    )
    ionic_steps: List[Dict[str, Any]] = Field(
        None, description="Energy, forces, and structure for each ionic step"
    )
    locpot: Dict[int, float] = Field(
        None, description="Average of the local potential along the crystal axes"
    )
    outcar: Dict[str, Any] = Field(
        None, description="Information extracted from the OUTCAR file"
    )
    force_constants: List[List[Matrix3D]] = Field(
        None, description="Force constants between every pair of atoms in the structure"
    )
    normalmode_eigenvals: List[float] = Field(
        None, description="Normal mode eigenvalues of phonon modes at Gamma"
    )
    normalmode_eigenvecs: List[Vector3D] = Field(
        None, description="Normal mode eigenvectors of phonon modes at Gamma"
    )
    run_stats: RunStatistics = Field(
        None, description="Summary of runtime statistics for this calculation"
    )

    @classmethod
    def from_vasp_outputs(
        cls, vasprun: Vasprun, outcar: Outcar, locpot: Optional[Locpot] = None
    ) -> "VaspOutputDoc":
        """
        Create a VASP output document from VASP outputs.

        Args:
            vasprun: A Vasprun object.
            outcar: An Outcar object.
            locpot: A Locpot object.

        Returns:
            The VASP calculation output document.
        """
        try:
            bandstructure = vasprun.get_band_structure()
            bandgap_info = bandstructure.get_band_gap()
            electronic_output = dict(
                vbm=bandstructure.get_vbm()["energy"],
                cbm=bandstructure.get_cbm()["energy"],
                bandgap=bandgap_info["energy"],
                is_gap_direct=bandgap_info["direct"],
                is_metal=bandstructure.is_metal(),
                direct_gap=bandstructure.get_direct_band_gap(),
                transition=bandgap_info["transition"],
            )
        except Exception:
            logger.warning("Error in parsing bandstructure")
            if vasprun.incar["IBRION"] == 1:
                logger.warning("Vasp doesn't properly output efermi for IBRION == 1")
            electronic_output = {}

        locpot_avg = None
        if locpot:
            locpot_avg = {i: locpot.get_average_along_axis(i) for i in range(3)}

        # parse force constants
        phonon_output = {}
        if hasattr(vasprun, "force_constants"):
            phonon_output = dict(
                force_constants=vasprun.force_constants.tolist(),
                normalmode_eigenvals=vasprun.normalmode_eigenvals.tolist(),
                normalmode_eigenvecs=vasprun.normalmode_eigenvecs.tolist(),
            )

        outcar_dict = outcar.as_dict()
        outcar_dict.pop("run_stats")

        structure = vasprun.final_structure
        if len(outcar.magnetization) != 0:
            # patch calculated magnetic moments into final structure
            magmoms = [m["tot"] for m in outcar.magnetization]
            structure.add_site_property("magmom", magmoms)

        return cls(
            structure=structure,
            energy=vasprun.final_energy,
            energy_per_atom=vasprun.final_energy / len(structure),
            epsilon_static=vasprun.epsilon_static or None,
            epsilon_static_wolfe=vasprun.epsilon_static_wolfe or None,
            epsilon_ionic=vasprun.epsilon_ionic or None,
            ionic_steps=vasprun.ionic_steps,
            locpot=locpot_avg,
            outcar=outcar_dict,
            run_stats=RunStatistics.from_outcar(outcar),
            **electronic_output,
            **phonon_output,
        )


class VaspCalcDoc(AtomateModel):
    """Full VASP calculation inputs and outputs."""

    dir_name: str = Field(None, description="The directory for this VASP calculation")
    vasp_version: str = Field(
        None, description="VASP version used to perform the calculation"
    )
    has_vasp_completed: Status = Field(
        None, description="Whether VASP completed the calculation successfully"
    )
    input: VaspInputDoc = Field(
        None, description="VASP input settings for the calculation"
    )
    output: VaspOutputDoc = Field(None, description="The VASP calculation output")
    completed_at: datetime = Field(
        None, description="Timestamp for when the calculation was completed"
    )
    task_name: str = Field(
        None, description="Name of task given by custodian (e.g., relax1, relax2)"
    )
    output_file_paths: Dict[str, str] = Field(
        None,
        description="Paths (relative to dir_name) of the VASP output files "
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
    def from_vasp_files(
        cls,
        dir_name: Union[Path, str],
        task_name: str,
        vasprun_file: Union[Path, str],
        outcar_file: Union[Path, str],
        volumetric_files: List[str],
        parse_dos: Union[str, bool] = "auto",
        parse_bandstructure: Union[str, bool] = "auto",
        average_locpot: bool = True,
        run_bader: bool = settings.vasp_run_bader,
        store_volumetric_data: Optional[
            Tuple[str]
        ] = settings.vasp_store_volumetric_data,
        vasprun_kwargs: Optional[Dict] = None,
    ) -> Tuple["VaspCalcDoc", Dict[VaspObject, Dict]]:
        """
        Create a VASP calculation document from a directory and file paths.

        Args:
            dir_name: The directory containing the calculation outputs.
            task_name: The task name.
            vasprun_file: Path to the vasprun.xml file, relative to dir_name.
            outcar_file: Path to the OUTCAR file, relative to dir_name.
            volumetric_files: Path to volumetric files, relative to dir_name.
            parse_dos: Whether to parse the DOS. Can be:
                - "auto": Only parse DOS if there are no ionic steps (NSW = 0).
                - True: Always parse DOS.
                - False: Never parse DOS.
            parse_bandstructure: How to parse the bandstructure. Can be:
                - "auto": Parse the bandstructure with projections for NSCF calculations
                  and decide automatically if it's line or uniform mode.
                - "line": Parse the bandstructure as a line mode calculation with
                  projections
                - True: Parse the bandstructure as a uniform calculation with
                  projections .
                - False: Parse the band structure without projects and just store
                  vbm, cbm, band_gap, is_metal and efermi rather than the full
                  band structure object.
            average_locpot: Whether to store the average of the LOCPOT along the
                crystal axes.
            run_bader: Whether to run bader on the charge density.
            store_volumetric_data: Which volumetric files to store.
            vasprun_kwargs: Additional keyword arguemnts that will be passed to
                to the Vasprun init.

        Returns:
            A VASP calculation document.
        """
        dir_name = Path(dir_name)
        vasprun_file = dir_name / vasprun_file
        outcar_file = dir_name / outcar_file

        vasprun_kwargs = vasprun_kwargs if vasprun_kwargs else {}
        vasprun = Vasprun(vasprun_file, **vasprun_kwargs)
        outcar = Outcar(outcar_file)
        completed_at = str(datetime.fromtimestamp(vasprun_file.stat().st_mtime))

        output_file_paths = _get_output_file_paths(volumetric_files)
        vasp_objects: Dict[VaspObject, Any] = _get_volumetric_data(
            dir_name, output_file_paths, store_volumetric_data
        )

        dos = _parse_dos(parse_dos, vasprun)
        if dos is not None:
            vasp_objects[VaspObject.DOS] = dos

        bandstructure = _parse_bandstructure(parse_bandstructure, vasprun)
        if bandstructure is not None:
            vasp_objects[VaspObject.BANDSTRUCTURE] = bandstructure

        bader = None
        if run_bader and "chgcar" in output_file_paths:
            suffix = "" if task_name == "standard" else f".{task_name}"
            bader = bader_analysis_from_path(dir_name, suffix=suffix)

        locpot = None
        if average_locpot:
            if VaspObject.LOCPOT in vasp_objects:
                locpot = vasp_objects[VaspObject.LOCPOT]
            elif VaspObject.LOCPOT in output_file_paths:
                locpot_file = dir_name / output_file_paths[VaspObject.LOCPOT]
                locpot = Locpot.from_file(locpot_file)

        input_doc = VaspInputDoc.from_vasprun(vasprun)
        output_doc = VaspOutputDoc.from_vasp_outputs(vasprun, outcar, locpot=locpot)

        has_vasp_completed = Status.SUCCESS if vasprun.converged else Status.FAILED
        return (
            cls(
                dir_name=str(dir_name),
                task_name=task_name,
                vasp_version=vasprun.vasp_version,
                has_vasp_completed=has_vasp_completed,
                completed_at=completed_at,
                input=input_doc,
                output=output_doc,
                output_file_paths={
                    k.name.lower(): v for k, v in output_file_paths.items()
                },
                bader=bader,
                run_type=run_type(input_doc.parameters),
                task_type=task_type(input_doc.dict()),
                calc_type=calc_type(input_doc.dict(), input_doc.parameters),
            ),
            vasp_objects,
        )


def _get_output_file_paths(volumetric_files: List[str]) -> Dict[VaspObject, str]:
    """
    Get the output file paths.for VASP output files from the list of volumetric files.

    Args:
        volumetric_files: A list of volumetric files associated with the calculation.

    Returns:
        A mapping between the VASP object type and the file path.
    """
    output_file_paths = {}
    for vasp_object in VaspObject:
        for volumetric_file in volumetric_files:
            if vasp_object.name in volumetric_file:
                output_file_paths[vasp_object] = volumetric_file
    return output_file_paths


def _get_volumetric_data(
    dir_name: Path,
    output_file_paths: Dict[VaspObject, str],
    store_volumetric_data: Optional[Tuple[str]],
) -> Dict[VaspObject, VolumetricData]:
    """
    Load volumetric data files from a directory.

    Args:
        dir_name: The directory containing the files.
        output_file_paths: A dictionary mapping the data type to file path relative to
            dir_name.
        store_volumetric_data: The volumetric data files to load. E.g.,
            `("chgcar", "locpot")

    Returns:
        A dictionary mapping the VASP object data type (VaspObject.LOCPOT,
        VaspObject.CHGCAR, etc) to the volumetric data object.
    """
    if store_volumetric_data is None or len(store_volumetric_data) == 0:
        return {}

    volumetric_data = {}
    for file_type, file in output_file_paths.items():
        if file_type.name not in store_volumetric_data:
            pass

        try:
            # assume volumetric data is all in CHGCAR format
            volumetric_data[file_type] = Chgcar.from_file(dir_name / file)
        except Exception:
            raise ValueError(f"Failed to parse {file_type} at {file}.")
    return volumetric_data


def _parse_dos(parse_mode: Union[str, bool], vasprun: Vasprun) -> Optional[Dict]:
    """Parse DOS. See VaspCalcDoc.from_vasp_files for supported arguments."""
    nsw = vasprun.incar.get("NSW", 0)
    dos = None
    if parse_mode is True or (parse_mode == "auto" and nsw < 1):
        dos = vasprun.complete_dos.as_dict()
    return dos


def _parse_bandstructure(
    parse_mode: Union[str, bool], vasprun: Vasprun
) -> Optional[Dict[str, Any]]:
    """Parse band structure. See VaspCalcDoc.from_vasp_files for supported arguments."""
    vasprun_file = vasprun.filename

    if parse_mode == "auto":
        if vasprun.incar.get("ICHARG", 0) > 10:
            # NSCF calculation
            bs_vrun = BSVasprun(vasprun_file, parse_projected_eigen=True)
            try:
                bs = bs_vrun.get_band_structure(line_mode=True)  # try parsing line mode
            except Exception:
                bs = bs_vrun.get_band_structure()  # treat as a regular calculation
        else:
            # Not a NSCF calculation
            bs_vrun = BSVasprun(vasprun_file, parse_projected_eigen=False)
            bs = bs_vrun.get_band_structure()

        # only save the bandstructure if not moving ions
        if vasprun.incar.get("NSW", 0) <= 1:
            return bs.as_dict()

    elif parse_mode:
        # legacy line/True behavior for bandstructure_mode
        bs_vrun = BSVasprun(vasprun_file, parse_projected_eigen=True)
        bs = bs_vrun.get_band_structure(line_mode=(parse_mode == "line"))
        return bs.as_dict()

    return None
