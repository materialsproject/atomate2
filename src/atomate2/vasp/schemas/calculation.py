"""Core definitions of a VASP calculation documents."""

import logging
from pathlib import Path
from shutil import which
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from jobflow.utils import ValueEnum
from pydantic import BaseModel, Extra, Field
from pydantic.datetime_parse import datetime
from pymatgen.command_line.bader_caller import bader_analysis_from_path
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.core.trajectory import Trajectory
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.core import OrbitalType
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.vasp import (
    BSVasprun,
    Locpot,
    Outcar,
    Poscar,
    Potcar,
    PotcarSingle,
    Vasprun,
    VolumetricData,
)

from atomate2 import SETTINGS
from atomate2.common.schemas.math import Matrix3D, Vector3D
from atomate2.vasp.schemas.calc_types import (
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
    "VaspObject",
    "PotcarSpec",
    "FrequencyDependentDielectric",
    "CalculationInput",
    "CalculationOutput",
    "RunStatistics",
    "Calculation",
    "IonicStep",
    "ElectronicStep",
    "ElectronPhononDisplacedStructures",
]


_BADER_EXE_EXISTS = True if (which("bader") or which("bader.exe")) else False


class Status(ValueEnum):
    """VASP calculation state."""

    SUCCESS = "successful"
    FAILED = "failed"


class VaspObject(ValueEnum):
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


class PotcarSpec(BaseModel):
    """Document defining a VASP POTCAR specification."""

    titel: str = Field(None, description="TITEL field from POTCAR header")
    hash: str = Field(None, description="md5 hash of POTCAR file")

    @classmethod
    def from_potcar_single(cls, potcar_single: PotcarSingle) -> "PotcarSpec":
        """
        Get a PotcarSpec from a PotcarSingle.

        Parameters
        ----------
        potcar_single
            A potcar single object.

        Returns
        -------
        PotcarSpec
            A potcar spec.
        """
        potcar_hash = potcar_single.get_potcar_hash()
        return cls(titel=potcar_single.symbol, hash=potcar_hash)

    @classmethod
    def from_potcar(cls, potcar: Potcar) -> List["PotcarSpec"]:
        """
        Get a list of PotcarSpecs from a Potcar.

        Parameters
        ----------
        potcar
            A potcar object.

        Returns
        -------
        list[PotcarSpec]
            A list of potcar specs.
        """
        return [cls.from_potcar_single(p) for p in potcar]


class CalculationInput(BaseModel):
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
    is_hubbard: bool = Field(False, description="Is this a Hubbard +U calculation")
    hubbards: Dict = Field(None, description="The hubbard parameters used")

    @classmethod
    def from_vasprun(cls, vasprun: Vasprun) -> "CalculationInput":
        """
        Create a VASP input document from a Vasprun object.

        Parameters
        ----------
        vasprun
            A vasprun object.

        Returns
        -------
        CalculationInput
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
            potcar=[s.split()[0] for s in vasprun.potcar_symbols],
            potcar_spec=vasprun.potcar_spec,
            potcar_type=[s.split()[0] for s in vasprun.potcar_symbols],
            parameters=dict(vasprun.parameters),
            lattice_rec=vasprun.initial_structure.lattice.reciprocal_lattice,
            is_hubbard=vasprun.is_hubbard,
            hubbards=vasprun.hubbards,
        )


class RunStatistics(BaseModel):
    """Summary of the run statistics for a VASP calculation."""

    average_memory: float = Field(0, description="The average memory used in kb")
    max_memory: float = Field(0, description="The maximum memory used in kb")
    elapsed_time: float = Field(0, description="The real time elapsed in seconds")
    system_time: float = Field(0, description="The system CPU time in seconds")
    user_time: float = Field(
        0, description="The user CPU time spent by VASP in seconds"
    )
    total_time: float = Field(0, description="The total CPU time for this calculation")
    cores: int = Field(0, description="The number of cores used by VASP")

    @classmethod
    def from_outcar(cls, outcar: Outcar) -> "RunStatistics":
        """
        Create a run statistics document from an Outcar object.

        Parameters
        ----------
        outcar
            An Outcar object.

        Returns
        -------
        RunStatistics
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
        run_stats = {}
        for k, v in mapping.items():
            stat = outcar.run_stats.get(k) or 0
            try:
                stat = float(stat)
            except ValueError:
                # sometimes the statistics are misformatted
                stat = 0

            run_stats[v] = stat

        return cls(**run_stats)


class FrequencyDependentDielectric(BaseModel):
    """Frequency-dependent dielectric data."""

    real: List[List[float]] = Field(
        None,
        description="Real part of the frequency dependent dielectric constant, given at"
        " each energy as 6 components according to XX, YY, ZZ, XY, YZ, ZX",
    )
    imaginary: List[List[float]] = Field(
        None,
        description="Imaginary part of the frequency dependent dielectric constant, "
        "given at each energy as 6 components according to XX, YY, ZZ, XY, "
        "YZ, ZX",
    )
    energy: List[float] = Field(
        None,
        description="Energies at which the real and imaginary parts of the dielectric"
        "constant are given",
    )

    @classmethod
    def from_vasprun(cls, vasprun: Vasprun) -> "FrequencyDependentDielectric":
        """
        Create a frequency-dependent dielectric calculation document from a vasprun.

        Parameters
        ----------
        vasprun : Vasprun
            A vasprun object.

        Returns
        -------
        FrequencyDependentDielectric
            A frequency-dependent dielectric document.
        """
        energy, real, imag = vasprun.dielectric
        return cls(real=real, imaginary=imag, energy=energy)


class ElectronPhononDisplacedStructures(BaseModel):
    """Document defining electron phonon displaced structures."""

    temperatures: List[float] = Field(
        None,
        description="The temperatures at which the electron phonon displacements "
        "were generated.",
    )
    structures: List[Structure] = Field(
        None, description="The displaced structures corresponding to each temperature."
    )


class ElectronicStep(BaseModel, extra=Extra.allow):  # type: ignore
    """Document defining the information at each electronic step.

    Note, not all the information will be available at every step.
    """

    alphaZ: float = Field(None, description="The alpha Z term.")
    ewald: float = Field(None, description="The ewald energy.")
    hartreedc: float = Field(None, description="Negative Hartree energy.")
    XCdc: float = Field(None, description="Negative exchange energy.")
    pawpsdc: float = Field(
        None, description="Negative potential energy with exchange-correlation energy."
    )
    pawaedc: float = Field(None, description="The PAW double counting term.")
    eentropy: float = Field(None, description="The entropy (T * S).")
    bandstr: float = Field(None, description="The band energy (from eigenvalues).")
    atom: float = Field(None, description="The atomic energy.")
    e_fr_energy: float = Field(None, description="The free energy.")
    e_wo_entrp: float = Field(None, description="The energy without entropy.")
    e_0_energy: float = Field(None, description="The internal energy.")


class IonicStep(BaseModel, extra=Extra.allow):  # type: ignore
    """Document defining the information at each ionic step."""

    e_fr_energy: float = Field(None, description="The free energy.")
    e_wo_entrp: float = Field(None, description="The energy without entropy.")
    e_0_energy: float = Field(None, description="The internal energy.")
    forces: List[Vector3D] = Field(None, description="The forces on each atom.")
    stress: Matrix3D = Field(None, description="The stress on the lattice.")
    electronic_steps: List[ElectronicStep] = Field(
        None, description="The electronic convergence steps."
    )
    structure: Structure = Field(None, description="The structure at this step.")


class CalculationOutput(BaseModel):
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
    mag_density: float = Field(
        None,
        description="The magnetization density, defined as total_mag/volume "
        "(units of A^-3)",
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
    frequency_dependent_dielectric: FrequencyDependentDielectric = Field(
        None,
        description="Frequency-dependent dielectric information from an LOPTICS "
        "calculation",
    )
    ionic_steps: List[IonicStep] = Field(
        None, description="Energy, forces, structure, etc. for each ionic step"
    )
    locpot: Dict[int, List[float]] = Field(
        None, description="Average of the local potential along the crystal axes"
    )
    outcar: Dict[str, Any] = Field(
        None, description="Information extracted from the OUTCAR file"
    )
    force_constants: List[List[Matrix3D]] = Field(
        None, description="Force constants between every pair of atoms in the structure"
    )
    normalmode_frequencies: List[float] = Field(
        None, description="Frequencies in THz of the normal modes at Gamma"
    )
    normalmode_eigenvals: List[float] = Field(
        None,
        description="Normal mode eigenvalues of phonon modes at Gamma. "
        "Note the unit changed between VASP 5 and 6.",
    )
    normalmode_eigenvecs: List[List[Vector3D]] = Field(
        None, description="Normal mode eigenvectors of phonon modes at Gamma"
    )
    elph_displaced_structures: ElectronPhononDisplacedStructures = Field(
        None,
        description="Electron-phonon displaced structures, generated by setting "
        "PHON_LMC = True.",
    )
    dos_properties: Dict[str, Dict[str, Dict[str, float]]] = Field(
        None,
        description="Element- and orbital-projected band properties (in eV) for the "
        "DOS. All properties are with respect to the Fermi level.",
    )
    run_stats: RunStatistics = Field(
        None, description="Summary of runtime statistics for this calculation"
    )

    @classmethod
    def from_vasp_outputs(
        cls,
        vasprun: Vasprun,
        outcar: Outcar,
        contcar: Poscar,
        locpot: Optional[Locpot] = None,
        elph_poscars: Optional[List[Path]] = None,
        store_trajectory: bool = False,
    ) -> "CalculationOutput":
        """
        Create a VASP output document from VASP outputs.

        Parameters
        ----------
        vasprun
            A Vasprun object.
        outcar
            An Outcar object.
        contcar
            A Poscar object.
        locpot
            A Locpot object.
        elph_poscars
            Path to displaced electron-phonon coupling POSCAR files generated using
            ``PHON_LMC = True``.
        store_trajectory
            Whether to store ionic steps as a pymatgen Trajectory object. If `True`,
            the `ionic_steps` field is left as None.

        Returns
        -------
            The VASP calculation output document.
        """
        try:
            bandstructure = vasprun.get_band_structure(efermi="smart")
            bandgap_info = bandstructure.get_band_gap()
            electronic_output = dict(
                efermi=bandstructure.efermi,
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
                logger.warning("VASP doesn't properly output efermi for IBRION == 1")
            electronic_output = {}

        freq_dependent_diel: Union[dict, FrequencyDependentDielectric] = {}
        try:
            freq_dependent_diel = FrequencyDependentDielectric.from_vasprun(vasprun)
        except KeyError:
            pass

        locpot_avg = None
        if locpot:
            locpot_avg = {
                i: locpot.get_average_along_axis(i).tolist() for i in range(3)
            }

        # parse force constants
        phonon_output = {}
        if hasattr(vasprun, "force_constants"):
            # convert eigenvalues to frequency
            eigs = -vasprun.normalmode_eigenvals
            frequencies = np.sqrt(np.abs(eigs)) * np.sign(eigs)

            # convert to THz in VASP 5 and lower; VASP 6 uses THz internally
            major_version = int(vasprun.vasp_version.split(".")[0])
            if major_version < 6:
                frequencies *= 15.633302

            phonon_output = dict(
                force_constants=vasprun.force_constants.tolist(),
                normalmode_frequencies=frequencies.tolist(),
                normalmode_eigenvals=vasprun.normalmode_eigenvals.tolist(),
                normalmode_eigenvecs=vasprun.normalmode_eigenvecs.tolist(),
            )

        outcar_dict = outcar.as_dict()
        outcar_dict.pop("run_stats")

        # use structure from CONTCAR as it is written to
        # greater precision than in the vasprun
        structure = contcar.structure
        mag_density = outcar.total_mag / structure.volume if outcar.total_mag else None

        if len(outcar.magnetization) != 0:
            # patch calculated magnetic moments into final structure
            magmoms = [m["tot"] for m in outcar.magnetization]
            structure.add_site_property("magmom", magmoms)

        # Parse DOS properties
        dosprop_dict = (
            _get_band_props(vasprun.complete_dos, structure)
            if hasattr(vasprun, "complete_dos")
            else {}
        )

        elph_structures: Dict[str, List[Any]] = {}
        if elph_poscars is not None:
            elph_structures.update({"temperatures": [], "structures": []})
            for elph_poscar in elph_poscars:
                temp = str(elph_poscar.name).replace("POSCAR.T=", "").replace(".gz", "")
                elph_structures["temperatures"].append(temp)
                elph_structures["structures"].append(Structure.from_file(elph_poscar))

        return cls(
            structure=structure,
            energy=vasprun.final_energy,
            energy_per_atom=vasprun.final_energy / len(structure),
            mag_density=mag_density,
            epsilon_static=vasprun.epsilon_static or None,
            epsilon_static_wolfe=vasprun.epsilon_static_wolfe or None,
            epsilon_ionic=vasprun.epsilon_ionic or None,
            frequency_dependent_dielectric=freq_dependent_diel,
            elph_displaced_structures=elph_structures,
            dos_properties=dosprop_dict,
            ionic_steps=vasprun.ionic_steps if not store_trajectory else None,
            locpot=locpot_avg,
            outcar=outcar_dict,
            run_stats=RunStatistics.from_outcar(outcar),
            **electronic_output,
            **phonon_output,
        )


class Calculation(BaseModel):
    """Full VASP calculation inputs and outputs."""

    dir_name: str = Field(None, description="The directory for this VASP calculation")
    vasp_version: str = Field(
        None, description="VASP version used to perform the calculation"
    )
    has_vasp_completed: Status = Field(
        None, description="Whether VASP completed the calculation successfully"
    )
    input: CalculationInput = Field(
        None, description="VASP input settings for the calculation"
    )
    output: CalculationOutput = Field(None, description="The VASP calculation output")
    completed_at: str = Field(
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
        contcar_file: Union[Path, str],
        volumetric_files: List[str] = None,
        elph_poscars: List[Path] = None,
        parse_dos: Union[str, bool] = False,
        parse_bandstructure: Union[str, bool] = False,
        average_locpot: bool = True,
        run_bader: bool = (SETTINGS.VASP_RUN_BADER and _BADER_EXE_EXISTS),
        strip_bandstructure_projections: bool = False,
        strip_dos_projections: bool = False,
        store_volumetric_data: Optional[
            Tuple[str]
        ] = SETTINGS.VASP_STORE_VOLUMETRIC_DATA,
        store_trajectory: bool = False,
        vasprun_kwargs: Optional[Dict] = None,
    ) -> Tuple["Calculation", Dict[VaspObject, Dict]]:
        """
        Create a VASP calculation document from a directory and file paths.

        Parameters
        ----------
        dir_name
            The directory containing the calculation outputs.
        task_name
            The task name.
        vasprun_file
            Path to the vasprun.xml file, relative to dir_name.
        outcar_file
            Path to the OUTCAR file, relative to dir_name.
        contcar_file
            Path to the CONTCAR file, relative to dir_name
        volumetric_files
            Path to volumetric files, relative to dir_name.
        elph_poscars
            Path to displaced electron-phonon coupling POSCAR files generated using
            ``PHON_LMC = True``, given relative to dir_name.
        parse_dos
            Whether to parse the DOS. Can be:

            - "auto": Only parse DOS if there are no ionic steps (NSW = 0).
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

        average_locpot
            Whether to store the average of the LOCPOT along the crystal axes.
        run_bader
            Whether to run bader on the charge density.
        strip_dos_projections
            Whether to strip the element and site projections from the density of
            states. This can help reduce the size of DOS objects in systems with many
            atoms.
        strip_bandstructure_projections
            Whether to strip the element and site projections from the band structure.
            This can help reduce the size of DOS objects in systems with many atoms.
        store_volumetric_data
            Which volumetric files to store.
        store_trajectory
            Whether to store the ionic steps in a pymatgen Trajectory object. if `True`,
            :obj:'.CalculationOutput.ionic_steps' is set to None to reduce duplicating
            information.
        vasprun_kwargs
            Additional keyword arguments that will be passed to the Vasprun init.

        Returns
        -------
        Calculation
            A VASP calculation document.
        """
        dir_name = Path(dir_name)
        vasprun_file = dir_name / vasprun_file
        outcar_file = dir_name / outcar_file
        contcar_file = dir_name / contcar_file

        vasprun_kwargs = vasprun_kwargs if vasprun_kwargs else {}
        volumetric_files = [] if volumetric_files is None else volumetric_files
        vasprun = Vasprun(vasprun_file, **vasprun_kwargs)
        outcar = Outcar(outcar_file)
        contcar = Poscar.from_file(contcar_file)
        completed_at = str(datetime.fromtimestamp(vasprun_file.stat().st_mtime))

        output_file_paths = _get_output_file_paths(volumetric_files)
        vasp_objects: Dict[VaspObject, Any] = _get_volumetric_data(
            dir_name, output_file_paths, store_volumetric_data
        )

        dos = _parse_dos(parse_dos, vasprun)
        if dos is not None:
            if strip_dos_projections:
                dos = Dos(dos.efermi, dos.energies, dos.densities)
            vasp_objects[VaspObject.DOS] = dos  # type: ignore

        bandstructure = _parse_bandstructure(parse_bandstructure, vasprun)
        if bandstructure is not None:
            if strip_bandstructure_projections:
                bandstructure.projections = {}
            vasp_objects[VaspObject.BANDSTRUCTURE] = bandstructure  # type: ignore

        bader = None
        if run_bader and VaspObject.CHGCAR in output_file_paths:
            suffix = "" if task_name == "standard" else f".{task_name}"
            bader = bader_analysis_from_path(dir_name, suffix=suffix)

        locpot = None
        if average_locpot:
            if VaspObject.LOCPOT in vasp_objects:
                locpot = vasp_objects[VaspObject.LOCPOT]  # type: ignore
            elif VaspObject.LOCPOT in output_file_paths:
                locpot_file = output_file_paths[VaspObject.LOCPOT]  # type: ignore
                locpot = Locpot.from_file(dir_name / locpot_file)

        input_doc = CalculationInput.from_vasprun(vasprun)

        output_doc = CalculationOutput.from_vasp_outputs(
            vasprun,
            outcar,
            contcar,
            locpot=locpot,
            elph_poscars=elph_poscars,
            store_trajectory=store_trajectory,
        )
        if store_trajectory:
            traj = Trajectory.from_structures(
                [d["structure"] for d in vasprun.ionic_steps],
                frame_properties=[IonicStep(**x).dict() for x in vasprun.ionic_steps],
                constant_lattice=False,
            )
            vasp_objects[VaspObject.TRAJECTORY] = traj  # type: ignore

        # MD run
        if vasprun.parameters.get("IBRION", -1) == 0:
            if vasprun.parameters.get("NSW", 0) == vasprun.nionic_steps:
                has_vasp_completed = Status.SUCCESS
            else:
                has_vasp_completed = Status.FAILED
        # others
        else:
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
    Get the output file paths for VASP output files from the list of volumetric files.

    Parameters
    ----------
    volumetric_files
        A list of volumetric files associated with the calculation.

    Returns
    -------
    Dict[VaspObject, str]
        A mapping between the VASP object type and the file path.
    """
    output_file_paths = {}
    for vasp_object in VaspObject:  # type: ignore
        for volumetric_file in volumetric_files:
            if vasp_object.name in str(volumetric_file):
                output_file_paths[vasp_object] = str(volumetric_file)
    return output_file_paths


def _get_volumetric_data(
    dir_name: Path,
    output_file_paths: Dict[VaspObject, str],
    store_volumetric_data: Optional[Tuple[str]],
) -> Dict[VaspObject, VolumetricData]:
    """
    Load volumetric data files from a directory.

    Parameters
    ----------
    dir_name
        The directory containing the files.
    output_file_paths
        A dictionary mapping the data type to file path relative to dir_name.
    store_volumetric_data
        The volumetric data files to load. E.g., `("chgcar", "locpot")

    Returns
    -------
    Dict[VaspObject, VolumetricData]
        A dictionary mapping the VASP object data type (`VaspObject.LOCPOT`,
        `VaspObject.CHGCAR`, etc) to the volumetric data object.
    """
    from pymatgen.io.vasp import Chgcar

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


def _parse_dos(parse_mode: Union[str, bool], vasprun: Vasprun) -> Optional[Dos]:
    """Parse DOS. See Calculation.from_vasp_files for supported arguments."""
    nsw = vasprun.incar.get("NSW", 0)
    dos = None
    if parse_mode is True or (parse_mode == "auto" and nsw < 1):
        dos = vasprun.complete_dos
    return dos


def _parse_bandstructure(
    parse_mode: Union[str, bool], vasprun: Vasprun
) -> Optional[BandStructure]:
    """Parse band structure. See Calculation.from_vasp_files for supported arguments."""
    vasprun_file = vasprun.filename

    if parse_mode == "auto":
        if vasprun.incar.get("ICHARG", 0) > 10:
            # NSCF calculation
            bs_vrun = BSVasprun(vasprun_file, parse_projected_eigen=True)
            try:
                # try parsing line mode
                bs = bs_vrun.get_band_structure(line_mode=True, efermi="smart")
            except Exception:
                # treat as a regular calculation
                bs = bs_vrun.get_band_structure(efermi="smart")
        else:
            # Not a NSCF calculation
            bs_vrun = BSVasprun(vasprun_file, parse_projected_eigen=False)
            bs = bs_vrun.get_band_structure(efermi="smart")

        # only save the bandstructure if not moving ions
        if vasprun.incar.get("NSW", 0) <= 1:
            return bs

    elif parse_mode:
        # legacy line/True behavior for bandstructure_mode
        bs_vrun = BSVasprun(vasprun_file, parse_projected_eigen=True)
        bs = bs_vrun.get_band_structure(line_mode=parse_mode == "line", efermi="smart")
        return bs

    return None


def _get_band_props(
    complete_dos: CompleteDos, structure: Structure
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Calculate band properties from a CompleteDos object and Structure.

    Parameters
    ----------
    complete_dos
        A CompleteDos object.
    structure
        a pymatgen Structure object.

    Returns
    -------
    Dict
        A dictionary of element and orbital-projected DOS properties.
    """
    dosprop_dict: Dict[str, Dict[str, Dict[str, float]]] = {}
    for el in structure.composition.elements:
        el_name = el.name
        dosprop_dict[el_name] = {}
        for orb_type in [
            OrbitalType.s,
            OrbitalType.p,
            OrbitalType.d,
        ]:
            orb_name = orb_type.name
            if (
                (el.block == "s" and orb_name in ["p", "d", "f"])
                or (el.block == "p" and orb_name in ["d", "f"])
                or (el.block == "d" and orb_name == "f")
            ):
                continue
            dosprops = {
                "filling": complete_dos.get_band_filling(band=orb_type, elements=[el]),
                "center": complete_dos.get_band_center(band=orb_type, elements=[el]),
                "bandwidth": complete_dos.get_band_width(band=orb_type, elements=[el]),
                "skewness": complete_dos.get_band_skewness(
                    band=orb_type, elements=[el]
                ),
                "kurtosis": complete_dos.get_band_kurtosis(
                    band=orb_type, elements=[el]
                ),
                "upper_edge": complete_dos.get_upper_band_edge(
                    band=orb_type, elements=[el]
                ),
            }
            dosprop_dict[el_name][orb_name] = dosprops

    return dosprop_dict
