"""Schemas for phonon documents."""

import copy
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from emmet.core.math import Matrix3D
from emmet.core.structure import StructureMetadata
from monty.json import MSONable
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon
from phonopy.units import VaspToTHz
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.io.phonopy import (
    get_ph_bs_symm_line,
    get_ph_dos,
    get_phonopy_structure,
    get_pmg_structure,
)
from pymatgen.io.vasp import Kpoints
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.symmetry.kpath import KPathSeek
from typing_extensions import Self

from atomate2.aims.utils.units import omegaToTHz

logger = logging.getLogger(__name__)


def get_factor(code: str) -> float:
    """
    Get the frequency conversion factor to THz for each code.

    Parameters
    ----------
    code: str
        The code to get the conversion factor for

    Returns
    -------
    float
        The correct conversion factor

    Raises
    ------
    ValueError
        If code is not defined
    """
    if code in ["ase", "forcefields", "vasp"]:
        return VaspToTHz
    if code == "aims":
        return omegaToTHz  # Based on CODATA 2002
    raise ValueError(f"Frequency conversion factor for code ({code}) not defined.")


class PhononComputationalSettings(BaseModel):
    """Collection to store computational settings for the phonon computation."""

    # could be optional and implemented at a later stage?
    npoints_band: int = Field("number of points for band structure computation")
    kpath_scheme: str = Field("indicates the kpath scheme")
    kpoint_density_dos: int = Field(
        "number of points for computation of free energies and densities of states",
    )


class ThermalDisplacementData(BaseModel):
    """Collection to store information on the thermal displacement matrices."""

    freq_min_thermal_displacements: float = Field(
        "cutoff frequency in THz to avoid numerical issues in the "
        "computation of the thermal displacement parameters"
    )
    thermal_displacement_matrix_cif: Optional[list[list[Matrix3D]]] = Field(
        None, description="field including thermal displacement matrices in CIF format"
    )
    thermal_displacement_matrix: Optional[list[list[Matrix3D]]] = Field(
        None,
        description="field including thermal displacement matrices in Cartesian "
        "coordinate system",
    )
    temperatures_thermal_displacements: Optional[list[int]] = Field(
        None,
        description="temperatures at which the thermal displacement matrices"
        "have been computed",
    )


class PhononUUIDs(BaseModel):
    """Collection to save all uuids connected to the phonon run."""

    optimization_run_uuid: Optional[str] = Field(
        None, description="optimization run uuid"
    )
    displacements_uuids: Optional[list[str]] = Field(
        None, description="The uuids of the displacement jobs."
    )
    static_run_uuid: Optional[str] = Field(None, description="static run uuid")
    born_run_uuid: Optional[str] = Field(None, description="born run uuid")


class ForceConstants(MSONable):
    """A force constants class."""

    def __init__(self, force_constants: list[list[Matrix3D]]) -> None:
        self.force_constants = force_constants


class PhononJobDirs(BaseModel):
    """Collection to save all job directories relevant for the phonon run."""

    displacements_job_dirs: Optional[list[Optional[str]]] = Field(
        None, description="The directories where the displacement jobs were run."
    )
    static_run_job_dir: Optional[Optional[str]] = Field(
        None, description="Directory where static run was performed."
    )
    born_run_job_dir: Optional[str] = Field(
        None, description="Directory where born run was performed."
    )
    optimization_run_job_dir: Optional[str] = Field(
        None, description="Directory where optimization run was performed."
    )
    taskdoc_run_job_dir: Optional[str] = Field(
        None, description="Directory where task doc was generated."
    )


class PhononBSDOSDoc(StructureMetadata, extra="allow"):  # type: ignore[call-arg]
    """Collection of all data produced by the phonon workflow."""

    structure: Optional[Structure] = Field(
        None, description="Structure of Materials Project."
    )

    phonon_bandstructure: Optional[PhononBandStructureSymmLine] = Field(
        None,
        description="Phonon band structure object.",
    )

    phonon_dos: Optional[PhononDos] = Field(
        None,
        description="Phonon density of states object.",
    )

    free_energies: Optional[list[float]] = Field(
        None,
        description="vibrational part of the free energies in J/mol per "
        "formula unit for temperatures in temperature_list",
    )

    heat_capacities: Optional[list[float]] = Field(
        None,
        description="heat capacities in J/K/mol per "
        "formula unit for temperatures in temperature_list",
    )

    internal_energies: Optional[list[float]] = Field(
        None,
        description="internal energies in J/mol per "
        "formula unit for temperatures in temperature_list",
    )
    entropies: Optional[list[float]] = Field(
        None,
        description="entropies in J/(K*mol) per formula unit"
        "for temperatures in temperature_list ",
    )

    temperatures: Optional[list[int]] = Field(
        None,
        description="temperatures at which the vibrational"
        " part of the free energies"
        " and other properties have been computed",
    )

    total_dft_energy: Optional[float] = Field(
        None, description="total DFT energy per formula unit in eV"
    )

    volume_per_formula_unit: Optional[float] = Field(
        None, description="volume per formula unit in Angstrom**3"
    )

    formula_units: Optional[int] = Field(None, description="Formula units per cell")

    has_imaginary_modes: Optional[bool] = Field(
        None, description="if true, structure has imaginary modes"
    )

    # needed, e.g. to compute Grueneisen parameter etc
    force_constants: Optional[ForceConstants] = Field(
        None, description="Force constants between every pair of atoms in the structure"
    )

    born: Optional[list[Matrix3D]] = Field(
        None,
        description="born charges as computed from phonopy. Only for symmetrically "
        "different atoms",
    )

    epsilon_static: Optional[Matrix3D] = Field(
        None, description="The high-frequency dielectric constant"
    )

    supercell_matrix: Optional[Matrix3D] = Field(
        None, description="matrix describing the supercell"
    )
    primitive_matrix: Optional[Matrix3D] = Field(
        None, description="matrix describing relationship to primitive cell"
    )

    code: Optional[str] = Field(
        None, description="String describing the code for the computation"
    )

    phonopy_settings: Optional[PhononComputationalSettings] = Field(
        None, description="Field including settings for Phonopy"
    )

    thermal_displacement_data: Optional[ThermalDisplacementData] = Field(
        None,
        description="Includes all data of the computation of the thermal displacements",
    )

    jobdirs: Optional[PhononJobDirs] = Field(
        None, description="Field including all relevant job directories"
    )

    uuids: Optional[PhononUUIDs] = Field(
        None, description="Field including all relevant uuids"
    )

    @classmethod
    def from_forces_born(
        cls,
        structure: Structure,
        supercell_matrix: np.array,
        displacement: float,
        sym_reduce: bool,
        symprec: float,
        use_symmetrized_structure: Union[str, None],
        kpath_scheme: str,
        code: str,
        displacement_data: dict[str, list],
        total_dft_energy: float,
        epsilon_static: Matrix3D = None,
        born: Matrix3D = None,
        **kwargs,
    ) -> Self:
        """Generate collection of phonon data.

        Parameters
        ----------
        structure: Structure object
        supercell_matrix: numpy array describing the supercell
        displacement: float
            size of displacement in angstrom
        sym_reduce: bool
            if True, phonopy will use symmetry
        symprec: float
            precision to determine kpaths,
            primitive cells and symmetry in phonopy and pymatgen
        use_symmetrized_structure: str
            primitive, conventional or None
        kpath_scheme: str
            kpath scheme to generate phonon band structure
        code: str
            which code was used for computation
        displacement_data:
            output of the displacement data
        total_dft_energy: float
            total energy in eV per cell
        epsilon_static: Matrix3D
            The high-frequency dielectric constant
        born: Matrix3D
            born charges
        **kwargs:
            additional arguments
        """
        additional_fields = kwargs.get("additional_fields", {})
        factor = get_factor(code)
        # This opens the opportunity to add support for other codes
        # that are supported by phonopy

        cell = get_phonopy_structure(structure)

        if use_symmetrized_structure == "primitive":
            primitive_matrix: Union[np.ndarray, str] = np.eye(3)
        else:
            primitive_matrix = "auto"

        # TARP: THIS IS BAD! Including for discussions sake
        if cell.magnetic_moments is not None and primitive_matrix == "auto":
            if np.any(cell.magnetic_moments != 0.0):
                raise ValueError(
                    "For materials with magnetic moments, "
                    "use_symmetrized_structure must be 'primitive'"
                )
            cell.magnetic_moments = None

        phonon = Phonopy(
            cell,
            supercell_matrix,
            primitive_matrix=primitive_matrix,
            factor=factor,
            symprec=symprec,
            is_symmetry=sym_reduce,
        )
        phonon.generate_displacements(distance=displacement)
        set_of_forces = [np.array(forces) for forces in displacement_data["forces"]]

        if born is not None and epsilon_static is not None:
            if len(structure) == len(born):
                borns, epsilon = symmetrize_borns_and_epsilon(
                    ucell=phonon.unitcell,
                    borns=np.array(born),
                    epsilon=np.array(epsilon_static),
                    symprec=symprec,
                    primitive_matrix=phonon.primitive_matrix,
                    supercell_matrix=phonon.supercell_matrix,
                    is_symmetry=kwargs.get("symmetrize_born", True),
                )
            else:
                raise ValueError(
                    "Number of born charges does not agree with number of atoms"
                )
            if code == "vasp" and not np.all(np.isclose(borns, 0.0)):
                phonon.nac_params = {
                    "born": borns,
                    "dielectric": epsilon,
                    "factor": 14.399652,
                }
            # Other codes could be added here
        else:
            borns = None
            epsilon = None

        # Produces all force constants
        phonon.produce_force_constants(forces=set_of_forces)

        filename_phonopy_yaml = kwargs.get("filename_phonopy_yaml", "phonopy.yaml")
        # if kwargs.get("filename_phonopy_yaml") is None:
        #    kwargs["filename_phonopy_yaml"] = "phonopy.yaml"

        # with phonopy.load("phonopy.yaml") the phonopy API can be used
        phonon.save(filename_phonopy_yaml)

        # get phonon band structure
        kpath_dict, kpath_concrete = PhononBSDOSDoc.get_kpath(
            structure=get_pmg_structure(phonon.primitive),
            kpath_scheme=kpath_scheme,
            symprec=symprec,
        )

        npoints_band = kwargs.get("npoints_band", 101)
        qpoints, connections = get_band_qpoints_and_path_connections(
            kpath_concrete, npoints=kwargs.get("npoints_band", 101)
        )

        # phonon band structures will always be computed
        filename_band_yaml = kwargs.get(
            "filename_band_yaml", "phonon_band_structure.yaml"
        )
        # filename_band_yaml = "phonon_band_structure.yaml"

        # TODO: potentially add kwargs to avoid computation of eigenvectors
        phonon.run_band_structure(
            qpoints,
            path_connections=connections,
            with_eigenvectors=kwargs.get("band_structure_eigenvectors", False),
            is_band_connection=kwargs.get("band_structure_eigenvectors", False),
        )
        phonon.write_yaml_band_structure(filename=filename_band_yaml)
        bs_symm_line = get_ph_bs_symm_line(
            filename_band_yaml, labels_dict=kpath_dict, has_nac=born is not None
        )
        new_plotter = PhononBSPlotter(bs=bs_symm_line)
        new_plotter.save_plot(
            filename=kwargs.get("filename_bs", "phonon_band_structure.pdf"),
            units=kwargs.get("units", "THz"),
        )

        # will determine if imaginary modes are present in the structure
        imaginary_modes = bs_symm_line.has_imaginary_freq(
            tol=kwargs.get("tol_imaginary_modes", 1e-5)
        )

        # gets data for visualization on website - yaml is also enough
        if kwargs.get("band_structure_eigenvectors"):
            bs_symm_line.write_phononwebsite("phonon_website.json")

        # get phonon density of states
        filename_dos_yaml = kwargs.get("filename_dos_yaml", "phonon_dos.yaml")
        # filename_dos_yaml = "phonon_dos.yaml"

        kpoint_density_dos = kwargs.get("kpoint_density_dos", 7_000)
        kpoint = Kpoints.automatic_density(
            structure=get_pmg_structure(phonon.primitive),
            kppa=kpoint_density_dos,
            force_gamma=True,
        )
        phonon.run_mesh(kpoint.kpts[0])
        phonon_dos_sigma = kwargs.get("phonon_dos_sigma")
        dos_use_tetrahedron_method = kwargs.get("dos_use_tetrahedron_method", True)
        phonon.run_total_dos(
            sigma=phonon_dos_sigma, use_tetrahedron_method=dos_use_tetrahedron_method
        )
        phonon.write_total_dos(filename=filename_dos_yaml)
        dos = get_ph_dos(filename_dos_yaml)
        new_plotter_dos = PhononDosPlotter()
        new_plotter_dos.add_dos(label="total", dos=dos)
        new_plotter_dos.save_plot(
            filename=kwargs.get("filename_dos", "phonon_dos.pdf"),
            units=kwargs.get("units", "THz"),
        )

        # compute vibrational part of free energies per formula unit
        temperature_range = np.arange(
            kwargs.get("tmin", 0), kwargs.get("tmax", 500), kwargs.get("tstep", 10)
        )

        free_energies = [
            dos.helmholtz_free_energy(
                temp=temp, structure=get_pmg_structure(phonon.primitive)
            )
            for temp in temperature_range
        ]

        entropies = [
            dos.entropy(temp=temp, structure=get_pmg_structure(phonon.primitive))
            for temp in temperature_range
        ]

        internal_energies = [
            dos.internal_energy(
                temp=temp, structure=get_pmg_structure(phonon.primitive)
            )
            for temp in temperature_range
        ]

        heat_capacities = [
            dos.cv(temp=temp, structure=get_pmg_structure(phonon.primitive))
            for temp in temperature_range
        ]

        # will compute thermal displacement matrices
        # for the primitive cell (phonon.primitive!)
        # only this is available in phonopy
        if kwargs.get("create_thermal_displacements"):
            phonon.run_mesh(
                kpoint.kpts[0], with_eigenvectors=True, is_mesh_symmetry=False
            )
            freq_min_thermal_displacements = kwargs.get(
                "freq_min_thermal_displacements", 0.0
            )
            phonon.run_thermal_displacement_matrices(
                t_min=kwargs.get("tmin_thermal_displacements", 0),
                t_max=kwargs.get("tmax_thermal_displacements", 500),
                t_step=kwargs.get("tstep_thermal_displacements", 100),
                freq_min=freq_min_thermal_displacements,
            )

            temperature_range_thermal_displacements = np.arange(
                kwargs.get("tmin_thermal_displacements", 0),
                kwargs.get("tmax_thermal_displacements", 500),
                kwargs.get("tstep_thermal_displacements", 100),
            )
            for idx, temp in enumerate(temperature_range_thermal_displacements):
                phonon.thermal_displacement_matrices.write_cif(
                    phonon.primitive, idx, filename=f"tdispmat_{temp}K.cif"
                )
            _disp_mat = phonon._thermal_displacement_matrices  # noqa: SLF001
            tdisp_mat = _disp_mat.thermal_displacement_matrices.tolist()

            tdisp_mat_cif = _disp_mat.thermal_displacement_matrices_cif.tolist()

        else:
            tdisp_mat = None
            tdisp_mat_cif = None

        formula_units = (
            structure.composition.num_atoms
            / structure.composition.reduced_composition.num_atoms
        )

        total_dft_energy_per_formula_unit = (
            total_dft_energy / formula_units if total_dft_energy is not None else None
        )

        volume_per_formula_unit = structure.volume / formula_units

        doc = cls.from_structure(
            structure=structure,
            meta_structure=structure,
            phonon_bandstructure=bs_symm_line,
            phonon_dos=dos,
            free_energies=free_energies,
            internal_energies=internal_energies,
            heat_capacities=heat_capacities,
            entropies=entropies,
            temperatures=temperature_range.tolist(),
            total_dft_energy=total_dft_energy_per_formula_unit,
            volume_per_formula_unit=volume_per_formula_unit,
            formula_units=formula_units,
            has_imaginary_modes=imaginary_modes,
            force_constants={"force_constants": phonon.force_constants.tolist()}
            if kwargs["store_force_constants"]
            else None,
            born=borns.tolist() if borns is not None else None,
            epsilon_static=epsilon.tolist() if epsilon is not None else None,
            supercell_matrix=phonon.supercell_matrix.tolist(),
            primitive_matrix=phonon.primitive_matrix.tolist(),
            code=code,
            thermal_displacement_data={
                "temperatures_thermal_displacements": temperature_range_thermal_displacements.tolist(),  # noqa: E501
                "thermal_displacement_matrix_cif": tdisp_mat_cif,
                "thermal_displacement_matrix": tdisp_mat,
                "freq_min_thermal_displacements": freq_min_thermal_displacements,
            }
            if kwargs.get("create_thermal_displacements")
            else None,
            jobdirs={
                "displacements_job_dirs": displacement_data["dirs"],
                "static_run_job_dir": kwargs["static_run_job_dir"],
                "born_run_job_dir": kwargs["born_run_job_dir"],
                "optimization_run_job_dir": kwargs["optimization_run_job_dir"],
                "taskdoc_run_job_dir": str(Path.cwd()),
            },
            uuids={
                "displacements_uuids": displacement_data["uuids"],
                "born_run_uuid": kwargs["born_run_uuid"],
                "optimization_run_uuid": kwargs["optimization_run_uuid"],
                "static_run_uuid": kwargs["static_run_uuid"],
            },
            phonopy_settings={
                "npoints_band": npoints_band,
                "kpath_scheme": kpath_scheme,
                "kpoint_density_dos": kpoint_density_dos,
            },
        )

        return doc.model_copy(update=additional_fields)

    @staticmethod
    def get_kpath(
        structure: Structure, kpath_scheme: str, symprec: float, **kpath_kwargs
    ) -> tuple:
        """Get high-symmetry points in k-space in phonopy format.

        Parameters
        ----------
        structure: Structure Object
        kpath_scheme: str
            string describing kpath
        symprec: float
            precision for symmetry determination
        **kpath_kwargs:
            additional parameters that can be passed to this method as a dict
        """
        if kpath_scheme in ("setyawan_curtarolo", "latimer_munro", "hinuma"):
            high_symm_kpath = HighSymmKpath(
                structure, path_type=kpath_scheme, symprec=symprec, **kpath_kwargs
            )
            kpath = high_symm_kpath.kpath
        elif kpath_scheme == "seekpath":
            high_symm_kpath = KPathSeek(structure, symprec=symprec, **kpath_kwargs)
            kpath = high_symm_kpath._kpath  # noqa: SLF001
        else:
            raise ValueError(f"Unexpected {kpath_scheme=}")

        path = copy.deepcopy(kpath["path"])

        for set_idx, label_set in enumerate(kpath["path"]):
            for lbl_idx, label in enumerate(label_set):
                path[set_idx][lbl_idx] = kpath["kpoints"][label]
        return kpath["kpoints"], path
