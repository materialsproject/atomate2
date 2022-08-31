from __future__ import annotations

import copy
import logging
from typing import List, Optional, Union

import numpy as np
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.symmetry import elaborate_borns_and_epsilon
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

from atomate2.common.schemas.math import Matrix3D

logger = logging.getLogger(__name__)

__all__ = ["PhononBSDOSDoc"]


class PhononComputationalSettings(BaseModel):
    """
    Collection to store computational settings for the phonon computation
    """

    # could be optional and implemented at a later stage?
    npoints_band: int = Field("number of points for band structure computation")
    kpath_scheme: str = Field("indicates the kpath scheme")
    kpoint_density_dos: int = Field(
        "number of points for computation of free energies and densities of states",
    )


class ThermalDisplacementData(BaseModel):
    """
    Collection to store information on the thermal displacement matrices
    """

    # TODO: check if we could add a connection
    #  to a pymatgen class for thermal displacement matrices
    freq_min_thermal_displacements: float = Field(
        "cutoff frequency in THz to avoid numerical issues in the "
        "computation of the thermal displacement parameters"
    )
    thermal_displacement_matrix_cif: List[List[Matrix3D]] = Field(
        None, description="field including thermal displacement matrices in cif format"
    )
    thermal_displacement_matrix: List[List[Matrix3D]] = Field(
        None,
        description="field including thermal displacement matrices in cartesian coordinate system",
    )
    temperatures_thermal_displacements: List[int] = Field(
        None,
        description="temperatures at which the thermal displacement matrices"
        "have been computed",
    )


class PhononUUIDs(BaseModel):
    """
    Collection to save all uuids connected to the phonon run
    """

    optimization_run_uuid: str = Field(None, description="optimization run uuid")
    displacements_uuids: List[str] = Field(
        None, description="The uuids of the displacement jobs."
    )
    static_run_uuid: str = Field(None, description="static run uuid")
    born_run_uuid: str = Field(None, description="born run uuid")


class PhononJobDirs(BaseModel):
    """
    Collection to save all job directories relevant for the phonon run
    """

    displacements_job_dirs: List[str] = Field(
        None, description="The directories where the displacement jobs were run."
    )
    static_run_job_dir: str = Field(
        None, description="Directory where static run was performed."
    )
    born_run_job_dir: str = Field(
        None, description="Directory where born run was performed."
    )
    optimization_run_job_dir: str = Field(
        None, description="Directory where optimization run was performed."
    )


class PhononBSDOSDoc(BaseModel):
    """
    Collection of all data produced by the phonon worklow
    """

    structure: Structure = Field(
        None,
        description="Structure of Materials Project.",
    )

    phonon_bandstructure: PhononBandStructureSymmLine = Field(
        None,
        description="Phonon band structure object.",
    )

    phonon_dos: PhononDos = Field(
        None,
        description="Phonon density of states object.",
    )

    free_energies: List[float] = Field(
        None,
        description="vibrational part of the free energies in kJ/mol per "
        "formula unit for temperatures in temperature_list",
    )

    temperatures: List[int] = Field(
        None,
        description="temperatures at which the vibrational"
        " part of the free energies"
        " and other properties have been computed",
    )

    total_dft_energy: Optional[float] = Field("total DFT energy per formula unit in eV")

    has_imaginary_modes: bool = Field(
        None, description="if true, structure has imaginary modes"
    )

    # needed, e.g. to compute Grueneisen parameter etc
    force_constants: List[List[Matrix3D]] = Field(
        None, description="Force constants between every pair of atoms in the structure"
    )

    born: List[Matrix3D] = Field(
        None,
        description="born charges as computed from phonopy. Only for symmetrically different atoms",
    )

    epsilon_static: Matrix3D = Field(
        None, description="The high-frequency dielectric constant"
    )

    supercell_matrix: Matrix3D = Field("matrix describing the supercell")
    primitive_matrix: Matrix3D = Field(
        "matrix describing relationship to primitive cell"
    )

    code: str = Field("String describing the code for the computation")

    phonopy_settings: PhononComputationalSettings = Field(
        "Field including settings for Phonopy"
    )

    thermal_displacement_data: Optional[ThermalDisplacementData] = Field(
        "Includes all data of the " "computation of the thermal displacements"
    )

    jobdirs: PhononJobDirs = Field("Field including all relevant job directories")

    uuids: PhononUUIDs = Field("Field including all relevant uuids")

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
    ):
        """
        generates  a collection of phonon data

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
            primitive, conentional or None
        kpath_scheme: str
            kpath scheme to generate phonon band structure
        code: str
            which code was used for computation
        displacement_data:
            output of the VASP displacement data
        total_dft_energy: float
            total energy in eV for whole cell
        epsilon_static: Matrix3D
            The high-frequency dielectric constant
        born: Matrix3D
            born charges
        **kwargs:
            additional arguments


        """

        if code == "vasp":
            factor = VaspToTHz
        # This opens the opportunity to add support for other codes
        # that are supported by phonopy

        cell = get_phonopy_structure(structure)

        if use_symmetrized_structure == "primitive" and kpath_scheme != "seekpath":
            primitive_matrix: Union[List[List[float]], str] = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        else:
            primitive_matrix = "auto"
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
                borns, epsilon, atom_indices = elaborate_borns_and_epsilon(
                    ucell=get_phonopy_structure(structure),
                    borns=np.array(born),
                    epsilon=np.array(epsilon_static),
                    symprec=symprec,
                    primitive_matrix=phonon.primitive_matrix,
                    supercell_matrix=phonon.supercell_matrix,
                )
            else:
                borns = born
                epsilon = epsilon_static
            if code == "vasp":
                if not np.all(np.isclose(borns, 0.0)):
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

        # with phonon.load("phonopy.yaml") the phonopy API can be used
        phonon.save("phonopy.yaml")

        # get phonon band structure
        kpath_dict, kpath_concrete = cls.get_kpath(
            structure=get_pmg_structure(phonon.primitive),
            kpath_scheme=kpath_scheme,
            symprec=symprec,
        )

        npoints_band = kwargs.get("npoints_band", 101)
        qpoints, connections = get_band_qpoints_and_path_connections(
            kpath_concrete, npoints=kwargs.get("npoints_band", 101)
        )

        # phonon band structures will always be cmouted
        filename_band_yaml = "phonon_band_structure.yaml"
        phonon.run_band_structure(
            qpoints, path_connections=connections, with_eigenvectors=True
        )
        phonon.write_yaml_band_structure(filename=filename_band_yaml)
        bs_symm_line = get_ph_bs_symm_line(
            filename_band_yaml, labels_dict=kpath_dict, has_nac=born is not None
        )
        new_plotter = PhononBSPlotter(bs=bs_symm_line)

        new_plotter.save_plot(
            "phonon_band_structure.eps",
            img_format=kwargs.get("img_format", "eps"),
            units=kwargs.get("units", "THz"),
        )

        # will determine if imaginary modes are present in the structure
        imaginary_modes = bs_symm_line.has_imaginary_freq(
            tol=kwargs.get("tol_imaginary_modes", 1e-5)
        )

        # gets data for visualization on website - yaml is also enough
        bs_symm_line.write_phononwebsite("phonon_website.json")

        # get phonon density of states
        filename_dos_yaml = "phonon_dos.yaml"

        kpoint_density_dos = kwargs.get("kpoint_density_dos", 7000)
        kpoint = Kpoints.automatic_density(
            structure=structure,
            kppa=kpoint_density_dos,
            force_gamma=True,
        )
        phonon.run_mesh(kpoint.kpts[0])
        phonon.run_total_dos()
        phonon.write_total_dos(filename=filename_dos_yaml)
        dos = get_ph_dos(filename_dos_yaml)
        new_plotter_dos = PhononDosPlotter()
        new_plotter_dos.add_dos(label="total", dos=dos)
        new_plotter_dos.save_plot(
            filename="phonon_dos.eps",
            img_format=kwargs.get("img_format", "eps"),
            units=kwargs.get("units", "THz"),
        )

        # compute vibrational part of free energies per formula unit
        temperature_range = np.arange(
            kwargs.get("tmin", 0), kwargs.get("tmax", 500), kwargs.get("tstep", 100)
        )
        free_energies = [
            dos.helmholtz_free_energy(structure=structure, t=temperature)
            for temperature in temperature_range
        ]

        # compute formula units to divide total dft energy by formula unit
        formula_units = (
            structure.composition.num_atoms
            / structure.composition.reduced_composition.num_atoms
        )
        # will compute thermal displacement matrices
        # for the primitive cell (phonon.primitive!)
        # only this is available in phonopy
        if kwargs["create_thermal_displacements"]:

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
            for i, temp in enumerate(temperature_range_thermal_displacements):
                phonon.thermal_displacement_matrices.write_cif(
                    phonon.primitive,
                    i,
                    filename="tdispmat_" + str(temp) + "K.cif",
                )
            tdisp_mat = (
                phonon._thermal_displacement_matrices.thermal_displacement_matrices.tolist()
            )

            tdisp_mat_cif = (
                phonon._thermal_displacement_matrices.thermal_displacement_matrices_cif.tolist()
            )

        else:
            tdisp_mat = None
            tdisp_mat_cif = None

        if total_dft_energy:
            total_dft_energy_per_formula_unit = total_dft_energy / formula_units
        else:
            total_dft_energy_per_formula_unit = None
        return cls(
            structure=structure,
            phonon_bandstructure=bs_symm_line,
            phonon_dos=dos,
            free_energies=free_energies,
            temperatures=temperature_range.tolist(),
            total_dft_energy=total_dft_energy_per_formula_unit,
            has_imaginary_modes=imaginary_modes,
            force_constants=phonon.force_constants.tolist()
            if kwargs["store_force_constants"]
            else None,
            born=borns.tolist() if borns is not None else None,
            epsilon_static=epsilon.tolist() if epsilon is not None else None,
            supercell_matrix=phonon.supercell_matrix.tolist(),
            primitive_matrix=phonon.primitive_matrix.tolist(),
            code="vasp",
            thermal_displacement_data={
                "temperatures_thermal_displacements": temperature_range_thermal_displacements.tolist(),
                "thermal_displacement_matrix_cif": tdisp_mat_cif,
                "thermal_displacement_matrix": tdisp_mat,
                "freq_min_thermal_displacements": freq_min_thermal_displacements,
            }
            if kwargs["create_thermal_displacements"]
            else None,
            jobdirs={
                "displacements_job_dirs": displacement_data["dirs"],
                "static_run_job_dir": kwargs["static_run_job_dir"],
                "born_run_job_dir": kwargs["born_run_job_dir"],
                "optimization_run_job_dir": kwargs["optimization_run_job_dir"],
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

    @staticmethod
    def get_kpath(
        structure: Structure, kpath_scheme: str, symprec: float, **kpath_kwargs
    ):
        """
        get high-symmetry points in k-space in phonopy format

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

        if kpath_scheme in [
            "setyawan_curtarolo",
            "latimer_munro",
            "all",
            "hinuma",
        ]:
            highsymmkpath = HighSymmKpath(
                structure, path_type=kpath_scheme, symprec=symprec, **kpath_kwargs
            )
            kpath = highsymmkpath.kpath
        elif kpath_scheme == "seekpath":
            highsymmkpath = KPathSeek(structure, symprec=symprec, **kpath_kwargs)
            kpath = highsymmkpath._kpath

        path = copy.deepcopy(kpath["path"])

        for ilabelset, labelset in enumerate(kpath["path"]):
            for ilabel, label in enumerate(labelset):
                path[ilabelset][ilabel] = kpath["kpoints"][label]
        return kpath["kpoints"], path
