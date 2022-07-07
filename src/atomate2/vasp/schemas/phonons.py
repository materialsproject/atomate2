from __future__ import annotations

import copy
import logging
from typing import List, Union

import numpy as np
from phonopy import Phonopy, load
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


class PhononBSDOSDoc(BaseModel):
    """
    Phonon band structures and density of states data.
    """

    structure: Structure = Field(
        None,
        description="Structure of Materials Project.",
    )

    ph_bs: PhononBandStructureSymmLine = Field(
        None,
        description="Phonon band structure object.",
    )

    ph_dos: PhononDos = Field(
        None,
        description="Phonon density of states object.",
    )

    # TODO: improve this data structure
    free_energy_list: List[float] = Field(
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

    total_energy: float = Field("total DFT energy per formula unit in eV")

    has_imaginary_modes: bool = Field(
        None, description="if true, structure has imaginary modes"
    )
    # copied from electron-phonon workflow

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
    dft_phononwebsite: dict = Field(
        "dict of the band structure information that can be visualized with phonon"
        "website (http://henriquemiranda.github.io/phononwebsite)"
    )

    supercell_matrix: Matrix3D = Field("matrix describing the supercell")
    primitive_matrix: Matrix3D = Field(
        "matrix describing relationship to primitive cell"
    )

    code: str = Field("String describing the code for the computation")
    thermal_displacement_matrix_cif: List[List[Matrix3D]] = Field(
        None, description="field including thermal displacement matrices in cif format"
    )
    thermal_displacement_matrix: List[List[Matrix3D]] = Field(
        None,
        description="field including thermal displacement matrices in cartesian coordinate system",
    )
    # could be optional and implemented at a later stage?
    npoints_band: int = Field("number of points for band structure computation")
    kpath_scheme: str = Field("indicates the kpath scheme")
    kpoint_density_dos: int = Field(
        "number of points for computation of free energies and densities of states",
    )
    freq_min_thermal_displacements: float = Field(
        "cutoff frequency in THz to avoid numerical issues in the "
        "computation of the thermal displacement parameters"
    )
    temperatures_thermal_displacements: List[int] = Field(
        None,
        description="temperatures at which the thermal displacement matrices"
        "have been computed",
    )

    displacements_uuids: List[str] = Field(
        None, description="The uuids of the displacement jobs."
    )
    displacements_job_dirs: List[str] = Field(
        None, description="The directories where the displacement jobs were run."
    )
    static_run_job_dir: str = Field(
        None, description="Directory where static run was performed."
    )
    static_run_uuid: str = Field(None, description="static run uuid")
    born_run_job_dir: str = Field(
        None, description="Directory where born run was performed."
    )
    born_run_uuid: str = Field(None, description="born run uuid")
    optimization_run_job_dir: str = Field(
        None, description="Directory where optimization run was performed."
    )
    optimization_run_uuid: str = Field(None, description="optimization run uuid")

    @classmethod
    def from_forces_born(
        cls,
        structure: Structure,
        supercell_matrix: np.array,
        displacement: float,
        sym_reduce: bool,
        symprec: float,
        use_standard_primitive: bool,
        kpath_scheme: str,
        code: str,
        displacement_data: dict[str, list],
        total_energy: float,
        epsilon_static: Matrix3D = None,
        born: Matrix3D = None,
        full_born: bool = True,
        **kwargs,
    ):
        """This will initialize
        the document starting from forces, and born information"""
        # Could we do this in a better way?

        # have to regenerate this object as I cannot make it a job output
        # TODO: other way?
        if code == "vasp":
            factor = VaspToTHz
        # TODO: add other codes?

        cell = get_phonopy_structure(structure)
        # TODO: check why this does not work!
        if use_standard_primitive and kpath_scheme != "seekpath":
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
            if full_born:

                # TODO: if this is a good way when user provide data
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
        else:
            borns = None
            epsilon = None
        # This the next part should be done as a part of the Scheme

        phonon.produce_force_constants(forces=set_of_forces)

        # still not working
        phonon.save("phonopy.yaml")
        phonon = load("phonopy.yaml")
        # get phonon band structure

        kpath_dict, kpath_concrete = cls.get_kpath(
            get_pmg_structure(phonon.primitive), kpath_scheme
        )

        # Okay or do I need to implement
        # a fallback in case "npoints" band is not implemented?
        qpoints, connections = get_band_qpoints_and_path_connections(
            kpath_concrete, npoints=kwargs["npoints_band"]
        )

        # add option to disable phonon bandstructure computation?
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
            img_format=kwargs["img_format"],
            units=kwargs["units"],
        )
        # add a free energy document?
        imaginary_modes = bs_symm_line.has_imaginary_freq(
            tol=kwargs["tol_imaginary_modes"]
        )

        # gets data for visualization on website - yaml is also enough
        phononwebsite_dict = bs_symm_line.as_phononwebsite()
        # TODO: should we include any other animation output?

        # get phonon density of states
        filename_dos_yaml = "phonon_dos.yaml"
        kpoint = Kpoints.automatic_density(
            structure=structure, kppa=kwargs["kpoint_density_dos"], force_gamma=True
        )
        phonon.run_mesh(kpoint.kpts[0])
        phonon.run_total_dos()
        phonon.write_total_dos(filename=filename_dos_yaml)
        dos = get_ph_dos(filename_dos_yaml)
        new_plotter_dos = PhononDosPlotter()
        new_plotter_dos.add_dos(label="total", dos=dos)
        new_plotter_dos.save_plot(
            filename="phonon_dos.eps",
            img_format=kwargs["img_format"],
            units=kwargs["units"],
        )

        # add tmin tmax tstep
        temperature_range = np.arange(kwargs["tmin"], kwargs["tmax"], kwargs["tstep"])
        free_energy = [
            dos.helmholtz_free_energy(structure=structure, t=temperature)
            for temperature in temperature_range
        ]

        # transfer the force constants to compute Grüneisen parameters?
        formula_units = (
            structure.composition.num_atoms
            / structure.composition.reduced_composition.num_atoms
        )
        if kwargs["create_thermal_displacements"]:
            phonon.run_mesh(
                kpoint.kpts[0], with_eigenvectors=True, is_mesh_symmetry=False
            )
            phonon.run_thermal_displacement_matrices(
                t_min=kwargs["tmin_thermal_displacements"],
                t_max=kwargs["tmax_thermal_displacements"],
                t_step=kwargs["tstep_thermal_displacements"],
                freq_min=kwargs["freq_min_thermal_displacements"],
            )

            # will compute thermal displacement matrices
            temperature_range_thermal_displacements = range(
                kwargs["tmin_thermal_displacements"],
                kwargs["tmax_thermal_displacements"],
                kwargs["tstep_thermal_displacements"],
            )
            for i, temp in enumerate(temperature_range_thermal_displacements):
                phonon.thermal_displacement_matrices.write_cif(
                    get_phonopy_structure(structure),
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

        return cls(
            structure=structure,
            ph_bs=bs_symm_line,
            ph_dos=dos,
            free_energy_list=free_energy,
            temperatures=temperature_range.tolist(),
            total_energy=total_energy / formula_units,
            has_imaginary_modes=imaginary_modes,
            force_constants=phonon.force_constants.tolist()
            if kwargs["store_force_constants"]
            else None,
            born=borns.tolist() if borns is not None else None,
            epsilon_static=epsilon.tolist() if epsilon is not None else None,
            dft_phononwebsite=phononwebsite_dict,
            supercell_matrix=phonon.supercell_matrix.tolist(),
            primitive_matrix=phonon.primitive_matrix.tolist(),
            code="vasp",
            thermal_displacement_matrix_cif=tdisp_mat_cif,
            thermal_displacement_matrix=tdisp_mat,
            npoints_band=kwargs["npoints_band"],
            kpath_scheme=kpath_scheme,
            kpoint_density_dos=kwargs["kpoint_density_dos"],
            freq_min_thermal_displacements=kwargs["freq_min"],
            temperatures_thermal_displacements=temperature_range_thermal_displacements,
            displacements_uuids=displacement_data["uuids"],
            displacements_job_dirs=displacement_data["dirs"],
            static_run_job_dir=kwargs["static_run_job_dir"],
            static_run_uuid=kwargs["static_run_uuid"],
            born_run_job_dir=kwargs["born_run_job_dir"],
            born_run_uuid=kwargs["born_run_uuid"],
            optimization_run_job_dir=kwargs["optimization_run_job_dir"],
            optimization_run_uuid=kwargs["optimization_run_uuid"],
        )

    @staticmethod
    def get_kpath(structure: Structure, kpath_scheme: str, **kpath_kwargs):
        """
        get high-symmetry points in k-space
        Args:
            structure: Structure Object
        Returns:
        """

        if kpath_scheme in [
            "setyawan_curtarolo",
            "hinuma",  # is the same as seekpath?!
            "latimer_munro",
            "all_pymatgen",
        ]:
            if kpath_scheme == "all_pymatgen":
                kpath_scheme = "all"
            highsymmkpath = HighSymmKpath(
                structure, path_type=kpath_scheme, **kpath_kwargs
            )
            kpath = highsymmkpath.kpath
        elif kpath_scheme == "seekpath":
            highsymmkpath = KPathSeek(structure, **kpath_kwargs)
            kpath = highsymmkpath._kpath

        path = copy.deepcopy(kpath["path"])

        for ilabelset, labelset in enumerate(kpath["path"]):
            for ilabel, label in enumerate(labelset):
                path[ilabelset][ilabel] = kpath["kpoints"][label]
        return kpath["kpoints"], path
