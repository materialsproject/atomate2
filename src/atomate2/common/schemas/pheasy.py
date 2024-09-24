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
import pickle

# import lib by jiongzhi zheng 
from ase.io import read
from hiphive import ClusterSpace, ForceConstantPotential, ForceConstants, enforce_rotational_sum_rules
from hiphive.cutoffs import estimate_maximum_cutoff
from hiphive.utilities import extract_parameters
import subprocess
from phonopy.file_IO import write_FORCE_CONSTANTS, parse_FORCE_CONSTANTS
from phonopy.interface.vasp import write_vasp
from phonopy.interface.vasp import read_vasp


from atomate2.aims.utils.units import omegaToTHz

# import some classmethod directly from phonons
from atomate2.common.schemas.phonons import get_factor
from atomate2.common.schemas.phonons import ThermalDisplacementData
from atomate2.common.schemas.phonons import PhononComputationalSettings
from atomate2.common.schemas.phonons import PhononUUIDs
from atomate2.common.schemas.phonons import PhononJobDirs



logger = logging.getLogger(__name__)

class Forceconstants(MSONable):
    """A force constants class."""

    def __init__(self, force_constants: list[list[Matrix3D]]) -> None:
        self.force_constants = force_constants

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

    total_dft_energy: Optional[float] = Field("total DFT energy per formula unit in eV")

    has_imaginary_modes: Optional[bool] = Field(
        None, description="if true, structure has imaginary modes"
    )

    # needed, e.g. to compute Grueneisen parameter etc
    force_constants: Optional[Forceconstants] = Field(
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

    supercell_matrix: Matrix3D = Field("matrix describing the supercell")
    primitive_matrix: Matrix3D = Field(
        "matrix describing relationship to primitive cell"
    )

    code: str = Field("String describing the code for the computation")

    phonopy_settings: PhononComputationalSettings = Field(
        "Field including settings for Phonopy"
    )

    thermal_displacement_data: Optional[ThermalDisplacementData] = Field(
        "Includes all data of the computation of the thermal displacements"
    )

    jobdirs: Optional[PhononJobDirs] = Field(
        "Field including all relevant job directories"
    )

    uuids: Optional[PhononUUIDs] = Field("Field including all relevant uuids")


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
        mp_id: str,
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

        # Start from here, we use the pheasy code to extract the force constants.
        supercell = phonon.get_supercell()
        write_vasp("POSCAR", cell)
        write_vasp("SPOSCAR", supercell)

        # get the force-displacement dataset from previous calculations
        set_of_forces = [np.array(forces) for forces in displacement_data["forces"]]
        set_of_forces_a_o = np.array(set_of_forces)

        # deduct the residual forces on the equilibrium structure
        set_of_forces_a_t = set_of_forces_a_o - set_of_forces_a_o[-1, :, :]
        set_of_forces_a = set_of_forces_a_t[:-1, :, :]
        set_of_disps = [np.array(disps.cart_coords) for disps in displacement_data["displaced_structures"]]

        # get the displacement dataset
        set_of_disps_m_o = np.round((set_of_disps - supercell.get_positions()), 
                                    decimals=16).astype('double')
        set_of_disps_m = set_of_disps_m_o[:-1, :, :]

        # get the number of displacements
        n_shape = set_of_disps_m.shape[0]

        # save the displacement and force matrix in the current directory
        # for the future use by pheasy code 
        with open("disp_matrix.pkl","wb") as file: 
             pickle.dump(set_of_disps_m,file)
        with open("force_matrix.pkl","wb") as file:
             pickle.dump(set_of_forces_a,file)

        # TODO: extract the anharmonic force constants
        Calc_anharmonic_FCs = False
        if Calc_anharmonic_FCs:
            from alm import ALM
            supercell_z = phonon.supercell
            lattice = supercell_z.cell
            positions = supercell_z.scaled_positions
            numbers = supercell_z.numbers
            natom = len(numbers)
            with ALM(lattice, positions, numbers) as alm:
                alm.define(1)
                alm.suggest()
                n_fp = alm._get_number_of_irred_fc_elements(1)

            num = int(np.ceil(n_fp / (3.0 * natom)))
            num_round = int(np.round((n_fp / (3.0 * natom))))

            if num > num_round:
                num_d = num
                displacement_t = 0.01
                phonon.generate_displacements(displacement_t)
                num_disp_t = len(phonon.displacements)
                int_num = int(num_disp_t / num_d)
                if int_num > 3:
                    num_d = int(np.ceil(int_num / 3.0))
                else:
                    num_d = int(np.ceil(int_num / 3.0) + 1)
            else:
                num_d = num
                displacement_t = 0.01
                phonon.generate_displacements(displacement_t)
                num_disp_t = len(phonon.displacements)
                int_num = int(num_disp_t / num_d)
                if int_num >= 3:
                    num_d = int(np.ceil(int_num / 3.0))
                else:
                    num_d = int(num + 1)

            displacement_f = 0.01
            phonon.generate_displacements(distance=displacement_f)
            disps = phonon.displacements

            f_disp_n = int(len(disps))
            if f_disp_n > 2:
                num_har = num_d
            else:
                num_har = f_disp_n
        else:
            num_har = n_shape


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

        prim = read('POSCAR')
        supercell = read('SPOSCAR')

        # Create the clusters and orbitals for second order force constants
        pheasy_cmd_1 = 'pheasy --dim "{0}" "{1}" "{2}" -s -w 2 --symprec "{3}" --nbody 2'.format(
            int(supercell_matrix[0][0]),
            int(supercell_matrix[1][1]),
            int(supercell_matrix[2][2]),
            float(symprec))
        
        # Create the null space to further reduce the free parameters for 
        # specific force constants and make them physically correct.
        pheasy_cmd_2 = 'pheasy --dim "{0}" "{1}" "{2}" -c --symprec "{3}" -w 2'.format(
            int(supercell_matrix[0][0]),
            int(supercell_matrix[1][1]),
            int(supercell_matrix[2][2]),
            float(symprec))
        
        # Generate the Compressive Sensing matrix,i.e., displacment matrix
        # for the input of machine leaning method.i.e., LASSO,
        pheasy_cmd_3 = 'pheasy --dim "{0}" "{1}" "{2}" -w 2 -d --symprec "{3}" \
            --ndata "{4}" --disp_file'.format(
            int(supercell_matrix[0][0]),
            int(supercell_matrix[1][1]),
            int(supercell_matrix[2][2]),
            float(symprec),
            int(num_har))


        # Here we set a criteria to determine which method to use to generate the force constants.
        # If the number of displacements is larger than 3, 
        # we will use the LASSO method to generate the force constants.
        # Otherwise, we will use the least-squred method to generate the force constants.
        phonon.generate_displacements(distance=displacement)
        disps = phonon.displacements
        num_judge = len(disps)

        if num_judge > 3:
           # Calculate the force constants using the LASSO method
           pheasy_cmd_4 = 'pheasy --dim "{0}" "{1}" "{2}" -f --full_ifc -w 2 --symprec "{3}" \
            -l LASSO --std --rasr BHH --ndata "{4}"'.format(
                int(supercell_matrix[0][0]),
                int(supercell_matrix[1][1]),
                int(supercell_matrix[2][2]),
                float(symprec),
                int(num_har))
        else:
            # Calculate the force constants using the least-squred method
            pheasy_cmd_4 = 'pheasy --dim "{0}" "{1}" "{2}" -f --full_ifc -w 2 --symprec "{3}" \
                  --rasr BHH --ndata "{4}"'.format(
                int(supercell_matrix[0][0]), 
                int(supercell_matrix[1][1]), 
                int(supercell_matrix[2][2]),
                float(symprec), 
                int(num_har))
            
        logger.info("Start running pheasy in cluster")

        subprocess.call(pheasy_cmd_1, shell=True)
        subprocess.call(pheasy_cmd_2, shell=True)
        subprocess.call(pheasy_cmd_3, shell=True)
        subprocess.call(pheasy_cmd_4, shell=True)

        force_constants = parse_FORCE_CONSTANTS(filename="FORCE_CONSTANTS")
        phonon.force_constants = force_constants
        phonon.symmetrize_force_constants()

        # with phonopy.load("phonopy.yaml") the phonopy API can be used
        phonon.save("phonopy.yaml")

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
        filename_band_yaml = "phonon_band_structure.yaml"

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

        # jiongzhi zheng
        if imaginary_modes:
            # Define a cluster space using the largest cutoff you can
            max_cutoff = estimate_maximum_cutoff(supercell) - 0.01
            cutoffs = [max_cutoff]  # only second order needed
            cs = ClusterSpace(prim, cutoffs)

            # import the phonopy force constants using the correct supercell also provided by phonopy
            fcs = ForceConstants.read_phonopy(supercell, 'FORCE_CONSTANTS')

            # Find the parameters that best fits the force constants given you cluster space
            parameters = extract_parameters(fcs, cs)

            # Enforce the rotational sum rules
            parameters_rot = enforce_rotational_sum_rules(cs, 
                                                          parameters, 
                                                          ['Huang','Born-Huang'], 
                                                          alpha=1e-6)
            
            # use the new parameters to make a fcp and then create the force constants and write to a phonopy file
            fcp = ForceConstantPotential(cs, parameters_rot)
            fcs = fcp.get_force_constants(supercell)
            fcs.write_to_phonopy('FORCE_CONSTANTS_new', format='text')

            force_constants = parse_FORCE_CONSTANTS(filename="FORCE_CONSTANTS_new")
            phonon.force_constants = force_constants
            phonon.symmetrize_force_constants()

            phonon.run_band_structure(qpoints, path_connections=connections, with_eigenvectors=True)
            phonon.write_yaml_band_structure(filename=filename_band_yaml)
            bs_symm_line = get_ph_bs_symm_line(filename_band_yaml, 
                                               labels_dict=kpath_dict, 
                                               has_nac=born is not None)

            new_plotter = PhononBSPlotter(bs=bs_symm_line)

            new_plotter.save_plot(filename=kwargs.get("filename_bs", "phonon_band_structure.pdf"),
                                  units=kwargs.get("units", "THz"))

            # new_plotter.save_plot("phonon_band_structure.eps",img_format=kwargs.get("img_format", "eps"),units=kwargs.get("units", "THz"),)

            imaginary_modes_hiphive = bs_symm_line.has_imaginary_freq(
            tol=kwargs.get("tol_imaginary_modes", 1e-5)
        )

        else:
           imaginary_modes_hiphive = False
           imaginary_modes = False

        if imaginary_modes_hiphive:
            pheasy_cmd_11 = 'pheasy --dim "{0}" "{1}" "{2}" -s -w 2 --c2 10.0 --symprec "{3}" --nbody 2'.format(
                int(supercell_matrix[0][0]), 
                int(supercell_matrix[1][1]), 
                int(supercell_matrix[2][2]),
                float(symprec))
            pheasy_cmd_12 = 'pheasy --dim "{0}" "{1}" "{2}" -c --symprec "{3}" --c2 10.0 -w 2'.format(
                int(supercell_matrix[0][0]), 
                int(supercell_matrix[1][1]), 
                int(supercell_matrix[2][2]),
                float(symprec))
            pheasy_cmd_13 = 'pheasy --dim "{0}" "{1}" "{2}" -w 2 -d --symprec "{3}" --c2 10.0 --ndata "{4}" --disp_file'.format(
                int(supercell_matrix[0][0]), 
                int(supercell_matrix[1][1]), 
                int(supercell_matrix[2][2]),
                float(symprec),
                int(num_har))
            
            displacement_f = 0.01
            phonon.generate_displacements(distance=displacement_f)
            disps = phonon.displacements
            num_judge = len(disps)

            if num_judge > 3:
                pheasy_cmd_14 = 'pheasy --dim "{0}" "{1}" "{2}" -f --c2 10.0 --full_ifc -w 2 --symprec "{3}" -l LASSO --std --rasr BHH --ndata "{4}"'.format(
                    int(supercell_matrix[0][0]),
                    int(supercell_matrix[1][1]), 
                    int(supercell_matrix[2][2]),
                    float(symprec), 
                    int(num_har))
            else:
                pheasy_cmd_14 = 'pheasy --dim "{0}" "{1}" "{2}" -f --full_ifc --c2 10.0 -w 2 --symprec "{3}" --rasr BHH --ndata "{4}"'.format(
                    int(supercell_matrix[0][0]), 
                    int(supercell_matrix[1][1]), 
                    int(supercell_matrix[2][2]),
                    float(symprec), 
                    int(num_har))

            logger.info("Start running pheasy in cluster")
            subprocess.call(pheasy_cmd_11, shell=True)
            subprocess.call(pheasy_cmd_12, shell=True)
            subprocess.call(pheasy_cmd_13, shell=True)
            subprocess.call(pheasy_cmd_14, shell=True)

            force_constants = parse_FORCE_CONSTANTS(filename="FORCE_CONSTANTS")
            phonon.force_constants = force_constants
            phonon.symmetrize_force_constants()

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

            new_plotter.save_plot(filename=kwargs.get("filename_bs", 
                                                      "phonon_band_structure.pdf"),
                                                      units=kwargs.get("units", "THz"))

            imaginary_modes_cutoff = bs_symm_line.has_imaginary_freq(
            tol=kwargs.get("tol_imaginary_modes", 1e-5))
            imaginary_modes = imaginary_modes_cutoff
            # new_plotter.save_plot(
            #    "phonon_band_structure.eps",
            #    img_format=kwargs.get("img_format", "eps"),
            #    units=kwargs.get("units", "THz"),
            #)
        else:
            pass

        # gets data for visualization on website - yaml is also enough
        if kwargs.get("band_structure_eigenvectors"):
            bs_symm_line.write_phononwebsite("phonon_website.json")

        # get phonon density of states
        filename_dos_yaml = "phonon_dos.yaml"

        kpoint_density_dos = kwargs.get("kpoint_density_dos", 7_000)
        kpoint = Kpoints.automatic_density(
            structure=get_pmg_structure(phonon.primitive),
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

        return cls.from_structure(
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
            has_imaginary_modes=imaginary_modes,
            force_constants={"force_constants": phonon.force_constants.tolist()}
            if kwargs["store_force_constants"]
            else None,
            born=borns.tolist() if borns is not None else None,
            epsilon_static=epsilon.tolist() if epsilon is not None else None,
            supercell_matrix=phonon.supercell_matrix.tolist(),
            primitive_matrix=phonon.primitive_matrix.tolist(),
            code=code,
            mp_id=mp_id,
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
