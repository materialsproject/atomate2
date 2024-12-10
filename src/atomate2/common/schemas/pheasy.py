"""Schemas for phonon documents."""

import copy
import logging
import pickle
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Union

import numpy as np

# import lib by jiongzhi zheng
from ase.io import read
from emmet.core.math import Matrix3D
from emmet.core.structure import StructureMetadata
from hiphive import (
    ClusterSpace,
    ForceConstantPotential,
    ForceConstants,
    enforce_rotational_sum_rules,
)
from hiphive.cutoffs import estimate_maximum_cutoff
from hiphive.utilities import extract_parameters
from monty.json import MSONable
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS, write_force_constants_to_hdf5
from phonopy.interface.vasp import write_vasp
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon
from pydantic import Field
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

# import some classmethod directly from phonons
from atomate2.common.schemas.phonons import (
    PhononComputationalSettings,
    PhononJobDirs,
    PhononUUIDs,
    ThermalDisplacementData,
    get_factor,
)

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
        num_displaced_supercells: int,
        cal_anhar_fcs: bool,
        displacement_anhar: float,
        num_disp_anhar: int,
        fcs_cutoff_radius: list[int],
        renorm_phonon: bool,
        cal_ther_cond: bool,
        ther_cond_mesh: list[int],
        ther_cond_temp: list[int],
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
        num_displaced_supercells: int
            number of displaced supercells
        cal_anhar_fcs: bool
            if True, anharmonic force constants will be computed
        displacement_anhar: float
            size of displacement in angstrom for anharmonic force constants
        num_disp_anhar: int
            number of displaced supercells for anharmonic force constants
        fcs_cutoff_radius: list
            cutoff radius for force constants
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

        # Create the phonon object using the phonopy API to write the POSCAR and
        # SPOSCAR files for the input of pheasy code.
        phonon = Phonopy(
            cell,
            supercell_matrix,
            primitive_matrix=primitive_matrix,
            factor=factor,
            symprec=symprec,
            is_symmetry=sym_reduce,
        )

        # Write the POSCAR and SPOSCAR files for the input of pheasy code
        supercell = phonon.get_supercell()
        write_vasp("POSCAR", cell)
        write_vasp("SPOSCAR", supercell)

        # get the force-displacement dataset from previous calculations
        dataset_forces = [np.array(forces) for forces in displacement_data["forces"]]
        dataset_forces_array = np.array(dataset_forces)
        # save to csv file
        import csv

        with open("dataset_forces.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerows(dataset_forces_array)

        # To deduct the residual forces on an equilibrium structure to eliminate the
        # fitting error
        dataset_forces_array_rr = dataset_forces_array - dataset_forces_array[-1, :, :]

        # force matrix on the displaced structures
        dataset_forces_array_disp = dataset_forces_array_rr[:-1, :, :]
        # dataset_disps = [
        #    np.array(disps.cart_coords)
        #    for disps in displacement_data["displaced_structures"]
        # ]

        # get the displacement dataset
        # dataset_disps_array_rr = np.round(
        #    (dataset_disps - supercell.get_positions()),
        #                            decimals=16
        # ).astype('double')
        # dataset_disps_array_use = dataset_disps_array_rr[:-1, :, :]

        # To handle the large dispalced distance in the dataset
        dataset_disps = [
            np.array(disps.frac_coords)
            for disps in displacement_data["displaced_structures"]
        ]
        logger.info(f"dataset_disps = {dataset_disps}")
        # save to csv file
        import csv

        with open("dataset_disps.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerows(dataset_disps)
        dataset_disps_array_rr = np.round(
            (dataset_disps - supercell.get_scaled_positions()), decimals=16
        ).astype("double")
        # save to csv file
        with open("dataset_disps_array_rr.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerows(dataset_disps_array_rr)
        dataset_disps_array_rr = np.where(
            dataset_disps_array_rr > 0.5,
            dataset_disps_array_rr - 1.0,
            dataset_disps_array_rr,
        )
        dataset_disps_array_rr = np.where(
            dataset_disps_array_rr < -0.5,
            dataset_disps_array_rr + 1.0,
            dataset_disps_array_rr,
        )

        # Transpose the displacement array on the
        # last two axes (atoms and coordinates)
        dataset_disps_array_rr_transposed = np.transpose(
            dataset_disps_array_rr, (0, 2, 1)
        )

        # Perform matrix multiplication with the transposed supercell.cell
        # 'ij' for supercell.cell.T and
        # 'nkj' for the transposed dataset_disps_array_rr
        dataset_disps_array_rr_cartesian = np.einsum(
            "ij,njk->nik", supercell.cell.T, dataset_disps_array_rr_transposed
        )
        # Transpose back to the original format
        dataset_disps_array_rr_cartesian = np.transpose(
            dataset_disps_array_rr_cartesian, (0, 2, 1)
        )

        dataset_disps_array_use = dataset_disps_array_rr_cartesian[:-1, :, :]

        # separate the dataset into harmonic and anharmonic parts
        if cal_anhar_fcs:
            try:
                from alm import ALM
            except ImportError:
                logging.exception(
                    "Error importing ALM. Please ensure the 'alm'"
                    "library is installed."
                )

            supercell_ph = phonon.supercell
            lattice = supercell_ph.cell
            positions = supercell_ph.scaled_positions
            numbers = supercell_ph.numbers
            natom = len(numbers)

            # get the number of free parameters of 2ND FCs from ALM, labeled as n_fp
            with ALM(lattice, positions, numbers) as alm:
                alm.define(1)
                alm.suggest()
                n_fp = alm._get_number_of_irred_fc_elements(1)  # noqa: SLF001

            # get the number of displaced supercells based on the
            # number of free parameters
            num = int(np.ceil(n_fp / (3.0 * natom)))

            # get the number of displaced supercells from phonopy to compared
            # with the number of 3, if the number of displaced supercells is
            # less than 3, we will use the finite displacement method to generate
            # the supercells. Otherwise, we will use the random displacement
            # method to generate the supercells.
            phonon.generate_displacements(distance=displacement)
            num_disp_f = len(phonon.displacements)
            if num_disp_f > 3:
                num_d = int(np.ceil(num * 1.8))
                num_har = num_d
            else:
                num_har = num_disp_f
        else:
            num_har = dataset_disps_array_use.shape[0]

        if cal_anhar_fcs:
            dataset_disps_array_use_har = dataset_disps_array_use[:num_har, :, :]
            dataset_forces_array_disp_har = dataset_forces_array_disp[:num_har, :, :]
            with open("disp_matrix.pkl", "wb") as file:
                pickle.dump(dataset_disps_array_use_har, file)
            with open("force_matrix.pkl", "wb") as file:
                pickle.dump(dataset_forces_array_disp_har, file)

        else:
            with open("disp_matrix.pkl", "wb") as file:
                pickle.dump(dataset_disps_array_use, file)
            with open("force_matrix.pkl", "wb") as file:
                pickle.dump(dataset_forces_array_disp, file)

        # get the born charges and dielectric constant
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

        prim = read("POSCAR")
        supercell = read("SPOSCAR")

        # Create the clusters and orbitals for second order force constants
        # For the variables: --w, --nbody, they are used to specify the order of the
        # force constants. in the near future, we will add the option to specify the
        # order of the force constants. And these two variables can be defined by the
        # users.
        pheasy_cmd_1 = (
            f"pheasy --dim {int(supercell_matrix[0][0])} "
            f"{int(supercell_matrix[1][1])} "
            f"{int(supercell_matrix[2][2])} "
            f"-s -w 2 --symprec {float(symprec)} --nbody 2"
        )

        # Create the null space to further reduce the free parameters for
        # specific force constants and make them physically correct.
        pheasy_cmd_2 = (
            f"pheasy --dim {int(supercell_matrix[0][0])} "
            f"{int(supercell_matrix[1][1])} "
            f"{int(supercell_matrix[2][2])} -c --symprec "
            f"{float(symprec)} -w 2"
        )

        # Generate the Compressive Sensing matrix,i.e., displacement matrix
        # for the input of machine leaning method.i.e., LASSO,
        pheasy_cmd_3 = (
            f"pheasy --dim {int(supercell_matrix[0][0])} "
            f"{int(supercell_matrix[1][1])} "
            f"{int(supercell_matrix[2][2])} -w 2 -d "
            f"--symprec {float(symprec)} "
            f"--ndata {int(num_har)} --disp_file"
        )

        # Here we set a criteria to determine which method to use to generate the
        # force constants. If the number of displacements is larger than 3, we
        # will use the LASSO method to generate the force constants. Otherwise,
        # we will use the least-squred method to generate the force constants.
        phonon.generate_displacements(distance=displacement)
        disps = phonon.displacements
        num_judge = len(disps)

        if num_judge > 3:
            # Calculate the force constants using the LASSO method due to the
            # random-displacement method Obviously, the rotaional invariance
            # constraint, i.e., tag: --rasr BHH, is enforced during the
            # fitting process.
            pheasy_cmd_4 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -f --full_ifc "
                f"-w 2 --symprec {float(symprec)} "
                f"-l LASSO --std --rasr BHH --ndata {int(num_har)}"
            )

        else:
            # Calculate the force constants using the least-squred method
            pheasy_cmd_4 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -f --full_ifc "
                f"-w 2 --symprec {float(symprec)} "
                f"--rasr BHH --ndata {int(num_har)}"
            )

        logger.info("Start running pheasy in cluster")

        subprocess.call(shlex.split(pheasy_cmd_1))
        logger.info(f"all files in cwd after cmd_1 are {list(Path.cwd().iterdir())}")
        subprocess.call(shlex.split(pheasy_cmd_2))
        logger.info(f"all files in cwd after cmd_2 are {list(Path.cwd().iterdir())}")
        subprocess.call(shlex.split(pheasy_cmd_3))
        logger.info(f"all files in cwd after cmd_3 are {list(Path.cwd().iterdir())}")
        # print the cwd
        logger.info(f"path before running cmd_4 is {Path.cwd()}")
        subprocess.call(shlex.split(pheasy_cmd_4))
        logger.info(f"path after running cmd_4 is {Path.cwd()}")
        # print all the files in the current directory
        logger.info(f"all files in cwd after cmd_4 are {list(Path.cwd().iterdir())}")

        # When this code is run on Github tests, it is failing because it is
        # not able to find the FORCE_CONSTANTS file. This is because the file is
        # somehow getting generated in some temp directory. Can you fix the bug?
        cwd = Path.cwd()
        fc_file = cwd / "FORCE_CONSTANT"

        if cal_anhar_fcs:
            # subprocess.call("rm -f disp_matrix.pkl force_matrix.pkl", shell=True)
            subprocess.run(
                ["/bin/rm", "-f", "disp_matrix.pkl", "force_matrix.pkl"], check=True
            )
            dataset_disps_array_use_anahr = dataset_disps_array_use[num_har:, :, :]
            dataset_forces_array_disp_anhar = dataset_forces_array_disp[num_har:, :, :]
            with open("disp_matrix.pkl", "wb") as file:
                pickle.dump(dataset_disps_array_use_anahr, file)
            with open("force_matrix.pkl", "wb") as file:
                pickle.dump(dataset_forces_array_disp_anhar, file)
            num_anhar = dataset_disps_array_use_anahr.shape[0]
        else:
            pass

        # We next begin to generate the anharmonic force constants up to fourth
        # order using the LASSO method

        if cal_anhar_fcs:
            pheasy_cmd_5 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -s -w 4 --symprec "
                f"{float(symprec)} "
                f"--nbody 2 3 3 --c3 {float(fcs_cutoff_radius[1]/1.89)} "
                f"--c4 {float(fcs_cutoff_radius[2]/1.89)}"
            )
            logger.info("pheasy_cmd_5 = %s", pheasy_cmd_5)

            pheasy_cmd_6 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -c --symprec "
                f"{float(symprec)} -w 4"
            )
            logger.info("pheasy_cmd_6 = %s", pheasy_cmd_6)
            pheasy_cmd_7 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -w 4 -d --symprec "
                f"{float(symprec)} "
                f"--ndata {int(num_anhar)} --disp_file"
            )
            logger.info("pheasy_cmd_7 = %s", pheasy_cmd_7)
            pheasy_cmd_8 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -f -w 4 --fix_fc2 "
                f"--symprec {float(symprec)} "
                f"--ndata {int(num_anhar)} "
            )
            logger.info("pheasy_cmd_8 = %s", pheasy_cmd_8)
            logger.info("Start running pheasy in cluster")

            subprocess.call(shlex.split(pheasy_cmd_5))
            subprocess.call(shlex.split(pheasy_cmd_6))
            subprocess.call(shlex.split(pheasy_cmd_7))
            subprocess.call(shlex.split(pheasy_cmd_8))
        else:
            pass

        # begin to renormzlize the phonon energies
        if renorm_phonon:
            pheasy_cmd_9 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -f -w 4 --fix_fc2 "
                f"--hdf5 --symprec {float(symprec)} "
                f"--ndata {int(num_anhar)}"
            )

            logger.info("Start running pheasy in cluster")
            subprocess.call(shlex.split(pheasy_cmd_9))

            # write the born charges and dielectric constant to the pheasy format

        else:
            pass

        # begin to convert the force constants to the phonopy and phono3py format
        # for the further lattice thermal conductivity calculations
        if cal_ther_cond:
            # convert the 2ND order force constants to the phonopy format
            fc_phonopy_text = parse_FORCE_CONSTANTS(filename="FORCE_CONSTANTS")
            write_force_constants_to_hdf5(fc_phonopy_text, filename="fc2.hdf5")

            # convert the 3RD order force constants to the phonopy format

            prim_hiphive = read("POSCAR")
            supercell_hiphive = read("SPOSCAR")
            fcs = ForceConstants.read_shengBTE(
                supercell_hiphive, "FORCE_CONSTANTS_3RD", prim_hiphive
            )
            fcs.write_to_phono3py("fc3.hdf5")

            phono3py_cmd = (
                f"phono3py --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} {int(supercell_matrix[2][2])} "
                f"--fc2 --fc3 --br --isotope --wigner "
                f"--mesh {ther_cond_mesh[0]} {ther_cond_mesh[1]} {ther_cond_mesh[2]} "
                f"--tmin {ther_cond_temp[0]} --tmax {ther_cond_temp[1]} "
                f"--tstep {ther_cond_temp[2]}"
            )

            subprocess.call(shlex.split(phono3py_cmd))
        else:
            pass

        # Read the force constants from the output file of pheasy code
        force_constants = parse_FORCE_CONSTANTS(filename=fc_file)
        phonon.force_constants = force_constants
        # symmetrize the force constants to make them physically correct based on
        # the space group symmetry of the crystal structure.
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

        # If imaginary modes are present, we first use the hiphive code to enforce
        # some symmetry constraints to eliminate the imaginary modes (generally work
        # for small imaginary modes near Gamma point). If the imaginary modes are
        # still present, we will use the pheasy code to generate the force constants
        # using a shorter cutoff (10 A) to eliminate the imaginary modes, also we
        # just want to remove the imaginary modes near Gamma point. In the future,
        # we will only use the pheasy code to do the job.

        if imaginary_modes:
            # Define a cluster space using the largest cutoff you can
            max_cutoff = estimate_maximum_cutoff(supercell) - 0.01
            cutoffs = [max_cutoff]  # only second order needed
            cs = ClusterSpace(prim, cutoffs)

            # import the phonopy force constants using the correct supercell also
            # provided by phonopy
            fcs = ForceConstants.read_phonopy(supercell, "FORCE_CONSTANTS")

            # Find the parameters that best fits the force constants given you
            # cluster space
            parameters = extract_parameters(fcs, cs)

            # Enforce the rotational sum rules
            parameters_rot = enforce_rotational_sum_rules(
                cs, parameters, ["Huang", "Born-Huang"], alpha=1e-6
            )

            # use the new parameters to make a fcp and then create the force
            # constants and write to a phonopy file
            fcp = ForceConstantPotential(cs, parameters_rot)
            fcs = fcp.get_force_constants(supercell)
            fcs.write_to_phonopy("FORCE_CONSTANTS_new", format="text")

            force_constants = parse_FORCE_CONSTANTS(filename="FORCE_CONSTANTS_new")
            phonon.force_constants = force_constants
            phonon.symmetrize_force_constants()

            phonon.run_band_structure(
                qpoints, path_connections=connections, with_eigenvectors=True
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

            # new_plotter.save_plot("phonon_band_structure.eps",
            # img_format=kwargs.get("img_format", "eps"),
            # units=kwargs.get("units", "THz"),)

            imaginary_modes_hiphive = bs_symm_line.has_imaginary_freq(
                tol=kwargs.get("tol_imaginary_modes", 1e-5)
            )

        else:
            imaginary_modes_hiphive = False
            imaginary_modes = False

        # Using a shorter cutoff (10 A) to generate the force constants to
        # eliminate the imaginary modes near Gamma point in phesay code
        if imaginary_modes_hiphive:
            pheasy_cmd_11 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -s -w 2 --c2 "
                f"10.0 --symprec {float(symprec)} "
                f"--nbody 2"
            )

            pheasy_cmd_12 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -c --symprec "
                f"{float(symprec)} --c2 10.0 -w 2"
            )

            pheasy_cmd_13 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -w 2 -d --symprec "
                f"{float(symprec)} --c2 10.0 "
                f"--ndata {int(num_har)} --disp_file"
            )

            phonon.generate_displacements(distance=displacement)
            disps = phonon.displacements
            num_judge = len(disps)

            if num_judge > 3:
                pheasy_cmd_14 = (
                    f"pheasy --dim {int(supercell_matrix[0][0])} "
                    f"{int(supercell_matrix[1][1])} "
                    f"{int(supercell_matrix[2][2])} -f --c2 10.0 "
                    f"--full_ifc -w 2 --symprec {float(symprec)} "
                    f"-l LASSO --std --rasr BHH --ndata {int(num_har)}"
                )

            else:
                pheasy_cmd_14 = (
                    f"pheasy --dim {int(supercell_matrix[0][0])} "
                    f"{int(supercell_matrix[1][1])} "
                    f"{int(supercell_matrix[2][2])} -f --full_ifc "
                    f"--c2 10.0 -w 2 --symprec {float(symprec)} "
                    f"--rasr BHH --ndata {int(num_har)}"
                )

            logger.info("Start running pheasy in cluster")

            subprocess.call(shlex.split(pheasy_cmd_11))
            subprocess.call(shlex.split(pheasy_cmd_12))
            subprocess.call(shlex.split(pheasy_cmd_13))
            subprocess.call(shlex.split(pheasy_cmd_14))

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

            new_plotter.save_plot(
                filename=kwargs.get("filename_bs", "phonon_band_structure.pdf"),
                units=kwargs.get("units", "THz"),
            )

            imaginary_modes_cutoff = bs_symm_line.has_imaginary_freq(
                tol=kwargs.get("tol_imaginary_modes", 1e-5)
            )
            imaginary_modes = imaginary_modes_cutoff
            # new_plotter.save_plot(
            #    "phonon_band_structure.eps",
            #    img_format=kwargs.get("img_format", "eps"),
            #    units=kwargs.get("units", "THz"),
            # )
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
            num_displaced_supercells=num_displaced_supercells,
            displacement_anhar=displacement_anhar,
            num_disp_anhar=num_disp_anhar,
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
