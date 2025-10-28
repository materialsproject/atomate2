"""Jobs for running phonon calculations with phonopy and pheasy."""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.io import read as ase_read
from emmet.core import __version__ as _emmet_core_version
from emmet.core.phonon import PhononBSDOSDoc
from hiphive import ClusterSpace, ForceConstantPotential, enforce_rotational_sum_rules
from hiphive import ForceConstants as HiPhiveForceConstants
from hiphive.cutoffs import estimate_maximum_cutoff
from hiphive.utilities import extract_parameters
from jobflow import job
from packaging.version import parse as parse_version
from phonopy.file_IO import parse_FORCE_CONSTANTS, write_force_constants_to_hdf5
from phonopy.interface.vasp import write_vasp
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon
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
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)

from atomate2.common.jobs.phonons import _generate_phonon_object, _get_kpath

if TYPE_CHECKING:
    from emmet.core.math import Matrix3D

logger = logging.getLogger(__name__)

try:
    from alm import ALM
except ImportError:
    ALM = None

_DEFAULT_FILE_PATHS = {
    "force_displacements": "dataset_forces.npy",
    "displacements": "dataset_disps.npy",
    "displacements_folded": "dataset_disps_array_rr.npy",
    "phonopy": "phonopy.yaml",
    "band_structure": "phonon_band_structure.yaml",
    "band_structure_plot": "phonon_band_structure.pdf",
    "dos": "phonon_dos.yaml",
    "dos_plot": "phonon_dos.pdf",
    "force_constants": "FORCE_CONSTANTS",
    "harmonic_displacements": "disp_matrix.npy",
    "anharmonic_displacements": "disp_matrix_anhar.npy",
    "harmonic_force_matrix": "force_matrix.npy",
    "anharmonic_force_matrix": "force_matrix_anhar.npy",
    "website": "phonon_website.json",
}


@job
def get_supercell_size(
    structure: Structure,
    min_length: float,
    max_atoms: int,
    force_90_degrees: bool,
    force_diagonal: bool,
) -> list[list[float]]:
    """
    Determine supercell size with given min_length and max_length.

    Parameters
    ----------
    structure: Structure Object
        Input structure that will be used to determine supercell
    min_length: float
        minimum length of cell in Angstrom
    max_length: float
        maximum length of cell in Angstrom
    prefer_90_degrees: bool
        if True, the algorithm will try to find a cell with 90 degree angles first
    allow_orthorhombic: bool
        if True, orthorhombic supercells are allowed
    **kwargs:
        Additional parameters that can be set.
    """
    transformation = CubicSupercellTransformation(
        min_length=min_length,
        max_atoms=max_atoms,
        force_90_degrees=force_90_degrees,
        force_diagonal=force_diagonal,
        angle_tolerance=1e-2,
        allow_orthorhombic=False,
    )
    transformation.apply_transformation(structure=structure)
    return transformation.transformation_matrix.transpose().tolist()


@job(data=[Structure])
def generate_phonon_displacements(
    structure: Structure,
    supercell_matrix: np.array,
    displacement: float,
    num_displaced_supercells: int,
    cal_anhar_fcs: bool,
    displacement_anhar: float,
    num_disp_anhar: int,
    fcs_cutoff_radius: list[int],
    sym_reduce: bool,
    symprec: float,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    code: str,
    random_seed: int | None = 103,
    verbose: bool = False,
) -> list[Structure]:
    """Generate small-distance perturbed structures with phonopy based on two ways.

    (we will directly use the pheasy to generate the supercell in the near future)
    1. finite-displacment method (one displaced atom) when the displacement number
    is less than 3. 2. random-displacement method (all-displaced atoms) when the
    displacement number is more than 3.

    Parameters
    ----------
    structure: Structure object
        Fully optimized input structure for phonon run
    supercell_matrix: np.array
        array to describe supercell matrix
    displacement: float
        displacement in Angstrom (default: 0.01)
    num_displaced_supercells: int
        number of displaced supercells defined by users
    cal_anhar_fcs: bool
        TODO : docstr
    displacement_anhar: float
        TODO : docstr
    sym_reduce: bool
        if True, symmetry will be used to generate displacements
    symprec: float
        precision to determine symmetry
    use_symmetrized_structure: str or None
        primitive, conventional or None
    kpath_scheme: str
        scheme to generate kpath
    code: str
        code to perform the computations
    random_seed : int | None = 103
        Random seed to use in generating randomly-displaced structures.
    verbose : bool = False
        Whether to log warnings.

    """
    # TODO: remove ALMODE dependence for 2nd order force constants
    if not ALM:
        raise ImportError(
            "Error importing ALM. Please ensure the 'alm' library is installed."
        )
    phonon = _generate_phonon_object(
        structure,
        supercell_matrix,
        displacement,
        sym_reduce,
        symprec,
        use_symmetrized_structure,
        kpath_scheme,
        code,
        verbose=verbose,
    )

    # 1. the ALM module is used to determine the number of free parameters
    # (irreducible force constants) corresponding to the second order
    # force constants (FCs) given a supercell.
    # 2. Based on the number of free parameters, we can determine how many
    # displaced supercells we need to use to extract the second order force
    # constants. Generally, the number of free parameters should be less than
    # 3 * natom(supercell) * num_displaced_supercells. However, the full rank
    # of the matrix can not always guarantee accurate results, you
    # may need to displace more random configurations. Use at least one or
    # two more configurations based on the suggested number of displacements.
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

    # get the number of displaced supercells based on the number of free parameters
    num_disp_sc = int(np.ceil(n_fp / (3.0 * natom)))

    if verbose:
        logger.info(
            f"There are {n_fp} free parameters for the second-order "
            "force constants (FCs)."
            f"There are {3 * natom * num_disp_sc} equations used to "
            "obtain the second-order FCs."
            "CAUTION: you may need to increase the number of "
            "displacements in some cases."
            "If the number of atoms in the supercell are less than 100 and "
            "all lattice constants are less than 10 Å, the user is advised "
            "to use 1-2 more randomly-displaced configurations."
        )

    # get the number of displaced supercells from phonopy to compared with the number
    # of 3, if the number of displaced supercells is less than 3, we will use the finite
    # displacement method to generate the supercells. Otherwise, we will use the random
    # displacement method to generate the supercells.
    if len(phonon.displacements) > 3:
        phonon.generate_displacements(
            distance=displacement,
            number_of_snapshots=(
                num_displaced_supercells
                if num_displaced_supercells != 0
                else int(np.ceil(num_disp_sc * 1.8)) + 1
            ),
            random_seed=random_seed,
        )

    supercells = phonon.supercells_with_displacements
    displacements = [get_pmg_structure(cell) for cell in supercells]

    # Here, the ALAMODE code is used to determine the number of
    # third and fourth-order FCs are needed for the supercell
    if cal_anhar_fcs:
        # Due to the cutoff radius of the force constants use the unit of Borh in ALM,
        # we need to convert the cutoff radius from Angstrom to Bohr.
        with ALM(lattice * 1.89, positions, numbers) as alm:
            # Define the force constants up to fourth order with a list of
            # cutoff radius
            alm.define(3, fcs_cutoff_radius)
            # Perform symmetry analysis and suggest irreducible force constants.
            alm.suggest()
            # Get the number of irreducible elements for both 3RD- and 4TH-order
            # force constants
            n_rd_anh = alm._get_number_of_irred_fc_elements(  # noqa: SLF001
                2
            ) + alm._get_number_of_irred_fc_elements(3)  # noqa: SLF001
            # we can determine how many displaced supercells we need to use to extract
            # the 3rd and 4th order force constants, and we can add a scaling factor
            # to reduce the number of displaced supercells due to we use the lasso
            # technique.
            num_d_anh = int(np.ceil(n_rd_anh / (3.0 * natom)))
            num_dis_cells_anhar = num_disp_anhar if num_disp_anhar != 0 else num_d_anh

        num_dis_cells_anhar = 20
        # generate the supercells for anharmonic force constants
        phonon.generate_displacements(
            distance=displacement_anhar,
            number_of_snapshots=num_dis_cells_anhar,
            random_seed=random_seed,
        )
        supercells = phonon.supercells_with_displacements
        displacements += [get_pmg_structure(cell) for cell in supercells]

    # add the equilibrium structure to the list for calculating
    # the residual forces.
    displacements.append(get_pmg_structure(phonon.supercell))
    return displacements


@job(
    output_schema=PhononBSDOSDoc,
    data=[PhononDos, PhononBandStructureSymmLine, "force_constants"],
)
def generate_frequencies_eigenvectors(
    structure: Structure,
    supercell_matrix: np.array,
    displacement: float,
    cal_anhar_fcs: bool,
    fcs_cutoff_radius: list[int],
    renorm_phonon: bool,
    cal_ther_cond: bool,
    ther_cond_mesh: list[int],
    ther_cond_temp: list[int],
    sym_reduce: bool,
    symprec: float,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    code: str,
    displacement_data: dict[str, list],
    total_dft_energy: float,
    epsilon_static: Matrix3D = None,
    born: Matrix3D = None,
    **kwargs,
) -> PhononBSDOSDoc:
    """
    Analyze the phonon runs and summarize the results.

    Parameters
    ----------
    structure: Structure object
        Fully optimized structure used for phonon runs
    supercell_matrix: np.array
        array to describe supercell
    displacement: float
        displacement in Angstrom used for supercell computation
    sym_reduce: bool
        if True, symmetry will be used in phonopy
    symprec: float
        precision to determine symmetry
    use_symmetrized_structure: str
        primitive, conventional, None are allowed
    kpath_scheme: str
        kpath scheme for phonon band structure computation
    code: str
        code to run computations
    displacement_data: dict
        outputs from displacements
    total_dft_energy: float
        total DFT energy in eV per cell
    epsilon_static: Matrix3D
        The high-frequency dielectric constant
    born: Matrix3D
        Born charges
    verbose : bool = False
        Whether to log error messages.
    kwargs: dict
        Additional parameters that are passed to PhononBSDOSDoc.from_forces_born
    """
    phonon = _generate_phonon_object(
        structure,
        supercell_matrix,
        displacement,
        sym_reduce,
        symprec,
        use_symmetrized_structure,
        kpath_scheme,
        code,
        verbose=False,
    )

    # Write the POSCAR and SPOSCAR files for the input of pheasy code
    supercell = phonon._supercell  # noqa: SLF001
    write_vasp("POSCAR", get_phonopy_structure(structure))
    write_vasp("SPOSCAR", supercell)

    # get the force-displacement dataset from previous calculations
    dataset_forces = np.array(displacement_data["forces"])
    np.save(_DEFAULT_FILE_PATHS["force_displacements"], dataset_forces)

    # To deduct the residual forces on an equilibrium structure to eliminate the
    # fitting error
    dataset_forces_array_rr = dataset_forces - dataset_forces[-1, :, :]

    # force matrix on the displaced structures
    dataset_forces_array_disp = dataset_forces_array_rr[:-1, :, :]

    # To handle the large dispalced distance in the dataset
    dataset_disps = np.array(
        [disps.frac_coords for disps in displacement_data["displaced_structures"]]
    )
    np.save(_DEFAULT_FILE_PATHS["displacements"], dataset_disps)

    dataset_disps_array_rr = np.round(
        (dataset_disps - supercell.get_scaled_positions()), decimals=16
    )
    np.save(_DEFAULT_FILE_PATHS["displacements_folded"], dataset_disps_array_rr)

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
    dataset_disps_array_rr_transposed = np.transpose(dataset_disps_array_rr, (0, 2, 1))

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
    num_har = dataset_disps_array_use.shape[0]
    if cal_anhar_fcs:
        if not ALM:
            raise ImportError(
                "Error importing ALM. Please ensure the 'alm' library is installed."
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
        num_disp_f = len(phonon.displacements)
        num_har = int(np.ceil(num * 1.8)) if num_disp_f > 3 else num_disp_f

    np.save(
        _DEFAULT_FILE_PATHS["harmonic_displacements"],
        dataset_disps_array_use[:num_har, :, :],
    )
    np.save(
        _DEFAULT_FILE_PATHS["harmonic_force_matrix"],
        dataset_forces_array_disp[:num_har, :, :],
    )

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

    prim = ase_read("POSCAR")
    supercell = ase_read("SPOSCAR")

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
    if len(phonon.displacements) > 3:
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
    subprocess.call(shlex.split(pheasy_cmd_2))
    subprocess.call(shlex.split(pheasy_cmd_3))
    subprocess.call(shlex.split(pheasy_cmd_4))

    # When this code is run on Github tests, it is failing because it is
    # not able to find the FORCE_CONSTANTS file. This is because the file is
    # somehow getting generated in some temp directory. Can you fix the bug?
    fc_file = Path(_DEFAULT_FILE_PATHS["force_constants"])

    if cal_anhar_fcs and fc_file.exists():
        np.save(
            _DEFAULT_FILE_PATHS["anharmonic_displacements"],
            dataset_disps_array_use[num_har:, :, :],
        )
        np.save(
            _DEFAULT_FILE_PATHS["anharmonic_force_matrix"],
            dataset_forces_array_disp[num_har:, :, :],
        )
        num_anhar = dataset_forces_array_disp.shape[0] - num_har

        # We next begin to generate the anharmonic force constants up to fourth
        # order using the LASSO method
        pheasy_cmd_5 = (
            f"pheasy --dim {int(supercell_matrix[0][0])} "
            f"{int(supercell_matrix[1][1])} "
            f"{int(supercell_matrix[2][2])} -s -w 4 --symprec "
            f"{float(symprec)} "
            f"--nbody 2 3 3 --c3 {float(fcs_cutoff_radius[1] / 1.89)} "
            f"--c4 {float(fcs_cutoff_radius[2] / 1.89)}"
        )

        pheasy_cmd_6 = (
            f"pheasy --dim {int(supercell_matrix[0][0])} "
            f"{int(supercell_matrix[1][1])} "
            f"{int(supercell_matrix[2][2])} -c --symprec "
            f"{float(symprec)} -w 4"
        )
        pheasy_cmd_7 = (
            f"pheasy --dim {int(supercell_matrix[0][0])} "
            f"{int(supercell_matrix[1][1])} "
            f"{int(supercell_matrix[2][2])} -w 4 -d --symprec "
            f"{float(symprec)} "
            f"--ndata {int(num_anhar)} --disp_file"
        )
        pheasy_cmd_8 = (
            f"pheasy --dim {int(supercell_matrix[0][0])} "
            f"{int(supercell_matrix[1][1])} "
            f"{int(supercell_matrix[2][2])} -f -w 4 --fix_fc2 "
            f"--symprec {float(symprec)} "
            f"--ndata {int(num_anhar)} "
        )

        subprocess.call(shlex.split(pheasy_cmd_5))
        subprocess.call(shlex.split(pheasy_cmd_6))
        subprocess.call(shlex.split(pheasy_cmd_7))
        subprocess.call(shlex.split(pheasy_cmd_8))

    # begin to renormzlize the phonon energies
    if renorm_phonon:
        pheasy_cmd_9 = (
            f"pheasy --dim {int(supercell_matrix[0][0])} "
            f"{int(supercell_matrix[1][1])} "
            f"{int(supercell_matrix[2][2])} -f -w 4 --fix_fc2 "
            f"--hdf5 --symprec {float(symprec)} "
            f"--ndata {int(num_anhar)}"
        )

        subprocess.call(shlex.split(pheasy_cmd_9))

        # write the born charges and dielectric constant to the pheasy format

    # begin to convert the force constants to the phonopy and phono3py format
    # for the further lattice thermal conductivity calculations
    if cal_ther_cond and fc_file.exists():
        # convert the 2ND order force constants to the phonopy format
        fc_phonopy_text = parse_FORCE_CONSTANTS(filename=fc_file)
        write_force_constants_to_hdf5(fc_phonopy_text, filename="fc2.hdf5")

        # convert the 3RD order force constants to the phonopy format

        prim_hiphive = ase_read("POSCAR")
        supercell_hiphive = ase_read("SPOSCAR")
        fcs = HiPhiveForceConstants.read_shengBTE(
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

    if fc_file.exists():
        # Read the force constants from the output file of pheasy code
        force_constants = parse_FORCE_CONSTANTS(filename=fc_file)
        phonon.force_constants = force_constants
        # symmetrize the force constants to make them physically correct based on
        # the space group symmetry of the crystal structure.
        phonon.symmetrize_force_constants()

    # with phonopy.load("phonopy.yaml") the phonopy API can be used
    phonon.save(_DEFAULT_FILE_PATHS["phonopy"])

    # get phonon band structure
    kpath_dict, kpath_concrete = _get_kpath(
        structure=get_pmg_structure(phonon.primitive),
        kpath_scheme=kpath_scheme,
        symprec=symprec,
    )

    npoints_band = kwargs.get("npoints_band", 101)
    qpoints, connections = get_band_qpoints_and_path_connections(
        kpath_concrete, npoints=kwargs.get("npoints_band", 101)
    )

    phonon.run_band_structure(
        qpoints,
        path_connections=connections,
        with_eigenvectors=kwargs.get("band_structure_eigenvectors", False),
        is_band_connection=kwargs.get("band_structure_eigenvectors", False),
    )
    # phonon.write_hdf5_band_structure(filename=_DEFAULT_FILE_PATHS["band_structure"])
    phonon.write_yaml_band_structure(filename=_DEFAULT_FILE_PATHS["band_structure"])
    bs_symm_line = get_ph_bs_symm_line(
        _DEFAULT_FILE_PATHS["band_structure"],
        labels_dict=kpath_dict,
        has_nac=born is not None,
    )

    bs_plot_file = kwargs.get("filename_bs", _DEFAULT_FILE_PATHS["band_structure_plot"])
    dos_plot_file = kwargs.get("filename_dos", _DEFAULT_FILE_PATHS["dos_plot"])

    new_plotter = PhononBSPlotter(bs=bs_symm_line)
    new_plotter.save_plot(
        filename=bs_plot_file,
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
        fcs = HiPhiveForceConstants.read_phonopy(supercell, "FORCE_CONSTANTS")

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
        new_fc_file = f"{_DEFAULT_FILE_PATHS['force_constants']}_short_cutoff"
        fcs.write_to_phonopy(new_fc_file, format="text")

        force_constants = parse_FORCE_CONSTANTS(filename=new_fc_file)
        phonon.force_constants = force_constants
        phonon.symmetrize_force_constants()

        phonon.run_band_structure(
            qpoints, path_connections=connections, with_eigenvectors=True
        )
        phonon.write_yaml_band_structure(filename=_DEFAULT_FILE_PATHS["band_structure"])
        bs_symm_line = get_ph_bs_symm_line(
            _DEFAULT_FILE_PATHS["band_structure"],
            labels_dict=kpath_dict,
            has_nac=born is not None,
        )

        new_plotter = PhononBSPlotter(bs=bs_symm_line)

        new_plotter.save_plot(
            filename=bs_plot_file,
            units=kwargs.get("units", "THz"),
        )

        imaginary_modes = bs_symm_line.has_imaginary_freq(
            tol=kwargs.get("tol_imaginary_modes", 1e-5)
        )

    # Using a shorter cutoff (10 A) to generate the force constants to
    # eliminate the imaginary modes near Gamma point in phesay code
    if imaginary_modes:
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

        if len(phonon.displacements) > 3:
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

        subprocess.call(shlex.split(pheasy_cmd_11))
        subprocess.call(shlex.split(pheasy_cmd_12))
        subprocess.call(shlex.split(pheasy_cmd_13))
        subprocess.call(shlex.split(pheasy_cmd_14))

        force_constants = parse_FORCE_CONSTANTS(filename=new_fc_file)
        phonon.force_constants = force_constants
        phonon.symmetrize_force_constants()

        phonon.save(_DEFAULT_FILE_PATHS["phonopy"])

        # get phonon band structure
        kpath_dict, kpath_concrete = _get_kpath(
            structure=get_pmg_structure(phonon.primitive),
            kpath_scheme=kpath_scheme,
            symprec=symprec,
        )

        npoints_band = kwargs.get("npoints_band", 101)
        qpoints, connections = get_band_qpoints_and_path_connections(
            kpath_concrete, npoints=kwargs.get("npoints_band", 101)
        )

        # phonon band structures will always be computed
        phonon.run_band_structure(
            qpoints, path_connections=connections, with_eigenvectors=True
        )
        phonon.write_yaml_band_structure(filename=_DEFAULT_FILE_PATHS["band_structure"])
        bs_symm_line = get_ph_bs_symm_line(
            _DEFAULT_FILE_PATHS["band_structure"],
            labels_dict=kpath_dict,
            has_nac=born is not None,
        )
        new_plotter = PhononBSPlotter(bs=bs_symm_line)

        new_plotter.save_plot(
            filename=bs_plot_file,
            units=kwargs.get("units", "THz"),
        )

        imaginary_modes = bs_symm_line.has_imaginary_freq(
            tol=kwargs.get("tol_imaginary_modes", 1e-5)
        )

    # gets data for visualization on website - yaml is also enough
    if kwargs.get("band_structure_eigenvectors"):
        bs_symm_line.write_phononwebsite(_DEFAULT_FILE_PATHS["website"])

    # get phonon density of states
    kpoint_density_dos = kwargs.get("kpoint_density_dos", 7_000)
    kpoint = Kpoints.automatic_density(
        structure=get_pmg_structure(phonon.primitive),
        kppa=kpoint_density_dos,
        force_gamma=True,
    )
    phonon.run_mesh(kpoint.kpts[0])
    phonon.run_total_dos()
    phonon.write_total_dos(filename=_DEFAULT_FILE_PATHS["dos"])
    dos = get_ph_dos(_DEFAULT_FILE_PATHS["dos"])
    new_plotter_dos = PhononDosPlotter()
    new_plotter_dos.add_dos(label="total", dos=dos)
    new_plotter_dos.save_plot(
        filename=dos_plot_file,
        units=kwargs.get("units", "THz"),
    )

    # will compute thermal displacement matrices
    # for the primitive cell (phonon.primitive!)
    # only this is available in phonopy
    if kwargs.get("create_thermal_displacements"):
        phonon.run_mesh(kpoint.kpts[0], with_eigenvectors=True, is_mesh_symmetry=False)
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

    cls_constructor = (
        "migrate_fields"
        if parse_version(_emmet_core_version) >= parse_version("0.85.1")
        else "from_structure"
    )
    return getattr(PhononBSDOSDoc, cls_constructor)(
        structure=structure,
        meta_structure=structure,
        phonon_bandstructure=bs_symm_line,
        phonon_dos=dos,
        total_dft_energy=total_dft_energy_per_formula_unit,
        has_imaginary_modes=imaginary_modes,
        force_constants=(
            {"force_constants": phonon.force_constants.tolist()}
            if kwargs.get("store_force_constants")
            else None
        ),
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
        post_process_settings={
            "npoints_band": npoints_band,
            "kpath_scheme": kpath_scheme,
            "kpoint_density_dos": kpoint_density_dos,
        },
    )
