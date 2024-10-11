"""Jobs for running phonon calculations."""

from __future__ import annotations

import contextlib
import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Flow, Response, job
from phonopy import Phonopy
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos

from atomate2.common.schemas.phonons import get_factor

if TYPE_CHECKING:
    from pathlib import Path

    from emmet.core.math import Matrix3D

    from atomate2.aims.jobs.base import BaseAimsMaker
    from atomate2.forcefields.jobs import ForceFieldStaticMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker

# move to here to avoid circular import
from atomate2.common.schemas.pheasy import Forceconstants, PhononBSDOSDoc


logger = logging.getLogger(__name__)


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
) -> list[Structure]:
    """

    Generate small-distance perturbed structures with phonopy based on two ways:
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
    sym_reduce: bool
        if True, symmetry will be used to generate displacements
    symprec: float
        precision to determine symmetry
    use_symmetrized_structure: str or None
        primitive, conventional or None
    kpath_scheme: str
        scheme to generate kpath
    code:
        code to perform the computations

    """

    warnings.warn(
        "Initial magnetic moments will not be considered for the determination "
        "of the symmetry of the structure and thus will be removed now.",
        stacklevel=1,
    )
    cell = get_phonopy_structure(
        structure.remove_site_property(property_name="magmom")
        if "magmom" in structure.site_properties
        else structure
    )
    factor = get_factor(code)

    # a bit of code repetition here as I currently
    # do not see how to pass the phonopy object?
    if use_symmetrized_structure == "primitive" and kpath_scheme != "seekpath":
        primitive_matrix: np.ndarray | str = np.eye(3)
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
    # create the phonopy object to get some information
    # for the displacement generation in ALM code.
    phonon = Phonopy(
        cell,
        supercell_matrix,
        primitive_matrix=primitive_matrix,
        factor=factor,
        symprec=symprec,
        is_symmetry=sym_reduce,
    )

    # 1. the ALM module is used to determine how many free parameters 
    # (irreducible force constants) of second order force constants (FCs)
    # within the supercell.
    # 2. Based on the number of free parameters, we can determine how many
    # displaced supercells we need to use to extract the second order force
    # constants. Generally, the number of free parameters should be less than
    # 3 * natom(supercell) * num_displaced_supercells. However, the full rank
    # of matrix can not always guarantee the accurate result sometimes, you
    # may need to displace more random configurations. At least use one or 
    # two more configurations based on the suggested number of displacements.

    try:
        from alm import ALM
    except ImportError as e:
        logging.error(
            f"Error importing ALM: {e}. Please ensure the 'alm'"
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
        n_fp = alm._get_number_of_irred_fc_elements(1)

    # get the number of displaced supercells based on the number of free parameters
    num = int(np.ceil(n_fp / (3.0 * natom)))

    # get the number of displaced supercells from phonopy to compared with the number
    # of 3, if the number of displaced supercells is less than 3, we will use the finite
    # displacement method to generate the supercells. Otherwise, we will use the random
    # displacement method to generate the supercells.  
    phonon.generate_displacements(distance=displacement)
    num_disp_f = len(phonon.displacements)
    if num_disp_f > 3:
        num_d = int(np.ceil(num * 1.8))
    else:
        pass
    
    logger.info(
        "The number of free parameters of Second Order Force Constants is %s",
                n_fp
    )
    logger.info("")  

    logger.info(
        "The Number of Equations Used to Obtain the 2ND FCs is %s",
                3 * natom * num
    )
    logger.info("")  

    logger.warning(
        "Be Careful!!! Full Rank of Matrix cannot always guarantee the correct result\
        sometimes.\n"
        "If the total atoms in the supercell are less than 100 and\n"
        "lattice constants are less than 10 angstroms,\n"
        "I highly suggest displacing more random configurations.\n"
        "At least use one or two more configurations based on the suggested\
        number of displacements."
    )
    logger.info("")

    if num_disp_f > 3:
        if num_displaced_supercells != 0:
            phonon.generate_displacements(
                distance=displacement, 
                number_of_snapshots=num_displaced_supercells, 
                random_seed=103,
            )
        else:
            phonon.generate_displacements(
                distance=displacement, 
                number_of_snapshots=num_d, 
                random_seed=103,
            )
    else:
        pass
    
    supercells = phonon.supercells_with_displacements
    displacements = [get_pmg_structure(cell) for cell in supercells]

    # Here, the ALM module is used to determine how many free parameters of third and
    # fourth order force constants (FCs) within the specific supercell.
    if cal_anhar_fcs:
            # Due to the cutoff radius of the force constants use the unit of Borh in ALM,
            # we need to convert the cutoff radius from Angstrom to Bohr.
            with ALM(lattice * 1.89, positions, numbers) as alm:
                # Define the force constants up to fourth order with a list of cutoff radius.
                alm.define(3, fcs_cutoff_radius)
                # Perform symmetry analysis and suggest irreducible force constants.
                alm.suggest()
                # Get the number of irreducible elements for both 3RD- and 4TH-order
                # force constants
                n_rd_anh = (
                    alm._get_number_of_irred_fc_elements(2) 
                    + alm._get_number_of_irred_fc_elements(3)
                )
                # we can determine how many displaced supercells we need to use to extract 
                # the third and fourth order force constants, and we can add a scaling factor
                # to reduce the number of displaced supercells due to we use the lasso
                # technique.
                num_d_anh = int(np.ceil(n_rd_anh / (3.0 * natom)))
                if num_disp_anhar != 0:
                    num_dis_cells_anhar = num_disp_anhar
                else:
                    num_dis_cells_anhar = num_d_anh

            # generate the supercells for anharmonic force constants
            phonon.generate_displacements(
                distance=displacement_anhar,
                number_of_snapshots=num_dis_cells_anhar,
                random_seed=103,
            )
            supercells = phonon.supercells_with_displacements
            displacements += [get_pmg_structure(cell) for cell in supercells]
    else:
        pass

    # add the equilibrium structure to the list for calculating
    # the residual forces.
    displacements.append(get_pmg_structure(phonon.supercell))
    return displacements


@job(
    output_schema=PhononBSDOSDoc,
    data=[PhononDos, PhononBandStructureSymmLine, Forceconstants],
)
def generate_frequencies_eigenvectors(
    structure: Structure,
    supercell_matrix: np.array,
    displacement: float,
    displacement_anhar: float,
    num_displaced_supercells: int,
    num_disp_anhar: int,
    cal_anhar_fcs: bool,
    fcs_cutoff_radius: list[int],
    sym_reduce: bool,
    symprec: float,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    code: str,
    mp_id: str,
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
    kwargs: dict
        Additional parameters that are passed to PhononBSDOSDoc.from_forces_born
    """
    return PhononBSDOSDoc.from_forces_born(
        structure=structure.remove_site_property(property_name="magmom")
        if "magmom" in structure.site_properties
        else structure,
        supercell_matrix=supercell_matrix,
        displacement=displacement,
        num_displaced_supercells=num_displaced_supercells,
        cal_anhar_fcs=cal_anhar_fcs,
        displacement_anhar=displacement_anhar,
        num_disp_anhar=num_disp_anhar,
        fcs_cutoff_radius=fcs_cutoff_radius,
        sym_reduce=sym_reduce,
        symprec=symprec,
        use_symmetrized_structure=use_symmetrized_structure,
        kpath_scheme=kpath_scheme,
        code=code,
        mp_id=mp_id,
        displacement_data=displacement_data,
        total_dft_energy=total_dft_energy,
        epsilon_static=epsilon_static,
        born=born,
        **kwargs,
    )


# I did not directly import this job from the phonon module
# because I modified the job to pass the displaced structures
# to the output.
@job(data=["forces", "displaced_structures"])
def run_phonon_displacements(
    displacements: list[Structure],
    structure: Structure,
    supercell_matrix: Matrix3D,
    phonon_maker: BaseVaspMaker | ForceFieldStaticMaker | BaseAimsMaker = None,
    prev_dir: str | Path = None,
    prev_dir_argname: str = None,
    socket: bool = False,
) -> Flow:
    """
    Run phonon displacements.

    Note, this job will replace itself with N displacement calculations,
    or a single socket calculation for all displacements.

    Parameters
    ----------
    displacements: Sequence
        All displacements to calculate
    structure: Structure object
        Fully optimized structure used for phonon computations.
    supercell_matrix: Matrix3D
        supercell matrix for meta data
    phonon_maker : .BaseVaspMaker or .ForceFieldStaticMaker or .BaseAimsMaker
        A maker to use to generate dispacement calculations
    prev_dir: str or Path
        The previous working directory
    prev_dir_argname: str
        argument name for the prev_dir variable
    socket: bool
        If True use the socket-io interface to increase performance
    """
    phonon_jobs = []
    outputs: dict[str, list] = {
        "displacement_number": [],
        "forces": [],
        "uuids": [],
        "dirs": [],
        "displaced_structures": [],
    }
    phonon_job_kwargs = {}
    if prev_dir is not None and prev_dir_argname is not None:
        phonon_job_kwargs[prev_dir_argname] = prev_dir

    if socket:
        phonon_job = phonon_maker.make(displacements, **phonon_job_kwargs)
        info = {
            "original_structure": structure,
            "supercell_matrix": supercell_matrix,
            "displaced_structures": displacements,
        }
        phonon_job.update_maker_kwargs(
            {"_set": {"write_additional_data->phonon_info:json": info}}, dict_mod=True
        )
        phonon_jobs.append(phonon_job)
        outputs["displacement_number"] = list(range(len(displacements)))
        outputs["uuids"] = [phonon_job.output.uuid] * len(displacements)
        outputs["dirs"] = [phonon_job.output.dir_name] * len(displacements)
        outputs["forces"] = phonon_job.output.output.all_forces
        # add the displaced structures, still need to be careful with the order,
        # experimental feature
        outputs["displaced_structures"] = displacements
    else:
        for idx, displacement in enumerate(displacements):
            if prev_dir is not None:
                phonon_job = phonon_maker.make(displacement, prev_dir=prev_dir)
            else:
                phonon_job = phonon_maker.make(displacement)
            phonon_job.append_name(f" {idx + 1}/{len(displacements)}")

            # we will add some meta data
            info = {
                "displacement_number": idx,
                "original_structure": structure,
                "supercell_matrix": supercell_matrix,
                "displaced_structure": displacement,
            }
            with contextlib.suppress(Exception):
                phonon_job.update_maker_kwargs(
                    {"_set": {"write_additional_data->phonon_info:json": info}},
                    dict_mod=True,
                )
            phonon_jobs.append(phonon_job)
            outputs["displacement_number"].append(idx)
            outputs["uuids"].append(phonon_job.output.uuid)
            outputs["dirs"].append(phonon_job.output.dir_name)
            outputs["forces"].append(phonon_job.output.output.forces)
            outputs["displaced_structures"].append(displacement)

    displacement_flow = Flow(phonon_jobs, outputs)
    return Response(replace=displacement_flow)
