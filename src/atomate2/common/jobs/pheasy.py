"""Jobs for running phonon calculations with phonopy and pheasy."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from jobflow import job
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_pmg_structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)

from atomate2.common.jobs.phonons import _generate_phonon_object_for_displacements

if TYPE_CHECKING:
    from emmet.core.math import Matrix3D

# move to here to avoid circular import
from atomate2.common.schemas.pheasy import Forceconstants, PhononBSDOSDoc

logger = logging.getLogger(__name__)

try:
    from alm import ALM
except ImportError:
    ALM = None


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
        raise ValueError(
            "Error importing ALM. Please ensure the 'alm' library is installed."
        )
    phonon = _generate_phonon_object_for_displacements(
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

    # get the number of displaced supercells from phonopy to compared with the number
    # of 3, if the number of displaced supercells is less than 3, we will use the finite
    # displacement method to generate the supercells. Otherwise, we will use the random
    # displacement method to generate the supercells.
    phonon.generate_displacements(distance=displacement)
    num_disp_f = len(phonon.displacements)

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

    if num_disp_f > 3:
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

    # Here, the ALAMODE copde is used to determine the number of
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
    renorm_phonon: bool,
    renorm_temp: list[int],
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
        renorm_phonon=renorm_phonon,
        renorm_temp=renorm_temp,
        cal_ther_cond=cal_ther_cond,
        ther_cond_mesh=ther_cond_mesh,
        ther_cond_temp=ther_cond_temp,
        sym_reduce=sym_reduce,
        symprec=symprec,
        use_symmetrized_structure=use_symmetrized_structure,
        kpath_scheme=kpath_scheme,
        code=code,
        displacement_data=displacement_data,
        total_dft_energy=total_dft_energy,
        epsilon_static=epsilon_static,
        born=born,
        **kwargs,
    )
