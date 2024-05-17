"""Jobs for running anharmonicity quantification."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Optional

from jobflow import Flow, Response, job
from phonopy import Phonopy
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)
from atomate2.common.jobs.phonons import generate_phonon_displacements, run_phonon_displacements
from jobflow import run_locally

from scipy.constants import physical_constants

# TODO: NEED TO CHANGE
from atomate2.common.schemas.phonons import ForceConstants, PhononBSDOSDoc, get_factor

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from emmet.core.math import Vector3D, Matrix3D

    from atomate2.aims.jobs.base import BaseAimsMaker
    from atomate2.forcefields.jobs import ForceFieldStaticMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker


logger = logging.getLogger(__name__)

@job
def get_force_constants(
    phonon: Phonopy,
    forces_DFT: list[np.ndarray]
) -> np.ndarray:
    """
    Get the force constants for a perturbed Phonopy object.

    Parameters
    ----------
    phonon: Phonopy
        Phonopy object
    forces_DFT: list[np.ndarray]
        DFT forces used to find the force constants
    """
    phonon.produce_force_constants(forces=forces_DFT)

    force_constants = phonon.get_force_constants()
    return force_constants

def make_phonon(
    structure: Structure,
    supercell: np.ndarray,
    code: str,
    symprec: float,
    sym_reduce: bool,
    use_symmetrized_structure: str | None,
    kpath_scheme: str, 
    force_constants: Optional[ForceConstants] = None,
) -> Phonopy:
    """
    Builds a Phonopy object

    Parameters
    ----------
    structure: Structure
        Fully optimized input structure to use for making phonon
    supercell: np.ndarray
        array to describe supercell
    code: str
        code to perform the computations
    symprec: float
        precision to determine symmetry
    sym_reduce: bool
        if True, symmetry will be used to generate displacements
    use_symmetrized_structure: str or None
        Primitive, conventional, or None
    kpath_scheme:
        scheme to generate kpath
    """
    if use_symmetrized_structure == "primitive" and kpath_scheme != "seekpath":
        primitive_matrix: np.ndarray | str = np.eye(3)
    else:
        primitive_matrix = "auto"

    # Generate Phonopy object
    cell = get_phonopy_structure(structure)
    phonon = Phonopy(cell, 
                    supercell_matrix = supercell,
                    primitive_matrix = primitive_matrix,
                    factor = get_factor(code),
                    symprec = symprec,
                    is_symmetry = sym_reduce)
    if force_constants is not None:
        phonon.force_constants = np.array(force_constants)
    return phonon

@job 
def build_dyn_mat(
    structure: Structure,
    force_constants: ForceConstants,
    supercell: np.ndarray,
    qpoints: list,
    code: str,
    symprec: float,
    sym_reduce: bool,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    kpath_concrete: list
) -> np.ndarray:
    """
    Builds and returns the dynamical matrix through Phonopy

    Parameters
    ----------
    structure: Structure
        Fully optimized input structure to use for making phonon
    force_constants: ForceConstants
        List of the force constants
    supercell: np.ndarray
        array to describe supercell
    qpoints: list
        list of q-points to calculate dynamic matrix at
    code: str
        code to perform the computations
    symprec: float
        precision to determine symmetry
    sym_reduce: bool
        if True, symmetry will be used to generate displacements
    use_symmetrized_structure: str or None
        Primitive, conventional, or None
    kpath_scheme: str
        scheme to generate kpath
    kpath_concrete: list
        list of paths
    """
    # Set Phonopy object's force constants
    phonon = make_phonon(structure = structure,
                        supercell = supercell,
                        code = code,
                        symprec = symprec,
                        sym_reduce = sym_reduce,
                        use_symmetrized_structure = use_symmetrized_structure,
                        kpath_scheme = kpath_scheme,
                        force_constants=force_constants)

    # Convert qpoints into array of 3-element arrays
    qpoints, _ = get_band_qpoints_and_path_connections(kpath_concrete)
    q_vectors = []
    for i in range(len(qpoints)):
        for j in range(len(qpoints[i])):
            q_vectors.append(qpoints[i][j])
    q_vectors = np.array(q_vectors)

    # To get dynamical matrix
    phonon.run_qpoints(q_vectors)
    dyn_mat = phonon.dynamical_matrix.dynamical_matrix
    return dyn_mat

@job
def get_emode_efreq(
    dynamical_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Calculate the eigenmodes and eigenfrequencies of a dynamical matrix.

    Parameters
    ----------
    dynamical_matrix: np.ndarray
        Dynamical matrix to be diagonalized
    """
    eig_freq, eig_mode = np.linalg.eigh(dynamical_matrix)
    eig_freq = np.sqrt(eig_freq)
    return eig_freq, eig_mode

@job
def displace_structure(
    structure: Structure,
    supercell: np.ndarray,
    code: str,
    symprec: float,
    sym_reduce: bool,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    force_constants: ForceConstants = None,
    temp: float = 300,
) -> np.ndarray:
    """
    Calculate the displaced structure (N x 3 matrix).
    Displaced supercell = Original supercell + Displacements

    Parameters
    ---------
    structure: Structure
        Structure to displace
    supercell: np.ndarray
        Supercell matrix for the structure (undisplaced)
    code: str
        Code to perform the computations
    symprec: float
        Precision to determine symmetry
    sym_reduce: bool
        If True, symmetry will be used to generate displacements
    use_symmetrized_structure: str or None
        Primitive, conventional, or None
    kpath_scheme: str
        Scheme to generate kpath
    force_constants: ForceConstants
        Force constants for the structure
    temp: float
        Temperature (in K) to displace structure at
    """
    phonon = make_phonon(structure, supercell, code, symprec, sym_reduce, use_symmetrized_structure, kpath_scheme, force_constants)
    supercell_structure = get_pmg_structure(phonon.supercell)
    # Make phonon where supercell and unit cell are both the correct supercell for the structure
    phonon = make_phonon(supercell_structure, np.eye(3), code, symprec, False, "primitive", "", force_constants)
    hz_to_THz_factor = 10**(-12)
    
    # Boltzmann constant in THz/K
    k_b = physical_constants["Boltzmann constant in Hz/K"][0] * hz_to_THz_factor

    positions = supercell_structure.cart_coords
    disp = np.zeros(positions.shape)

    phonon.run_qpoints([[0,0,0]])
    eig_val, eig_vec = np.linalg.eigh(
        phonon.dynamical_matrix.dynamical_matrix
    )
    freqs = phonon.get_frequencies([0,0,0])
    eig_val = np.sqrt(eig_val)
    inv_sqrt_mass = np.array([site.species.weight for site in supercell_structure.sites])**(-0.5)
    for s in range(3, len(eig_val)):
        zeta = (-1) ** s
        mean_amp = np.sqrt(2 * k_b * temp) / freqs[s] * zeta
        disp += eig_vec[s, :].reshape((-1, 3)).real * mean_amp

    # Normalize displacements by square root of masses
    for ii in range(disp.shape[0]):
        positions[ii, :] += disp[ii, :] * inv_sqrt_mass[ii]
    return positions

# Idea: Run run_phonon_displacements (phonon version) with structure = get_pmg_structure(displaced supercell),
# supercell_matrix = displaced supercell, and displacements = 0
@job
def run_phonon_displacements_anharm(
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
    }
    phonon_job_kwargs = {}
    if prev_dir is not None and prev_dir_argname is not None:
        phonon_job_kwargs[prev_dir_argname] = prev_dir


@job
def get_anharmonic_force(
    original_structure: Structure,
    displaced_positions: np.ndarray,
    force_constants: ForceConstants,
    displacements: list[list[Vector3D]],
    DFT_forces: list[list[Vector3D]],
) -> np.ndarray:
    """
    # TODO: Fix docstring
    Uses DFT calculated forces ( F^DFT ) and harmonic approximation forces ( F^(2) )
    to find the anharmonic force via F^A = F^DFT - F^(2)

    Parameters
    ----------
    structure: Structure
        Fully optimized input structure to use for making phonon
    supercell: np.ndarray
        array to describe supercell
    displacement: float
        displacement in Angstrom
    code: str
        code to perform the computations
    symprec: float
        precision to determine symmetry
    sym_reduce: bool
        if True, symmetry will be used to generate displacements
    use_symmetrized_structure: str or None
        Primitive, conventional, or None
    kpath_scheme:
        scheme to generate kpath
    DFT_forces: list[np.ndarray]
        Matrix of DFT_forces
    """
    # Generate the displaced structure in Pymatgen format
    lattice = original_structure.lattice
    species = original_structure.species
    displaced_structure = Structure(lattice, species, displaced_positions)


    fc = np.array(force_constants)
    DFT_forces_np = [np.array(DFT_force) for DFT_force in DFT_forces]
    harmonic_forces = [-fc @ np.array(disp) for disp in displacements]
    anharmonic_force = [DFT_force_np - harmonic_force for DFT_force_np, harmonic_force in zip(DFT_forces_np, harmonic_forces)]
    return anharmonic_force

@job
def calc_sigma_A_oneshot(
    anharmonic_force: np.ndarray,
    DFT_forces: list[np.ndarray]
) -> float:
    """
    Calculates the one-shot approximation of sigma_A as the RMSE of the harmonic model 
    divided by the standard deviation of the force distribution.

    Parameters
    ----------
    anharmonic_force: np.ndarray
        Matrix of anharmonic forces
    DFT_forces: list[np.ndarray]
        Matrix of DFT forces
    """
    DFT_forces_np = np.array(DFT_forces)
    return np.std(np.ndarray.flatten(anharmonic_force))/np.std(np.ndarray.flatten(DFT_forces_np))