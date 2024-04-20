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
    from emmet.core.math import Matrix3D

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
    return phonon

@job 
def build_dyn_mat(
    structure: Structure,
    force_constants: list,
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
    force_constants: list
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
    kpath_scheme:
        scheme to generate kpath
    kpath_concrete:
        list of paths
    """
    # Set Phonopy object's force constants
    phonon = make_phonon(structure = structure,
                        supercell = supercell,
                        code = code,
                        symprec = symprec,
                        sym_reduce = sym_reduce,
                        use_symmetrized_structure = use_symmetrized_structure,
                        kpath_scheme = kpath_scheme)
    phonon.force_constants = force_constants

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
    phonon: Phonopy,
    eig_vec: np.ndarray,
    eig_val: np.ndarray,
    temp: float = 300,
) -> np.ndarray:
    """
    Calculate the displaced structure (N x 3 matrix).

    Paramters
    ---------
    phonon: Phonopy
        Phonopy object containing undisplaced structure
    eig_vec: np.ndarray
        Eigenmodes from diagonalized dynamical matrix
    eig_val: np.ndarray
        Eigenfrequencies from diagonalized dynamical matrix
    temp: float
        Temperature to calculate displacement vector at (default is 300 K)
    """
    hz_to_THz_factor = 10**(-12)
    # Boltzmann constant in THz/K
    k_b = physical_constants["Boltzmann constant in Hz/K"][0] * hz_to_THz_factor
    displacements = np.zeros(shape = np.shape(phonon.supercell))
    dir = ["x", "y", "z"]

    # Get displacement vectors for each atom and each direction
    for atom in range(len(phonon.masses)):
        for alpha in range(len(dir)):
            inv_sqrt_mass = (phonon.masses[atom])**(-1/2)
            for s in range(len(displacements)):
                zeta = (-1)**s 
                mean_amp = np.sqrt(2*k_b*temp)/eig_val[s]
                e_sI_alpha = eig_vec[:, 3*atom + alpha]
                displacements[atom, alpha] += zeta * mean_amp * e_sI_alpha
            displacements[atom, alpha] *= inv_sqrt_mass
    
    # Displace the supercell positions
    displaced_supercell = phonon.supercell - displacements

    return displaced_supercell

@job
def get_anharmonic_force(
    structure: Structure,
    supercell: np.ndarray,
    displacement: float,
    code: str,
    symprec: float,
    sym_reduce: bool,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    DFT_forces: list[np.ndarray]
) -> np.ndarray:
    """
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
    phonon = make_phonon(structure = structure,
                        supercell = supercell,
                        code = code,
                        symprec = symprec,
                        sym_reduce = sym_reduce,
                        use_symmetrized_structure = use_symmetrized_structure,
                        kpath_scheme = kpath_scheme)
    
    disp_job = generate_phonon_displacements(structure = structure,
                                             supercell_matrix = supercell,
                                             displacement = displacement,
                                             sym_reduce = sym_reduce,
                                             symprec = symprec,
                                             use_symmetrized_structure = use_symmetrized_structure,
                                             kpath_scheme = kpath_scheme,
                                             code = code)
    disp_response = run_locally(disp_job, ensure_success=True)
    displacements = disp_response[disp_job.uuid][1].output

    run_displacements = run_phonon_displacements(displacements = displacements,
                                                 structure = structure,
                                                 supercell_matrix = supercell)
    responses = run_locally(run_displacements, create_folders=True, ensure_success=True)
    #TODO: Figure out how to run run_phonon_displacements
    DFT_forces_np = np.array(DFT_forces)
    harmonic_force = phonon.forces
    anharmonic_force = DFT_forces_np - harmonic_force
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