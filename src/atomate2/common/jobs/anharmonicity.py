"""Jobs for running anharmonicity quantification."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Optional

from jobflow import Flow, Response, job
from phonopy import Phonopy
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)
from atomate2.common.jobs.phonons import generate_phonon_displacements, run_phonon_displacements

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

@job(data=[Structure])
def generate_phonon_displacements(
    structure: Structure,
    supercell_matrix: np.array,
    displacement: float,
    sym_reduce: bool,
    symprec: float,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    code: str,
) -> tuple[list[Structure], Phonopy]:
    """
    Generate displaced structures with phonopy.

    Parameters
    ----------
    structure: Structure object
        Fully optimized input structure for phonon run
    supercell_matrix: np.array
        array to describe supercell matrix
    displacement: float
        displacement in Angstrom
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
    cell = get_phonopy_structure(structure)
    factor = get_factor(code)

    # a bit of code repetition here as I currently
    # do not see how to pass the phonopy object?
    if use_symmetrized_structure == "primitive" and kpath_scheme != "seekpath":
        primitive_matrix: list[list[float]] | str = [
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

    supercells = phonon.supercells_with_displacements

    return ([get_pmg_structure(cell) for cell in supercells], phonon)

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

@job  
def build_dyn_mat(
    phonon: Phonopy
) -> Optional[np.ndarray]:
    """
    Gets the dynamical matrix of a Phonopy object
    
    Parameters
    ----------
    phonon: Phonopy
        Phonopy object to find dynamical matrix for
    """

    # Gets dynamical matrix in form of DynamicalMatrix class (defined by Phonopy)
    dynamical_matrix_obj = phonon.dynamical_matrix()
    dynamical_matrix = dynamical_matrix_obj.dynamical_matrix()
    return dynamical_matrix

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
    phonon: Phonopy,
    DFT_forces: list[np.ndarray]
) -> np.ndarray:
    """
    Uses DFT calculated forces ( F^DFT ) and harmonic approximation forces ( F^(2) )
    to find the anharmonic force via F^A = F^DFT - F^(2)

    Parameters
    ----------
    phonon: Phonopy
        The phonon object to get F^(2) from
    DFT_forces: list[np.ndarray]
        Matrix of DFT_forces
    """

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