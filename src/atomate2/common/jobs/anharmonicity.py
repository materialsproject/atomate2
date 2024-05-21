"""Jobs for running anharmonicity quantification."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Optional

from jobflow import Flow, Response, job
from phonopy import Phonopy
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from pymatgen.core import Structure, Lattice
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
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

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
        phonon.force_constants = force_constants.force_constants
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
    # TODO: Change docstring
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

    eig_freq, eig_mode = np.linalg.eigh(dyn_mat)
    eig_freq = np.sqrt(eig_freq)
    print(eig_freq)
    print(eig_mode)
    return (eig_freq, eig_mode)


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
    return (eig_freq, eig_mode)

@job
def displace_structure(
    structure: Structure,
    supercell: np.ndarray,
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
    supercell_structure = structure.make_supercell(scaling_matrix=supercell, in_place=False)
    
    positions = supercell_structure.cart_coords
    disp = np.zeros(positions.shape)
    
    force_constants_2D = np.array(force_constants.force_constants).swapaxes(1, 2).reshape(2 * (len(supercell_structure) * 3,))
    masses = np.array([site.species.weight for site in supercell_structure.sites])
    rminv = (masses**-0.5).repeat(3)
    dynamical_matrix = force_constants_2D * rminv[:, None] * rminv[None, :]

    eig_val, eig_vec = np.linalg.eigh(
        dynamical_matrix
    )
    eig_val = np.sqrt(eig_val[3:])
    X_acs = eig_vec[:, 3:].reshape((-1, 3, len(eig_val)))

    # gauge eigenvectors: largest value always positive
    for ii in range(X_acs.shape[-1]):
        vec = X_acs[:, :, ii]
        max_arg = np.argmax(abs(vec))
        X_acs[:, :, ii] *= np.sign(vec.flat[max_arg])

    inv_sqrt_mass = masses**(-0.5)
    zetas = (-1) ** np.arange(len(eig_val))
    A_s = np.sqrt(temp) / eig_val * zetas
    disp = (A_s * X_acs).sum(axis=2) * inv_sqrt_mass[:, None]

    print("\n\n\n", disp, "\n\n")

    # Normalize displacements by square root of masses
    for ii in range(disp.shape[0]):
        positions[ii, :] += disp[ii, :] * inv_sqrt_mass[ii]

    return Structure(
        lattice=supercell_structure.lattice, 
        species=supercell_structure.species, 
        coords=positions,
        coords_are_cartesian=True,
    )

# Idea: Run run_phonon_displacements (phonon version) with structure = get_pmg_structure(displaced supercell),
# supercell_matrix = displaced supercell, and displacements = 0
@job
def run_phonon_displacements_anharm(
    displacements: list[Structure],
    structure: Structure,
    supercell_matrix: Matrix3D,
    force_eval_maker: BaseVaspMaker | ForceFieldStaticMaker | BaseAimsMaker = None,
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
    force_eval_maker : .BaseVaspMaker or .ForceFieldStaticMaker or .BaseAimsMaker
        A maker to use to generate dispacement calculations
    prev_dir: str or Path
        The previous working directory
    prev_dir_argname: str
        argument name for the prev_dir variable
    socket: bool
        If True use the socket-io interface to increase performance
    """
    force_eval_jobs = []
    outputs: dict[str, list] = {
        "displacement_number": [],
        "forces": [],
        "uuids": [],
        "dirs": [],
    }
    force_eval_job_kwargs = {}
    if prev_dir is not None and prev_dir_argname is not None:
        force_eval_job_kwargs[prev_dir_argname] = prev_dir


@job
def get_sigma_a(
    force_constants: ForceConstants,
    structure: Structure,
    supercell_matrix: Matrix3D,
    displaced_structures: list,
) -> float:
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
    print(displaced_structures)
    supercell_structure = structure.make_supercell(
        scaling_matrix = supercell_matrix,
        in_place = False
    )
    force_constants_2D = np.swapaxes(
        force_constants.force_constants,
        1,
        2,
    ).reshape(2 * (len(supercell_structure) * 3,))
    displacements = [np.array(disp_data) for disp_data in displaced_structures["displacements"]]
    harmonic_forces = [(-force_constants_2D @ displacement.flatten()).reshape((-1, 3)) for displacement in displacements]

    # anharmonic_force = [DFT_force_np - harmonic_force for DFT_force_np, harmonic_force in zip(DFT_forces_np, harmonic_forces)]
    dft_forces = [np.array(disp_data) for disp_data in displaced_structures["forces"]]
    anharmonic_forces = [
        dft_force - harmonic_force for dft_force, harmonic_force in zip(dft_forces, harmonic_forces)  
    ]

    return np.std(anharmonic_forces) / np.std(dft_forces)

@job(data=["forces", "displaced_structures"])
def run_displacements(
    displacements: list[Structure],
    structure: Structure,
    supercell_matrix: Matrix3D,
    force_eval_maker: BaseVaspMaker | ForceFieldStaticMaker | BaseAimsMaker = None,
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
    force_eval_maker : .BaseVaspMaker or .ForceFieldStaticMaker or .BaseAimsMaker
        A maker to use to generate dispacement calculations
    prev_dir: str or Path
        The previous working directory
    prev_dir_argname: str
        argument name for the prev_dir variable
    socket: bool
        If True use the socket-io interface to increase performance
    """
    force_eval_jobs = []
    outputs: dict[str, list] = {
        "displacements": [],
        "forces": [],
        "uuids": [],
        "dirs": [],
    }
    force_eval_job_kwargs = {}
    if prev_dir is not None and prev_dir_argname is not None:
        force_eval_job_kwargs[prev_dir_argname] = prev_dir

    reference_structure = structure.make_supercell(
        scaling_matrix = supercell_matrix,
        in_place = False
    )
    if socket:
        force_eval_job = force_eval_maker.make(displacements, **force_eval_job_kwargs)
        info = {
            "original_structure": structure,
            "supercell_matrix": supercell_matrix,
            "displaced_structures": displacements,
        }
        force_eval_job.update_maker_kwargs(
            {"_set": {"write_additional_data->phonon_info:json": info}}, dict_mod=True
        )
        force_eval_jobs.append(force_eval_job)
        outputs["displacements"] = [displacement.cart_coords - reference_structure.cart_coords for displacement in displacements]
        outputs["uuids"] = [force_eval_job.output.uuid] * len(displacements)
        outputs["dirs"] = [force_eval_job.output.dir_name] * len(displacements)
        outputs["forces"] = force_eval_job.output.output.all_forces
    else:
        for idx, displacement in enumerate(displacements):
            if prev_dir is not None:
                force_eval_job = force_eval_maker.make(displacement, prev_dir=prev_dir)
            else:
                force_eval_job = force_eval_maker.make(displacement)
            force_eval_job.append_name(f" {idx + 1}/{len(displacements)}")

            # we will add some meta data
            info = {
                "original_structure": structure,
                "supercell_matrix": supercell_matrix,
                "displaced_structure": displacement,
            }
            with contextlib.suppress(Exception):
                force_eval_job.update_maker_kwargs(
                    {"_set": {"write_additional_data->phonon_info:json": info}},
                    dict_mod=True,
                )
            force_eval_jobs.append(force_eval_job)
            outputs["displacements"].append(displacement.cart_coords - reference_structure.cart_coords)
            outputs["uuids"].append(force_eval_job.output.uuid)
            outputs["dirs"].append(force_eval_job.output.dir_name)
            outputs["forces"].append(force_eval_job.output.output.forces)

    displacement_flow = Flow(force_eval_jobs, outputs)
    return Response(replace=displacement_flow)

# @job
# def calc_sigma_A_oneshot(
#     anharmonic_force: np.ndarray,
#     DFT_forces: list[np.ndarray]
# ) -> float:
#     """
#     Calculates the one-shot approximation of sigma_A as the RMSE of the harmonic model 
#     divided by the standard deviation of the force distribution.

#     Parameters
#     ----------
#     anharmonic_force: np.ndarray
#         Matrix of anharmonic forces
#     DFT_forces: list[np.ndarray]
#         Matrix of DFT forces
#     """
#     anharmonic_force = np.array(anharmonic_force)
#     DFT_forces_np = np.array(DFT_forces)
#     return np.std(np.ndarray.flatten(anharmonic_force))/np.std(np.ndarray.flatten(DFT_forces_np))