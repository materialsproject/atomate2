"""Jobs for running anharmonicity quantification."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Flow, Response, job
from phonopy import Phonopy
from pymatgen.core import Structure
from pymatgen.core.units import kb
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2.aims.schemas.calculation import Calculation
from atomate2.aims.utils.units import omegaToTHz
from atomate2.common.schemas.anharmonicity import AnharmonicityDoc

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from atomate2.aims.jobs.base import BaseAimsMaker
    from atomate2.common.schemas.phonons import ForceConstants, PhononBSDOSDoc
    from atomate2.forcefields.jobs import ForceFieldStaticMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker


logger = logging.getLogger(__name__)


class ImaginaryModeError(Exception):
    """Exception raised when an imaginary mode is detected.

    Attributes
    ----------
    largest_mode:
        The largest eigenmode to check the sign of
    message:
        Explanation of the error
    """

    def __init__(self, largest_mode: float) -> None:
        self.largest_mode = largest_mode
        self.message = f"""Structure has imaginary modes: the largest optical
                        eigenmode {largest_mode} < 0.0001"""
        super().__init__(self.message)


@job
def get_phonon_supercell(
    structure: Structure,
    supercell_matrix: np.ndarray,
) -> Structure:
    """Get the phonon supercell of a structure.

    Parameters
    ----------
    structure: Structure
        The structure to get phonon supercell for
    supercell_matrix: np.ndarray
        Supercell matrix for the phonon

    Returns
    -------
    Structure
        The phonopy structure
    """
    cell = get_phonopy_structure(structure)
    phonon = Phonopy(
        cell,
        supercell_matrix,
    )
    return get_pmg_structure(phonon.supercell)


def get_sigma_per_site(
    structure: Structure,
    forces_dft: np.ndarray,
    forces_harmonic: np.ndarray,
) -> list[tuple]:
    """Compute a site-resolved sigma^A.

    This will calculate a sigma^A value for every
    site in the structure.

    Parameters
    ----------
    structure: Structure
        The structure to use for the calculation
    forces_dft: np.ndarray
        DFT calculated forces
    forces_harmonic: np.ndarray
        Harmonic approximation of the forces

    Returns
    -------
    list[tuple]
        List of tuples in the form
        ({wyckoff symbol: sites}, sigma^A) for
        all the sites in the structure
    """
    # Ensure that DFT and harmonic forces are in np format
    forces_dft = np.array(forces_dft)
    forces_harmonic = np.array(forces_harmonic)

    # Check shapes of forces
    if len(np.shape(forces_dft)) == 2:
        forces_dft = np.expand_dims(forces_dft, axis=0)
        forces_harmonic = np.expand_dims(forces_harmonic, axis=0)

    sg_analyzer = SpacegroupAnalyzer(structure)
    sym_dataset = sg_analyzer.get_symmetry_dataset()
    wycks = np.array(sym_dataset.wyckoffs)
    sites = np.array(structure.sites)

    if np.shape(forces_dft)[1] == 3 * structure.num_sites:
        sites = sites.repeat(3)
        wycks = wycks.repeat(3)

    unique_sites = np.unique(wycks)
    site_to_wyckoff: dict = {wyck: [] for wyck in unique_sites}
    for idx, val in enumerate(wycks):
        site_to_wyckoff[val].append(sites[idx])

    sigma_sites = []
    for s in unique_sites:
        mask = wycks == s
        f_dft = forces_dft[:, mask]
        f_ha = forces_harmonic[:, mask]
        f_anharm = f_dft - f_ha
        sigma_sites.append(calc_sigma_a(f_anharm, f_dft))

    return [
        (
            {
                list(site_to_wyckoff.keys())[i]: site_to_wyckoff[
                    list(site_to_wyckoff.keys())[i]
                ]
            },
            sigma_sites[i],
        )
        for i in range(len(sigma_sites))
    ]


def get_sigma_per_element(
    structure: Structure,
    forces_dft: np.ndarray,
    forces_harmonic: np.ndarray,
) -> list[tuple[str, float]]:
    """Compute an element-resolved sigma^A.

    Every element type in the structure will have a
    sigma^A value calculated for it.

    Parameters
    ----------
    structure: Structure
        The structure to use for the calculation
    forces_dft: np.ndarray
        DFT calculated forces
    forces_harmonic: np.ndarray
        Harmonic approximation of the forces

    Returns
    -------
    list[tuple[str, float]]
        List of tuples in the form (atom symbol, sigma^A)
        for all the atoms in the structure.
    """
    # Ensure that DFT and harmonic forces are in np format
    forces_dft = np.array(forces_dft)
    forces_harmonic = np.array(forces_harmonic)

    # Check shapes of forces
    if len(np.shape(forces_dft)) == 2:
        forces_dft = np.expand_dims(forces_dft, axis=0)
        forces_harmonic = np.expand_dims(forces_harmonic, axis=0)

    atom_numbers = np.array([site.specie.number for site in structure.sites])

    if np.shape(forces_dft)[1] == 3 * structure.num_sites:
        atom_numbers = atom_numbers.repeat(3)

    symbols = np.array([site.specie.name for site in structure.sites])
    unique_atoms = np.unique(atom_numbers)

    sigma_atom = []
    unique_symbols = []
    for u in unique_atoms:
        # Find atoms of type u in the structure
        mask = atom_numbers == u
        # Add symbol for atom u to list of examined atoms
        unique_symbols.append(symbols[mask][0])
        # Take forces belonging to this atom
        f_dft = forces_dft[:, mask]
        f_ha = forces_harmonic[:, mask]
        f_anharm = f_dft - f_ha
        # Calculate sigma^A for this atom
        sigma_atom.append(calc_sigma_a(f_anharm, f_dft))

    return [(unique_symbols[i], sigma_atom[i]) for i in range(len(sigma_atom))]


def box_muller(
    eig_vals: np.ndarray,
    eig_vecs: np.ndarray,
    temp: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Get a normally distributed random variable displacement.

    Uses the Box-Muller transform (https://en.wikipedia.org/wiki/Box-Muller_transform)

    Parameters
    ----------
    eig_vals: np.ndarray
        Vector of harmonic eigenvalues (with first 3 removed for translational modes)
    eig_vecs: np.ndarray
        Matrix of harmonic eigenvectors (with first 3 removed for translational modes)
    temp: float
        Temperature in K to find velocity and displacement at
    rng: np.random.Generator
        Seeded random number generator

    Returns
    -------
    np.ndarray
        Array with iid normally distributed displacements
    """
    n_eigvals = eig_vals.shape[0]
    spread = np.sqrt(-2.0 * np.log(1.0 - rng.random(size=n_eigvals)))

    # Assign amplitudes (A_s) and phases (phi_s)
    a_s = spread * (np.sqrt(temp * kb) / eig_vals)
    phi_s = 2.0 * np.pi * rng.random(size=n_eigvals)

    # Get displacement (not normalized by sqrt(masses) yet)
    return (a_s * np.cos(phi_s) * eig_vecs).sum(axis=2)


@job
def displace_structure(
    phonon_supercell: Structure,
    force_constants: ForceConstants = None,
    temp: float = 300,
    one_shot: bool = True,
    seed: int | None = None,
    mode_resolved: bool = False,
    n_samples: int = 1,
) -> list[Structure]:
    """Calculate the displaced structure.

    Procedure defined in doi.org/10.1103/PhysRevB.94.075125.
    Displaced supercell = Original supercell + Displacements

    Parameters
    ----------
    phonon_supercell: Structure
        Supercell to distort
    force_constants: ForceConstants
        Force constants calculated from phonopy
    temp: float
        Temperature (in K) to displace structure at
    one_shot: bool
        If false, uses a normally distributed random number for zeta.
        The default is true.
    seed: int | None
        Seed to use for the random number generator
        (only used if one_shot_approx == False)
    mode_resolved: bool
        If true, calculate the mode-resolved sigma^A. This is false by default.
    n_samples: int
        Number of samples to generate (must be >= 1).
        An error is raised if n_samples == 1 and mode_resolved == True

    Returns
    -------
    list[Structure]
        List of displaced Pymatgen structures
    """
    if n_samples != 1 and one_shot:
        raise ValueError(
            f"One-shot can't be used unless n_sample == 1 not {n_samples}."
        )
    if n_samples < 1:
        raise ValueError(f"n_sample must be >= 1, not {n_samples}.")
    if n_samples == 1 and mode_resolved:
        raise ValueError("n_sample must be > 1 when mode_resolved == True.")
    rng = np.random.default_rng(seed)

    coords = phonon_supercell.cart_coords
    disp = np.zeros(coords.shape)

    masses = np.array([site.species.weight for site in phonon_supercell.sites])
    dynamical_matrix = build_dynmat(force_constants, phonon_supercell)
    eig_val, eig_vec = get_eigens(dynamical_matrix)

    # Check for imaginary modes
    if eig_val[3] < 0.0001:
        raise ImaginaryModeError(eig_val[3])
    eig_val = eig_val[3:]
    x_acs = eig_vec[:, 3:].reshape((-1, 3, len(eig_val)))

    # gauge eigenvectors: largest value always positive
    for ii in range(x_acs.shape[-1]):
        vec = x_acs[:, :, ii]
        max_arg = np.argmax(abs(vec))
        x_acs[:, :, ii] *= np.sign(vec.flat[max_arg])

    inv_sqrt_mass = masses ** (-0.5)
    if one_shot:
        zetas = (-1) ** np.arange(len(eig_val))
        a_s = np.sqrt(temp * kb) / eig_val * zetas
        disp = (a_s * x_acs).sum(axis=2) * inv_sqrt_mass[:, None]
        return [
            Structure(
                lattice=phonon_supercell.lattice,
                species=phonon_supercell.species,
                coords=coords + disp,
                coords_are_cartesian=True,
            )
        ]

    samples = []
    for _ in range(n_samples):
        disp = box_muller(eig_val, x_acs, temp, rng) * inv_sqrt_mass[:, None]
        samples.append(
            Structure(
                lattice=phonon_supercell.lattice,
                species=phonon_supercell.species,
                coords=coords + disp,
                coords_are_cartesian=True,
            )
        )
    return samples


def build_dynmat(
    force_constants: ForceConstants,
    structure: Structure,
) -> np.ndarray:
    """Build the dynamical matrix.

    Parameters
    ----------
    force_constants: ForceConstants
        Force constants calculated by Phonopy
    structure: Structure
        Structure as a Pymatgen Structure object

    Returns
    -------
    np.ndarray
        The dynamical matrix
    """
    force_constants_2d = (
        np.array(force_constants.force_constants)
        .swapaxes(1, 2)
        .reshape(2 * (len(structure) * 3,))
    )
    masses = np.array([site.species.weight for site in structure.sites])
    rminv = (masses**-0.5).repeat(3)
    return force_constants_2d * rminv[:, None] * rminv[None, :]


def get_eigens(dynmat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the eigenmodes and eigenfrequencies from the dynamical matrix.

    Parameters
    ----------
    dynmat: np.ndarray
        Dynamical matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple with first element being the array of eigenmodes and the second being the
        matrix of eigenfrequencies
    """
    eig_val, eig_vec = np.linalg.eigh(dynmat)
    eig_val = np.sqrt(eig_val)
    return (eig_val, eig_vec)


def get_sigma_a_per_mode(
    force_constants: ForceConstants,
    structure: Structure,
    dft_forces: np.ndarray,
    harmonic_forces: np.ndarray,
) -> list[tuple[float, float]]:
    """Calculate sigma^A for each mode.

    Parameters
    ----------
    force_constants: ForceConstants
        Force constants calculated by Phonopy
    structure: Structure
        Structure as a Pymatgen Structure object
    dft_forces: np.ndarray
        Array of DFT forces
    harmonic_forces: np.ndarray
        Array of harmonically approximated forces

    Returns
    -------
    list[tuple[float, float]]
        List of tuples in the form (mode frequency in THz, Sigma^A)
        for all the modes in the structure
    """
    # Projecting the forces
    dynamical_matrix = build_dynmat(force_constants, structure)
    eig_val, eig_vec = get_eigens(dynamical_matrix)
    eig_val = eig_val[3:] * omegaToTHz
    # Projection matrix P
    p = eig_vec.T
    masses = np.array([site.species.weight for site in structure.sites])
    m = masses ** (-0.5)
    # Project the forces
    dft_forces = m.reshape((len(structure), 1)) * dft_forces
    dft_proj = np.array([p @ (f.flatten()) for f in np.array(dft_forces)])[:, 3:]
    harmonic_forces = m.reshape((len(structure), 1)) * harmonic_forces
    harmonic_proj = np.array([p @ (f.flatten()) for f in np.array(harmonic_forces)])[
        :, 3:
    ]
    # Calculate sigma^A for each mode
    mode_sigma_vals = []
    for mode in np.unique(eig_val):
        dft_vals = np.array(
            [dft_proj[:, idx] for idx in range(len(eig_val)) if eig_val[idx] == mode]
        ).flatten()
        harmonic_vals = np.array(
            [
                harmonic_proj[:, idx]
                for idx in range(len(eig_val))
                if eig_val[idx] == mode
            ]
        ).flatten()
        anharmonic_vals = [
            dft_force - harmonic_force
            for dft_force, harmonic_force in zip(dft_vals, harmonic_vals)
        ]
        sigma_a = np.std(anharmonic_vals) / np.std(dft_vals)
        mode_sigma_vals.append((mode, sigma_a))

    return mode_sigma_vals


@job
def get_forces(
    force_constants: ForceConstants,
    phonon_supercell: Structure,
    displaced_structures: dict[str, list],
) -> list[np.ndarray]:
    """Calculate the DFT forces and harmonic forces.

    Parameters
    ----------
    force_constants: ForceConstants
        The force constants calculated by phonopy
    phonon_supercell: Structure
        The supercell used for the phonon calculation
    displaced_structures: dict[str, list]
        The output of run_displacements

    Returns
    -------
    list[np.ndarray]
        List of forces in the form [DFT forces, harmonic forces]
    """
    force_constants_2d = np.swapaxes(
        force_constants.force_constants,
        1,
        2,
    ).reshape(2 * (len(phonon_supercell) * 3,))
    if isinstance(displaced_structures["coords"][0], Calculation):
        displacements = [
            disp_data.output.structure.cart_coords - phonon_supercell.cart_coords
            for disp_data in displaced_structures["coords"]
        ]
    else:
        displacements = [
            np.array(disp_data) - phonon_supercell.cart_coords
            for disp_data in displaced_structures["coords"]
        ]

    harmonic_forces = [
        (-force_constants_2d @ displacement.flatten()).reshape((-1, 3))
        for displacement in displacements
    ]

    dft_forces = [np.array(disp_data) for disp_data in displaced_structures["forces"]]

    return [dft_forces, harmonic_forces]


@job
def get_sigmas(
    dft_forces: np.ndarray,
    harmonic_forces: np.ndarray,
    force_constants: ForceConstants,
    structure: Structure,
    one_shot: bool,
    element_resolved: bool = False,
    mode_resolved: bool = False,
    site_resolved: bool = False,
) -> dict:
    """Get the desired sigma^A measures.

    Parameters
    ----------
    dft_forces: np.ndarray
        Array of DFT forces
    harmonic_forces: np.ndarray
        Array of forces from harmonic model
    force_constants: ForceConstants
        ForceConstants object containing a list of force constants
    structure: Structure
        The structure to calculate sigma^A on
    one_shot: bool
        If true, find the one-shot approximation. If false, find the
        full sigma^A number.
    element_resolved: bool
        If true, resolve sigma^A to the different elements.
        Default is false.
    mode_resolved: bool
        If true, resolve sigma^A to the different modes.
        Default is false.
    site_resolved: bool
        If true, resolve sigma^A to the different sites.
        Default is false.

    Returns
    -------
    dict[str, Union[float, list]]
        Dictionary in the form {sigma^A type: float/list with sigma^A values}
        that contains all the sigma^A values
    """
    dft_forces = np.array(dft_forces)
    harmonic_forces = np.array(harmonic_forces)
    anharmonic_forces = np.array(
        [
            dft_force - harmonic_force
            for dft_force, harmonic_force in zip(dft_forces, harmonic_forces)
        ]
    )

    sigma_dict: dict[str, Any] = {}

    if mode_resolved:
        sigma_dict["mode-resolved"] = get_sigma_a_per_mode(
            force_constants,
            structure,
            dft_forces,
            harmonic_forces,
        )
    if element_resolved:
        sigma_dict["element-resolved"] = get_sigma_per_element(
            structure,
            dft_forces,
            harmonic_forces,
        )
    if site_resolved:
        sigma_dict["site-resolved"] = get_sigma_per_site(
            structure,
            dft_forces,
            harmonic_forces,
        )
    if one_shot:
        sigma_dict["one-shot"] = calc_sigma_a(anharmonic_forces, dft_forces)
    else:
        sigma_dict["full"] = calc_sigma_a(anharmonic_forces, dft_forces)

    return sigma_dict


def calc_sigma_a(
    anharmonic_forces: np.ndarray,
    dft_forces: np.ndarray,
) -> float:
    """Calculate sigma^A.

    Parameters
    ----------
    anharmonic_forces: np.ndarray
        Array of anharmonic forces (dft - harmonic)
    dft_forces: np.ndarray
        Array of DFT forces

    Returns
    -------
    float
        sigma^A number
    """
    return np.std(anharmonic_forces) / np.std(dft_forces)


@job(data=["forces", "displaced_structures"])
def run_displacements(
    displacements: list[Structure],
    phonon_supercell: Structure,
    force_eval_maker: BaseVaspMaker | ForceFieldStaticMaker | BaseAimsMaker = None,
    prev_dir: str | Path = None,
    prev_dir_argname: str = None,
    socket: bool = False,
) -> Flow:
    """Run displaced structures.

    Note, this job will replace itself with N displacement calculations,
    or a single socket calculation for all displacements.

    Parameters
    ----------
    displacements: Sequence
        All displacements to calculate
    phonon_supercell: Structure
        The supercell used for the phonon calculation
    force_eval_maker : .BaseVaspMaker or .ForceFieldStaticMaker or .BaseAimsMaker
        A maker to use to generate dispacement calculations
    prev_dir: str or Path
        The previous working directory
    prev_dir_argname: str
        argument name for the prev_dir variable
    socket: bool
        If True use the socket-io interface to increase performance

    Returns
    -------
    Flow
        Flow calculating DFT forces on displaced structures
    """
    force_eval_jobs = []
    outputs: dict[str, list] = {
        "coords": [],
        "forces": [],
        "uuids": [],
        "dirs": [],
    }
    force_eval_job_kwargs = {}
    if prev_dir is not None and prev_dir_argname is not None:
        force_eval_job_kwargs[prev_dir_argname] = prev_dir

    if socket:
        force_eval_job = force_eval_maker.make(displacements, **force_eval_job_kwargs)
        info = {
            "phonon_supercell": phonon_supercell,
            "displaced_structures": displacements,
        }
        force_eval_job.update_maker_kwargs(
            {"_set": {"write_additional_data->anharmonicity_info:json": info}},
            dict_mod=True,
        )
        force_eval_jobs.append(force_eval_job)
        outputs["coords"] = force_eval_job.output.calcs_reversed
        outputs["uuids"] = [force_eval_job.output.uuid] * len(displacements)
        outputs["dirs"] = [force_eval_job.output.dir_name] * len(displacements)
        outputs["forces"] = force_eval_job.output.output.all_forces
    else:
        for idx, displacement in enumerate(displacements):
            if prev_dir is not None:
                force_eval_job = force_eval_maker.make(displacement, prev_dir=prev_dir)
            else:
                force_eval_job = force_eval_maker.make(displacement)
            force_eval_job.append_name(
                f" anharmonicity quant. {idx + 1}/{len(displacements)}"
            )

            # we will add some meta data
            info = {
                "phonon_supercell": phonon_supercell,
                "displaced_structure": displacement,
            }
            with contextlib.suppress(Exception):
                force_eval_job.update_maker_kwargs(
                    {"_set": {"write_additional_data->anharmonicity_info:json": info}},
                    dict_mod=True,
                )

            force_eval_jobs.append(force_eval_job)
            outputs["coords"].append(force_eval_job.output.structure.cart_coords)
            outputs["uuids"].append(force_eval_job.output.uuid)
            outputs["dirs"].append(force_eval_job.output.dir_name)
            outputs["forces"].append(force_eval_job.output.output.forces)

    displacement_flow = Flow(force_eval_jobs, outputs)
    return Response(replace=displacement_flow)


@job
def store_results(
    sigma_dict: dict[str, list | float],
    phonon_doc: PhononBSDOSDoc,
    one_shot: bool,
    temp: float,
    n_samples: int,
    seed: int | None,
) -> AnharmonicityDoc:
    """
    Store the results in a schema object.

    Parameters
    ----------
    sigma_dict: dict[str, Union[list, float]]
        Dictionary of computed sigma^A values.
        Possible contents are full, one-shot, atom-resolved, and
        mode-resolved.
    phonon_doc: PhononBSDOSDoc
        Info from phonon workflow to be stored
    one_shot: bool
        Whether the one-shot approximation was used (true) or not (false)
    temp: float
        Temperature (in K) to displace structure at
    n_samples: int
        How many displaced structures to sample
    seed: int | None
        What random seed was used to displace structures

    Returns
    -------
    AnharmonicityDoc
        Document containing information on workflow results
    """
    return AnharmonicityDoc.from_phonon_doc_sigma(
        sigma_dict=sigma_dict,
        phonon_doc=phonon_doc,
        one_shot=one_shot,
        temp=temp,
        n_samples=n_samples,
        seed=seed,
    )
