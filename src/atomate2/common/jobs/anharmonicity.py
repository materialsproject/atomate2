"""Jobs for running anharmonicity quantification."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.random import rand
from jobflow import Flow, Response, job
from phonopy import Phonopy
from pymatgen.core import Structure
from pymatgen.core.units import kb
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from atomate2.aims.schemas.calculation import Calculation

from atomate2.common.schemas.anharmonicity import AnharmonicityDoc

if TYPE_CHECKING:
    from pathlib import Path

    from atomate2.aims.jobs.base import BaseAimsMaker
    from atomate2.common.schemas.phonons import ForceConstants, PhononBSDOSDoc
    from atomate2.forcefields.jobs import ForceFieldStaticMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker
    


logger = logging.getLogger(__name__)


@job
def get_phonon_supercell(phonon_doc: PhononBSDOSDoc) -> Structure:
    """Get the phonon supercell of a structure.

    Parameters
    ----------
    phonon_doc: PhononBSDOSDoc
        The output of a phonopy workflow

    Returns
    -------
    Structure
        The phonopy structure
    """
    cell = get_phonopy_structure(phonon_doc.structure)
    phonon = Phonopy(
        cell,
        phonon_doc.supercell_matrix,
    )
    return get_pmg_structure(phonon.supercell)

def box_muller(
    eig_vals: np.ndarray,
    eig_vecs: np.ndarray,
    temp: float,
) -> np.ndarray:
    """Get a normally distributed random variable displacement
    Uses the Box-Muller transform (https://en.wikipedia.org/wiki/Box–Muller_transform)

    Parameters
    ----------
    eig_vals: np.ndarray
        Vector of harmonic eigenvalues (with first 3 removed for translational modes)
    eig_vecs: np.ndarray
        Matrix of harmonic eigenvectors (with first 3 removed for translational modes)
    temp: float
        Temperature in K to find velocity and displacement at
    """
    n_eigvals = eig_vals.shape[0]
    spread = np.sqrt(-2.0 * np.log(1.0 - rand(n_eigvals)))

    # Assign amplitudes (A_s) and phases (phi_s)
    A_s = spread * (np.sqrt(temp * kb)/eig_vals)
    phi_s = 2.0 * np.pi * rand(n_eigvals)

    # Get displacement (not normalized by sqrt(masses) yet)
    d_ac = (A_s * np.cos(phi_s) * eig_vecs).sum(axis=2)

    return d_ac

@job
def displace_structure(
    phonon_supercell: Structure,
    force_constants: ForceConstants = None,
    temp: float = 300,
    one_shot: bool = True
) -> Structure:
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
    """
    coords = phonon_supercell.cart_coords
    disp = np.zeros(coords.shape)

    force_constants_2d = (
        np.array(force_constants.force_constants)
        .swapaxes(1, 2)
        .reshape(2 * (len(phonon_supercell) * 3,))
    )

    masses = np.array([site.species.weight for site in phonon_supercell.sites])
    rminv = (masses**-0.5).repeat(3)
    dynamical_matrix = force_constants_2d * rminv[:, None] * rminv[None, :]

    eig_val, eig_vec = np.linalg.eigh(dynamical_matrix)
    eig_val = np.sqrt(eig_val[3:])
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
    elif not one_shot:
        disp = box_muller(eig_val, x_acs, temp) * inv_sqrt_mass[:, None]
    
    return Structure(
        lattice=phonon_supercell.lattice,
        species=phonon_supercell.species,
        coords=coords + disp,
        coords_are_cartesian=True,
    )


@job
def get_sigma_a(
    force_constants: ForceConstants,
    phonon_supercell: Structure,
    displaced_structures: dict[str, list],
) -> float:
    """Calculate sigma^A.

    Parameters
    ----------
    force_constants: ForceConstants
        The force constants calculated by phonopy
    phonon_supercell: Structure
        The supercell used for the phonon calculation
    displaced_structures: dict[str, list]
        The output of run_displacements
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

    anharmonic_forces = [
        dft_force - harmonic_force
        for dft_force, harmonic_force in zip(dft_forces, harmonic_forces)
    ]

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
    sigma_A: float,
    phonon_doc: PhononBSDOSDoc,
    one_shot: bool
) -> AnharmonicityDoc:
    """
    Stores the results in a schema object

    Parameters
    ----------
    sigma_A: float
        Sigma^A value to be stored
    phonon_doc: PhononBSDOSDoc
        Info from phonon workflow to be stored
    one_shot: bool
        If false, uses a normally distributed random number for zeta.
        The default is true.
    """
    return AnharmonicityDoc.store_data(
        sigma_A=sigma_A,
        phonon_doc=phonon_doc,
        one_shot=one_shot,
    )