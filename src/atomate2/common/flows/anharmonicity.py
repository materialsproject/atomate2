"""Flows for anharmonicity quantification"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

# TODO: NEED TO CHANGE
from atomate2.common.jobs.phonons import (
    generate_frequencies_eigenvectors,
    get_supercell_size,
    get_total_energy_per_cell,
    run_phonon_displacements,
)
from atomate2.common.jobs.utils import structure_to_conventional, structure_to_primitive
from phonopy import Phonopy
from atomate2.common.jobs.anharmonicity import (
    generate_phonon_displacements,
    get_force_constants,
    build_dyn_mat,
    get_emode_efreq,
    displace_structure,
    get_anharmonic_force,
    calc_sigma_A_oneshot
)

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

    from atomate2.aims.jobs.base import BaseAimsMaker
    from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker

SUPPORTED_CODES = ["vasp", "aims", "forcefields"]

@dataclass
class BaseAnharmonicityMaker(Maker):
    name: str = "anharmonicity"
    displacement: float = 0.01
    fc: np.ndarray = None
    phonon: Phonopy = None
    dynamical_matrix: np.ndarray = None
    eigenmodes: np.ndarray = None
    eigenfreq: np.ndarray = None
    displaced_supercell: np.ndarray = None
    DFT_forces: list[np.ndarray] = []
    anharmonic_forces: np.ndarray
    sigma_A_oneshot: float

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
        supercell_matrix: Matrix3D | None = None,
        temperature: float = 300
    ) -> Flow:
        jobs = []
        if supercell_matrix is None:
            supercell_job = get_supercell_size(
                structure,
                self.min_length,
                self.prefer_90_degrees,
                **self.get_supercell_size_kwargs,
            )
            jobs.append(supercell_job)
            supercell_matrix = supercell_job.output
        
        # Get phonopy object
        displacements = generate_phonon_displacements(
            structure = structure,
            supercell_matrix = supercell_matrix,
            displacement = self.displacement,
            sym_reduce = self.sym_reduce,
            symprec = self.symprec,
            use_symmetrized_structure = self.use_symmetrized_structure,
            kpath_scheme = self.kpath_scheme,
            code = self.code,
        )
        jobs.append(displacements)

        # Recover Phonopy object for use
        self.phonon = displacements.output[1]

        # Run displacement calculations
        displacement_calcs = run_phonon_displacements(
            displacements = displacements.output,
            structure = structure,
            supercell_matrix = supercell_matrix,
            phonon_maker = self.phonon_displacement_maker,
            socket = self.socket,
            prev_dir_argname = self.prev_calc_dir_argname,
            prev_dir = prev_dir,
        )
        jobs.append(displacement_calcs)

        # Recover DFT calculated forces
        self.DFT_forces = displacement_calcs.output["forces"]

        # Get force constants
        force_consts = get_force_constants(
            phonon = self.phonon,
            forces_DFT = self.DFT_forces
        )
        jobs.append(force_consts)
        self.fc = force_consts.output

        # Build Dynamical Matrix and get it
        dyn_mat = build_dyn_mat(
            phonon = self.phonon,
        )
        jobs.append(dyn_mat)
        self.dynamical_matrix = dyn_mat.output

        # Calculate eigenmodes and eigenfrequencies
        eig_calc = get_emode_efreq(
            dynamical_matrix = self.dynamical_matrix
        )
        jobs.append(eig_calc)
        self.eigenfreq, self.eigenmodes = eig_calc.output

        # Generate the displaced supercell 
        # TODO: Check if this is really needed or if Phonopy already found F^(2)
        displace_supercell = displace_structure(
            phonon = self.phonon,
            eig_vec = self.eigenmodes,
            eig_val = self.eigenmodes,
            temp = temperature
        )
        jobs.append(displace_supercell)
        self.displaced_supercell = displace_supercell.output

        # Calculate the anharmonic contribution to the forces
        get_force_anharmonic = get_anharmonic_force(
            phonon = self.phonon,
            DFT_forces = self.DFT_forces
        )
        jobs.append(get_force_anharmonic)
        self.anharmonic_forces = get_force_anharmonic.output

        # Calculate oneshot approximation of sigma_A
        calc_sigma_A_os = calc_sigma_A_oneshot(
            anharmonic_force = self.anharmonic_forces,
            DFT_forces = self.DFT_forces   
        )
        jobs.append(calc_sigma_A_os)
        self.sigma_A_oneshot = calc_sigma_A_os.output