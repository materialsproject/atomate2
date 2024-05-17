"""Flows for anharmonicity quantification"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

# TODO: NEED TO CHANGE
from atomate2.common.jobs.phonons import (
    generate_phonon_displacements,
    generate_frequencies_eigenvectors,
    get_supercell_size,
    get_total_energy_per_cell,
    run_phonon_displacements,
)
from atomate2.common.jobs.utils import structure_to_conventional, structure_to_primitive
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from atomate2.common.jobs.anharmonicity import (
    get_force_constants,
    build_dyn_mat,
    get_emode_efreq,
    displace_structure,
    get_anharmonic_force,
    calc_sigma_A_oneshot,
)
from atomate2.common.schemas.phonons import ForceConstants, PhononBSDOSDoc

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
    sym_reduce: bool = True
    symprec: float = 1e-4
    use_symmetrized_structure: str | None = None
    kpath_scheme: str = "seekpath"
    code: str = None
    phonon_displacement_maker: ForceFieldStaticMaker | BaseVaspMaker | BaseAimsMaker = (
        None
    )
    socket: bool = False
    store_force_constants: bool = True

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
        supercell_matrix: Matrix3D | None = None,
        temperature: float = 300,
        total_dft_energy_per_formula_unit: float | None = None,
        npoints_band: float = 101
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
        
        # Computation of static energy
        total_dft_energy = None
        static_run_job_dir = None
        static_run_uuid = None
        if (self.static_energy_maker is not None) and (
            total_dft_energy_per_formula_unit is None
        ):
            static_job_kwargs = {}
            if self.prev_calc_dir_argname is not None:
                static_job_kwargs[self.prev_calc_dir_argname] = prev_dir
            static_job = self.static_energy_maker.make(
                structure=structure, **static_job_kwargs
            )
            jobs.append(static_job)
            total_dft_energy = static_job.output.output.energy
            static_run_job_dir = static_job.output.dir_name
            static_run_uuid = static_job.output.uuid
            prev_dir = static_job.output.dir_name
        elif total_dft_energy_per_formula_unit is not None:
            # to make sure that one can reuse results from Doc
            compute_total_energy_job = get_total_energy_per_cell(
                total_dft_energy_per_formula_unit, structure
            )
            jobs.append(compute_total_energy_job)
            total_dft_energy = compute_total_energy_job.output

        # Get phonopy object
        phonon_displacements = generate_phonon_displacements(
            structure = structure,
            supercell_matrix = supercell_matrix,
            displacement = self.displacement,
            sym_reduce = self.sym_reduce,
            symprec = self.symprec,
            use_symmetrized_structure = self.use_symmetrized_structure,
            kpath_scheme = self.kpath_scheme,
            code = self.code,
        )
        jobs.append(phonon_displacements)

        # Recover Phonopy object for use
        displacements = phonon_displacements.output

        # Run displacement calculations
        displacement_calcs = run_phonon_displacements(
            displacements = displacements,
            structure = structure,
            supercell_matrix = supercell_matrix,
            phonon_maker = self.phonon_displacement_maker,
            prev_dir = prev_dir,
            prev_dir_argname = self.prev_calc_dir_argname,
            socket = self.socket,
        )
        jobs.append(displacement_calcs)

        # Recover DFT calculated forces
        self.DFT_forces = displacement_calcs.output["forces"]

        phonon_collect = generate_frequencies_eigenvectors(
            supercell_matrix = supercell_matrix,
            displacement = self.displacement,
            sym_reduce = self.sym_reduce,
            symprec = self.symprec,
            use_symmetrized_structure = self.use_symmetrized_structure,
            kpath_scheme = self.kpath_scheme,
            code = self.code,
            structure = structure,
            displacement_data = displacement_calcs.output,
            total_dft_energy = total_dft_energy,
            store_force_constants = self.store_force_constants
        )

        jobs.append(phonon_collect)
        phononBSDOS = phonon_collect.output
        self.fc = phononBSDOS.force_constants

        # # Get force constants (I think this is redundant due to PhononBSDOSDoc)
        # force_consts = get_force_constants(
        #     phonon = self.phonon,
        #     forces_DFT = self.DFT_forces
        # )
        # jobs.append(force_consts)
        # self.fc = force_consts.output

        # get phonon band structure
        kpath_dict, kpath_concrete = PhononBSDOSDoc.get_kpath(
            structure = structure,
            kpath_scheme = self.kpath_scheme,
            symprec = self.symprec,
        )

        # Get q points
        qpoints, connections = get_band_qpoints_and_path_connections(
            kpath_concrete,     
            npoints = npoints_band
)

        # Build Dynamical Matrix and get it
        dyn_mat = build_dyn_mat(
            structure = structure,
            force_constants = self.fc,
            supercell = supercell_matrix,
            qpoints = qpoints,
            code = self.code,
            symprec = self.symprec,
            sym_reduce = self.sym_reduce,
            use_symmetrized_structure = self.use_symmetrized_structure,
            kpath_scheme = self.kpath_scheme,
            kpath_concrete = kpath_concrete
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
        displace_supercell = displace_structure(
            supercell = supercell_matrix,
            masses = [site.species.weight for site in phononBSDOS.structure],
            eig_vec = self.eigenmodes,
            eig_val = self.eigenfreq,
            temp = temperature
        )
        jobs.append(displace_supercell)
        self.displaced_supercell = displace_supercell.output

        # Generate displaced supercell as pymatgen structure
        lattice = structure.lattice
        species = structure.species
        displaced_structure = Structure(lattice, species, self.displaced_supercell)

        # Get harmonic forces using displaced structure
        displacements_harmonic = generate_phonon_displacements(
            displaced_structure,
            self.displaced_supercell,
            0,
            self.sym_reduce,
            self.symprec,
            self.use_symmetrized_structure,
            self.kpath_scheme,
            self.code
        )
        jobs.append(displacements_harmonic)
        harmonic_disp_calcs = run_phonon_displacements(
            displacements_harmonic.output,
            displaced_structure,
            self.displaced_supercell,
            self.phonon_displacement_maker,
            prev_dir,
            self.prev_calc_dir_argname,
            self.socket
        )
        jobs.append(harmonic_disp_calcs)
        harmonic_force = harmonic_disp_calcs.output["forces"]

        """
        # TODO: Implement this function. Take guidance from run_phonon_displacements()
        # I want the function to also return the displacements
        # Run it using the displaced structure from displace_structure() above
        calculated_displacements = run_anharmonic_displacements()
        """

        # Calculate the anharmonic contribution to the forces
        get_force_anharmonic = get_anharmonic_force(
            phononBSDOS.force_constants,
            harmonic_force,
            DFT_forces = self.DFT_forces,
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