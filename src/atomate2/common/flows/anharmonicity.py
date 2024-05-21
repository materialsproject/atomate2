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
    # get_force_constants,
    # build_dyn_mat,
    # get_emode_efreq,
    get_sigma_a,
    displace_structure,
    run_displacements,
    # get_anharmonic_force,
    # calc_sigma_A_oneshot,
    # make_displaced_structure,
)
from atomate2.common.schemas.phonons import ForceConstants, PhononBSDOSDoc
from pymatgen.core.structure import Structure

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
    structure: Structure = None
    dynamical_matrix: np.ndarray = None
    eigenmodes: np.ndarray = None
    eigenfreq: np.ndarray = None
    displaced_supercell: np.ndarray = None
    DFT_forces: list[np.ndarray] = field(default_factory=lambda: [])
    anharmonic_forces: np.ndarray = None
    sigma_A_oneshot: float = None
    sym_reduce: bool = True
    symprec: float = 1e-4
    use_symmetrized_structure: str | None = None
    kpath_scheme: str = "seekpath"
    code: str = None
    born_maker: ForceFieldStaticMaker | BaseVaspMaker | None = None
    phonon_displacement_maker: ForceFieldStaticMaker | BaseVaspMaker | BaseAimsMaker = (
        None
    )
    socket: bool = True
    store_force_constants: bool = True
    bulk_relax_maker: ForceFieldRelaxMaker | BaseVaspMaker | BaseAimsMaker | None = None
    static_energy_maker: ForceFieldRelaxMaker | BaseVaspMaker | BaseAimsMaker | None = (
        None
    )

    def make(
        self,
        structure: Structure,
        socket: bool = True,
        prev_dir: str | Path | None = None,
        supercell_matrix: Matrix3D | None = None,
        temperature: float = 300,
        total_dft_energy_per_formula_unit: float | None = None,
        npoints_band: float = 101
    ) -> Flow:
        self.socket = socket
        self.structure = structure
        use_symmetrized_structure = self.use_symmetrized_structure
        kpath_scheme = self.kpath_scheme
        valid_structs = (None, "primitive", "conventional")
        if use_symmetrized_structure not in valid_structs:
            raise ValueError(
                f"Invalid {use_symmetrized_structure=}, use one of {valid_structs}"
            )

        if use_symmetrized_structure != "primitive" and kpath_scheme != "seekpath":
            raise ValueError(
                f"You can't use {kpath_scheme=} with the primitive standard "
                "structure, please use seekpath"
            )

        valid_schemes = ("seekpath", "hinuma", "setyawan_curtarolo", "latimer_munro")
        if kpath_scheme not in valid_schemes:
            raise ValueError(
                f"{kpath_scheme=} is not implemented, use one of {valid_schemes}"
            )

        if self.code is None or self.code not in SUPPORTED_CODES:
            raise ValueError(
                "The code variable must be passed and it must be a supported code."
                f" Supported codes are: {SUPPORTED_CODES}"
            )

        jobs = []

        if self.use_symmetrized_structure == "primitive":
            # These structures are compatible with many
            # of the kpath algorithms that are used for Materials Project
            prim_job = structure_to_primitive(structure, self.symprec)
            jobs.append(prim_job)
            structure = prim_job.output
        elif self.use_symmetrized_structure == "conventional":
            # it could be beneficial to use conventional standard structures to arrive
            # faster at supercells with right angles
            conv_job = structure_to_conventional(structure, self.symprec)
            jobs.append(conv_job)
            structure = conv_job.output

        optimization_run_job_dir = None
        optimization_run_uuid = None

        if self.bulk_relax_maker is not None:
            # optionally relax the structure
            bulk_kwargs = {}
            if self.prev_calc_dir_argname is not None:
                bulk_kwargs[self.prev_calc_dir_argname] = prev_dir
            bulk = self.bulk_relax_maker.make(structure, **bulk_kwargs)
            jobs.append(bulk)
            structure = bulk.output.structure
            prev_dir = bulk.output.dir_name
            optimization_run_job_dir = bulk.output.dir_name
            optimization_run_uuid = bulk.output.uuid
        
        # self.structure = structure
        """
        try:
            species = [site.species for site in structure]
        except TypeError:
            return bulk
        """
        # jobs = []
        if supercell_matrix is None:
            supercell_job = get_supercell_size(
                self.structure,
                self.min_length,
                self.prefer_90_degrees,
                **self.get_supercell_size_kwargs,
            )
            jobs.append(supercell_job)
            supercell_matrix = supercell_job.output

        # Issue: To implement prev_calc_dir_argname separately would basically be copy/pasting
        # the code from phonon flows and would require making anharmonicity flows inheriting 
        # this flow in each DFT code folder
        # ------------------------------------------------------------------------------------
        # Idea: Can I reuse the phonon flow to do the static/bulk maker parts?
        # This would allow me to get total_dft_energy from the static maker result
        
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
                structure=self.structure, **static_job_kwargs
            )
            jobs.append(static_job)
            total_dft_energy = static_job.output.output.energy
            static_run_job_dir = static_job.output.dir_name
            static_run_uuid = static_job.output.uuid
            prev_dir = static_job.output.dir_name
        elif total_dft_energy_per_formula_unit is not None:
            # to make sure that one can reuse results from Doc
            compute_total_energy_job = get_total_energy_per_cell(
                total_dft_energy_per_formula_unit, self.structure
            )
            jobs.append(compute_total_energy_job)
            total_dft_energy = compute_total_energy_job.output
        
        # Get phonopy object
        phonon_displacements = generate_phonon_displacements(
            structure = self.structure,
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
            structure = self.structure,
            supercell_matrix = supercell_matrix,
            phonon_maker = self.phonon_displacement_maker,
            prev_dir = prev_dir,
            prev_dir_argname = self.prev_calc_dir_argname,
            socket = self.socket,
        )
        jobs.append(displacement_calcs)

        # Recover DFT calculated forces
        self.DFT_forces = displacement_calcs.output["forces"]

        # Computation of BORN charges
        born_run_job_dir = None
        born_run_uuid = None
        if self.born_maker is not None and (born is None or epsilon_static is None):
            born_kwargs = {}
            if self.prev_calc_dir_argname is not None:
                born_kwargs[self.prev_calc_dir_argname] = prev_dir
            born_job = self.born_maker.make(structure, **born_kwargs)
            jobs.append(born_job)

            # I am not happy how we currently access "born" charges
            # This is very vasp specific code aims and forcefields
            # do not support this at the moment, if this changes we have
            # to update this section
            epsilon_static = born_job.output.calcs_reversed[0].output.epsilon_static
            born = born_job.output.calcs_reversed[0].output.outcar["born"]
            born_run_job_dir = born_job.output.dir_name
            born_run_uuid = born_job.output.uuid

        phonon_collect = generate_frequencies_eigenvectors(
            supercell_matrix = supercell_matrix,
            displacement = self.displacement,
            sym_reduce = self.sym_reduce,
            symprec = self.symprec,
            use_symmetrized_structure = self.use_symmetrized_structure,
            kpath_scheme = self.kpath_scheme,
            code = self.code,
            structure = self.structure,
            displacement_data = displacement_calcs.output,
            total_dft_energy = total_dft_energy,
            static_run_job_dir=static_run_job_dir,
            static_run_uuid=static_run_uuid,
            born_run_job_dir=born_run_job_dir,
            born_run_uuid=born_run_uuid,
            optimization_run_job_dir=optimization_run_job_dir,
            optimization_run_uuid=optimization_run_uuid,
            store_force_constants = self.store_force_constants
        )

        jobs.append(phonon_collect)
        phononBSDOS = phonon_collect.output
        self.fc = phononBSDOS.force_constants
        # self.structure = phononBSDOS.structure

        # # Get force constants (I think this is redundant due to PhononBSDOSDoc)
        # force_consts = get_force_constants(
        #     phonon = self.phonon,
        #     forces_DFT = self.DFT_forces
        # )
        # jobs.append(force_consts)
        # self.fc = force_consts.output

        # get phonon band structure
        kpath_dict, kpath_concrete = PhononBSDOSDoc.get_kpath(
            structure = self.structure,
            kpath_scheme = self.kpath_scheme,
            symprec = self.symprec,
        )

        # Get q points
        qpoints, connections = get_band_qpoints_and_path_connections(
            kpath_concrete,     
            npoints = npoints_band
        )  

        # Build Dynamical Matrix and get it
        # dyn_mat = build_dyn_mat(
        #     structure = self.structure,
        #     force_constants = self.fc,
        #     supercell = supercell_matrix,
        #     qpoints = qpoints,
        #     code = self.code,
        #     symprec = self.symprec,
        #     sym_reduce = self.sym_reduce,
        #     use_symmetrized_structure = self.use_symmetrized_structure,
        #     kpath_scheme = self.kpath_scheme,
        #     kpath_concrete = kpath_concrete
        # )
        # jobs.append(dyn_mat)
        # #self.dynamical_matrix = dyn_mat.output
        # self.eigenfreq = dyn_mat.output[0]
        # self.eigenmodes = dyn_mat.output[1]
        # print("Dynamical matrix created (line 307)")
        
        # # Calculate eigenmodes and eigenfrequencies
        # eig_calc = get_emode_efreq(
        #     dynamical_matrix = self.dynamical_matrix
        # )
        # jobs.append(eig_calc)
        # self.eigenfreq = eig_calc.output[0]
        # self.eigenmodes = eig_calc.output[1]
        
        # print("Eigs found (line 317)")

        # Generate the displaced supercell 
        displace_supercell = displace_structure(
            structure = self.structure,
            supercell = supercell_matrix,
            force_constants = self.fc,
            temp = temperature
        )
        jobs.append(displace_supercell)
        self.displaced_supercell = displace_supercell.output

        displacement_calcs = run_displacements(
            displacements=[displace_supercell.output],
            structure=structure,
            supercell_matrix=supercell_matrix,
            force_eval_maker=self.phonon_displacement_maker,
            socket=self.socket,
            prev_dir_argname=self.prev_calc_dir_argname,
            prev_dir=prev_dir,
        )
        jobs.append(displacement_calcs)
        # harmonic force = -force_constants @ d.flatten()
        

        print("Displaced positions found (line 330)")

        # # Generate displaced supercell as pymatgen structure
        # lattice = self.structure.make_supercell(
        #     scaling_matrix = supercell_matrix,
        #     in_place = False
        #     ).lattice
        # species = self.structure.make_supercell(
        #     scaling_matrix = supercell_matrix,
        #     in_place = False
        # ).species
        # get_displaced_structure = make_displaced_structure(lattice, species, self.displaced_supercell)
        # jobs.append(get_displaced_structure)
        # displaced_structure = get_displaced_structure.output

        # print("Displaced supercell made (line 339)")

        # # Get harmonic forces using displaced structure
        # displacements_harmonic = generate_phonon_displacements(
        #     displaced_structure,
        #     np.eye(3),
        #     0,
        #     self.sym_reduce,
        #     self.symprec,
        #     self.use_symmetrized_structure,
        #     self.kpath_scheme,
        #     self.code
        # )
        # jobs.append(displacements_harmonic)

        # print("Harmonic displacements generated (line 354)")

        # harmonic_disp_calcs = run_phonon_displacements(
        #     displacements_harmonic.output,
        #     displaced_structure,
        #     np.eye(3),
        #     #self.displaced_supercell,
        #     self.phonon_displacement_maker,
        #     prev_dir,
        #     self.prev_calc_dir_argname,
        #     self.socket
        # )
        # jobs.append(harmonic_disp_calcs)
        # harmonic_force = harmonic_disp_calcs.output["forces"]

        # print("Harmonic displacements ran (line 368)")

        """
        # TODO: Implement this function. Take guidance from run_phonon_displacements()
        # I want the function to also return the displacements
        # Run it using the displaced structure from displace_structure() above
        calculated_displacements = run_anharmonic_displacements()
        """

        # Calculate the anharmonic contribution to the forces
        # get_force_anharmonic = get_anharmonic_force(
        #     phononBSDOS.force_constants,
        #     harmonic_force,
        #     DFT_forces = self.DFT_forces,
        # )
        # jobs.append(get_force_anharmonic)
        # self.anharmonic_forces = get_force_anharmonic.output

        # print("Anharmonic force calculated (line 386)")

        # Calculate oneshot approximation of sigma_A
        calc_sigma_A_os = get_sigma_a(
            phononBSDOS.force_constants,
            structure,
            supercell_matrix,
            displacement_calcs.output,
        )
        jobs.append(calc_sigma_A_os)
        self.sigma_A_oneshot = calc_sigma_A_os.output

        print("Sigma A found (line 396)")

        return Flow(jobs, calc_sigma_A_os.output)

    # Note: Came from aims/flows/phonons.py
    # (Might be different for different DFT codes, but this might work as a first-pass solution)
    @property
    def prev_calc_dir_argname(self) -> str:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
        return "prev_dir"