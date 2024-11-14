"""Common Flow for calculating harmonic & anharmonic props of phonon."""

# Basic Python packages
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from jobflow import Flow, Maker

from atomate2.common.jobs.hiphive import (
    generate_frequencies_eigenvectors,
    generate_phonon_displacements,
)

# Jobflow packages
from atomate2.common.jobs.phonons import (
    get_supercell_size,
    get_total_energy_per_cell,
    run_phonon_displacements,
)
from atomate2.common.jobs.utils import structure_to_conventional, structure_to_primitive
from atomate2.forcefields.jobs import (
    CHGNetStaticMaker,
    ForceFieldRelaxMaker,
    ForceFieldStaticMaker,
)

# Atomate2 packages
from atomate2.vasp.jobs.phonons import PhononDisplacementMaker
from atomate2.vasp.sets.core import StaticSetGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure  # type: ignore  # noqa: PGH003

    from atomate2.vasp.flows.core import DoubleRelaxMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker

SUPPORTED_CODES = ["vasp", "forcefields"]

logger = logging.getLogger(__name__)

__all__ = ["BaseHiphiveMaker"]

__author__ = "Alex Ganose, Junsoo Park, Zhuoying Zhu, Hrushikesh Sahasrabuddhe"
__email__ = "aganose@lbl.gov, jsyony37@lbl.gov, zyzhu@lbl.gov, hpsahasrabuddhe@lbl.gov"


@dataclass
class BaseHiphiveMaker(Maker, ABC):
    """
    Workflow to calc. interatomic force constants and vibrational props using hiPhive.

    A summary of the workflow is as follows:
    1. Structure relaxtion
    2. Calculate a supercell transformation matrix that brings the
       structure as close as cubic as possible, with all lattice lengths
       greater than 5 nearest neighbor distances. Then, perturb the atomic sites
       for each supercell using a Fixed displacement rattle procedure. The atoms are
       perturbed roughly according to a normal deviation around the average value.
       A number of standard deviation perturbation distances are included. Multiple
       supercells may be generated for each perturbation distance. Then, run static
       VASP calculations on each perturbed supercell to calculate atomic forces.
       Then, aggregate the forces and the perturbed structures.
    3. Conduct the fit atomic force constants using the regression schemes in hiPhive.
    4. Perform phonon renormalization at finite temperature - useful when unstable
       modes exist
    5. Output the interatomic force constants, and phonon band structure and density of
       states to the database
    6. Solve the lattice thermal conductivity using ShengBTE and output to the database.

    Args
    ----------
    name : str
        Name of the flows produced by this maker.
    bulk_relax_maker (BaseVaspMaker | None):
        The VASP input generator for bulk relaxation,
        default is DoubleRelaxMaker using TightRelaxMaker.
    phonon_displacement_maker (BaseVaspMaker | None):
        The VASP input generator for phonon displacement calculations,
        default is PhononDisplacementMaker.
    ff_displacement_maker (BaseVaspMaker | None):
        The force field displacement maker, default is CHGNetStaticMaker.
    min_length (float):
        Minimum length of supercell lattice vectors in Angstroms, default is 13.0.
    prefer_90_degrees (bool):
        Whether to prefer 90 degree angles in supercell matrix,
        default is True.
    supercell_matrix_kwargs (dict):
        Keyword arguments for supercell matrix calculation, default is {}.
    IMAGINARY_TOL (float):
        Imaginary frequency tolerance in THz, default is 0.025.
    MESH_DENSITY (float):
        Mesh density for phonon calculations, default is 100.0.
    T_QHA (list):
        Temperatures for phonopy thermodynamic calculations,
        default is [0, 100, 200, ..., 2000].
    T_RENORM (list):
        Temperatures for renormalization calculations, default is [1500].
    T_KLAT (int):
        Temperature for lattice thermal conductivity calculation, default is 300.
    FIT_METHOD (str):
        Method for fitting force constants, default is "rfe".
    RENORM_METHOD (str):
        Method for renormalization, default is 'pseudoinverse'.
    RENORM_NCONFIG (int):
        Number of configurations for renormalization, default is 5.
    RENORM_CONV_THRESH (float):
        Convergence threshold for renormalization in meV/atom, default is 0.1.
    RENORM_MAX_ITER (int):
        Maximum iterations for renormalization, default is 30.
    THERM_COND_SOLVER (str):
        Solver for lattice thermal conductivity, default is "almabte". Other options
        include "shengbte" and "phono3py".
    """

    name: str = "Lattice-Dynamics"
    bulk_relax_maker: DoubleRelaxMaker | ForceFieldRelaxMaker | None = None
    phonon_displacement_maker: BaseVaspMaker | ForceFieldStaticMaker | None = field(
        default_factory=lambda:PhononDisplacementMaker(
            input_set_generator=StaticSetGenerator(auto_lreal=True)
        )
    )
    ff_displacement_maker: ForceFieldStaticMaker | None = field(
        default_factory=CHGNetStaticMaker
    )
    supercell_matrix_kwargs: dict = field(default_factory=dict)
    IMAGINARY_TOL = 0.025  # in THz
    MESH_DENSITY = 100.0  # should always be a float
    T_QHA: ClassVar[list[int]] = [
        i * 100 for i in range(21)
    ]  # Temp. for phonopy calc. of thermo. properties (free energy etc.)
    FIT_METHOD = "least-squares" #least-squares #omp #rfe #elasticnet
    sym_reduce: bool = True
    symprec: float = 1e-4
    displacement: float = 0.01
    min_length: float | None = 20.0
    prefer_90_degrees: bool = True
    get_supercell_size_kwargs: dict = field(default_factory=dict)
    use_symmetrized_structure: str | None = None
    bulk_relax_maker: ForceFieldRelaxMaker | BaseVaspMaker | None = None
    static_energy_maker: ForceFieldRelaxMaker | BaseVaspMaker | None = (
        None
    )
    born_maker: ForceFieldStaticMaker | BaseVaspMaker | None = None
    phonon_displacement_maker: ForceFieldStaticMaker | BaseVaspMaker = (
        None
    )
    create_thermal_displacements: bool = True
    generate_frequencies_eigenvectors_kwargs: dict = field(default_factory=dict)
    kpath_scheme: str = "seekpath"
    code: str = None
    store_force_constants: bool = True
    socket: bool = False

    def make(
        self,
        mpid: str,
        structure: Structure,
        bulk_modulus: float,
        # supercell_matrix: list[list[int]] | None = None,
        supercell_matrix: Matrix3D | None = None,
        fit_method: str | None = FIT_METHOD,
        disp_cut: float | None = None,
        cutoffs: list[list[float]] | None = None,
        prev_dir: str | Path | None = None,
        renormalize: bool = True,
        mesh_density: float = MESH_DENSITY,
        imaginary_tol: float = IMAGINARY_TOL,
        temperature_qha: float | list | dict = T_QHA,
        n_structures: float = 1,
        fixed_displs: float | None = None,
        born: list[Matrix3D] | None = None,
        epsilon_static: Matrix3D | None = None,
        total_dft_energy_per_formula_unit: float | None = None,
    ) -> Flow:
        """
        Make flow to calculate the harmonic & anharmonic properties of phonon.

        Parameters
        ----------
        mpid (str):
            The Materials Project ID (MPID) of the material.
        structure (Structure):
            The A pymatgen structure of the material.
        bulk_modulus (float):
            Bulk modulus of the material in GPa.
        supercell_matrix (list[list[int]], optional):
            Supercell transformation matrix, default is None.
        fit_method (str, optional):
            Method for fitting force constants using hiphive, default is None.
        disp_cut (float, optional):
            Cutoff distance for displacements in Angstroms, default is None.
        cutoffs (List[List[float]], optional):
            List of cutoff distances for different force constants fitting,
            default is None.
        prev_dir (str | Path | None, optional):
            Previous RELAX calculation directory to use for copying outputs.,
            default is None.
        calculate_lattice_thermal_conductivity (bool, optional):
            Calculate lattice thermal conductivity, default is True.
        renormalize (bool, optional):
            Perform renormalization, default is False.
        renormalize_temperature (float | List | Dict, optional):
            Temperatures for renormalization, default is T_RENORM.
        renormalize_method (str, optional):
            Method for renormalization, default is RENORM_METHOD.
        renormalize_nconfig (int, optional):
            Number of configurations for renormalization, default is RENORM_NCONFIG.
        renormalize_conv_thresh (float, optional):
            Convergence threshold for renormalization in meV/atom,
            default is RENORM_CONV_THRESH.
        renormalize_max_iter (int, optional):
            Maximum iterations for renormalization, default is RENORM_MAX_ITER.
        renormalize_thermal_expansion_iter (bool, optional):
            Include thermal expansion during renormalization iterations,
            default is False.
        mesh_density (float, optional):
            Mesh density for phonon calculations, default is MESH_DENSITY.
        thermal_conductivity_temperature (float | List | Dict, optional):
            Temperatures for thermal conductivity calculations, default is T_KLAT.
        imaginary_tol (float, optional):
            Imaginary frequency tolerance in THz, default is IMAGINARY_TOL.
        temperature_qha (float, optional):
            Temperatures for phonopy thermodynamic calculations, default is T_QHA.
        n_structures (float, optional):
            Number of structures to consider for calculations, default is None.
        fixed_displs (float, optional):
            Avg value of atomic displacement in Angstroms, default is None.
        """
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
        # outputs = []

        # TODO: should this be after or before structural optimization as the
        #  optimization could change the symmetry we could add a tutorial and point out
        #  that the structure should be nearly optimized before the phonon workflow
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

        # if supercell_matrix is None, supercell size will be determined after relax
        # maker to ensure that cell lengths are really larger than threshold
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


        # get a phonon object from phonopy
        displacements = generate_phonon_displacements(
            structure=structure,
            supercell_matrix=supercell_matrix,
            fixed_displs=[0.01, 0.03, 0.08, 0.1],
            sym_reduce=self.sym_reduce,
            symprec=self.symprec,
            use_symmetrized_structure=self.use_symmetrized_structure,
            kpath_scheme=self.kpath_scheme,
            code=self.code,
        )
        jobs.append(displacements)

        # perform the phonon displacement calculations
        displacement_calcs = run_phonon_displacements(
            displacements=displacements.output,
            structure=structure,
            supercell_matrix=supercell_matrix,
            phonon_maker=self.phonon_displacement_maker,
            socket=self.socket,
            prev_dir_argname=self.prev_calc_dir_argname,
            prev_dir=prev_dir,
        )
        jobs.append(displacement_calcs)

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

        logger.info("Generating phonon frequencies and eigenvectors")
        print("Generating phonon frequencies and eigenvectors")
        phonon_collect = generate_frequencies_eigenvectors(
            supercell_matrix=supercell_matrix,
            displacement=self.displacement,
            sym_reduce=self.sym_reduce,
            symprec=self.symprec,
            use_symmetrized_structure=self.use_symmetrized_structure,
            kpath_scheme=self.kpath_scheme,
            code=self.code,
            structure=structure,
            displacement_data=displacement_calcs.output,
            epsilon_static=epsilon_static,
            born=born,
            total_dft_energy=total_dft_energy,
            static_run_job_dir=static_run_job_dir,
            static_run_uuid=static_run_uuid,
            born_run_job_dir=born_run_job_dir,
            born_run_uuid=born_run_uuid,
            optimization_run_job_dir=optimization_run_job_dir,
            optimization_run_uuid=optimization_run_uuid,
            create_thermal_displacements=self.create_thermal_displacements,
            store_force_constants=self.store_force_constants,
            bulk_modulus=bulk_modulus,
            **self.generate_frequencies_eigenvectors_kwargs,
        )

        jobs.append(phonon_collect)

        return Flow(jobs=jobs, output=phonon_collect.output, name=f"{mpid}_"
                                                    f"{disp_cut}_"
                                                    f"{cutoffs}_"
                                                    f"{self.name}")

    @property
    @abstractmethod
    def prev_calc_dir_argname(self) -> str | None:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
