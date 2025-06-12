"""Flow for calculating (an)harmonic FCs and phonon energy renorma. with pheasy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.common.jobs.pheasy import (
    generate_frequencies_eigenvectors,
    generate_phonon_displacements,
    get_supercell_size,
    run_phonon_displacements,
)
from atomate2.common.jobs.phonons import get_total_energy_per_cell
from atomate2.common.jobs.utils import structure_to_conventional, structure_to_primitive

if TYPE_CHECKING:
    from pathlib import Path

    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

    from atomate2.aims.jobs.base import BaseAimsMaker
    from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker

SUPPORTED_CODES = frozenset(("vasp", "aims", "forcefields"))


@dataclass
class BasePhononMaker(Maker, ABC):
    """Maker to calculate harmonic phonons with LASSO-based ML code Pheasy.

    Calculate the zero-K harmonic phonons of a material and higher-order FCs.
    Initially, a tight structural relaxation is performed to obtain a structure
    without forces on the atoms. Subsequently, supercells with all atoms displaced
    by a small amplitude (generally using 0.01 A) are generated and accurate forces
    are computed for these structures for the second order force constants. With the
    help of pheasy (LASSO technique), these forces are then converted into a dynamical
    matrix. In this Workflow, we separate the harmonic phonon calculations and
    anharmonic force constants calculations. To correct for polarization effects, a
    correction of the dynamical matrix based on BORN charges can be performed. Finally,
    phonon densities of states, phonon band structures and thermodynamic properties
    are computed. For the anharmonic force constants, the supercells with all atoms
    displaced by a larger amplitude (generally using 0.08 A) are generated and accurate
    forces are computed for these structures. With the help of pheasy (LASSO technique),
    the third- and fourth-order force constants are extracted at once.

    .. Note::
        It is heavily recommended to symmetrize the structure before passing it to
        this flow. Otherwise, a different space group might be detected and too many
        displacement calculations will be required for pheasy phonon calculation. It
        is recommended to check the convergence parameters here and adjust them if
        necessary. The default might not be strict enough for your specific case.
        Additionally, for high-throughoput calculations, it is recommended to calculate
        the residual forces on the atoms in the supercell after the relaxation. Then the
        forces on displaced supercells can deduct the residual forces to reduce the
        error in the dynamical matrix.

    Parameters
    ----------
    name : str
        Name of the flow produced by this maker.
    sym_reduce : bool
        Whether to reduce the number of deformations using symmetry.
    symprec : float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in pheasy, we recommend to
        use the value of 1e-3.
    displacement: float
        displacement distance for phonons, for most cases 0.01 A is a good choice,
        but it can be increased to 0.02 A for heavier elements.
    num_displaced_supercells: int
        number of displacements to be generated using a random-displacement approach
        for harmonic phonon calculations. The default value is 0 and the number of
        displacements is automatically determined by the number of atoms in the
        supercell and its space group.
    cal_anhar_fcs: bool
        if set to True, anharmonic force constants(FCs) up to fourth-order FCs will
        be calculated. The default value is False, and only harmonic phonons will
        be calculated.
    displacement_anhar: float
        displacement distance for anharmonic force constants(FCs) up to fourth-order
        FCs, for most cases 0.08 A is a good choice, but it can be increased to 0.1 A.
    num_disp_anhar: int
        number of displacements to be generated using a random-displacement approach
        for anharmonic phonon calculations. The default value is 0 and the number of
        displacements is automatically determined by the number of atoms in the
        supercell, cutoff distance for anharmonic FCs its space group. generally,
        50 large-distance displacements are enough for most cases.
    fcs_cutoff_radius: list
        cutoff distance for anharmonic force constants(FCs) up to fourth-order FCs.
        The default value is [-1, 12, 10], which means that the cutoff distance for
        second-order FCs is the Wigner-Seitz cell boundary and the cutoff distance
        for third-order FCs is 12 Borh, and the cutoff distance for fourth-order FCs
        is 10 Bohr. Generally, the default value is good enough.
    min_length: float
        minimum length of lattice constants will be used to create the supercell,
        the default value is 14.0 A. In most cases, the default value is good
        enough, but it can be increased for larger supercells.
    prefer_90_degrees: bool
        if set to True, supercell algorithm will first try to find a supercell
        with 3 90 degree angles.
    get_supercell_size_kwargs: dict
        kwargs that will be passed to get_supercell_size to determine supercell size
    use_symmetrized_structure: str
        allowed strings: "primitive", "conventional", None

        - "primitive" will enforce to start the phonon computation
          from the primitive standard structure
          according to Setyawan, W., & Curtarolo, S. (2010).
          High-throughput electronic band structure calculations:
          Challenges and tools. Computational Materials Science,
          49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010.
          This makes it possible to use certain k-path definitions
          with this workflow. Otherwise, we must rely on seekpath
        - "conventional" will enforce to start the phonon computation
          from the conventional standard structure
          according to Setyawan, W., & Curtarolo, S. (2010).
          High-throughput electronic band structure calculations:
          Challenges and tools. Computational Materials Science,
          49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010.
          We will however use seekpath and primitive structures
          as determined by from phonopy to compute the phonon band structure
    bulk_relax_maker: .ForceFieldRelaxMaker, .BaseAimsMaker, .BaseVaspMaker, or None
        A maker to perform a tight relaxation on the bulk.
        Set to ``None`` to skip the
        bulk relaxation
    static_energy_maker: .ForceFieldRelaxMaker, .BaseAimsMaker, .BaseVaspMaker, or None
        A maker to perform the computation of the DFT energy on the bulk.
        Set to ``None`` to skip the
        static energy computation
    born_maker: .ForceFieldStaticMaker, .BaseAsimsMaker, .BaseVaspMaker, or None
        Maker to compute the BORN charges.
    phonon_displacement_maker: .ForceFieldStaticMaker, .BaseAimsMaker, .BaseVaspMaker
        Maker used to compute the forces for a supercell.
    generate_frequencies_eigenvectors_kwargs : dict
        Keyword arguments passed to :obj:`generate_frequencies_eigenvectors`.
    create_thermal_displacements: bool
        Bool that determines if thermal_displacement_matrices are computed
    kpath_scheme: str
        scheme to generate kpoints. Please be aware that
        you can only use seekpath with any kind of cell
        Otherwise, please use the standard primitive structure
        Available schemes are:
        "seekpath", "hinuma", "setyawan_curtarolo", "latimer_munro".
        "seekpath" and "hinuma" are the same definition but
        seekpath can be used with any kind of unit cell as
        it relies on phonopy to handle the relationship
        to the primitive cell and not pymatgen
    code: str
        determines the dft or force field code.
    mp_id: str
        The mp_id of the material in the Materials Project database.
    store_force_constants: bool
        if True, force constants will be stored
    socket: bool
        If True, use the socket for the calculation

    """

    name: str = "phonon"
    sym_reduce: bool = True
    symprec: float = 1e-3
    displacement: float = 0.01
    num_displaced_supercells: int = 0
    cal_anhar_fcs: bool = False
    displacement_anhar: float = 0.08
    num_disp_anhar: int = 0
    fcs_cutoff_radius: list = field(
        default_factory=lambda: [-1, 12, 10]
    )  # units in Bohr
    renorm_phonon: bool = False
    renorm_temp: list = field(default_factory=lambda: [100, 700, 100])
    cal_ther_cond: bool = False
    ther_cond_mesh: list = field(default_factory=lambda: [20, 20, 20])
    ther_cond_temp: list = field(default_factory=lambda: [100, 700, 100])
    min_length: float | None = 8.0
    max_atoms: float | None = 200
    force_90_degrees: bool = True
    force_diagonal: bool = True
    allow_orthorhombic: bool = False
    get_supercell_size_kwargs: dict = field(default_factory=dict)
    use_symmetrized_structure: str | None = None
    bulk_relax_maker: ForceFieldRelaxMaker | BaseVaspMaker | BaseAimsMaker | None = None
    static_energy_maker: ForceFieldRelaxMaker | BaseVaspMaker | BaseAimsMaker | None = (
        None
    )
    born_maker: ForceFieldStaticMaker | BaseVaspMaker | None = None
    phonon_displacement_maker: ForceFieldStaticMaker | BaseVaspMaker | BaseAimsMaker = (
        None
    )
    create_thermal_displacements: bool = False
    generate_frequencies_eigenvectors_kwargs: dict = field(default_factory=dict)
    kpath_scheme: str = "seekpath"
    code: str = None
    mp_id: str = None
    store_force_constants: bool = True
    socket: bool = False

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
        born: list[Matrix3D] | None = None,
        epsilon_static: Matrix3D | None = None,
        total_dft_energy_per_formula_unit: float | None = None,
        supercell_matrix: Matrix3D | None = None,
    ) -> Flow:
        """Make flow to calculate the phonon properties.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object. Please start with a structure
            that is nearly fully optimized as the internal optimizers
            have very strict settings!
        prev_dir : str or Path or None
            A previous calculation directory to use for copying outputs.
        born: Matrix3D
            Instead of recomputing born charges and epsilon, these values can also be
            provided manually. If born and epsilon_static are provided, the born run
            will be skipped it can be provided in the VASP convention with information
            for every atom in unit cell. Please be careful when converting structures
            within in this workflow as this could lead to errors
        epsilon_static: Matrix3D
            The high-frequency dielectric constant to use instead of recomputing born
            charges and epsilon. If born, epsilon_static are provided, the born run
            will be skipped
        total_dft_energy_per_formula_unit: float
            It has to be given per formula unit (as a result in corresponding Doc).
            Instead of recomputing the energy of the bulk structure every time, this
            value can also be provided in eV. If it is provided, the static run will be
            skipped. This energy is the typical output dft energy of the dft workflow.
            No conversion needed.
        supercell_matrix: list
            Instead of min_length, also a supercell_matrix can be given, e.g.
            [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]
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
        # maker to ensure that cell lengths are really larger than threshold.
        # Note that If one wants to calculate the lattice thermal conductivity,
        # the supercell dimensions should be forced to be diagonal, e.g.
        # supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        if supercell_matrix is None:
            supercell_job = get_supercell_size(
                structure,
                self.min_length,
                self.max_atoms,
                self.force_90_degrees,
                self.force_diagonal,
                self.allow_orthorhombic,
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

        # get a phonon object from pheasy code using the random-displacement approach
        displacements = generate_phonon_displacements(
            structure=structure,
            supercell_matrix=supercell_matrix,
            displacement=self.displacement,
            num_displaced_supercells=self.num_displaced_supercells,
            cal_anhar_fcs=self.cal_anhar_fcs,
            displacement_anhar=self.displacement_anhar,
            num_disp_anhar=self.num_disp_anhar,
            fcs_cutoff_radius=self.fcs_cutoff_radius,
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

        phonon_collect = generate_frequencies_eigenvectors(
            supercell_matrix=supercell_matrix,
            displacement=self.displacement,
            num_displaced_supercells=self.num_displaced_supercells,
            cal_anhar_fcs=self.cal_anhar_fcs,
            displacement_anhar=self.displacement_anhar,
            num_disp_anhar=self.num_disp_anhar,
            fcs_cutoff_radius=self.fcs_cutoff_radius,
            renorm_phonon=self.renorm_phonon,
            renorm_temp=self.renorm_temp,
            cal_ther_cond=self.cal_ther_cond,
            ther_cond_mesh=self.ther_cond_mesh,
            ther_cond_temp=self.ther_cond_temp,
            sym_reduce=self.sym_reduce,
            symprec=self.symprec,
            use_symmetrized_structure=self.use_symmetrized_structure,
            kpath_scheme=self.kpath_scheme,
            code=self.code,
            mp_id=self.mp_id,
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
            **self.generate_frequencies_eigenvectors_kwargs,
        )

        jobs.append(phonon_collect)

        # create a flow including all jobs for a phonon computation
        return Flow(jobs, phonon_collect.output)

    @property
    @abstractmethod
    def prev_calc_dir_argname(self) -> str | None:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
