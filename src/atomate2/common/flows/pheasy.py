"""Flow for calculating (an)harmonic FCs and phonon energy renorma. with pheasy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from atomate2.common.flows.phonons import BasePhononMaker as PurePhonopyMaker
from atomate2.common.jobs.pheasy import (
    generate_frequencies_eigenvectors,
    generate_phonon_displacements,
    get_supercell_size,
    run_phonon_displacements,
)

if TYPE_CHECKING:
    from pathlib import Path

    from emmet.core.math import Matrix3D
    from jobflow import Flow, Job
    from pymatgen.core.structure import Structure

    from atomate2.aims.jobs.base import BaseAimsMaker
    from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker

SUPPORTED_CODES = frozenset(("vasp", "aims", "forcefields"))


@dataclass
class BasePhononMaker(PurePhonopyMaker, ABC):
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
    get_supercell_size_kwargs: dict = field(default_factory=dict)
    use_symmetrized_structure: Literal["primitive", "conventional"] | None = None
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
    store_force_constants: bool = True
    socket: bool = False

    def get_displacements(
        self, structure: Structure, supercell_matrix: Matrix3D
    ) -> Job | Flow:
        """
        Get displaced supercells.

        Parameters
        ----------
        structure: Structure
        supercell_matrix: Matrix3D

        Returns
        -------
        Job|Flow
        """
        return generate_phonon_displacements(
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

    def run_displacements(
        self,
        displacements: Job | Flow,
        prev_dir: str | Path | None,
        structure: Structure,
        supercell_matrix: Matrix3D,
    ) -> Job | Flow:
        """
        Perform displacement calculations.

        Parameters
        ----------
        displacements: Job | Flow
        prev_dir: str | Path | None
        structure: Structure
        supercell_matrix:  Matrix3D

        Returns
        -------
        Job | Flow
        """
        # perform the phonon displacement calculations
        return run_phonon_displacements(
            displacements=displacements.output,
            structure=structure,
            supercell_matrix=supercell_matrix,
            phonon_maker=self.phonon_displacement_maker,
            socket=self.socket,
            prev_dir_argname=self.prev_calc_dir_argname,
            prev_dir=prev_dir,
        )

    def get_results(
        self,
        born: Matrix3D,
        born_run_job_dir: str,
        born_run_uuid: str,
        displacement_calcs: Job | Flow,
        epsilon_static: Matrix3D,
        optimization_run_job_dir: str,
        optimization_run_uuid: str,
        static_run_job_dir: str,
        static_run_uuid: str,
        structure: Structure,
        supercell_matrix: Matrix3D | None,
        total_dft_energy: float,
    ) -> Job | Flow:
        """
        Calculate the harmonic phonons etc.

        Parameters
        ----------
        born: Matrix3D
        born_run_job_dir:  str
        born_run_uuid: str
        displacement_calcs: Job | Flow
        epsilon_static: Matrix3D
        optimization_run_job_dir:str
        optimization_run_uuid:str
        static_run_job_dir:str
        static_run_uuid:str
        structure: Structure
        supercell_matrix: Matrix3D
        total_dft_energy: float

        Returns
        -------
        Job | Flow
        """
        return generate_frequencies_eigenvectors(
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

    def get_supercell_matrix(self, structure: Structure) -> Job | Flow:
        """
        Get supercell matrix.

        Parameters
        ----------
        structure: Structure

        Returns
        -------
        Job|Flow
        """
        return get_supercell_size(
            structure,
            self.min_length,
            self.max_atoms,
            self.force_90_degrees,
            self.force_diagonal,
        )

    @property
    @abstractmethod
    def prev_calc_dir_argname(self) -> str | None:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
