"""Flow for calculating harmonic and anharmonic FCs with pheasy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pymatgen.util.due import Doi, due

from atomate2.common.flows.phonons import BasePhononMaker as PurePhonopyMaker
from atomate2.common.jobs.pheasy import (
    _check_anharmonic_settings,
    generate_frequencies_eigenvectors,
    generate_phonon_displacements,
    get_supercell_size,
)
from atomate2.common.jobs.phonons import run_phonon_displacements

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from emmet.core.math import Matrix3D
    from jobflow import Flow, Job
    from pymatgen.core.structure import Structure

    from atomate2.aims.jobs.base import BaseAimsMaker
    from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker

SUPPORTED_CODES = frozenset(("vasp", "aims", "forcefields"))


@due.dcite(
    Doi("10.26434/chemrxiv.15004632/v1"),
    description="Materials Project's Harmonic Phonon Database.",
)
@due.dcite(
    Doi("10.48550/arXiv.2508.01020"),
    description="Pheasy code for (an)harmonic force constants.",
)
@due.dcite(
    Doi("10.1088/0953-8984/26/22/225402"),
    description="ALM, used to count the free force constants.",
)
@dataclass
class BasePhononMaker(PurePhonopyMaker, ABC):
    """Maker to calculate harmonic phonons and anharmonic FCs with Pheasy.

    Calculate the zero-K harmonic phonons of a material and, optionally, its
    anharmonic FCs. Initially, a tight structural relaxation is performed to obtain
    a structure without forces on the atoms. Subsequently, displaced supercells
    (generally using 0.01 A) are generated and accurate forces are computed for
    them. If phonopy needs more than three finite displacements, all atoms are
    displaced randomly and pheasy fits the second order force constants with LASSO.
    Otherwise, the finite displacements are used with a least-squares fit. These
    force constants are then converted into a dynamical matrix. In this Workflow,
    we separate the harmonic phonon calculations and anharmonic force constants
    calculations. To correct for polarization effects, a
    correction of the dynamical matrix based on BORN charges can be performed. Finally,
    phonon densities of states, phonon band structures and thermodynamic properties
    are computed. For the anharmonic force constants, the supercells with all atoms
    displaced by a larger amplitude (generally using 0.08 A) are generated and accurate
    forces are computed for these structures. With the help of pheasy (LASSO technique),
    the third-order (and optionally fourth-order) force constants are extracted.

    .. Note::
        It is heavily recommended to symmetrize the structure before passing it to
        this flow. Otherwise, a different space group might be detected and too many
        displacement calculations will be required for pheasy phonon calculation. It
        is recommended to check the convergence parameters here and adjust them if
        necessary. The default might not be strict enough for your specific case.
        The residual forces of the undisplaced supercell are always subtracted from
        the forces of the displaced supercells.

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
        supercell and its space group. Not used when phonopy needs at most three
        finite displacements.
    cal_anhar_fcs: bool
        if set to True, anharmonic force constants(FCs) up to order
        anhar_max_order will be calculated. The default value is False, and only
        harmonic phonons will be calculated.
    displacement_anhar: float
        displacement distance for the anharmonic force constants(FCs). The default
        is 0.08 A.
    num_disp_anhar: int
        number of displacements to be generated using a random-displacement approach
        for anharmonic phonon calculations. A non-zero value is used as given. The
        default value is 0, and then the number is set so that the fit has 100 force
        equations per free force constant, with at least 20 displaced supercells.
        Above 600 the job stops. The free force constants are counted with ALM, which
        uses its own symmetry search, for this supercell and fcs_cutoff_radius. The
        second-order FCs are included when "one-shot" is requested.
    anhar_max_order: int
        highest order of the anharmonic FCs, 3 or 4. The default of 3 fits the
        third-order FCs. 4 also fits fourth-order FCs.
    anhar_fit_methods: Sequence[Literal["cocktail", "one-shot"]]
        how the anharmonic FCs are fitted to the large-distance dataset.
        "cocktail" keeps the second-order FCs fixed to the harmonic fit and fits
        only the higher orders. "one-shot" fits the second- and higher-order FCs
        together from the same dataset and writes them to the "one_shot" folder.
        Both can be requested. The default is ("cocktail",). The cocktail FCs are
        written to the job folder, and the one-shot FCs, including fc2.hdf5, to its
        one_shot subfolder: FORCE_CONSTANTS_3RD and fc3.hdf5, plus
        FORCE_CONSTANTS_4TH and fc4.hdf5 at fourth order. They are not stored in the
        output document. The LASSO fits are seeded, so a run is reproducible. With
        few displaced supercells, the one-shot result depends on that seed.
    anhar_alpha_min: int
        base-10 exponent of the smallest LASSO penalty tried by cross-validation
        in the anharmonic fits. It must be an integer below -2. The default of -12
        is below pheasy's own default of -6. If the chosen penalty lands on either
        end of the search, 10**anhar_alpha_min or pheasy's 1e-2, a warning is
        raised.
    fcs_cutoff_radius: list
        cutoff distance in Bohr for each FC order, starting at second order. The
        default value is [-1, 12, 10]. The first entry is not used, since pheasy
        fits the second-order FCs without a cutoff. The second and third entries
        are the cutoffs for third- and fourth-order FCs. The cutoff of each fitted
        order, up to anhar_max_order, must be positive. Longer cutoffs increase
        the number of free FCs, and with it the number of displaced supercells.
    min_length: float
        minimum length of lattice constants will be used to create the supercell,
        the default value is 8.0 A. It can be increased for larger supercells.
    max_atoms: float | None
        maximum number of atoms in the supercell.
    force_90_degrees: bool
        if set to True, only supercells with three 90 degree angles are allowed.
    force_diagonal: bool
        if set to True, only diagonal supercell matrices are allowed. The pheasy
        commands take the diagonal of the supercell matrix.
    get_supercell_size_kwargs: dict
        not used by this workflow.
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
    anhar_max_order: int = 3
    anhar_fit_methods: Sequence[Literal["cocktail", "one-shot"]] = ("cocktail",)
    anhar_alpha_min: int = -12
    fcs_cutoff_radius: list = field(
        default_factory=lambda: [-1, 12, 10]
    )  # units in Bohr
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

    def __post_init__(self) -> None:
        """Check the anharmonic settings before any calculation is run."""
        _check_anharmonic_settings(
            self.anhar_max_order,
            self.anhar_fit_methods,
            self.cal_anhar_fcs,
            self.fcs_cutoff_radius,
            self.anhar_alpha_min,
            self.num_disp_anhar,
        )

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
            anhar_max_order=self.anhar_max_order,
            anhar_fit_methods=self.anhar_fit_methods,
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
            store_displaced_structures=True,
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
            cal_anhar_fcs=self.cal_anhar_fcs,
            fcs_cutoff_radius=self.fcs_cutoff_radius,
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
            num_displaced_supercells=self.num_displaced_supercells,
            anhar_max_order=self.anhar_max_order,
            anhar_fit_methods=self.anhar_fit_methods,
            anhar_alpha_min=self.anhar_alpha_min,
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
        Job | Flow
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
