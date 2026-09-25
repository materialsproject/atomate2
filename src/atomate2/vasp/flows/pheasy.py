"""Define the VASP PhononMaker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from atomate2 import SETTINGS
from atomate2.common.flows.pheasy import BasePhononMaker
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import DielectricMaker, StaticMaker, TightRelaxMaker
from atomate2.vasp.jobs.phonons import PhononDisplacementMaker
from atomate2.vasp.sets.core import StaticSetGenerator

if TYPE_CHECKING:
    from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class PhononMaker(BasePhononMaker):
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
        this flow. Otherwise, a different space group might be detected and too
        many displacement calculations will be generated.
        It is recommended to check the convergence parameters here and
        adjust them if necessary. The default might not be strict enough
        for your specific case.

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
    prefer_90_degrees: bool
        not used by this workflow, which uses force_90_degrees instead.
    allow_orthorhombic: bool
        not used by this workflow.
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
    bulk_relax_maker: .BaseVaspMaker, or None
        A maker to perform a tight relaxation on the bulk.
        Set to ``None`` to skip the
        bulk relaxation
    static_energy_maker: .BaseVaspMaker, or None
        A maker to perform the computation of the DFT energy on the bulk.
        Set to ``None`` to skip the
        static energy computation
    born_maker: .BaseVaspMaker, or None
        Maker to compute the BORN charges.
    phonon_displacement_maker:  .BaseVaspMaker
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
    code: str = "vasp"
        determines the DFT code. currently only vasp is implemented.
        This keyword might enable the implementation of other codes
        in the future
    store_force_constants: bool
        if True, force constants will be stored
    socket: bool
        If True, use the socket for the calculation
    """

    name: str = "phonon"
    sym_reduce: bool = True
    symprec: float = SETTINGS.PHONON_SYMPREC
    cal_anhar_fcs: bool = False
    displacement: float = 0.01
    displacement_anhar: float = 0.08
    num_displaced_supercells: int = 0
    num_disp_anhar: int = 0
    fcs_cutoff_radius: list = field(default_factory=lambda: [-1, 12, 10])
    min_length: float | None = 8.0
    max_atoms: float | None = 200
    force_90_degrees: bool = True
    force_diagonal: bool = True
    allow_orthorhombic: bool = False
    prefer_90_degrees: bool = True
    get_supercell_size_kwargs: dict = field(default_factory=dict)
    use_symmetrized_structure: Literal["primitive", "conventional"] | None = None
    create_thermal_displacements: bool = False
    generate_frequencies_eigenvectors_kwargs: dict = field(default_factory=dict)
    kpath_scheme: str = "seekpath"
    store_force_constants: bool = True
    socket: bool = False
    code: str = "vasp"
    bulk_relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )
    static_energy_maker: BaseVaspMaker | None = field(
        default_factory=lambda: StaticMaker(
            input_set_generator=StaticSetGenerator(auto_ispin=True)
        )
    )
    born_maker: BaseVaspMaker | None = field(default_factory=DielectricMaker)
    phonon_displacement_maker: BaseVaspMaker = field(
        default_factory=PhononDisplacementMaker
    )

    @property
    def prev_calc_dir_argname(self) -> str:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
        return "prev_dir"
