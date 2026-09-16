"""Define the VASP PhononMaker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from atomate2 import SETTINGS
from atomate2.common.flows.hiphive import BasePhononMaker
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import DielectricMaker, StaticMaker, TightRelaxMaker
from atomate2.vasp.jobs.phonons import PhononDisplacementMaker
from atomate2.vasp.sets.core import StaticSetGenerator

if TYPE_CHECKING:
    from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class PhononMaker(BasePhononMaker):
    """Maker to calculate harmonic phonons with the cluster-expansion code hiPhive.

    Calculate the zero-K harmonic phonons of a material and higher-order FCs.
    Initially, a tight structural relaxation is performed to obtain a structure
    without forces on the atoms. Subsequently, supercells with all atoms displaced
    by a small amplitude (generally using 0.01 A) are generated and accurate forces
    are computed for these structures for the second order force constants. hiPhive
    builds a cluster space from the supercell symmetry and fits the force constants
    to those forces by regression. In this Workflow, we separate the harmonic phonon
    calculations and anharmonic force constants calculations. To correct for
    polarization effects, a correction of the dynamical matrix based on BORN charges
    can be performed. Finally, phonon densities of states, phonon band structures and
    thermodynamic properties are computed. For the anharmonic force constants, the
    supercells with all atoms displaced by a larger amplitude (generally using 0.08 A)
    are generated and accurate forces are computed for these structures. The third-
    and fourth-order force constants are fitted from the same cluster space.

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
        and to handle all symmetry-related tasks in hiPhive, we recommend to
        use the value of 1e-3.
    displacement: float
        displacement distance for phonons, for most cases 0.01 A is a good choice,
        but it can be increased to 0.02 A for heavier elements.
    num_displaced_supercells: int
        number of displacements to be generated using a random-displacement approach
        for harmonic phonon calculations. The default value is 0 and the number of
        displacements is automatically determined by the number of atoms in the
        supercell and its space group.
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
    displacement: float = 0.01
    num_displaced_supercells: int = 0
    cutoff_2nd: float | None = None
    fit_method: str = "lasso"
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
