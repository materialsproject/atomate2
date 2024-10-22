"""Flows for calculating phonons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from atomate2 import SETTINGS
from atomate2.common.flows.phonons import BasePhononMaker
from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker


@dataclass
class PhononMaker(BasePhononMaker):
    """
    Maker to calculate harmonic phonons with a force field.

    Calculate the harmonic phonons of a material. Initially, a tight structural
    relaxation is performed to obtain a structure without forces on the atoms.
    Subsequently, supercells with one displaced atom are generated and accurate
    forces are computed for these structures. With the help of phonopy, these
    forces are then converted into a dynamical matrix. To correct for polarization
    effects, a correction of the dynamical matrix based on BORN charges can
    be performed. The BORN charges can be supplied manually.
    Finally, phonon densities of states, phonon band structures
    and thermodynamic properties are computed.

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
        Name of the flows produced by this maker.
    sym_reduce : bool
        Whether to reduce the number of deformations using symmetry.
    symprec : float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy
    displacement: float
        displacement distance for phonons
    min_length: float
        min length of the supercell that will be built
    prefer_90_degrees: bool
        if set to True, supercell algorithm will first try to find a supercell
        with 3 90 degree angles
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
    bulk_relax_maker : .ForceFieldRelaxMaker or None
        A maker to perform a tight relaxation on the bulk.
        Set to ``None`` to skip the
        bulk relaxation
    static_energy_maker : .ForceFieldRelaxMaker or None
        A maker to perform the computation of the DFT energy on the bulk.
        Set to ``None`` to skip the
        static energy computation
    born_maker: .ForceFieldStaticMaker or None
        Maker to compute the BORN charges.
    phonon_displacement_maker : .ForceFieldStaticMaker or None
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
    symprec: float = SETTINGS.PHONON_SYMPREC
    displacement: float = 0.01
    min_length: float | None = 20.0
    prefer_90_degrees: bool = True
    get_supercell_size_kwargs: dict = field(default_factory=dict)
    use_symmetrized_structure: Literal["primitive", "conventional"] | None = None
    bulk_relax_maker: ForceFieldRelaxMaker | None = field(
        default_factory=lambda: ForceFieldRelaxMaker(
            force_field_name="CHGNet", relax_kwargs={"fmax": 0.00001}
        )
    )
    static_energy_maker: ForceFieldStaticMaker | None = field(
        default_factory=lambda: ForceFieldStaticMaker(force_field_name="CHGNet")
    )
    phonon_displacement_maker: ForceFieldStaticMaker = field(
        default_factory=lambda: ForceFieldStaticMaker(force_field_name="CHGNet")
    )
    create_thermal_displacements: bool = False
    generate_frequencies_eigenvectors_kwargs: dict = field(default_factory=dict)
    kpath_scheme: str = "seekpath"
    store_force_constants: bool = True
    code: str = "forcefields"
    born_maker: ForceFieldStaticMaker | None = None

    @property
    def prev_calc_dir_argname(self) -> None:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
        return
