"""Flows for calculating Grueneisen-Parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

from atomate2.common.flows.gruneisen import BaseGruneisenMaker
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.flows.phonons import PhononMaker
from atomate2.vasp.jobs.core import (
    StaticMaker,
    TightRelaxConstVolMaker,
    TightRelaxMaker,
)
from atomate2.vasp.jobs.phonons import PhononDisplacementMaker

# TODO: treat prev_vasp_dir correctly


@dataclass
class GruneisenMaker(BaseGruneisenMaker):
    """
    Maker to calculate Grueneisen parameters with a force field and Phonopy.

    Calculate the harmonic phonons of a material for and compute Grueneisen parameters.
    Initially, a tight structural relaxation is performed to obtain a structure without
    forces on the atoms. The optimized structure (ground state) is further expanded and
    shrunk by 1 % of its volume. Subsequently, supercells with one displaced atom are
    generated for all the three structures (ground state, expanded and shrunk volume)
    and accurate forces are computed for these structures. With the help of phonopy,
    these forces are then converted into a dynamical matrix. This dynamical matrix of
    three structures is then used as input phonopy Grueneisen api to compute Grueneisen
    parameters are computed.


    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    bulk_relax_maker: .ForceFieldRelaxMaker, .BaseAimsMaker, .BaseVaspMaker, or None
        A maker to perform an initial tight relaxation on the bulk.
    code: str
        determines the dft or force field code.
    const_vol_relax_maker: .ForceFieldRelaxMaker, .BaseAimsMaker,
        .BaseVaspMaker, or None. A maker to perform a tight relaxation
        on the expanded and shrunk structures at constant volume.
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
    mesh: tuple|float
        Mesh numbers along a, b, c axes used for Grueneisen parameter computation.
        Or float to indicate a kpoint density.
    phonon_displacement_maker: .StaticMaker | None
    phonon_static_maker: .StaticMaker | None
    phonon_maker_kwargs: dict
    perc_vol: float
        Percent volume to shrink and expand ground state structure
    compute_gruneisen_param_kwargs: dict
        Keyword arguments passed to :obj:`compute_gruneisen_param`.
    """

    name: str = "Gruneisen"
    bulk_relax_maker: TightRelaxMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )
    code: str = "vasp"

    const_vol_relax_maker: TightRelaxMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(
            TightRelaxConstVolMaker()
        )
    )
    kpath_scheme: str = "seekpath"
    phonon_displacement_maker: PhononDisplacementMaker | None = field(
        default_factory=PhononDisplacementMaker
    )
    phonon_maker_kwargs: dict = field(default_factory=dict)
    vol: float = 0.01
    mesh: tuple | float = 7_000
    compute_gruneisen_param_kwargs: dict = field(default_factory=dict)
    symprec: float = 1e-4

    @property
    def prev_calc_dir_argname(self) -> str:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
        return "prev_dir"

    def initialize_phonon_maker(
        self,
        phonon_displacement_maker: PhononDisplacementMaker,
        phonon_static_maker: StaticMaker,
        bulk_relax_maker: TightRelaxMaker | None,
        phonon_maker_kwargs: dict,
        symprec: float = 1e-4,
    ) -> PhononMaker | None:
        """Initialize Phonon Maker.

        Parameters
        ----------
        phonon_displacement_maker: .ForceFieldStaticMaker|None
            Computes Forces for displaced structures in
            harmonic phonon runs
        phonon_static_maker: .ForceFieldStaticMaker|None
            Additional static maker to compute
            energies and volume after optimization
        bulk_relax_maker: .ForceFieldRelaxMaker|None
            Relax Maker for Phonon Run. Typically None.
        phonon_maker_kwargs: dict
            Dict to set additional info for phonons.

        Returns
        -------
        .PhononMaker
        """
        return PhononMaker(
            phonon_displacement_maker=phonon_displacement_maker,
            static_energy_maker=phonon_static_maker,
            bulk_relax_maker=bulk_relax_maker,
            symprec=symprec,
            **phonon_maker_kwargs,
        )
