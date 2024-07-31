"""Flows for calculating Grueneisen-Parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

from atomate2 import SETTINGS
from atomate2.common.flows.gruneisen import BaseGruneisenMaker
from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.forcefields.jobs import CHGNetRelaxMaker, ForceFieldRelaxMaker


@dataclass
class GruneisenMaker(BaseGruneisenMaker):
    """
    Maker to calculate Grueneisen parameters with a force field and Phonopy.

    Calculate Grueneisen parameters by a finite volume change approach based on
    harmonic phonons.
    Initially, a tight structural relaxation is performed to obtain a structure without
    forces on the atoms. The optimized structure (ground state) is further expanded and
    shrunk by 1 % of its volume. Subsequently, supercells with one displaced atom are
    generated for all the three structures (ground state, expanded and shrunk volume)
    and accurate forces are computed for these structures. With the help of phonopy,
    these forces are then converted into a dynamical matrix. This dynamical matrix of
    three structures is then used as an input for the phonopy Grueneisen api
    to compute Grueneisen parameters.


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
    phonon_maker: .PhononMaker
        PhononMaker to run the phonon workflow.
    perc_vol: float
        Percent volume to shrink and expand ground state structure
    compute_gruneisen_param_kwargs: dict
        Keyword arguments passed to :obj:`compute_gruneisen_param`.
    symprec: float
        Symmetry precision for symmetry checks and phonon runs.
    """

    name: str = "Gruneisen"
    bulk_relax_maker: ForceFieldRelaxMaker | None = field(
        default_factory=lambda: CHGNetRelaxMaker(relax_kwargs={"fmax": 0.00001})
    )
    code: str = "forcefields"
    const_vol_relax_maker: ForceFieldRelaxMaker = field(
        default_factory=lambda: CHGNetRelaxMaker(
            relax_kwargs={"fmax": 0.00001}, relax_cell=False
        )
    )
    kpath_scheme: str = "seekpath"
    phonon_maker: PhononMaker = field(
        default_factory=lambda: PhononMaker(
            bulk_relax_maker=None, static_energy_maker=None
        )
    )
    perc_vol: float = 0.01
    mesh: tuple | float = 7_000
    compute_gruneisen_param_kwargs: dict = field(default_factory=dict)
    symprec: float = SETTINGS.PHONON_SYMPREC

    @property
    def prev_calc_dir_argname(self) -> None:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
        return
