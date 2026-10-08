"""Define the force field Debye-Waller maker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.common.flows.debye_waller import BaseDebyeWallerMaker
from atomate2.forcefields.flows.phonons import PhononMaker

if TYPE_CHECKING:
    from typing_extensions import Self

    from atomate2.forcefields import MLFF


@dataclass
class DebyeWallerMaker(BaseDebyeWallerMaker):
    """
    Maker to calculate Debye-Waller factors with a force field and phonopy.

    The phonon flow gives the harmonic force constants. phonopy then gives the
    Cartesian thermal displacement matrix U of each site of the primitive cell.
    The X-ray, neutron and electron diffraction patterns are computed with
    pymatgen with the Debye-Waller factors exp(-2 pi^2 g^T U g) at each
    temperature, and once without them. The frequencies are not renormalized
    with temperature.

    By default, the phonon flow uses MACE-MP-0. Use :obj:`from_force_field_name`
    to run it with another force field.

    This workflow is new and has not been tested widely. It might still change
    in future versions.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    phonon_maker: .PhononMaker
        The force field phonon maker. It must have store_force_constants=True.
    temperatures: list[float]
        Temperatures in K, not negative.
    mesh: tuple[int, int, int] | float
        q-point mesh for the thermal displacements, or a q-point density used
        as kppa in pymatgen's Kpoints.automatic_density for the primitive cell.
    freq_min: float
        Modes with abs(f) below this frequency in THz are left out. The three
        acoustic modes at Gamma are always left out.
    include_imaginary_modes: bool
        Also include the modes below -freq_min, as if their frequency were real
        with the same magnitude.
    xrd_kwargs: dict or None
        Keyword arguments of pymatgen's XRDCalculator.
    nd_kwargs: dict or None
        Keyword arguments of pymatgen's NDCalculator.
    tem_kwargs: dict or None
        Keyword arguments of pymatgen's TEMCalculator.
    """

    phonon_maker: PhononMaker = field(
        default_factory=lambda: PhononMaker.from_force_field_name("MACE-MP-0")
    )

    @classmethod
    def from_force_field_name(
        cls,
        force_field_name: str | MLFF | dict,
        calculator_kwargs: dict | None = None,
        relax_initial_structure: bool = True,
        **kwargs,
    ) -> Self:
        """
        Create a Debye-Waller maker whose phonon flow uses one force field.

        Parameters
        ----------
        force_field_name: str or .MLFF or dict
            The name of the force field.
        calculator_kwargs: dict or None
            Keyword arguments passed to the force field calculator.
        relax_initial_structure: bool
            Whether to relax the structure before the phonon calculation.
        **kwargs
            Further keyword arguments passed to DebyeWallerMaker. A phonon maker
            given here replaces the one built for the force field.

        Returns
        -------
        DebyeWallerMaker
        """
        phonon_maker = PhononMaker.from_force_field_name(
            force_field_name,
            calculator_kwargs=calculator_kwargs,
            relax_initial_structure=relax_initial_structure,
        )
        return cls(**{"phonon_maker": phonon_maker, **kwargs})
