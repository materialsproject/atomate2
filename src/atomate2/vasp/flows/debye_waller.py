"""Define the VASP Debye-Waller maker."""

from __future__ import annotations

from dataclasses import dataclass, field

from atomate2.common.flows.debye_waller import BaseDebyeWallerMaker
from atomate2.vasp.flows.phonons import PhononMaker


@dataclass
class DebyeWallerMaker(BaseDebyeWallerMaker):
    """
    Maker to calculate Debye-Waller factors with VASP and phonopy.

    The phonon flow gives the Cartesian thermal displacement matrix U of each
    site of the primitive cell with phonopy. The X-ray, neutron and electron
    diffraction patterns are computed with pymatgen with the Debye-Waller
    factors exp(-2 pi^2 g^T U g) at each temperature, and once without them.
    The frequencies are not renormalized with temperature.

    By default, the phonon flow is the atomate2 VASP phonon flow. It computes
    the Born charges and the dielectric tensor for the non-analytical
    correction.

    This workflow is new and has not been tested widely. It might still change
    in future versions.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    phonon_maker: .PhononMaker
        The VASP phonon maker. It must have create_thermal_displacements=True.
    xrd_kwargs: dict or None
        Keyword arguments of pymatgen's XRDCalculator.
    nd_kwargs: dict or None
        Keyword arguments of pymatgen's NDCalculator.
    tem_kwargs: dict or None
        Keyword arguments of pymatgen's TEMCalculator.
    """

    phonon_maker: PhononMaker = field(default_factory=PhononMaker)
