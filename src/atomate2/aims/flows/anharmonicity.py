"""Defines the anharmonicity quantification workflows for FHI-aims."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from atomate2.common.flows.anharmonicity import BaseAnharmonicityMaker

if TYPE_CHECKING:
    from atomate2.aims.flows.phonons import PhononMaker


@dataclass
class AnharmonicityMaker(BaseAnharmonicityMaker):
    """
    Maker to calculate the anharmonicity score of a material.

    Calculate sigma^A as defined in doi.org/10.1103/PhysRevMaterials.4.083809, by
    first calculating the phonons for a material and then generating the one-shot
    sample and calculating the DFT and harmonic forces.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    phonon_maker: BasePhononMaker
        The maker to generate the phonon model
    """

    name: str = "anharmonicity"
    phonon_maker: PhononMaker = None

    @property
    def prev_calc_dir_argname(self) -> str:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
        return "prev_dir"
