"""Flow for phonons with Quantum ESPRESSO and pheasy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jobflow import Flow

from atomate2 import SETTINGS
from atomate2.common.flows.pheasy import BasePhononMaker
from atomate2.espresso.jobs import (
    BaseEspressoMaker,
    EspressoDielectricMaker,
    EspressoRelaxMaker,
    EspressoStaticMaker,
    rotate_to_standard_orientation,
)

if TYPE_CHECKING:
    from pymatgen.core import Structure


@dataclass
class PhononMaker(BasePhononMaker):
    """
    Maker to calculate phonons with Quantum ESPRESSO and pheasy.

    The flow first rotates the structure to the spglib standard orientation, so
    that pw.x finds all symmetry operations. pw.x then relaxes the structure
    (vc-relax) and computes the forces of the displaced supercells, and pheasy
    fits the force constants. ph.x computes the dielectric tensor and the Born
    effective charges for the non-analytical term correction. The dielectric job
    needs a nonmagnetic insulator and does not check for one, so set
    ``born_maker=None`` for metals and magnetic materials. See
    :obj:`atomate2.common.flows.pheasy.BasePhononMaker` for the other parameters.

    Parameters
    ----------
    symprec : float
        The symmetry precision of the phonon calculation.
    bulk_relax_maker : .BaseEspressoMaker or None
        A maker to relax the bulk structure. Set to ``None`` to skip the
        relaxation.
    static_energy_maker : .BaseEspressoMaker or None
        A maker to compute the total energy of the bulk structure. Set to
        ``None`` to skip it.
    born_maker : .EspressoDielectricMaker or None
        A maker to compute the dielectric tensor and the Born effective charges.
        Set to ``None`` to skip the non-analytical term correction.
    phonon_displacement_maker : .BaseEspressoMaker
        A maker to compute the forces of the displaced supercells.
    code : str
        The code that computes the forces.
    """

    symprec: float = SETTINGS.PHONON_SYMPREC
    bulk_relax_maker: BaseEspressoMaker | None = field(
        default_factory=EspressoRelaxMaker
    )
    static_energy_maker: BaseEspressoMaker | None = field(
        default_factory=EspressoStaticMaker
    )
    born_maker: EspressoDielectricMaker | None = field(
        default_factory=EspressoDielectricMaker
    )
    phonon_displacement_maker: BaseEspressoMaker = field(
        default_factory=EspressoStaticMaker
    )
    code: str = "espresso"

    @property
    def prev_calc_dir_argname(self) -> None:
        """
        Name of the argument that passes the previous calculation directory.

        The Quantum ESPRESSO jobs start from the structure only, so it is None.
        """
        return

    def make(self, structure: Structure, *args: Any, **kwargs: Any) -> Flow:
        """
        Make the flow from the structure in the spglib standard orientation.

        Born charges and dielectric tensors given as arguments must be in that
        orientation. See :obj:`atomate2.common.flows.pheasy.BasePhononMaker.make`
        for the parameters.
        """
        rotate = rotate_to_standard_orientation(structure, self.symprec)
        phonons = super().make(rotate.output, *args, **kwargs)
        return Flow([rotate, phonons], phonons.output, name=self.name)
