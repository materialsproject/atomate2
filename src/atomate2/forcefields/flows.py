"""Flows to combine a force field relaxation with another job (e.g. DFT relaxation)."""

from __future__ import annotations

from dataclasses import dataclass, field

from jobflow import Flow, Maker
from pymatgen.core.structure import Structure

from atomate2.forcefields.jobs import CHGNetRelaxMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import RelaxMaker


@dataclass
class CHGNetVaspRelaxMaker(Maker):
    """
    Maker to (pre)relax a structure using CHGNet and then run VASP.

    Parameters
    ----------
    name : str
        Name of the flow produced by this maker.
    chgnet_maker : .CHGNetRelaxMaker
        Maker to generate a CHGNet relaxation job.
    vasp_maker : .BaseVaspMaker
        Maker to generate a VASP relaxation job.

    """

    name: str = "CHGNet relax followed by a VASP relax"
    chgnet_maker: CHGNetRelaxMaker = field(default_factory=CHGNetRelaxMaker)
    vasp_maker: BaseVaspMaker = field(default_factory=RelaxMaker)

    def make(self, structure: Structure):
        """
        Create a flow with a CHGNet (pre)relaxation followed by a VASP relaxation.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure.

        Returns
        -------
        Flow
            A flow containing a CHGNet relaxation followed by a VASP relaxation

        """
        chgnet_relax_job = self.chgnet_maker.make(structure)
        chgnet_relax_job.name = "CHGNet pre-relax"

        vasp_job = self.vasp_maker.make(chgnet_relax_job.output.structure)

        return Flow([chgnet_relax_job, vasp_job], vasp_job.output, name=self.name)
