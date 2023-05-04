"""Flows to combine a force field relaxation with another job (e.g. DFT relaxation)."""

from __future__ import annotations

from dataclasses import dataclass, field

from jobflow import Flow, Maker
from pymatgen.core.structure import Structure

from atomate2.forcefields.jobs import CHGNetRelaxMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import RelaxMaker


@dataclass
class CHGNetToVaspMaker(Maker):
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

    name: str = "CHGNet relax to VASP relax"
    CHGNet_maker: CHGNetRelaxMaker = field(default_factory=CHGNetRelaxMaker)
    VASP_maker: BaseVaspMaker = field(default_factory=RelaxMaker)

    def make(self, structure: Structure):
        """
        Create a flow with a CHGNet (pre)relaxation followed by a VASP relaxation.

        Parameters
        ----------
        structure: ~pymatgen.core.structure.Structure
            A pymatgen structure.

        Returns
        -------
        Flow
            A flow containing a CHGNet relaxation followed by a VASP relaxation

        """
        CHGNet_relax_job = self.CHGNet_maker.make(structure)
        CHGNet_relax_job.name = "CHGNet pre-relax job"

        VASP_job = self.VASP_maker.make(CHGNet_relax_job.output.structure)

        return Flow([CHGNet_relax_job, VASP_job], VASP_job.output, name=self.name)
