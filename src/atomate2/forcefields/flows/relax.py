"""Flows to combine a force field relaxation with another job (e.g. DFT relaxation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.forcefields import MLFF
from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from atomate2.vasp.jobs.core import RelaxMaker

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

    from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class CHGNetVaspRelaxMaker(Maker):
    """
    Maker to (pre)relax a structure using CHGNet and then run VASP.

    Parameters
    ----------
    name : str
        Name of the flow produced by this maker.
    chgnet_maker : .ForceFieldRelaxMaker
        Maker to generate a CHGNet relaxation job.
    vasp_maker : .BaseVaspMaker
        Maker to generate a VASP relaxation job.
    """

    name: str = f"{MLFF.CHGNet} relax followed by a VASP relax"
    chgnet_maker: ForceFieldRelaxMaker = field(
        default_factory=lambda: ForceFieldRelaxMaker(force_field_name="CHGNet")
    )
    vasp_maker: BaseVaspMaker = field(default_factory=RelaxMaker)

    def make(self, structure: Structure) -> Flow:
        """Create a flow with a CHGNet (pre)relaxation followed by a VASP relaxation.

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
        chgnet_relax_job.name = f"{MLFF.CHGNet} pre-relax"

        vasp_job = self.vasp_maker.make(chgnet_relax_job.output.structure)

        return Flow([chgnet_relax_job, vasp_job], vasp_job.output, name=self.name)


@dataclass
class M3GNetVaspRelaxMaker(Maker):
    """
    Maker to (pre)relax a structure using M3GNet and then run VASP.

    Parameters
    ----------
    name : str
        Name of the flow produced by this maker.
    m3gnet_maker : .M3GNetRelaxMaker
        Maker to generate a M3GNet relaxation job.
    vasp_maker : .BaseVaspMaker
        Maker to generate a VASP relaxation job.
    """

    name: str = f"{MLFF.M3GNet} relax followed by a VASP relax"
    m3gnet_maker: ForceFieldRelaxMaker = field(
        default_factory=lambda: ForceFieldRelaxMaker(force_field_name="M3GNet")
    )
    vasp_maker: BaseVaspMaker = field(default_factory=RelaxMaker)

    def make(self, structure: Structure) -> Flow:
        """Create a flow with a M3GNet (pre)relaxation followed by a VASP relaxation.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure.

        Returns
        -------
        Flow
            A flow containing a M3GNet relaxation followed by a VASP relaxation
        """
        m3gnet_relax_job = self.m3gnet_maker.make(structure)
        m3gnet_relax_job.name = f"{MLFF.M3GNet} pre-relax"

        vasp_job = self.vasp_maker.make(m3gnet_relax_job.output.structure)

        return Flow([m3gnet_relax_job, vasp_job], vasp_job.output, name=self.name)
