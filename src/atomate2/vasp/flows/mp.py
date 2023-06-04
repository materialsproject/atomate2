"""
Module defining flows for Materials Project r2SCAN workflows.

Reference: https://doi.org/10.1103/PhysRevMaterials.6.013801
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure

    from atomate2.vasp.jobs.base import BaseVaspMaker

from atomate2.vasp.jobs.mp import MPPreRelaxMaker, MPRelaxMaker

__all__ = ["MP2023RelaxMaker"]


@dataclass
class MP2023RelaxMaker(Maker):
    """
    Maker to perform a VASP r2SCAN relaxation workflow with MP settings.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    pre_relax_maker : .BaseVaspMaker
        Maker to generate the first relaxation.
    relax_maker : .BaseVaspMaker
        Maker to generate the second relaxation.
    """

    name: str = "MP 2023 Relax"
    pre_relax_maker: BaseVaspMaker = field(default_factory=MPPreRelaxMaker)
    relax_maker: BaseVaspMaker = field(default_factory=MPRelaxMaker)

    def make(self, structure: Structure, prev_vasp_dir: str | Path | None = None):
        """
        Create a flow with two chained relaxations.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_vasp_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing two relaxations.
        """
        pre_relax = self.pre_relax_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
        relax = self.relax_maker.make(
            structure=pre_relax.output.structure,
            bandgap=pre_relax.output.bandgap,
            prev_vasp_dir=pre_relax.output.dir_name,
        )
        return Flow([pre_relax, relax], relax.output, name=self.name)
