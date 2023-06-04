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

__all__ = ["MPMetaGGARelax"]


@dataclass
class MPMetaGGARelax(Maker):
    """
    Maker to perform a VASP r2SCAN relaxation workflow with MP settings.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    initial_relax_maker : .BaseVaspMaker
        Maker to generate the first relaxation.
    initial_static_maker : .BaseVaspMaker
        Maker to generate the static calculation before the relaxation.
    final_relax_maker : .BaseVaspMaker
        Maker to generate the second relaxation.
    """

    name: str = "MP Meta-GGA Relax"
    initial_relax_maker: BaseVaspMaker | None = field(default_factory=MPPreRelaxMaker)
    initial_static_maker: BaseVaspMaker | None = None
    final_relax_maker: BaseVaspMaker | None = field(default_factory=MPRelaxMaker)

    def make(self, structure: Structure, prev_vasp_dir: str | Path | None = None):
        """
        Create a flow consisting of a cheap pre-relaxation step and a high-quality
        relaxation step. An optional static calculation can be performed before and
        after the relaxation.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_vasp_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing the MP relaxation workflow.
        """
        # Define initial parameters
        bandgap = 0.0
        jobs = []

        # Run a pre-relaxation (typically PBEsol)
        if self.initial_relax_maker:
            initial_relax = self.initial_relax_maker.make(
                structure, prev_vasp_dir=prev_vasp_dir
            )
            output = initial_relax.output
            structure = output.structure
            bandgap = output.bandgap
            prev_vasp_dir = output.dir_name
            jobs += [initial_relax]

        # Run a static calculation (typically r2SCAN) before the relaxation.
        # See https://doi.org/10.1038/s41524-022-00881-w
        if self.initial_static_maker:
            initial_static = self.initial_static_maker.make(
                structure,
                bandgap=bandgap,
                prev_vasp_dir=prev_vasp_dir,
            )
            output = initial_static.output
            structure = output.structure
            bandgap = output.bandgap
            prev_vasp_dir = output.dir_name
            jobs += [initial_static]

        # Run a relaxation (typically r2SCAN)
        if self.final_relax_maker:
            final_relax = self.final_relax_maker.make(
                structure=structure,
                bandgap=bandgap,
                prev_vasp_dir=prev_vasp_dir,
            )
            output = final_relax.output
            structure = output.structure
            bandgap = output.bandgap
            prev_vasp_dir = output.dir_name
            jobs += [final_relax]

        return Flow(jobs, output, name=self.name)
