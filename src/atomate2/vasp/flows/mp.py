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

from atomate2.vasp.jobs.mp import MPPreRelaxMaker, MPRelaxMaker, MPStaticMaker

__all__ = ["MPMetaGGARelax"]


@dataclass
class MPMetaGGARelax(Maker):
    """
    Maker to perform a VASP r2SCAN relaxation workflow with MP settings.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    pre_relax_maker : .BaseVaspMaker
        Maker to generate the first relaxation.
    pre_static_maker : .BaseVaspMaker
        Maker to generate the static calculation before the relaxation.
    relax_maker : .BaseVaspMaker
        Maker to generate the second relaxation.
    static_maker : .BaseVaspMaker
        Maker to generate the static calculation after the relaxation.
    """

    name: str = "MP Meta-GGA Relax"
    pre_relax_maker: BaseVaspMaker | None = field(default_factory=MPPreRelaxMaker)
    pre_static_maker: BaseVaspMaker | None = None
    relax_maker: BaseVaspMaker | None = field(default_factory=MPRelaxMaker)
    static_maker: BaseVaspMaker | None = field(default_factory=MPStaticMaker)

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
        if self.pre_relax_maker:
            pre_relax = self.pre_relax_maker.make(
                structure, prev_vasp_dir=prev_vasp_dir
            )
            output = pre_relax.output
            structure = output.structure
            bandgap = output.bandgap
            prev_vasp_dir = output.dir_name
            jobs += [pre_relax]

        # Run a static calculation (typically r2SCAN) before the relaxation.
        # See https://doi.org/10.1038/s41524-022-00881-w
        if self.pre_static_maker:
            pre_static = self.pre_static_maker.make(
                structure,
                bandgap=bandgap,
                prev_vasp_dir=prev_vasp_dir,
            )
            output = pre_static.output
            structure = output.structure
            bandgap = output.bandgap
            prev_vasp_dir = output.dir_name
            jobs += [pre_static]

        # Run a relaxation (typically r2SCAN)
        if self.relax_maker:
            relax = self.relax_maker.make(
                structure=structure,
                bandgap=bandgap,
                prev_vasp_dir=prev_vasp_dir,
            )
            output = relax.output
            structure = output.structure
            bandgap = output.bandgap
            prev_vasp_dir = output.dir_name
            jobs += [relax]

        # Run a final static calculation (typically r2SCAN)
        if self.static_maker:
            static = self.static_maker.make(
                structure, bandgap=bandgap, prev_vasp_dir=prev_vasp_dir
            )
            output = static.output
            structure = output.structure
            bandgap = output.bandgap
            prev_vasp_dir = output.dir_name
            jobs += [static]

        return Flow(jobs, output, name=self.name)
