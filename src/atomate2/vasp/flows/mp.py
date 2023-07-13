"""
Module defining flows for Materials Project r2SCAN workflows.

Reference: https://doi.org/10.1103/PhysRevMaterials.6.013801
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from jobflow import Flow, Maker

from atomate2.vasp.jobs.mp import (
    MPGGARelaxMaker,
    MPGGAStaticMaker,
    MPMetaGGARelaxMaker,
    MPMetaGGAStaticMaker,
    MPPreRelaxMaker,
)

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Job
    from pymatgen.core.structure import Structure

    from atomate2.vasp.jobs.base import BaseVaspMaker


__all__ = ["MPGGARelax", "MPMetaGGARelax"]


@dataclass
class MPGGARelax(Maker):
    """
    Maker to perform a VASP GGA relaxation workflow with MP settings.

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

    name: str = "MP GGA Relax"
    initial_maker: BaseVaspMaker | None = field(default_factory=MPGGARelaxMaker)
    final_relax_maker: BaseVaspMaker = field(default_factory=MPGGARelaxMaker)
    final_static_maker: BaseVaspMaker | None = field(default_factory=MPGGAStaticMaker)
    copy_vasp_files: Sequence[str] | None = ("WAVECAR", "CHGCAR")

    def make(self, structure: Structure, prev_vasp_dir: str | Path | None = None):
        """
        Create a 2-step flow with a double relaxation followed by a high-quality static.

        An optional static calculation can be performed before the relaxation.

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
        jobs: list[Job] = []

        # Run a pre-relaxation
        if self.initial_maker:
            initial_relax = self.initial_maker.make(
                structure, prev_vasp_dir=prev_vasp_dir
            )
            jobs += [initial_relax]

        self.final_relax_maker.input_set_generator.additional_vasp_files = (
            self.copy_vasp_files
        )
        final_relax = self.final_relax_maker.make(
            structure=initial_relax.output.structure,
            prev_vasp_dir=initial_relax.output.dir_name,
        )
        output = final_relax.output
        jobs += [final_relax]

        if self.final_static_maker:
            # Run a static calculation
            self.final_static_maker.input_set_generator.additional_vasp_files = (
                self.copy_vasp_files
            )

            final_static = self.final_static_maker.make(
                structure=output.structure, prev_vasp_dir=output.dir_name
            )
            output = final_static.output
            jobs += [final_static]

        return Flow(jobs, output, name=self.name)


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
    initial_maker: BaseVaspMaker | None = field(default_factory=MPPreRelaxMaker)
    final_relax_maker: BaseVaspMaker = field(default_factory=MPMetaGGARelaxMaker)
    final_static_maker: BaseVaspMaker | None = field(
        default_factory=MPMetaGGAStaticMaker
    )
    copy_vasp_files: Sequence[str] | None = ("WAVECAR", "CHGCAR")

    def make(self, structure: Structure, prev_vasp_dir: str | Path | None = None):
        """
        Create a 2-step flow with a cheap pre-relaxation followed by a high-quality one.

        An optional static calculation can be performed before the relaxation.

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
        jobs: list[Job] = []

        # Run a pre-relaxation (typically PBEsol)
        if self.initial_maker:
            initial_relax = self.initial_maker.make(
                structure, prev_vasp_dir=prev_vasp_dir
            )
            jobs += [initial_relax]

        self.final_relax_maker.input_set_generator.additional_vasp_files = (
            self.copy_vasp_files
        )
        final_relax = self.final_relax_maker.make(
            structure=initial_relax.output.structure,
            prev_vasp_dir=initial_relax.output.dir_name,
        )
        output = final_relax.output
        jobs += [final_relax]

        if self.final_static_maker:
            # Run a static calculation (typically r2SCAN)
            self.final_static_maker.input_set_generator.additional_vasp_files = (
                self.copy_vasp_files
            )

            final_static = self.final_static_maker.make(
                structure=output.structure, prev_vasp_dir=output.dir_name
            )
            output = final_static.output
            jobs += [final_static]

        return Flow(jobs, output, name=self.name)
