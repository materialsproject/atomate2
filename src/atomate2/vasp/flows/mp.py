"""
Module defining flows for Materials Project workflows.

Reference: https://doi.org/10.1103/PhysRevMaterials.6.013801

In case of questions, consult @Andrew-S-Rosen, @esoteric-ephemera or @janosh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.mp import (
    MPGGARelaxMaker,
    MPGGAStaticMaker,
    MPMetaGGARelaxMaker,
    MPMetaGGAStaticMaker,
)

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Job
    from pymatgen.core.structure import Structure


@dataclass
class MPGGADoubleRelaxMaker(DoubleRelaxMaker):
    """MP GGA double relaxation workflow.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker1 : .BaseVaspMaker
        Maker to generate the first relaxation.
    relax_maker2 : .BaseVaspMaker
        Maker to generate the second relaxation.
    """

    name: str = "MP GGA double relax"
    relax_maker1: Maker | None = field(default_factory=MPGGARelaxMaker)
    relax_maker2: Maker = field(
        default_factory=lambda: MPGGARelaxMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")}
        )
    )


@dataclass
class MPMetaGGADoubleRelaxMaker(DoubleRelaxMaker):
    """MP meta-GGA double relaxation workflow.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker1 : .BaseVaspMaker
        Maker to generate the first relaxation.
    relax_maker2 : .BaseVaspMaker
        Maker to generate the second relaxation.
    """

    name: str = "MP GGA double relax"
    relax_maker1: Maker | None = field(default_factory=MPMetaGGARelaxMaker)
    relax_maker2: Maker = field(
        default_factory=lambda: MPMetaGGARelaxMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")}
        )
    )


@dataclass
class MPGGADoubleRelaxStatic(Maker):
    """
    Maker to perform a VASP GGA relaxation workflow with MP settings.

    Only the middle job performing a PBE relaxation is non-optional.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker
        Maker to generate the relaxation.
    static_maker : .BaseVaspMaker
        Maker to generate the static calculation before the relaxation.
    """

    name: str = "MP GGA relax"
    relax_maker: Maker = field(default_factory=MPGGADoubleRelaxMaker)
    static_maker: Maker | None = field(
        default_factory=lambda: MPGGAStaticMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")}
        )
    )

    def make(self, structure: Structure, prev_vasp_dir: str | Path | None = None):
        """
        1, 2 or 3-step flow with optional pre-relax and final static jobs.

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

        relax_job = self.relax_maker.make(
            structure=structure, prev_vasp_dir=prev_vasp_dir
        )
        output = relax_job.output
        jobs += [relax_job]

        if self.static_maker:
            # Run a static calculation
            static_job = self.static_maker.make(
                structure=output.structure, prev_vasp_dir=output.dir_name
            )
            output = static_job.output
            jobs += [static_job]

        return Flow(jobs, output, name=self.name)


@dataclass
class MPMetaGGADoubleRelaxStatic(MPGGADoubleRelaxMaker):
    """
    Flow with optional pre-relax and final static jobs.

    Only the middle job performing a meta-GGA relaxation is non-optional.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker
        Maker to generate the relaxation.
    static_maker : .BaseVaspMaker
        Maker to generate the static calculation before the relaxation.
    """

    name: str = "MP meta-GGA relax"
    relax_maker: Maker = field(default_factory=MPMetaGGADoubleRelaxMaker)
    static_maker: Maker | None = field(
        default_factory=lambda: MPMetaGGAStaticMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")}
        )
    )

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

        relax_job = self.relax_maker.make(
            structure=structure, prev_vasp_dir=prev_vasp_dir
        )
        output = relax_job.output
        jobs += [relax_job]

        if self.static_maker:
            # Run a static calculation (typically r2SCAN)
            static_job = self.static_maker.make(
                structure=output.structure, prev_vasp_dir=output.dir_name
            )
            output = static_job.output
            jobs += [static_job]

        return Flow(jobs, output=output, name=self.name)
