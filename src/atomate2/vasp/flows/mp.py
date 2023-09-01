"""
Module defining flows for Materials Project workflows.

Reference: https://doi.org/10.1103/PhysRevMaterials.6.013801
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

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

    from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class MPGGADoubleRelaxMaker(DoubleRelaxMaker):
    """MP GGA double relaxation workflow.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    pre_relax_maker : .BaseVaspMaker
        Maker to generate the first relaxation.
    relax_maker : .BaseVaspMaker
        Maker to generate the second relaxation.
    copy_vasp_files : Sequence[str] or None
        VASP files to copy from the previous calculation directory.
    """

    name: str = "MP GGA double relax"
    pre_relax_maker: BaseVaspMaker | None = field(default_factory=MPGGARelaxMaker)
    relax_maker: BaseVaspMaker = field(default_factory=MPGGARelaxMaker)
    copy_vasp_files: Sequence[str] | None = ("WAVECAR", "CHGCAR")

    def __post_init__(self):
        """Set the copy_vasp_kwargs for the relax_maker."""
        self.relax_maker.copy_vasp_kwargs.setdefault(
            "additional_vasp_files", self.copy_vasp_files
        )


@dataclass
class MPMetaGGADoubleRelaxMaker(DoubleRelaxMaker):
    """MP meta-GGA double relaxation workflow.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    pre_relax_maker : .BaseVaspMaker
        Maker to generate the first relaxation.
    relax_maker : .BaseVaspMaker
        Maker to generate the second relaxation.
    copy_vasp_files : Sequence[str] or None
        VASP files to copy from the previous calculation directory.
    """

    name: str = "MP GGA double relax"
    pre_relax_maker: BaseVaspMaker | None = field(default_factory=MPMetaGGARelaxMaker)
    relax_maker: BaseVaspMaker = field(default_factory=MPMetaGGARelaxMaker)
    copy_vasp_files: Sequence[str] | None = ("WAVECAR", "CHGCAR")

    def __post_init__(self):
        """Set the copy_vasp_kwargs for the relax_maker."""
        self.relax_maker.copy_vasp_kwargs.setdefault(
            "additional_vasp_files", self.copy_vasp_files
        )


@dataclass
class MPGGADoubleRelaxStatic(Maker):
    """
    Maker to perform a VASP GGA relaxation workflow with MP settings.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    pre_relax_maker : .BaseVaspMaker
        Maker to generate the first relaxation.
    relax_maker : .BaseVaspMaker
        Maker to generate the second relaxation.
    static_maker : .BaseVaspMaker
        Maker to generate the static calculation before the relaxation.
    copy_vasp_files : Sequence[str] or None
        VASP files to copy from the previous calculation directory.
    """

    name: str = "MP GGA relax"
    relax_maker: BaseVaspMaker = field(default_factory=MPGGADoubleRelaxMaker)
    static_maker: BaseVaspMaker | None = field(default_factory=MPGGAStaticMaker)
    copy_vasp_files: Sequence[str] | None = ("WAVECAR", "CHGCAR")
    GGA_plus_U: bool = False

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
        self.relax_maker.copy_vasp_kwargs.setdefault(
            "additional_vasp_files", self.copy_vasp_files
        )
        jobs: list[Job] = []

        self.relax_maker.copy_vasp_kwargs = {
            "additional_vasp_files": self.copy_vasp_files
        }
        relax_job = self.relax_maker.make(
            structure=structure, prev_vasp_dir=prev_vasp_dir
        )
        output = relax_job.output
        jobs += [relax_job]

        if self.static_maker:
            # Run a static calculation
            self.static_maker.copy_vasp_kwargs = {
                "additional_vasp_files": self.copy_vasp_files
            }

            static_job = self.static_maker.make(
                structure=output.structure, prev_vasp_dir=output.dir_name
            )
            output = static_job.output
            jobs += [static_job]

        return Flow(jobs, output, name=self.name)


@dataclass
class MPMetaGGADoubleRelaxStatic(MPGGADoubleRelaxMaker):
    """
    1, 2 or 3-step flow with optional pre-relax and final static jobs.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    pre_relax_maker : .BaseVaspMaker
        Maker to generate the first relaxation.
    relax_maker : .BaseVaspMaker
        Maker to generate the second relaxation.
    static_maker : .BaseVaspMaker
        Maker to generate the static calculation before the relaxation.
    copy_vasp_files : Sequence[str] or None
        VASP files to copy from the previous calculation directory.
    """

    name: str = "MP Meta-GGA relax"
    relax_maker: BaseVaspMaker = field(default_factory=MPMetaGGADoubleRelaxMaker)
    static_maker: BaseVaspMaker | None = field(default_factory=MPMetaGGAStaticMaker)
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
        self.relax_maker.copy_vasp_kwargs.setdefault(
            "additional_vasp_files", self.copy_vasp_files
        )
        jobs: list[Job] = []

        self.relax_maker.copy_vasp_kwargs = {
            "additional_vasp_files": self.copy_vasp_files
        }
        relax_job = self.relax_maker.make(
            structure=structure, prev_vasp_dir=prev_vasp_dir
        )
        output = relax_job.output
        jobs += [relax_job]

        if self.static_maker:
            # Run a static calculation (typically r2SCAN)
            self.static_maker.copy_vasp_kwargs = {
                "additional_vasp_files": self.copy_vasp_files
            }
            static_job = self.static_maker.make(
                structure=output.structure, prev_vasp_dir=output.dir_name
            )
            output = static_job.output
            jobs += [static_job]

        return Flow(jobs, output=output, name=self.name)
