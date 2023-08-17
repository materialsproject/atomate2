"""
Module defining flows for Materials Project workflows.

Reference: https://doi.org/10.1103/PhysRevMaterials.6.013801
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from jobflow import Flow, Maker

from monty.serialization import loadfn
from pkg_resources import resource_filename

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


__all__ = ["MPGGARelax", "MPGGADoubleRelaxMaker", "MPMetaGGARelax", "MPMetaGGADoubleRelaxMaker"]


@dataclass
class _MPGenericThreeStep(Maker):
    """
    - Generic maker for three-step Materials Project-adjacent jobs
    - Only one maker is required
    - Used to define GGA, meta-GGA workflows

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
    name : str = 'MP generic three-step'
    optional_maker_1 : BaseVaspMaker | None = None
    required_maker : BaseVaspMaker = field(default_factory=MPGGARelaxMaker)
    optional_maker_2 : BaseVaspMaker | None = None
    copy_vasp_files: Sequence[str] | None = ("WAVECAR", "CHGCAR")

    def make(self, structure: Structure, prev_vasp_dir: str | Path | None = None):
        """
        Create a 3-step flow with two optional steps (usually an optional relax, and final high-quality static)

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

        # Run first optional maker
        if self.optional_maker_1:
            optional_job_1 = self.optional_maker_1.make(
                structure, prev_vasp_dir=prev_vasp_dir
            )
            jobs += [optional_job_1]
            structure = optional_job_1.output.structure
            prev_vasp_dir = optional_job_1.output.dir_name

        self.required_maker.copy_vasp_kwargs = {
            "additional_vasp_files": self.copy_vasp_files
        }
        required_job = self.required_maker.make(
            structure=structure, prev_vasp_dir=prev_vasp_dir
        )
        output = required_job.output
        jobs += [required_job]

        if self.optional_maker_2:
            # Run second optional job
            self.optional_maker_2.copy_vasp_kwargs = {
                "additional_vasp_files": self.copy_vasp_files
            }

            optional_job_2 = self.optional_maker_2.make(
                structure=output.structure, prev_vasp_dir=output.dir_name
            )
            output = optional_job_2.output
            jobs += [optional_job_2]

        return Flow(jobs, output, name=self.name)

@dataclass
class MPGGARelax(_MPGenericThreeStep):

    """
    Maker to perform a VASP GGA relaxation workflow with MP settings.
    Additional kwarg GGA_plus_U to enable / disable GGA + U calculations
    By default, only do GGA (LDAU = False)

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
    optional_maker_1 : BaseVaspMaker | None = field(default_factory=MPGGARelaxMaker)
    required_maker : BaseVaspMaker | None = field(default_factory=MPGGARelaxMaker)
    optional_maker_2 : BaseVaspMaker | None = field(default_factory=MPGGAStaticMaker)
    copy_vasp_files: Sequence[str] | None = ("WAVECAR", "CHGCAR")
    GGA_plus_U : bool = False

    def make(self,structure : Structure, **kwargs):            
        for mkr in [self.optional_maker_1, self.required_maker, self.optional_maker_2]:
            if mkr:
                mkr.input_set_generator.config_dict['INCAR']['LDAU'] = self.GGA_plus_U
        return super().make(structure, **kwargs)

@dataclass
class MPGGADoubleRelaxMaker(_MPGenericThreeStep):

    """
    Maker to perform a VASP GGA relaxation workflow with MP settings.
    Additional kwarg GGA_plus_U to enable / disable GGA + U calculations
    By default, only do GGA (LDAU = False)

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

    name: str = "MP GGA Double Relax"
    optional_maker_1 : BaseVaspMaker | None = field(default_factory=MPGGARelaxMaker)
    required_maker : BaseVaspMaker | None = field(default_factory=MPGGARelaxMaker)
    optional_maker_2 : BaseVaspMaker | None = None
    copy_vasp_files: Sequence[str] | None = ("WAVECAR", "CHGCAR")
    GGA_plus_U : bool = False

    def make(self,structure : Structure, **kwargs):
        for mkr in [self.optional_maker_1, self.required_maker, self.optional_maker_2]:
            if mkr:
                mkr.input_set_generator.config_dict['INCAR']['LDAU'] = self.GGA_plus_U
        return super().make(structure, **kwargs)


@dataclass
class MPMetaGGARelax(_MPGenericThreeStep):
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
    optional_maker_1 : BaseVaspMaker | None = field(default_factory=MPPreRelaxMaker)
    required_maker : BaseVaspMaker | None = field(default_factory=MPMetaGGARelaxMaker)
    optional_maker_2 : BaseVaspMaker | None = field(default_factory=MPMetaGGAStaticMaker)
    copy_vasp_files: Sequence[str] | None = ("WAVECAR", "CHGCAR")


@dataclass
class MPMetaGGADoubleRelaxMaker(_MPGenericThreeStep):
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

    name: str = "MP Meta-GGA Double Relax"
    optional_maker_1 : BaseVaspMaker | None = field(default_factory=MPPreRelaxMaker)
    required_maker : BaseVaspMaker | None = field(default_factory=MPMetaGGARelaxMaker)
    optional_maker_2 : BaseVaspMaker | None = None
    copy_vasp_files: Sequence[str] | None = ("WAVECAR", "CHGCAR")
