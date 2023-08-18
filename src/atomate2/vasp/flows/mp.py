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

from atomate2.vasp.sets.base import VaspInputGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Job
    from pymatgen.core.structure import Structure

    from atomate2.vasp.jobs.base import BaseVaspMaker


__all__ = ["MPGGADoubleRelaxMaker", "MPGGARelax" , "MPMetaGGADoubleRelaxMaker", "MPMetaGGARelax"]


@dataclass
class MPGGADoubleRelaxMaker(Maker):
    """
    - Double relaxation using Materials Project GGA parameters
    - Only one maker is optional
    - Also used as base class to define further GGA and meta-GGA workflows

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    inital_relax_maker : .BaseVaspMaker
        Maker to generate the first relaxation.
    final_relax_maker : .BaseVaspMaker
        Maker to generate the second/final relaxation
    optional_final_static_maker : .BaseVaspMaker = None by default
        Optional maker to generate a final static
    GGA_plus_U : bool = False, used to easily enable/disable +U corrections
    """
    name : str = 'MP GGA Double Relax Maker'
    inital_relax_maker : BaseVaspMaker = field(default_factory=MPGGARelaxMaker)
    final_relax_maker : BaseVaspMaker = field(default_factory=MPGGARelaxMaker)
    optional_final_static_maker : BaseVaspMaker | None = None
    copy_vasp_files: Sequence[str] | None = ("WAVECAR", "CHGCAR")
    GGA_plus_U : bool = False

    def make(self, structure: Structure, prev_vasp_dir: str | Path | None = None):
        """
        Create a 3-step flow with one optional step (usually an optional, final, high-quality static)

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

        # Required initial relaxation        
        initial_relax = self.inital_relax_maker.make(
            structure, prev_vasp_dir=prev_vasp_dir
        )
        jobs += [initial_relax]
        structure = initial_relax.output.structure
        prev_vasp_dir = initial_relax.output.dir_name

        # Required second/final relaxation   
        self.final_relax_maker.copy_vasp_kwargs = {
            "additional_vasp_files": self.copy_vasp_files
        }
        final_relax = self.final_relax_maker.make(
            structure=structure, prev_vasp_dir=prev_vasp_dir
        )
        output = final_relax.output
        jobs += [final_relax]

        if self.optional_final_static_maker:
            # Run optional final static
            self.optional_final_static_maker.copy_vasp_kwargs = {
                "additional_vasp_files": self.copy_vasp_files
            }

            optional_static = self.optional_final_static_maker.make(
                structure=output.structure, prev_vasp_dir=output.dir_name
            )
            output = optional_static.output
            jobs += [optional_static]

        for mkr in jobs:
            mkr.input_set_generator.config_dict['INCAR']['LDAU'] = self.GGA_plus_U

        return Flow(jobs, output, name=self.name)

@dataclass
class MPGGARelax(MPGGADoubleRelaxMaker):

    """
    - Double relaxation + final static using Materials Project GGA parameters

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    inital_relax_maker : .BaseVaspMaker
        Maker to generate the first relaxation. (PBE GGA relax)
    final_relax_maker : .BaseVaspMaker
        Maker to generate the second/final relaxation
    optional_final_static_maker : .BaseVaspMaker = .BaseVaspMaker by default
        Optional maker to generate a final static
    GGA_plus_U : bool = False, used to easily enable/disable +U corrections
    """

    name: str = "MP GGA Relax"
    inital_relax_maker : BaseVaspMaker = field(default_factory=MPGGARelaxMaker)
    final_relax_maker : BaseVaspMaker = field(default_factory=MPGGARelaxMaker)
    optional_final_static_maker : BaseVaspMaker | None = field(default_factory=MPGGAStaticMaker)
    copy_vasp_files: Sequence[str] | None = ("WAVECAR", "CHGCAR")
    GGA_plus_U : bool = False


@dataclass
class MPMetaGGADoubleRelaxMaker(MPGGADoubleRelaxMaker):
    """
    - Double relaxation using Materials Project r2SCAN Meta-GGA parameters

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    inital_relax_maker : .BaseVaspMaker
        Maker to generate the first relaxation (PBEsol GGA relax)
    final_relax_maker : .BaseVaspMaker
        Maker to generate the second/final relaxation (r2SCAN meta-GGA relax)
    optional_final_static_maker : .BaseVaspMaker = .BaseVaspMaker by default
        Optional maker to generate a final static
    GGA_plus_U : bool = False, used to easily enable/disable +U corrections
    """

    name: str = "MP Meta-GGA Double Relax"
    inital_relax_maker : BaseVaspMaker | None = field(default_factory=MPPreRelaxMaker)
    final_relax_maker : BaseVaspMaker | None = field(default_factory=MPMetaGGARelaxMaker)
    optional_final_static_maker : BaseVaspMaker | None = None
    copy_vasp_files: Sequence[str] | None = ("WAVECAR", "CHGCAR")
    GGA_plus_U : bool = False


@dataclass
class MPMetaGGARelax(MPGGADoubleRelaxMaker):
    """
    - Double relaxation + final static using Materials Project r2SCAN Meta-GGA parameters

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    inital_relax_maker : .BaseVaspMaker
        Maker to generate the first relaxation. (PBEsol GGA relax)
    final_relax_maker : .BaseVaspMaker
        Maker to generate the second/final relaxation (r2SCAN meta-GGA relax)
    optional_final_static_maker : .BaseVaspMaker = .BaseVaspMaker by default
        Optional maker to generate a final static (r2SCAN meta-GGA static)
    GGA_plus_U : bool = False, used to easily enable/disable +U corrections
    """

    name: str = "MP Meta-GGA Relax"
    inital_relax_maker : BaseVaspMaker | None = field(default_factory=MPPreRelaxMaker)
    final_relax_maker : BaseVaspMaker | None = field(default_factory=MPMetaGGARelaxMaker)
    optional_final_static_maker : BaseVaspMaker | None = field(default_factory=MPMetaGGAStaticMaker)
    copy_vasp_files: Sequence[str] | None = ("WAVECAR", "CHGCAR")
    GGA_plus_U : bool = False

