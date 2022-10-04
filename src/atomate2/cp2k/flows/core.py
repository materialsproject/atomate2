"""Core VASP flows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from jobflow import Flow, Maker, Job
from pymatgen.core.structure import Structure

from atomate2.cp2k.jobs.base import BaseCp2kMaker
from atomate2.cp2k.jobs.core import (
    StaticMaker, RelaxMaker, CellOptMaker, 
    HybridStaticMaker, HybridRelaxMaker, HybridCellOptMaker,
    NonSCFMaker, MDMaker
)
from atomate2.cp2k.schemas.calculation import Cp2kObject

__all__ = [
]

@dataclass
class DoubleRelaxMaker(Maker):
    """
    Maker to perform a double CP2K relaxation.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker1 : .BaseCp2kMaker
        Maker to use to generate the first relaxation.
    relax_maker2 : .BaseCp2kMaker
        Maker to use to generate the second relaxation.
    """

    name: str = "double relax"
    relax_maker1: BaseCp2kMaker = field(default_factory=RelaxMaker)
    relax_maker2: BaseCp2kMaker = field(default_factory=RelaxMaker)

    def make(self, structure: Structure, prev_cp2k_dir: str | Path | None = None):
        """
        Create a flow with two chained relaxations.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_cp2k_dir : str or Path or None
            A previous Cp2k calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing two relaxations.
        """
        relax1 = self.relax_maker1.make(structure, prev_cp2k_dir=prev_cp2k_dir)
        relax1.name += " 1"

        relax2 = self.relax_maker2.make(
            relax1.output.structure, prev_cp2k_dir=relax1.output.dir_name
        )
        relax2.name += " 2"

        return Flow([relax1, relax2], relax2.output, name=self.name)

    @classmethod
    def from_relax_maker(cls, relax_maker: BaseCp2kMaker):
        """
        Instantiate the DoubleRelaxMaker with two relax makers of the same type.

        Parameters
        ----------
        relax_maker : .BaseCp2kMaker
            Maker to use to generate the first and second relaxations.
        """
        return cls(
            relax_maker1=deepcopy(relax_maker), relax_maker2=deepcopy(relax_maker)
        )


@dataclass
class BandStructureMaker(Maker):
    """
    Maker to generate Cp2k band structures.

    This is a static calculation followed by two non-self-consistent field calculations,
    one uniform and one line mode.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    bandstructure_type : str
        The type of band structure to generate. Options are "line", "uniform" or "both".
    static_maker : .BaseCp2kMaker
        The maker to use for the static calculation.
    bs_maker : .BaseCp2kMaker
        The maker to use for the non-self-consistent field calculations.
    """

    name: str = "band structure"
    bandstructure_type: str = "both"
    static_maker: BaseCp2kMaker = field(default_factory=StaticMaker)
    bs_maker: BaseCp2kMaker = field(default_factory=NonSCFMaker)

    def make(self, structure: Structure, prev_cp2k_dir: str | Path | None = None):
        """
        Create a band structure flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_cp2k_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A band structure flow.
        """
        static_job = self.static_maker.make(structure, prev_cp2k_dir=prev_cp2k_dir)
        jobs = [static_job]

        outputs = {}
        if self.bandstructure_type in ("both", "uniform"):
            uniform_job = self.bs_maker.make(
                static_job.output.structure,
                prev_cp2k_dir=static_job.output.dir_name,
                mode="uniform",
            )
            uniform_job.name += " uniform"
            jobs.append(uniform_job)
            output = {
                "uniform": uniform_job.output,
                "uniform_bs": uniform_job.output.cp2k_objects[Cp2kObject.BANDSTRUCTURE],
            }
            outputs.update(output)

        if self.bandstructure_type in ("both", "line"):
            line_job = self.bs_maker.make(
                static_job.output.structure,
                prev_cp2k_dir=static_job.output.dir_name,
                mode="line",
            )
            line_job.name += " line"
            jobs.append(line_job)
            output = {
                "line": line_job.output,
                "line_bs": line_job.output.cp2k_objects[Cp2kObject.BANDSTRUCTURE],
            }
            outputs.update(output)

        if self.bandstructure_type not in ("both", "line", "uniform"):
            raise ValueError(
                f"Unrecognised bandstructure type {self.bandstructure_type}"
            )

        return Flow(jobs, outputs, name=self.name)


@dataclass
class RelaxBandStructureMaker(Maker):
    """
    Make to create a flow with a relaxation and then band structure calculations.

    By default, this workflow generates relaxations using the :obj:`.DoubleRelaxMaker`.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseCp2kMaker
        The maker to use for the static calculation.
    band_structure_maker : .BaseCp2kMaker
        The maker to use for the line and uniform band structure calculations.
    """

    name: str = "relax and band structure"
    relax_maker: BaseCp2kMaker = field(default_factory=DoubleRelaxMaker)
    band_structure_maker: BaseCp2kMaker = field(default_factory=BandStructureMaker)

    def make(self, structure: Structure, prev_cp2k_dir: str | Path | None = None):
        """
        Run a relaxation and then calculate the uniform and line mode band structures.

        Parameters
        ----------
        structure: .Structure
            A pymatgen structure object.
        prev_cp2k_dir : str or Path or None
            A previous CP2K calculation directory to copy output files from.

        Returns
        -------
        Flow
            A relax and band structure flow.
        """
        relax_job = self.relax_maker.make(structure, prev_cp2k_dir=prev_cp2k_dir)
        bs_flow = self.band_structure_maker.make(
            relax_job.output.structure, prev_cp2k_dir=relax_job.output.dir_name
        )

        return Flow([relax_job, bs_flow], bs_flow.output, name=self.name)

@dataclass
class HybridFlowMaker(Maker):

    hybrid_functional: str = "PBE0"
    initialize_with_pbe: bool = field(default=True)
    initialize_maker: BaseCp2kMaker = field(default_factory=StaticMaker)
    hybrid_maker: BaseCp2kMaker = field(default_factory=HybridStaticMaker) 

    def __post_init__(self):
        self.hybrid_maker.hybrid_functional = self.hybrid_functional

    def make(self, structure: Structure, prev_cp2k_dir: str | Path | None = None) -> Job:
        jobs = []
        if self.initialize_with_pbe:
            initialization = self.initialize_maker.make(structure, prev_cp2k_dir)
            jobs.append(initialization)
        hyb = self.hybrid_maker.make(
            initialization.output.structure if self.initialize_with_pbe else structure, 
            prev_cp2k_dir=initialization.output.dir_name if self.initialize_with_pbe else prev_cp2k_dir
        ) 
        jobs.append(hyb)
        return Flow(jobs, output=hyb.output, name=self.name)

@dataclass
class HybridStaticFlowMaker(HybridFlowMaker):
    """
    Maker to perform a PBE restart to hybrid static flow

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    pbe_maker : .BaseCp2kMaker
        Maker to use to generate PBE restart file for hybrid calc
    hybrid_maker : .BaseCp2kMaker
        Maker to use to generate the second relaxation.
    """

    name: str = "pbe to hybrid static"

@dataclass
class HybridRelaxFlowMaker(HybridFlowMaker):
    """
    Maker to perform a PBE restart to hybrid relax flow

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    pbe_maker : .BaseCp2kMaker
        Maker to use to generate PBE restart file for hybrid calc
    hybrid_maker : .BaseCp2kMaker
        Maker to use to generate the second relaxation.
    """

    name: str = "pbe to hybrid relaxation"
    hybrid_maker: BaseCp2kMaker = field(default_factory=HybridRelaxMaker)

@dataclass
class HybridCellOptFlowMaker(HybridFlowMaker):
    """
    Maker to perform a PBE restart to hybrid cell opt flow

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    pbe_maker : .BaseCp2kMaker
        Maker to use to generate PBE restart file for hybrid calc
    hybrid_maker : .BaseCp2kMaker
        Maker to use to generate the second relaxation.
    """

    name: str = "pbe to hybrid cell opt"
    hybrid_maker: BaseCp2kMaker = field(default_factory=HybridCellOptMaker)