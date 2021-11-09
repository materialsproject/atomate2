"""Core VASP flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from jobflow import Flow, Maker
from pymatgen.core.structure import Structure

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import (
    HSEBSMaker,
    HSEStaticMaker,
    NonSCFMaker,
    RelaxMaker,
    StaticMaker,
)
from atomate2.vasp.schemas.calculation import VaspObject

__all__ = [
    "DoubleRelaxMaker",
    "BandStructureMaker",
    "HSEBandStructureMaker",
    "RelaxBandStructureMaker",
]


@dataclass
class DoubleRelaxMaker(Maker):
    """
    Maker to perform a double VASP relaxation.

    Parameters
    ----------
    name
        Name of the flows produced by this maker.
    relax_maker
        Maker to use to generate the relaxations.
    """

    name: str = "double relax"
    relax_maker: BaseVaspMaker = field(default_factory=RelaxMaker)

    def make(self, structure: Structure, prev_vasp_dir: Union[str, Path] = None):
        """
        Create a flow with two chained relaxations.

        Parameters
        ----------
        structure
            A pymatgen structure object.
        prev_vasp_dir
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing two relaxations.
        """
        relax1 = self.relax_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
        relax1.name += " 1"

        relax2 = self.relax_maker.make(
            relax1.output.structure, prev_vasp_dir=relax1.output.dir_name
        )
        relax2.name += " 2"

        return Flow([relax1, relax2], relax2.output, name=self.name)


@dataclass
class BandStructureMaker(Maker):
    """
    Maker to generate VASP band structures.

    This is a static calculation followed by two non-self-consistent field calculations,
    one uniform and one line mode.

    Parameters
    ----------
    name
        Name of the flows produced by this maker.
    bandstructure_type
        The type of band structure to generate. Options are "line", "uniform" or "both".
    static_maker
        The maker to use for the static calculation.
    bs_maker
        The maker to use for the non-self-consistent field calculations.
    """

    name: str = "band structure"
    bandstructure_type: str = "both"
    static_maker: BaseVaspMaker = field(default_factory=StaticMaker)
    bs_maker: BaseVaspMaker = field(default_factory=NonSCFMaker)

    def make(self, structure, prev_vasp_dir=None):
        """
        Create a band structure flow.

        Parameters
        ----------
        structure
            A pymatgen structure object.
        prev_vasp_dir
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A band structure flow.
        """
        static_job = self.static_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
        jobs = [static_job]

        outputs = {}
        if self.bandstructure_type in ("both", "uniform"):
            uniform_job = self.bs_maker.make(
                structure, prev_vasp_dir=static_job.output.dir_name, mode="uniform"
            )
            uniform_job.name += " uniform"
            jobs.append(uniform_job)
            output = {
                "uniform": uniform_job.output,
                "uniform_bs": uniform_job.output.vasp_objects[VaspObject.BANDSTRUCTURE],
            }
            outputs.update(output)

        if self.bandstructure_type in ("both", "line"):
            line_job = self.bs_maker.make(
                structure, prev_vasp_dir=static_job.output.dir_name, mode="line"
            )
            line_job.name += " line"
            jobs.append(line_job)
            output = {
                "line": line_job.output,
                "line_bs": line_job.output.vasp_objects[VaspObject.BANDSTRUCTURE],
            }
            outputs.update(output)

        if self.bandstructure_type not in ("both", "line", "uniform"):
            raise ValueError(
                f"Unrecognised bandstructure type {self.bandstructure_type}"
            )

        return Flow(jobs, outputs, name=self.name)


@dataclass
class HSEBandStructureMaker(BandStructureMaker):
    """
    Maker to generate VASP HSE band structures.

    This is a HSE06 static calculation followed by one HSE06 uniform calculation and
    one HSE06 line mode calculation.

    Parameters
    ----------
    name
        Name of the flows produced by this maker.
    bandstructure_type
        The type of band structure to generate. Options are "line", "uniform" or "both".
    static_maker
        The maker to use for the static calculation.
    bs_maker
        The maker to use for the line and uniform band structure calculations.
    """

    name: str = "hse band structure"
    bandstructure_type: str = "both"
    static_maker: BaseVaspMaker = field(default_factory=HSEStaticMaker)
    bs_maker: BaseVaspMaker = field(default_factory=HSEBSMaker)


@dataclass
class RelaxBandStructureMaker(Maker):
    """
    Make to create a flow with a relaxation and then band structure calculations.

    By default, this workflow generates relaxations using the :obj:`.DoubleRelaxMaker`.

    Parameters
    ----------
    name
        Name of the flows produced by this maker.
    bandstructure_type
        The type of band structure to generate. Options are "line", "uniform" or "both".
    static_maker
        The maker to use for the static calculation.
    bs_maker
        The maker to use for the line and uniform band structure calculations.
    """

    name: str = "relax and band structure"
    relax_maker: BaseVaspMaker = field(default_factory=DoubleRelaxMaker)
    band_structure_maker: BaseVaspMaker = field(default_factory=BandStructureMaker)

    def make(self, structure, prev_vasp_dir=None):
        """
        Run a relaxation and then calculate the uniform and line mode band structures.

        Parameters
        ----------
        structure
            A pymatgen structure object.
        prev_vasp_dir
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A relax and band structure flow.
        """
        relax_job = self.relax_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
        bs_flow = self.band_structure_maker.make(
            relax_job.output.structure, prev_vasp_dir=relax_job.output.dir_name
        )

        return Flow([relax_job, bs_flow], bs_flow.output, name=self.name)
