"""Core VASP flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
from atomate2.vasp.sets.core import HSEBSSetGenerator, NonSCFSetGenerator

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
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker
        Maker to use to generate the relaxations.
    """

    name: str = "double relax"
    relax_maker: BaseVaspMaker = field(default_factory=RelaxMaker)

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
    name : str
        Name of the flows produced by this maker.
    bandstructure_type : str
        The type of band structure to generate. Options are "line", "uniform" or "both".
    static_maker : .BaseVaspMaker
        The maker to use for the static calculation.
    bs_maker : .BaseVaspMaker
        The maker to use for the non-self-consistent field calculations.
    """

    name: str = "band structure"
    bandstructure_type: str = "both"
    static_maker: BaseVaspMaker = field(default_factory=StaticMaker)
    bs_maker: BaseVaspMaker = field(default_factory=NonSCFMaker)

    def make(self, structure: Structure, prev_vasp_dir: str | Path | None = None):
        """
        Create a band structure flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_vasp_dir : str or Path or None
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
    name : str
        Name of the flows produced by this maker.
    bandstructure_type : str
        The type of band structure to generate. Options are "line", "uniform" or "both".
    static_maker : .BaseVaspMaker
        The maker to use for the static calculation.
    bs_maker : .BaseVaspMaker
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
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker
        The maker to use for the static calculation.
    band_structure_maker : .BaseVaspMaker
        The maker to use for the line and uniform band structure calculations.
    """

    name: str = "relax and band structure"
    relax_maker: BaseVaspMaker = field(default_factory=DoubleRelaxMaker)
    band_structure_maker: BaseVaspMaker = field(default_factory=BandStructureMaker)

    def make(self, structure: Structure, prev_vasp_dir: str | Path | None = None):
        """
        Run a relaxation and then calculate the uniform and line mode band structures.

        Parameters
        ----------
        structure: .Structure
            A pymatgen structure object.
        prev_vasp_dir : str or Path or None
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


@dataclass
class OpticsMaker(BaseVaspMaker):
    """
    Maker to create optical absorption calculation VASP jobs.

    This workflow contains an initial static calculation, and then a non-self-consistent
    field calculation with LOPTICS set. The purpose of the static calculation is
    i) to determine if the material needs magnetism set, and ii) to determine the total
    number of bands (the second calculation contains 1.3 * number of bands as the
    initial static) as often the highest bands are not properly converged in VASP.

    .. Note::
        The magnetism will be disabled in the non-self-consistent field calculation if
        all MAGMOMs are less than 0.02.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    static_maker : .BaseVaspMaker
        The maker to use for the static calculation.
    band_structure_maker : .BaseVaspMaker
        The maker to use for the uniform optics calculation.
    """

    name: str = "static and optics"
    static_maker: BaseVaspMaker = field(default_factory=StaticMaker)
    band_structure_maker: BaseVaspMaker = field(
        default_factory=lambda: NonSCFMaker(
            name="optics",
            input_set_generator=NonSCFSetGenerator(optics=True),
        )
    )

    def make(self, structure: Structure, prev_vasp_dir: str | Path | None = None):
        """
        Run a static and then a non-scf optics calculation.

        Parameters
        ----------
        structure: .Structure
            A pymatgen structure object.
        prev_vasp_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A static and nscf with optics flow.
        """
        static_job = self.static_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
        nscf_job = self.band_structure_maker.make(
            static_job.output.structure, prev_vasp_dir=static_job.output.dir_name
        )
        return Flow([static_job, nscf_job], nscf_job.output, name=self.name)


@dataclass
class HSEOpticsMaker(BaseVaspMaker):
    """
    Maker to create HSE optical absorption calculation VASP jobs.

    This workflow contains an initial HSE static calculation, and then a uniform band
    structure calculation with LOPTICS set. The purpose of the static calculation is
    i) to determine if the material needs magnetism set and ii) to determine the total
    number of bands (the second calculation contains 1.3 * number of bands as the
    initial static) as often the highest bands are not properly converged in VASP.

    .. Note::
        The magnetism will be disabled in the uniform optics calculation if all MAGMOMs
        are less than 0.02.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    static_maker : .BaseVaspMaker
        The maker to use for the static calculation.
    band_structure_maker : .BaseVaspMaker
        The maker to use for the uniform optics calculation.
    """

    name: str = "hse static and optics"
    static_maker: BaseVaspMaker = field(default_factory=HSEStaticMaker)
    band_structure_maker: BaseVaspMaker = field(
        default_factory=lambda: HSEBSMaker(
            name="hse optics",
            input_set_generator=HSEBSSetGenerator(optics=True, mode="uniform"),
        )
    )

    def make(self, structure: Structure, prev_vasp_dir: str | Path | None = None):
        """
        Run a static and then a non-scf optics calculation.

        Parameters
        ----------
        structure: .Structure
            A pymatgen structure object.
        prev_vasp_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A static and nscf with optics flow.
        """
        static_job = self.static_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
        bs_job = self.band_structure_maker.make(
            static_job.output.structure, prev_vasp_dir=static_job.output.dir_name
        )
        return Flow([static_job, bs_job], bs_job.output, name=self.name)
