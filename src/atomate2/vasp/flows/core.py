"""Core VASP flows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from emmet.core.vasp.calculation import VaspObject
from jobflow import Flow, Maker

from atomate2.vasp.jobs.core import (
    HSEBSMaker,
    HSEStaticMaker,
    NonSCFMaker,
    RelaxMaker,
    StaticMaker,
)
from atomate2.vasp.sets.core import HSEBSSetGenerator, NonSCFSetGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Job
    from pymatgen.core.structure import Structure
    from typing_extensions import Self

    from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class DoubleRelaxMaker(Maker):
    """
    Maker to perform a double VASP relaxation.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker1 : .BaseVaspMaker
        Maker to use to generate the first relaxation.
    relax_maker2 : .BaseVaspMaker
        Maker to use to generate the second relaxation.
    """

    name: str = "double relax"
    relax_maker1: BaseVaspMaker | None = field(default_factory=RelaxMaker)
    relax_maker2: BaseVaspMaker = field(default_factory=RelaxMaker)

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """Create a flow with two chained relaxations.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing two relaxations.
        """
        jobs: list[Job] = []
        if self.relax_maker1:
            # Run a pre-relaxation
            relax1 = self.relax_maker1.make(structure, prev_dir=prev_dir)
            relax1.name += " 1"
            jobs += [relax1]
            structure = relax1.output.structure
            prev_dir = relax1.output.dir_name

        relax2 = self.relax_maker2.make(structure, prev_dir=prev_dir)
        relax2.name += " 2"
        jobs += [relax2]

        return Flow(jobs, output=relax2.output, name=self.name)

    @classmethod
    def from_relax_maker(cls, relax_maker: BaseVaspMaker) -> Self:
        """
        Instantiate the DoubleRelaxMaker with two relax makers of the same type.

        Parameters
        ----------
        relax_maker : .BaseVaspMaker
            Maker to use to generate the first and second relaxations.
        """
        return cls(
            relax_maker1=deepcopy(relax_maker), relax_maker2=deepcopy(relax_maker)
        )


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

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """Create a band structure flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A band structure flow.
        """
        static_job = self.static_maker.make(structure, prev_dir=prev_dir)
        jobs = [static_job]

        outputs = {}
        bandstructure_type = self.bandstructure_type
        if bandstructure_type in ("both", "uniform"):
            uniform_job = self.bs_maker.make(
                static_job.output.structure,
                prev_dir=static_job.output.dir_name,
                mode="uniform",
            )
            uniform_job.name += " uniform"
            jobs.append(uniform_job)
            output = {
                "uniform": uniform_job.output,
                "uniform_bs": uniform_job.output.vasp_objects[VaspObject.BANDSTRUCTURE],
            }
            outputs.update(output)

        if bandstructure_type in ("both", "line"):
            line_job = self.bs_maker.make(
                static_job.output.structure,
                prev_dir=static_job.output.dir_name,
                mode="line",
            )
            line_job.name += " line"
            jobs.append(line_job)
            output = {
                "line": line_job.output,
                "line_bs": line_job.output.vasp_objects[VaspObject.BANDSTRUCTURE],
            }
            outputs.update(output)

        if bandstructure_type not in ("both", "line", "uniform"):
            raise ValueError(f"Unrecognised {bandstructure_type=}")

        return Flow(jobs, outputs, name=self.name)


@dataclass
class UniformBandStructureMaker(Maker):
    """
    Maker to generate uniform VASP band structure.

    This is a static calculation followed by a uniform non-self-consistent field
    calculations.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    static_maker : .BaseVaspMaker
        The maker to use for the static calculation.
    bs_maker : .BaseVaspMaker
        The maker to use for the non-self-consistent field calculations.
    """

    name: str = "uniform band structure"
    static_maker: BaseVaspMaker = field(default_factory=StaticMaker)
    bs_maker: BaseVaspMaker = field(default_factory=NonSCFMaker)

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """Create a uniform band structure flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A uniform band structure flow.
        """
        static_job = self.static_maker.make(structure, prev_dir=prev_dir)
        uniform_job = self.bs_maker.make(
            static_job.output.structure,
            prev_dir=static_job.output.dir_name,
            mode="uniform",
        )
        uniform_job.name += " uniform"
        jobs = [static_job, uniform_job]
        return Flow(jobs, uniform_job.output, name=self.name)


@dataclass
class LineModeBandStructureMaker(Maker):
    """
    Maker to generate line mode VASP band structure.

    This is a static calculation followed by a line mode non-self-consistent field
    calculations.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    static_maker : .BaseVaspMaker
        The maker to use for the static calculation.
    bs_maker : .BaseVaspMaker
        The maker to use for the non-self-consistent field calculations.
    """

    name: str = "line band structure"
    static_maker: BaseVaspMaker = field(default_factory=StaticMaker)
    bs_maker: BaseVaspMaker = field(default_factory=NonSCFMaker)

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """Create a line mode band structure flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A line mode band structure flow.
        """
        static_job = self.static_maker.make(structure, prev_dir=prev_dir)
        line_job = self.bs_maker.make(
            static_job.output.structure,
            prev_dir=static_job.output.dir_name,
            mode="line",
        )
        line_job.name += " line"
        jobs = [static_job, line_job]
        return Flow(jobs, line_job.output, name=self.name)


@dataclass
class HSEBandStructureMaker(BandStructureMaker):
    """
    Maker to generate VASP HSE band structures.

    This is an HSE06 static calculation followed by one HSE06 uniform calculation and
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
class HSEUniformBandStructureMaker(UniformBandStructureMaker):
    """
    Maker to generate VASP HSE uniform band structures.

    This is an HSE06 static calculation followed by an HSE06 uniform calculation.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    static_maker : .BaseVaspMaker
        The maker to use for the static calculation.
    bs_maker : .BaseVaspMaker
        The maker to use for the uniform band structure calculation.
    """

    name: str = "hse band structure"
    static_maker: BaseVaspMaker = field(default_factory=HSEStaticMaker)
    bs_maker: BaseVaspMaker = field(default_factory=HSEBSMaker)


@dataclass
class HSELineModeBandStructureMaker(LineModeBandStructureMaker):
    """
    Maker to generate VASP HSE line mode band structures.

    This is an HSE06 static calculation followed by an HSE06 line mode calculation.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    static_maker : .BaseVaspMaker
        The maker to use for the static calculation.
    bs_maker : .BaseVaspMaker
        The maker to use for the line-mode band structure calculation.
    """

    name: str = "hse band structure"
    static_maker: BaseVaspMaker = field(default_factory=HSEStaticMaker)
    bs_maker: BaseVaspMaker = field(default_factory=HSEBSMaker)


@dataclass
class RelaxBandStructureMaker(Maker):
    """
    Maker to create a flow with a relaxation and then band structure calculations.

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

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """Run a relaxation, then calculate the uniform and line mode band structures.

        Parameters
        ----------
        structure: .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A relax and band structure flow.
        """
        relax_job = self.relax_maker.make(structure, prev_dir=prev_dir)
        bs_flow = self.band_structure_maker.make(
            relax_job.output.structure, prev_dir=relax_job.output.dir_name
        )

        return Flow([relax_job, bs_flow], bs_flow.output, name=self.name)


@dataclass
class OpticsMaker(Maker):
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

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """Run a static and then a non-scf optics calculation.

        Parameters
        ----------
        structure: .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A static and nscf with optics flow.
        """
        static_job = self.static_maker.make(structure, prev_dir=prev_dir)
        nscf_job = self.band_structure_maker.make(
            static_job.output.structure, prev_dir=static_job.output.dir_name
        )
        return Flow([static_job, nscf_job], nscf_job.output, name=self.name)


@dataclass
class HSEOpticsMaker(Maker):
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

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """Run a static and then a non-scf optics calculation.

        Parameters
        ----------
        structure: .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A static and nscf with optics flow.
        """
        static_job = self.static_maker.make(structure, prev_dir=prev_dir)
        bs_job = self.band_structure_maker.make(
            static_job.output.structure, prev_dir=static_job.output.dir_name
        )
        return Flow([static_job, bs_job], bs_job.output, name=self.name)
