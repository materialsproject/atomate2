"""Core VASP flows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Job, Maker

from atomate2.cp2k.jobs.core import (
    HybridCellOptMaker,
    HybridRelaxMaker,
    HybridStaticMaker,
    NonSCFMaker,
    RelaxMaker,
    StaticMaker,
)
from atomate2.cp2k.schemas.calculation import Cp2kObject
from atomate2.cp2k.sets.base import recursive_update

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure
    from typing_extensions import Self

    from atomate2.cp2k.jobs.base import BaseCp2kMaker


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
    relax_maker1: Maker = field(default_factory=RelaxMaker)
    relax_maker2: Maker = field(default_factory=RelaxMaker)

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """Create a flow with two chained relaxations.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous Cp2k calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing two relaxations.
        """
        relax1 = self.relax_maker1.make(structure, prev_dir=prev_dir)
        relax1.name += " 1"

        relax2 = self.relax_maker2.make(
            relax1.output.structure, prev_dir=relax1.output.dir_name
        )
        relax2.name += " 2"

        return Flow([relax1, relax2], relax2.output, name=self.name)

    @classmethod
    def from_relax_maker(cls, relax_maker: BaseCp2kMaker) -> Self:
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
    static_maker: Maker = field(default_factory=StaticMaker)
    bs_maker: Maker = field(default_factory=NonSCFMaker)

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
                "uniform_bs": uniform_job.output.cp2k_objects[Cp2kObject.BANDSTRUCTURE],
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
                "line_bs": line_job.output.cp2k_objects[Cp2kObject.BANDSTRUCTURE],
            }
            outputs.update(output)

        if bandstructure_type not in ("both", "line", "uniform"):
            raise ValueError(f"Unrecognised {bandstructure_type=}")

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
    relax_maker: Maker = field(default_factory=DoubleRelaxMaker)
    band_structure_maker: Maker = field(default_factory=BandStructureMaker)

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """Run a relaxation, then calculate the uniform and line mode band structures.

        Parameters
        ----------
        structure: .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous CP2K calculation directory to copy output files from.

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
class HybridFlowMaker(Maker):
    """
    Maker to create hybrid flows.

    Parameters
    ----------
    hybrid_functional
        built-in hybrid functional to use
    initialize_with_pbe
        Whether or not to attach a pre-hybrid flow that can be used to
        kickstart the hybrid flow. This is treated differently than just
        stiching flows together, because of the screening done in
        __post_init__
    pbe_maker
        Maker for the initialization
    hybrid_maker
        Maker for the hybrid job
    """

    hybrid_functional: str = "PBE0"
    initialize_with_pbe: bool = field(default=True)
    pbe_maker: Maker = field(default=lambda: StaticMaker(store_output_data=False))
    hybrid_maker: Maker = field(default_factory=HybridStaticMaker)

    def __post_init__(self) -> None:
        """
        Post init updates.

        Set the user-specified hybrid_functional and activate initial density matrix
        screening if restarting from a PBE calculation.

        Initializing with PBE allows CP2K to screen exchange integrals using
        the PBE density matrix, which creates huge speed-ups. Rarely causes
        problems so it is done as a default here.
        """
        updates: dict[str, dict[str, str | bool]] = {
            "activate_hybrid": {"hybrid_functional": self.hybrid_functional}
        }
        if self.initialize_with_pbe:
            updates["activate_hybrid"].update(
                {"screen_on_initial_p": True, "screen_p_forces": True}
            )
        self.hybrid_maker.input_set_generator.user_input_settings = recursive_update(
            updates, self.hybrid_maker.input_set_generator.user_input_settings
        )

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Job:
        """Make a hybrid flow.

        Parameters
        ----------
        structure: .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous CP2K calculation directory to copy output files from.

        Returns
        -------
        Flow
            A hybrid flow with a possible PBE initialization step
        """
        jobs = []
        if self.initialize_with_pbe:
            initialization = self.pbe_maker.make(structure, prev_dir)
            jobs.append(initialization)
        hyb = self.hybrid_maker.make(
            initialization.output.structure if self.initialize_with_pbe else structure,
            prev_dir=initialization.output.dir_name
            if self.initialize_with_pbe
            else prev_dir,
        )
        jobs.append(hyb)
        return Flow(jobs, output=hyb.output, name=self.name)


@dataclass
class HybridStaticFlowMaker(HybridFlowMaker):
    """
    Maker to perform a PBE restart to hybrid static flow.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    pbe_maker : .BaseCp2kMaker
        Maker to use to generate PBE restart file for hybrid calc
    hybrid_maker : .BaseCp2kMaker
        Maker to use to generate the second relaxation.
    """

    name: str = "hybrid static flow"


@dataclass
class HybridRelaxFlowMaker(HybridFlowMaker):
    """
    Maker to perform a PBE restart to hybrid relax flow.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    pbe_maker : .BaseCp2kMaker
        Maker to use to generate PBE restart file for hybrid calc
    hybrid_maker : .BaseCp2kMaker
        Maker to use to generate the second relaxation.
    """

    name: str = "hybrid relax flow"
    hybrid_maker: Maker = field(default_factory=HybridRelaxMaker)


@dataclass
class HybridCellOptFlowMaker(HybridFlowMaker):
    """
    Maker to perform a PBE restart to hybrid cell opt flow.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    pbe_maker : .BaseCp2kMaker
        Maker to use to generate PBE restart file for hybrid calc
    hybrid_maker : .BaseCp2kMaker
        Maker to use to generate the second relaxation.
    """

    name: str = "hybrid cell opt flow"
    hybrid_maker: Maker = field(default_factory=HybridCellOptMaker)
