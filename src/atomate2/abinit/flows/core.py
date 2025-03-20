"""Core abinit flow makers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.abinit.jobs.core import (
    LineNonSCFMaker,
    RelaxMaker,
    StaticMaker,
    UniformNonSCFMaker,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure

    from atomate2.abinit.jobs.base import BaseAbinitMaker


@dataclass
class BandStructureMaker(Maker):
    """
    Maker to generate abinit band structures.

    This is a static calculation followed by two non-self-consistent field
    calculations, one uniform and one line mode.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    scf_maker : .BaseAbinitMaker
        The maker to use for the static calculation.
    bs_maker : .BaseAbinitMaker
        The maker to use for the non-self-consistent field calculations.
    """

    name: str = "band structure - dos"
    static_maker: BaseAbinitMaker = field(default_factory=StaticMaker)
    bs_maker: BaseAbinitMaker | None = field(default_factory=LineNonSCFMaker)
    dos_maker: BaseAbinitMaker | None = field(default_factory=UniformNonSCFMaker)

    def make(
        self,
        structure: Structure,
        restart_from: str | Path | None = None,
    ) -> Flow:
        """Create a band structure flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        restart_from : str or Path or None
            One previous directory to restart from.

        Returns
        -------
        Flow
            A band structure flow.
        """
        static_job = self.static_maker.make(structure, restart_from=restart_from)
        jobs = [static_job]

        if self.dos_maker:
            uniform_job = self.dos_maker.make(
                prev_outputs=static_job.output.dir_name,
            )
            jobs.append(uniform_job)

        if self.bs_maker:
            line_job = self.bs_maker.make(
                prev_outputs=static_job.output.dir_name,
            )
            jobs.append(line_job)

        return Flow(jobs, name=self.name)


@dataclass
class RelaxFlowMaker(Maker):
    """
    Maker to generate a relaxation flow with abinit.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relaxation_makers : .BaseAbinitMaker
        The maker or list of makers to use for the relaxation flow.
    """

    name: str = "relaxation"
    relaxation_makers: list[Maker] = field(
        default_factory=lambda: [
            RelaxMaker.ionic_relaxation(),
            RelaxMaker.full_relaxation(),
        ]
    )

    def make(
        self,
        structure: Structure | None = None,
        restart_from: str | Path | None = None,
    ) -> Flow:
        """Create a relaxation flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        restart_from : str or Path or None
            One previous directory to restart from.

        Returns
        -------
        Flow
            A relaxation flow.
        """
        relax_job1 = self.relaxation_makers[0].make(
            structure=structure, restart_from=restart_from
        )
        jobs = [relax_job1]
        for rlx_maker in self.relaxation_makers[1:]:
            rlx_job = rlx_maker.make(restart_from=jobs[-1].output.dir_name)
            jobs.append(rlx_job)
        return Flow(jobs, output=jobs[-1].output, name=self.name)

    @classmethod
    def ion_ioncell_relaxation(cls, *args, **kwargs) -> Flow:
        """Create a double relaxation (ionic relaxation + full relaxation)."""
        ion_rlx_maker = RelaxMaker.ionic_relaxation(*args, **kwargs)
        ioncell_rlx_maker = RelaxMaker.full_relaxation(*args, **kwargs)
        return cls(relaxation_makers=[ion_rlx_maker, ioncell_rlx_maker])
