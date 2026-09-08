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

__all__ = ["BandStructureMaker", "RelaxFlowMaker"]


@dataclass
class BandStructureMaker(Maker):
    """
    Maker to generate abinit band structures and density of states.

    This flow consists of a static self-consistent calculation followed by
    two non-self-consistent field calculations: one with a uniform k-point
    mesh for density of states and one with a line-mode k-path for the
    band structure.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    static_maker : .BaseAbinitMaker
        The maker to use for the initial static self-consistent calculation.
    bs_maker : .BaseAbinitMaker or None
        The maker to use for the line-mode non-self-consistent field
        calculation to generate the band structure. If None, the band
        structure calculation will be skipped.
    dos_maker : .BaseAbinitMaker or None
        The maker to use for the uniform k-point non-self-consistent field
        calculation to generate the density of states. If None, the DOS
        calculation will be skipped.
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
        """
        Create a band structure and density of states flow.

        This method creates a workflow consisting of:
        1. A static self-consistent field calculation
        2. A uniform k-point non-SCF calculation for DOS (if dos_maker is set)
        3. A line-mode k-point non-SCF calculation for band structure
           (if bs_maker is set)

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object defining the crystal structure.
        restart_from : str or Path or None
            Path to a previous calculation directory to restart from.

        Returns
        -------
        Flow
            A jobflow Flow containing the static job and any requested
            non-self-consistent field jobs for band structure and DOS.
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
    Maker to generate a sequential relaxation flow.

    This flow runs multiple relaxation calculations in sequence, where each
    calculation uses the output structure from the previous one. By default,
    it performs ionic relaxation followed by full relaxation (ions + cell).

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relaxation_makers : list[Maker]
        A list of Maker objects to use for the sequential relaxation steps.
        Each maker in the list will be run in order, with each subsequent
        calculation using the relaxed structure from the previous step.
        Defaults to [ionic_relaxation, full_relaxation].
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
        """
        Create a sequential relaxation flow.

        The first relaxation maker uses the provided structure and optional
        restart directory. Each subsequent relaxation uses the relaxed
        structure from the previous calculation, allowing for progressive
        relaxation (e.g., ions first, then ions + cell).

        Parameters
        ----------
        structure : Structure or None
            A pymatgen Structure object. If None, must provide restart_from.
        restart_from : str or Path or None
            Path to a previous calculation directory to restart the first
            relaxation from. Allows reusing wavefunctions and density.

        Returns
        -------
        Flow
            A jobflow Flow containing all sequential relaxation jobs.
            The Flow output is set to the output of the final relaxation.
        """
        relax_job1 = self.relaxation_makers[0].make(
            structure=structure, restart_from=restart_from
        )
        jobs = [relax_job1]
        for rlx_maker in self.relaxation_makers[1:]:
            rlx_job = rlx_maker.make(prev_outputs=jobs[-1].output.dir_name)
            jobs.append(rlx_job)
        return Flow(jobs, output=jobs[-1].output, name=self.name)

    @classmethod
    def ion_ioncell_relaxation(cls, *args, **kwargs) -> RelaxFlowMaker:
        """
        Create a RelaxFlowMaker with ionic then full relaxation.

        This convenience classmethod creates a two-step relaxation workflow:
        first relaxing only ionic positions, then performing a full relaxation
        of both ions and cell parameters.

        Parameters
        ----------
        *args
            Positional arguments passed to both RelaxMaker.ionic_relaxation()
            and RelaxMaker.full_relaxation().
        **kwargs
            Keyword arguments passed to both RelaxMaker.ionic_relaxation()
            and RelaxMaker.full_relaxation().

        Returns
        -------
        RelaxFlowMaker
            A RelaxFlowMaker instance configured with two sequential makers:
            ionic relaxation followed by full relaxation.
        """
        ion_rlx_maker = RelaxMaker.ionic_relaxation(*args, **kwargs)
        ioncell_rlx_maker = RelaxMaker.full_relaxation(*args, **kwargs)
        return cls(relaxation_makers=[ion_rlx_maker, ioncell_rlx_maker])
