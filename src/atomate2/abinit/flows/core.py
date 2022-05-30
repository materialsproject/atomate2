"""Core abinit flow makers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from jobflow import Flow, Maker
from pymatgen.core.structure import Structure

from atomate2.abinit.jobs.base import BaseAbinitMaker
from atomate2.abinit.jobs.core import NonScfMaker, RelaxMaker, ScfMaker


@dataclass
class LineBandStructureMaker(Maker):
    """
    Maker to generate line abinit band structure.

    This is a static calculation followed by a non-self-consistent field
    calculations.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    scf_maker : .BaseAbinitMaker
        The maker to use for the static calculation.
    bs_maker : .BaseAbinitMaker
        The maker to use for the non-self-consistent field calculations.
    """

    name: str = "line band structure"
    scf_maker: BaseAbinitMaker = field(default_factory=ScfMaker)
    bs_maker: BaseAbinitMaker = field(default_factory=NonScfMaker)

    def make(
        self,
        structure: Structure,
        restart_from: Optional[Union[str, Path]] = None,
    ):
        """
        Create a line mode band structure flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        restart_from : str or Path or None
            One previous directory to restart from.

        Returns
        -------
        Flow
            A line mode band structure flow.
        """
        scf_job = self.scf_maker.make(structure, restart_from=restart_from)
        line_job = self.bs_maker.make(
            prev_outputs=scf_job.output.dirname,
        )
        jobs = [scf_job, line_job]
        return Flow(jobs, line_job.output, name=self.name)


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
    relaxation_makers: Union[Maker, List[Maker]] = field(default_factory=RelaxMaker)

    def make(
        self,
        structure: Optional[Structure] = None,
        restart_from: Optional[Union[str, Path]] = None,
    ):
        """
        Create a relaxation flow.

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
        if isinstance(self.relaxation_makers, Maker):
            relaxation_makers = [self.relaxation_makers]
        else:
            relaxation_makers = self.relaxation_makers
        relax_job1 = relaxation_makers[0].make(
            structure=structure, restart_from=restart_from
        )
        jobs = [relax_job1]
        for rlx_maker in relaxation_makers[1:]:
            rlx_job = rlx_maker.make(
                # structure=jobs[-1].output.structure, restart_from=jobs[-1].output
                restart_from=jobs[-1].output
            )
            jobs.append(rlx_job)
        return Flow(jobs, output=jobs[-1].output, name=self.name)

    @classmethod
    def ion_ioncell_relaxation(cls):
        """Create a double relaxation (ionic relaxation + full relaxation)."""
        ion_rlx_maker = RelaxMaker.ionic_relaxation()
        ioncell_rlx_maker = RelaxMaker.full_relaxation()
        return cls(relaxation_makers=[ion_rlx_maker, ioncell_rlx_maker])
