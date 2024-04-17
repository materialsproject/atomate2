"""Core abinit flow makers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from jobflow import Flow, Maker
from pymatgen.core.structure import Structure

from atomate2.abinit.jobs.base import BaseAbinitMaker
from atomate2.abinit.jobs.core import NonSCFMaker, StaticMaker, ConvergenceMaker
from atomate2.abinit.jobs.gw import ScreeningMaker, SigmaMaker
from atomate2.abinit.powerups import update_user_abinit_settings


@dataclass
class GWbandsMaker(Maker):
    """
    Maker to generate bands for GW caculation.
    """

    name: str = "Bands calculation"
    scf_maker: StaticMaker = field(default_factory=StaticMaker)
    nscf_maker: NonSCFMaker = field(default_factory=NonSCFMaker)

    def make(
        self,
        structure: Structure,
        restart_from: Optional[Union[str, Path]] = None,
    ):
        """
        Create a G0W0 flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        restart_from : str or Path or None
            One previous directory to restart from.

        Returns
        -------
        Flow
            A G0W0 flow.
        """

        scf_job = self.scf_maker.make(
            structure, 
            restart_from=restart_from)
        nscf_job = self.nscf_maker.make(
            prev_outputs=[scf_job.output.dir_name], 
            mode="uniform",
        )
        return Flow([scf_job, nscf_job], output=nscf_job.output, name=self.name)

@dataclass
class G0W0Maker(Maker):
    """
    Maker to generate G0W0 flows.
    """

    name: str = "G0W0 calculation"
    nscf_output: str = None
    scr_maker: BaseAbinitMaker = field(default_factory=ScreeningMaker)
    sigma_maker: BaseAbinitMaker = field(default_factory=SigmaMaker)

    def make(
        self,
        structure: Structure,
        restart_from: Optional[Union[str, Path]] = None
    ):
        """
        Create a G0W0 flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        restart_from : str or Path or None
            One previous directory to restart from.

        Returns
        -------
        Flow
            A G0W0 flow.
        """
		 
        scr_job = self.scr_maker.make(
            prev_outputs=[self.nscf_output], 
        )
        m_scr_job = update_user_abinit_settings(
            flow=scr_job, 
            abinit_updates={"iomode": 3}
        )
        sigma_job = self.sigma_maker.make(
            prev_outputs=[self.nscf_output, scr_job.output.dir_name],
        )
        m_sigma_job = update_user_abinit_settings(
            flow=sigma_job, 
            abinit_updates={"gw_qprange": 0, "iomode": 3}
        )
        return Flow([m_scr_job, m_sigma_job], output=m_sigma_job.output, name=self.name)

@dataclass
class G0W0ConvergenceMaker(Maker):
    """
    Maker to generate convergence of G0W0 calculations.
    """

    name: str = "G0W0 calculation"
    scf_maker: StaticMaker = field(default_factory=StaticMaker)
    nscf_maker: NonSCFMaker = field(default_factory=NonSCFMaker)
    scr_makers: List[ScreeningMaker] = field(default_factory=lambda: [ScreeningMaker()])
    sigma_makers: List[SigmaMaker] = field(default_factory=lambda: [SigmaMaker()])

    def __post_init__(self):
        # TODO: make some checks on the input sets, e.g.:
        #  - non scf has to be uniform
        #  - set istwfk ? or check that it is "*1" ?
        #  - kpoint shifts ?
        pass

    def make(
        self,
        structure: Structure,
        restart_from: Optional[Union[str, Path]] = None,
    ):
        """
        Create a convergence G0W0 flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        restart_from : str or Path or None
            One previous directory to restart from.

        Returns
        -------
        Flow
            A G0W0 flow.
        """

        scf_job = self.scf_maker.make(structure, restart_from=restart_from)
        nscf_job = self.nscf_maker.make(
            prev_outputs=scf_job.output.dir_name, mode="uniform"
        )
        jobs = [scf_job, nscf_job]
        for scr_maker in self.scr_makers:
            scr_job = scr_maker.make(prev_outputs=nscf_job.output.dir_name)
            jobs.append(scr_job)
            for sigma_maker in self.sigma_makers:
                sigma_job = sigma_maker.make(
                    prev_outputs=[nscf_job.output.dir_name, scr_job.output.dir_name]
                )
                jobs.append(sigma_job)

        return Flow(jobs, name=self.name)


@dataclass
class PeriodicGWConvergenceMaker(Maker):
    """
    A maker to perform a GW workflow with automatic convergence in FHI-aims.

    Parameters
    ----------
    name : str
        A name for the job
    criterion_name: str
        A name for the convergence criterion. Must be in the run results
    epsilon: float
        A difference in criterion value for subsequent runs
    convergence_field: str
        An input parameter that changes to achieve convergence
    convergence_steps: list | tuple
        An iterable of the possible values for the convergence field.
        If the iterable is depleted and the convergence is not reached,
        that the job is failed
    """

    name: str = "GW convergence"
    criterion_name: str = "bandgap"
    epsilon: float = 0.1
    convergence_field: str = field(default_factory=str)
    convergence_steps: list = field(default_factory=list)

    #def __post_init__(self):
        # TODO: make some checks on the input sets, e.g.:
        #  - non scf has to be uniform
        #  - set istwfk ? or check that it is "*1" ?
        #  - kpoint shifts ?
        #  - check nbands in nscf is >= nband in screening and sigma
    #    pass

    def make(
        self,
        structure: Structure,
        restart_from: Optional[Union[str, Path]] = None,
    ):
	
        #scf_job = self.scf_maker.make(structure, restart_from=restart_from)
        #nscf_job = self.nscf_maker.make(
        #    prev_outputs=scf_job.output.dir_name, mode="uniform"
        #)
            #scr_job = self.scr_maker.make(prev_outputs=["../nscf"],abinit_settings={self.convergence_field: value})
        #static = GWbandsMaker().make(structure)
        gw_maker = G0W0Maker(nscf_output='/home/ucl/modl/tbiswas/abinit_run/nscf')
        convergence = ConvergenceMaker(
            maker=gw_maker,
            epsilon=self.epsilon,
            criterion_name=self.criterion_name,
            convergence_field=self.convergence_field,
            convergence_steps=self.convergence_steps,
        )
        gw = convergence.make(structure)
        return Flow([gw], gw.output, name=self.name)

