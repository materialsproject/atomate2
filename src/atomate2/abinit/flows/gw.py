"""Core abinit flow makers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from jobflow import Flow, Maker
from pymatgen.core.structure import Structure

from atomate2.abinit.jobs.base import BaseAbinitMaker
from atomate2.abinit.jobs.core import NonSCFMaker, StaticMaker, ConvergenceMaker
from atomate2.abinit.jobs.gw import ScreeningMaker, SigmaMaker, BSEMaker
from atomate2.abinit.powerups import update_user_abinit_settings, update_factory_kwargs, update_user_kpoints_settings 
from pymatgen.io.abinit.abiobjects import KSampling


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
    Maker to perform G0W0 calculation from previous GWbands calculation.

    This is a screening calculation followed by a sigma calculations, 
    one can perform QP corrections only for bandedges (useful for 
    convergence calculations) or at all k-points.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    scr_maker : .BaseAbinitMaker
        The maker to use for the screening calculation.
    sigma_maker : .BaseAbinitMaker
        The maker to use for the sigma calculations.
    gw_qprange: int
           0 - Compute the QP corrections only for the fundamental and the direct gap

        +num - Compute the QP corrections for all the k-points in the irreducible zone
               , and include num bands above and below the Fermi level.

        -num - Compute the QP corrections for all the k-points in the irreducible zone. 
               Include all occupied states and num empty states.

    """

    name: str = "G0W0 calculation"
    scr_maker: BaseAbinitMaker = field(default_factory=ScreeningMaker)
    sigma_maker: BaseAbinitMaker = field(default_factory=SigmaMaker)
    gw_qprange: int = 0

    def make(
        self,
        structure: Structure,
        prev_outputs: str = None, 
        restart_from: Optional[Union[str, Path]] = None
    ):
        """
        Create a G0W0 flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_outputs : str 
            One previous directory where ncsf 
            calculation were performed.
        restart_from : str or Path or None
            One previous directory to restart from.

        Returns
        -------
        Flow
            A G0W0 flow.
        """
		 
        scr_job = self.scr_maker.make(
            prev_outputs=prev_outputs, 
        )
        m_scr_job = update_user_abinit_settings(
            flow=scr_job, 
            abinit_updates={"iomode": 3}
        )
        sigma_job = self.sigma_maker.make(
            prev_outputs=[prev_outputs, scr_job.output.dir_name],
        )
        m_sigma_job = update_user_abinit_settings(
            flow=sigma_job, 
            abinit_updates={"gw_qprange": self.gw_qprange, "iomode": 3}
        )
        return Flow([m_scr_job, m_sigma_job], output=m_sigma_job.output, name=self.name)

@dataclass
class G0W0ConvergenceMaker(Maker):
    """
    Maker to generate convergence of G0W0 calculations.

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
	
        static = GWbandsMaker().make(structure)
        gw_maker = G0W0Maker()
        convergence = ConvergenceMaker(
            maker=gw_maker,
            epsilon=self.epsilon,
            criterion_name=self.criterion_name,
            convergence_field=self.convergence_field,
            convergence_steps=self.convergence_steps,
        )
        gw = convergence.make(structure, prev_outputs=static.output.dir_name)
        return Flow([static, gw], gw.output, name=self.name)

@dataclass
class BSEmdfMaker(Maker):

    bs_nband: int  
    mdf_epsinf: float   
    name: str = "BSE mdf calculation"
    scf_maker: BaseAbinitMaker = field(default_factory=StaticMaker)
    nscf_maker: BaseAbinitMaker = field(default_factory=NonSCFMaker)
    bse_maker: BaseAbinitMaker = field(default_factory=BSEMaker)
    bs_loband: int = 1 
    mbpt_sciss: float = 0.0
    kppa: int = 100
    shiftk: tuple = (0.11, 0.12, 0.13)
 
    def make(
        self,
        structure: Structure,
        restart_from: Optional[Union[str, Path]] = None
    ):
		 
        scf_job = self.scf_maker.make(
            structure, 
            restart_from=restart_from
        )
        nscf_job = self.nscf_maker.make(
            prev_outputs=[scf_job.output.dir_name], 
            mode="uniform",
        )
        #njob=update_user_abinit_settings(
        #    flow=nscf_job,
        #    abinit_updates={
        #      'rfelfd':   1,
        #      'rfdir': (1, 1, 1),         
        #      'nqpt':  1,
        #      'qpt':   (0.0, 0.0, 0.0),
        #      'kptopt':   2,         
        #      'iscf':  -2,
        #      'tolwfr': 1e-22}
        #) 
        nscf_job = update_user_kpoints_settings(
            flow=nscf_job, 
            kpoints_updates=KSampling.automatic_density(
            structure=structure, 
            kppa=self.kppa,
            shifts=self.shiftk, 
            chksymbreak=0)
        )
        bse_job = self.bse_maker.make(
            prev_outputs=[nscf_job.output.dir_name, 
            "/home/ucl/modl/tbiswas/scratch/abinit_run/sigma"],
        )
        bse_job=update_factory_kwargs(
            flow=bse_job, factory_updates={
            'bs_loband': self.bs_loband, 
            'bs_nband': self.bs_nband, 
            'mdf_epsinf': self.mdf_epsinf, 
            'mbpt_sciss': self.mbpt_sciss}
        )
        return Flow([scf_job, nscf_job, bse_job], output=nscf_job.output, name=self.name)
