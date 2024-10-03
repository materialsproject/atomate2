"""Core abinit flow makers."""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from jobflow import Flow, Maker, Response, job
from pymatgen.core.structure import Structure

from atomate2.abinit.jobs.base import BaseAbinitMaker
from atomate2.abinit.jobs.core import NonSCFMaker, StaticMaker, ConvergenceMaker
from atomate2.abinit.jobs.gw import ScreeningMaker, SigmaMaker
from atomate2.abinit.powerups import update_user_abinit_settings


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
    gw_qprange: int
           0 - Compute the QP corrections only for the fundamental and the direct gap

        +num - Compute the QP corrections for all the k-points in the irreducible zone
               , and include num bands above and below the Fermi level.

        -num - Compute the QP corrections for all the k-points in the irreducible zone. 
               Include all occupied states and num empty states.

    joblist : list[str]
        Steps of GW calculations to be included.
        Default is ["scf", "nscf", "scr", "sigma"],
        which creates a worflow to perform the 
        entire GW calculations.
    scf_maker : .BaseAbinitMaker
        The maker to use for the scf calculation.
    nscf_maker : .BaseAbinitMaker
        The maker to use for the nscf calculations.
    scr_maker : .BaseAbinitMaker
        The maker to use for the screening calculation.
    sigma_maker : .BaseAbinitMaker
        The maker to use for the sigma calculations.
    """

    name: str = "G0W0 calculation"
    gw_qprange: int = 0
    joblist: List = field(default_factory=lambda: ["scf", "nscf", "scr", "sigma"])
    scf_maker: BaseAbinitMaker = field(default_factory=StaticMaker)
    nscf_maker: BaseAbinitMaker = field(default_factory=NonSCFMaker)
    scr_maker: BaseAbinitMaker = field(default_factory=ScreeningMaker)
    sigma_maker: BaseAbinitMaker = field(default_factory=SigmaMaker)

    def make(
        self,
        structure: Structure,
        prev_outputs: Optional[list] = [],
        restart_from: Optional[Union[str, Path]] = None,
    ):
        """
        Create a G0W0 flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_outputs : str 
            List of previous directory where scf, ncsf and scr calculations were done.
        restart_from : str or Path or None
            One previous directory to restart from.

        Returns
        -------
        Flow
            A G0W0 flow.
        """
        joblist=self.joblist
        jobs=[]
        result={}
        #SCF step
        scf_job = self.scf_maker.make(
            structure, 
            restart_from=restart_from
        )
        if "scf" in joblist:
            if len(prev_outputs)!=0:
                raise RuntimeError("No previous calculation needed in prev_outputs")
            jobs.append(scf_job)
            prev_outputs=[scf_job.output.dir_name]
            result=scf_job.output
  
        #NSCF step
        if "nscf" in joblist:
            if len(prev_outputs)!=1:
                raise RuntimeError("Previous SCF calculation needed in prev_outputs")
            nscf_job = self.nscf_maker.make(
                prev_outputs=[prev_outputs[0]], 
                mode="uniform",
            )
            jobs.append(nscf_job)
            prev_outputs=[nscf_job.output.dir_name]
            result=nscf_job.output

        #SCR step
        if "scr" in joblist:
            if len(prev_outputs)!=1:
                raise RuntimeError("Previous SCF and NSCF calculation needed in prev_outputs")
            scr_job = self.scr_maker.make(
                prev_outputs=[prev_outputs[0]], 
            )
            scr_job = update_user_abinit_settings(
                flow=scr_job, 
                abinit_updates={"iomode": 3}
            )
            jobs.append(scr_job)
            prev_outputs.append(scr_job.output.dir_name)
            result=scr_job.output

        #SIGMA step
        if "sigma" in joblist:
            if len(prev_outputs)!=2:
                raise RuntimeError("Previous SCF, NSCF and SCR calculation needed in prev_outputs")
            sigma_job = self.sigma_maker.make(
                prev_outputs=[prev_outputs[0], prev_outputs[1]],
            )
            sigma_job = update_user_abinit_settings(
                flow=sigma_job, 
                abinit_updates={"gw_qprange": self.gw_qprange, "iomode": 3}
            )
            jobs.append(sigma_job)
            result=sigma_job.output

        return Flow(jobs, output=result, name=self.name)

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

    def make(
        self,
        structure: Structure,
        restart_from: Optional[Union[str, Path]] = None,
    ):  

        NSCF_FIELDS = ["nband","ngkpt"]
        SCR_FIELDS = ["ecuteps"]
        SIGMA_FIELDS = ["ecutsigx"]

        SUPPORTED_FIELDS = NSCF_FIELDS + SCR_FIELDS + SIGMA_FIELDS

        if self.convergence_field not in SUPPORTED_FIELDS: 
            raise RuntimeError("convergence_field not supported yet")

        if self.convergence_field in NSCF_FIELDS:
            static_job = G0W0Maker(joblist=["scf"]).make(structure)
            gw_maker = G0W0Maker(joblist=["nscf","scr","sigma"])
            flow=[static_job] 
            prev_outputs=[static_job.output.dir_name]

        if self.convergence_field in SCR_FIELDS: 
            static_job = G0W0Maker(joblist=["scf","nscf"]).make(structure)
            gw_maker = G0W0Maker(joblist=["scr","sigma"])
            flow=[static_job] 
            prev_outputs=[static_job.output.dir_name]

        if self.convergence_field in SIGMA_FIELDS: 
            pre_static_job = G0W0Maker(joblist=["scf","nscf"]).make(structure)
            static_job = G0W0Maker(joblist=["scr"]).make(structure, prev_outputs=[pre_static_job.output.dir_name])
            gw_maker = G0W0Maker(joblist=["sigma"])
            flow=[pre_static_job,static_job] 
            prev_outputs=[pre_static_job.output.dir_name, static_job.output.dir_name]

        convergence = ConvergenceMaker(
            maker = gw_maker,
            epsilon = self.epsilon,
            criterion_name = self.criterion_name,
            convergence_field = self.convergence_field,
            convergence_steps = self.convergence_steps,
            )

        gw_job = convergence.make(structure, prev_outputs=prev_outputs)
        flow.append(gw_job)
        return Flow(flow, gw_job.output, name=self.name)

