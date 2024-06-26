"""Core abinit flow makers."""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from jobflow import Flow, Maker, Response, job
from pymatgen.core.structure import Structure

from atomate2.abinit.jobs.base import BaseAbinitMaker
from atomate2.abinit.jobs.core import NonSCFMaker, StaticMaker, ConvergenceMaker
from atomate2.abinit.jobs.bse import BSEmdfMaker, BSEscrMaker
from atomate2.abinit.powerups import update_user_abinit_settings, update_factory_kwargs, update_user_kpoints_settings 
from pymatgen.io.abinit.abiobjects import KSampling
from atomate2.abinit.schemas.task import AbinitTaskDoc, ConvergenceSummary 


@dataclass
class BSEFlowMaker(Maker):

    name: str = "BSE mdf calculation"
    nscf_maker: BaseAbinitMaker = field(default_factory=NonSCFMaker)
    bse_maker: BaseAbinitMaker = field(default_factory=BSEmdfMaker)
    kppa: int = 1000
    shifts: tuple = (0.11, 0.22, 0.33)
    mbpt_sciss: float = 0.0
    mdf_epsinf: float = None
    enwinbse: float = 3.0
 
    def make(
        self,
        structure: Structure,
        prev_outputs: Union[str, Path, list[str]], 
    ):

        nscf_job = self.nscf_maker.make(
            prev_outputs=prev_outputs[0], 
            mode="uniform",
        )

        nscf_job = update_user_kpoints_settings(
            flow=nscf_job, 
            kpoints_updates=KSampling.automatic_density(
            structure=structure, 
            kppa=self.kppa,
            shifts=self.shifts, 
            chksymbreak=0)
        )
        nscf_job = update_user_abinit_settings(
            flow=nscf_job, 
            abinit_updates={"nstep": 50}
        )
        bse_prepjob = self.find_bse_params(
            nscf_job.output.output.bandlims, 
            self.enwinbse, 
            nscf_job.output.output.direct_gap
        )  

        if len(prev_outputs)==2:
            prev_outputs=[nscf_job.output.dir_name, prev_outputs[1]]
        else: 
            prev_outputs=[nscf_job.output.dir_name]

        bse_job = self.bse_maker.make(
            prev_outputs=prev_outputs,
            mbpt_sciss=self.mbpt_sciss,
            bs_loband=bse_prepjob.output["bs_loband"],
            nband=bse_prepjob.output["nband"],
            mdf_epsinf=self.mdf_epsinf,
            bs_freq_mesh=bse_prepjob.output["bs_freq_mesh"]
        )
        jobs=[nscf_job, bse_prepjob, bse_job]

        return Flow(jobs, output=bse_job.output, name=self.name)

    @job(name="Find BSE parameters")
    def find_bse_params(self, bandlims, enwinbse, directgap):
        vband=[]
        cband=[]
        for bandlim in bandlims:
            spin=bandlim[0]
            iband=bandlim[1]+1
            enemin=bandlim[2]
            enemax=bandlim[3]
            if enemin>0:
                if enemin<directgap+enwinbse:
                    cband.append(iband)
            if enemax<=0:
                if abs(enemax)<abs(enwinbse):
                    vband.append(iband)
        output={"nband": max(cband),
                "bs_loband": min(vband),
                "bs_freq_mesh": [0, enwinbse+directgap, 0.01]}
        return Response(output=output)



@dataclass
class BSEConvergenceMaker(Maker):
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

    name: str = "BSE convergence"
    scf_maker: StaticMaker = field(default_factory=StaticMaker)
    bse_maker: Maker = field(default_factory=Maker)
    criterion_name: str = "emacro"
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
	
        static_job = self.scf_maker.make(
            structure, 
            restart_from=restart_from
        )
        convergence = ConvergenceMaker(
            maker=self.bse_maker,
            epsilon=self.epsilon,
            criterion_name=self.criterion_name,
            convergence_field=self.convergence_field,
            convergence_steps=self.convergence_steps,
        )

        bse = convergence.make(structure, prev_outputs=[static_job.output.dir_name])

        return Flow([static_job, bse], bse.output, name=self.name)

@dataclass
class BSEMultiShiftedMaker(Maker):
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
    cards: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.cards:
            self.cards = self.create_cards()

    def create_cards(self):
        return ['King', 'Queen']



    name: str = "BSE Mutiple Shifted Grid"
    scf_maker: StaticMaker = field(default_factory=StaticMaker)
    bse_maker: Maker = field(default_factory=Maker)
    shiftks: list = None

    def make(
        self,
        structure: Structure,
        restart_from: Optional[Union[str, Path]] = None,
    ):
	
        jobs=[]
        spectra=[] 
        static_job = self.scf_maker.make(
            structure, 
            restart_from=restart_from
        )
        jobs.append(static_job)
        for idx, shifts in enumerate(self.shiftks):
            bse_job = self.bse_maker.make(
                    structure=structure,
                    prev_outputs=[static_job.output.dir_name],
                    )
            bse_job = update_user_abinit_settings(
                    flow=bse_job, 
                    abinit_updates={
                        "shiftk": shifts}
                    )
            bse_job.append_name(append_str=f" {idx}")
            jobs.append(bse_job)
            spectra.append(
                    bse_job.output.output.emacro,
                    )             
        avg_job=self.calc_average_spectra(spectra)  
        jobs.append(avg_job)
        return Flow(jobs, output=avg_job.output, name=self.name)

    @job(name="Calculate average spectra")
    def calc_average_spectra(self, spectra):
        for idx, spectrum in enumerate(spectra):
            if idx==0:
                mesh0=spectrum[0]
                teps2=spectrum[1]
            else:
                mesh=spectrum[0]
                int_eps2=np.interp(mesh0, mesh, spectrum[1])
                teps2=np.add(teps2, int_eps2)
        teps2=np.array(teps2)*(1./len(spectra))
        conv_res=[mesh0, teps2]
        return Response(output=conv_res)


