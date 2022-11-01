"""Flows for calculating elastic constants."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Maker, OnMissing
from pymatgen.core.structure import Structure

from atomate2 import SETTINGS
from atomate2.common.schemas.math import Matrix3D
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import RelaxMaker
from atomate2.vasp.jobs.core import PolarizationMaker
from atomate2.vasp.jobs.ferroelectric import polarization_analysis, interpolate_structures

__all__ = ["FerroelectricMaker"]

logger = logging.getLogger(__name__)

@dataclass
class FerroelectricMaker(Maker):
    """
    Maker to calculate polarization of a polar material.

    
    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    nimages: int
        Number of interpolations calculated from polar to nonpolar structures, 
        including the nonpolar.
    """

    name: str = "ferroelectric"
    nimages: int = 9
    symprec: float = SETTINGS.SYMPREC
    relax: bool | tuple = False
    bulk_relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(RelaxMaker())
    )
    lcalcpol_maker: BaseVaspMaker = field(default_factory=PolarizationMaker)

    def make(
        self,
        polar_structure: Structure,
        nonpolar_structure: Structure,
        prev_vasp_dir: str | Path | None = None,
    ):
        """
        Make flow to calculate the polarization

        Parameters
        ----------
        polar_structure : .Structure
            A pymatgen structure of polar phase.
        nonpolar_structure : .Structure
            A pymatgen structure of nonpolar phase.
        prev_vasp_dir : str or Path or None
            A previous vasp calculation directory to use for copying outputs.
        """
        
        jobs = []
        outputs = {}
        prev_vasp_dir_p,prev_vasp_dir_np = None, None
        
        if isinstance(self.relax,bool):
            self.relax = (self.relax,self.relax) 
        
        if self.relax[0]:
            # optionally relax the polar structure
            relax_p = self.bulk_relax_maker.make(polar_structure)
            relax_p.append_name(" polar")
            jobs.append(relax_p)
            polar_structure = relax_p.output.structure
            prev_vasp_dir_p = relax_p.output.dir_name

        logger.info(f'{type(polar_structure)}')
        
        polar_lcalcpol = self.lcalcpol_maker.make(polar_structure,
                                                  prev_vasp_dir=prev_vasp_dir_p)
        polar_lcalcpol.name += " polar"
        jobs.append(polar_lcalcpol)
        polar_structure = polar_lcalcpol.output.structure
        
        if self.relax[1]:
            # optionally relax the nonpolar structure
            relax_np = self.bulk_relax_maker.make(nonpolar_structure)
            relax_np.append_name(" nonpolar")
            jobs.append(relax_np)
            nonpolar_structure = relax_np.output.structure
            prev_vasp_dir_np = relax_np.output.dir_name

        nonpolar_lcalcpol = self.lcalcpol_maker.make(nonpolar_structure,
                                                     prev_vasp_dir=prev_vasp_dir_np)
        nonpolar_lcalcpol.append_name(" nonpolar")
        jobs.append(nonpolar_lcalcpol)
        nonpolar_structure = nonpolar_lcalcpol.output.structure
        

        interp_lcalcpol = interpolate_structures(polar_structure,
                                                 nonpolar_structure,
                                                 self.nimages)
        jobs.append(interp_lcalcpol)
        
        pol_analysis = polarization_analysis([nonpolar_lcalcpol.output,
                                              polar_lcalcpol.output,
                                              interp_lcalcpol.output])
        jobs.append(pol_analysis)
        
        # allow some of the deformations to fail
        # fit_tensor.config.on_missing_references = OnMissing.NONE

        flow = Flow(
            jobs=jobs,
            output=pol_analysis.output,
            name=self.name,
        )
        return flow
