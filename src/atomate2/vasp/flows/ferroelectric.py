"""Flows for calculating elastic constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Maker, OnMissing
from pymatgen.core.structure import Structure

from atomate2 import SETTINGS
from atomate2.common.schemas.math import Matrix3D
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import TightRelaxMaker
from atomate2.vasp.jobs.core import PolarizationMaker
from atomate2.vasp.jobs.ferroelectric import polarization_analysis

__all__ = ["FerroelectricMaker"]


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
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
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
        structure : .Structure
            A pymatgen structure.
        prev_vasp_dir : str or Path or None
            A previous vasp calculation directory to use for copying outputs.
        """
        
        jobs = []
        outputs = {}

        if isinstance(self.relax,bool):
            self.relax = (self.relax,self.relax) 
        
        if self.relax[0]:
            # optionally relax the polar structure
            bulk = self.bulk_relax_maker.make(polar_structure,
                                              prev_vasp_dir=prev_vasp_dir)
            jobs.append(bulk)
            polar_structure = bulk.output.structure
            prev_vasp_dir = bulk.output.dir_name
        
        polar_lcalcpol = self.lcalcpol_maker.make(polar_structure,
                                                  prev_vasp_dir=prev_vasp_dir)

        if self.relax[1]:
            # optionally relax the nonpolar structure
            bulk = self.bulk_relax_maker.make(nonpolar_structure,
                                              prev_vasp_dir=prev_vasp_dir)
            jobs.append(bulk)
            nonpolar_structure = bulk.output.structure
            prev_vasp_dir = bulk.output.dir_name

        nonpolar_lcalcpol = self.lcalcpol_maker.make(nonpolar_structure,
                                                     prev_vasp_dir=prev_vasp_dir)
        jobs.append(polar_lcalcpol)
        jobs.append(nonpolar_lcalcpol)

        outputs = {
            "polar_lcalcpol": polar_lcalcpol.output,
        }

        interpolations = []
        interp_structures = polar_structure.interpolate(nonpolar_structure,nimages,True)
        
        for i,interp_structure in enumerate(interp_structures[1:]):
            interpolation = self.lcalcpol_maker.make(interp_structure)
            jobs.append(interpolation)
            outputs.update({f'interpolation_{i}_lcalcpol':interpolation.output})

        outputs.update({"nonpolar_lcalcpol": nonpolar_lcalcpol.output})
        
        polarization_analysis = polarization_analysis(outputs)
        jobs.append(polarization_analysis)
        
        # allow some of the deformations to fail
        # fit_tensor.config.on_missing_references = OnMissing.NONE

        flow = Flow(
            jobs=jobs,
            output=polarization_analysis.output,
            name=self.name,
        )
        return flow
