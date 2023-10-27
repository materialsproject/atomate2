"""Flows for calculating the polarization of a polar material."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Maker
from pymatgen.core.structure import Structure

from atomate2 import SETTINGS
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import PolarizationMaker, RelaxMaker
from atomate2.vasp.jobs.ferroelectric import (
    interpolate_structures,
    add_interpolation_flow,
    polarization_analysis,
    get_polarization_output,
)

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
    relax_maker: BaseVaspMaker or None or tuple
        None to avoid relaxation of both polar and nonpolar structures
        BaseVaspMaker to relax both structures (default)
        tuple of BaseVaspMaker and None to control relaxation for each structure
    lcalcpol_maker: BaseVaspMaker
       Vasp maker to compute the polarization of each structure
    """

    name: str = "ferroelectric"
    nimages: int = 9
    relax_maker: BaseVaspMaker | None | tuple = field(
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
        Make flow to calculate the polarization.

        Parameters
        ----------
        polar_structure : .Structure
            A pymatgen structure of the polar phase.
        nonpolar_structure : .Structure
            A pymatgen structure of the nonpolar phase.
        prev_vasp_dir : str or Path or None
            A previous vasp calculation directory to use for copying outputs.
        """
        jobs = []
        prev_vasp_dir_p, prev_vasp_dir_np = None, None

        if not isinstance(self.relax_maker, tuple):
            self.relax_maker = (self.relax_maker, self.relax_maker)

        if self.relax_maker[0]:
            # optionally relax the polar structure
            relax_p = self.relax_maker[0].make(polar_structure)
            relax_p.append_name(" polar")
            jobs.append(relax_p)
            polar_structure = relax_p.output.structure
            prev_vasp_dir_p = relax_p.output.dir_name

        logger.info(f"{type(polar_structure)}")

        polar_lcalcpol = self.lcalcpol_maker.make(
            polar_structure, prev_vasp_dir=prev_vasp_dir_p
        )
        polar_lcalcpol.append_name(" polar")
        jobs.append(polar_lcalcpol)
        polar_structure = polar_lcalcpol.output.structure

        if self.relax_maker[1]:
            # optionally relax the nonpolar structure
            relax_np = self.relax_maker[1].make(nonpolar_structure)
            relax_np.append_name(" nonpolar")
            jobs.append(relax_np)
            nonpolar_structure = relax_np.output.structure
            prev_vasp_dir_np = relax_np.output.dir_name

        nonpolar_lcalcpol = self.lcalcpol_maker.make(
            nonpolar_structure, prev_vasp_dir=prev_vasp_dir_np
        )
        nonpolar_lcalcpol.append_name(" nonpolar")
        jobs.append(nonpolar_lcalcpol)
        nonpolar_structure = nonpolar_lcalcpol.output.structure

        interp_structs_job = interpolate_structures(
            polar_structure, nonpolar_structure, self.nimages
        )
        jobs.append(interp_structs_job)

        prev_interp_dir = interp_structs_job.output
        add_interp_flow = add_interpolation_flow(prev_interp_dir,
                                                 self.lcalcpol_maker)

        pol_analysis = polarization_analysis(
            get_polarization_output(nonpolar_lcalcpol),
            get_polarization_output(polar_lcalcpol),
            add_interp_flow.output,
        )
        
        jobs.append(add_interp_flow)
        jobs.append(pol_analysis)

        flow = Flow(
            jobs=jobs,
            output=pol_analysis.output,
            name=self.name,
        )
        return flow
