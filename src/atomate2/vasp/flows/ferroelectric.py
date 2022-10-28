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
from atomate2.vasp.jobs.elastic import (
    ElasticRelaxMaker,
    fit_elastic_tensor,
    generate_elastic_deformations,
    run_elastic_deformations,
)

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
    relax: bool = False
    bulk_relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )
    elastic_relax_maker: BaseVaspMaker = field(default_factory=ElasticRelaxMaker)


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

        if isinstance(relax,bool):
            relax = (relax,relax)

        if relax[0]:
            # optionally relax the polar structure
            bulk = self.bulk_relax_maker.make(polar_structure,
                                              prev_vasp_dir=prev_vasp_dir)
            jobs.append(bulk)
            polar_structure = bulk.output.structure
            prev_vasp_dir = bulk.output.dir_name

        if relax[1]:
            # optionally relax the nonpolar structure
            bulk = self.bulk_relax_maker.make(nonpolar_structure,
                                              prev_vasp_dir=prev_vasp_dir)
            jobs.append(bulk)
            nonpolar_structure = bulk.output.structure
            prev_vasp_dir = bulk.output.dir_name

        polar_lcalcpol = self.polar_lcalcpol_maker.make(nonpolar_structure,
                                                        prev_vasp_dir=prev_vasp_dir)
        nonpolar_lcalcpol = self.nonpolar_lcalcpol_maker.make(nonpolar_structure,
                                                        prev_vasp_dir=prev_vasp_dir)

        
        deformations = generate_elastic_deformations(
            structure,
            order=self.order,
            sym_reduce=self.sym_reduce,
            symprec=self.symprec,
            **self.generate_elastic_deformations_kwargs,
        )
        vasp_deformation_calcs = run_elastic_deformations(
            structure,
            deformations.output,
            prev_vasp_dir=prev_vasp_dir,
            elastic_relax_maker=self.elastic_relax_maker,
        )
        fit_tensor = fit_elastic_tensor(
            structure,
            vasp_deformation_calcs.output,
            equilibrium_stress=equilibrium_stress,
            order=self.order,
            symprec=self.symprec if self.sym_reduce else None,
            **self.fit_elastic_tensor_kwargs,
        )

        # allow some of the deformations to fail
        fit_tensor.config.on_missing_references = OnMissing.NONE

        jobs += [deformations, vasp_deformation_calcs, fit_tensor]

        flow = Flow(
            jobs=jobs,
            output=fit_tensor.output,
            name=self.name,
        )
        return flow
