"""Flows for calculating the elastic constant."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from jobflow import Flow, Maker, OnMissing
from pymatgen.core.structure import Structure

from atomate2.common.schemas.math import Matrix3D
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.elastic import (
    ElasticRelaxMaker,
    fit_elastic_tensor,
    generate_elastic_deformations,
    run_elastic_deformations,
)

__all__ = ["ElasticMaker"]


@dataclass
class ElasticMaker(Maker):
    """
    Maker to calculate elastic constants.

    Parameters
    ----------
    name
        Name of the flows produced by this maker.
    order
        Order of the tensor expansion to be determined. Can be either 2 or 3.
    sym_reduce
        Whether to reduce the number of deformations using symmetry.
    elastic_relax_maker
        Maker used to generate elastic relaxations.
    generate_elastic_deformations_kwargs
        Keyword arguments passed to :obj:`generate_elastic_deformations`.
    fit_elastic_tensor_kwargs
        Keyword arguments passed to :obj:`fit_elastic_tensor`.
    """

    name = "elastic"
    order: int = 2
    sym_reduce: bool = True
    elastic_relax_maker: BaseVaspMaker = field(default_factory=ElasticRelaxMaker)
    generate_elastic_deformations_kwargs: dict = field(default_factory=dict)
    fit_elastic_tensor_kwargs: dict = field(default_factory=dict)

    def make(
        self,
        structure: Structure,
        prev_vasp_dir: Union[str, Path] = None,
        equilibrium_stress: Matrix3D = None,
    ):
        """
        Make flow to calculate the elastic constant.

        .. Note::
            It is heavily recommended to symmetrize the structure before passing it to
            this flow. Otherwise, the symmetry reduction routines will not be as
            effective at reducing the total number of deformations needed.

        Parameters
        ----------
        structure
            A pymatgen structure.
        prev_vasp_dir
            A previous vasp calculation directory to use for copying outputs.
        equilibrium_stress
            The equilibrium stress of the (relaxed) structure, if known.
        """
        deformations = generate_elastic_deformations(
            structure,
            order=self.order,
            sym_reduce=self.sym_reduce,
            **self.generate_elastic_deformations_kwargs
        )
        vasp_deformation_calcs = run_elastic_deformations(
            structure,
            deformations.output["deformations"],
            symmetry_ops=deformations.output["symmetry_ops"],
            prev_vasp_dir=prev_vasp_dir,
            elastic_relax_maker=self.elastic_relax_maker,
        )
        fit_tensor = fit_elastic_tensor(
            structure,
            vasp_deformation_calcs.output,
            equilibrium_stress=equilibrium_stress,
            order=self.order,
            **self.fit_elastic_tensor_kwargs
        )

        # allow some of the deformations to fail
        fit_tensor.config.on_missing_references = OnMissing.NONE

        flow = Flow(
            jobs=[deformations, vasp_deformation_calcs, fit_tensor],
            output=fit_tensor.output,
            name=self.name,
        )
        return flow
