"""Flows for calculating the elastic constant."""

from __future__ import annotations

import typing
from dataclasses import dataclass, field

from jobflow import Flow, Maker

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.elastic import (
    ElasticRelaxMaker,
    fit_elastic_tensor,
    generate_elastic_deformations,
    run_elastic_deformations,
)

if typing.TYPE_CHECKING:
    from pathlib import Path
    from typing import Union

    from pymatgen.core.structure import Structure

    from atomate2.common.schemas.math import Matrix3D


@dataclass
class ElasticMaker(Maker):
    """Maker to calculate elastic constants."""

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

        Parameters
        ----------
        structure
            A pymatgen structure.
        prev_vasp_dir
            A previous vasp calculation directory to use for copying outputs.
        equilibrium_stress
            The equilibrium stress of the (relaxed) structure, if known.
        """
        # make sure we don't overwrite settings in kwargs
        if "order" not in self.generate_elastic_deformations_kwargs:
            self.generate_elastic_deformations_kwargs["order"] = self.order

        if "sym_reduce" not in self.generate_elastic_deformations_kwargs:
            self.generate_elastic_deformations_kwargs["sym_reduce"] = self.sym_reduce

        if "order" not in self.fit_elastic_tensor_kwargs:
            self.fit_elastic_tensor_kwargs["order"] = self.order

        deformations = generate_elastic_deformations(
            structure, **self.generate_elastic_deformations_kwargs
        )
        vasp_deformation_calcs = run_elastic_deformations(
            structure,
            deformations.output["deformation"],
            symmetry_ops=deformations.output["symmetry_ops"],
            prev_vasp_dir=prev_vasp_dir,
            elastic_relax_maker=self.elastic_relax_maker,
        )
        fit_tensor = fit_elastic_tensor(
            structure,
            vasp_deformation_calcs.output,
            equilibrium_stress=equilibrium_stress,
            **self.fit_elastic_tensor_kwargs
        )

        flow = Flow(
            jobs=[deformations, vasp_deformation_calcs, fit_tensor],
            output=fit_tensor.output,
            name=self.name,
        )
        return flow
