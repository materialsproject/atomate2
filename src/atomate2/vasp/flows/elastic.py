"""Flows for calculating the elastic constant."""

from __future__ import annotations

import typing
from dataclasses import dataclass, field

from jobflow import Flow, Maker

from atomate2.vasp.jobs.elastic import (
    FitElasticTensorMaker,
    GenerateElasticDeformationsMaker,
    RunElasticDeformationsMaker,
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
    generate_deformations_maker: GenerateElasticDeformationsMaker = field(
        default_factory=GenerateElasticDeformationsMaker
    )
    run_deformations_maker: RunElasticDeformationsMaker = field(
        default_factory=RunElasticDeformationsMaker
    )
    fit_tensor_maker: FitElasticTensorMaker = field(
        default_factory=FitElasticTensorMaker
    )

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
        deformations = self.generate_deformations_maker.make(structure)
        vasp_deformation_calcs = self.run_deformations_maker.make(
            structure,
            deformations.output["deformation"],
            symmetry_ops=deformations.output["symmetry_ops"],
            prev_vasp_dir=prev_vasp_dir,
        )
        fit_tensor = self.fit_tensor_maker.make(
            structure,
            vasp_deformation_calcs.output,
            equilibrium_stress=equilibrium_stress,
        )

        flow = Flow(
            jobs=[deformations, vasp_deformation_calcs, fit_tensor],
            output=fit_tensor.output,
            name=self.name,
        )
        return flow
