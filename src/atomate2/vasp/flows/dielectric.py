"""Flows for calculating dielectric properties."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.common.flows.dielectric import BaseDielectricMaker
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import DielectricMaker, TightRelaxMaker

if TYPE_CHECKING:
    from jobflow import OutputReference

    from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class RelaxDielectricMaker(BaseDielectricMaker):
    """
    Maker to create a flow with a relaxation and a static dielectric tensor calculation.

    The dielectric calculation uses density functional perturbation theory to obtain
    the electronic and ionic contributions to the dielectric and piezoelectric
    tensors, as well as the Born effective charges. As these are sensitive to residual
    forces, by default the structure is first relaxed using a :obj:`.DoubleRelaxMaker`
    with tight convergence parameters. The results are collected in a
    :obj:`.DielectricDocument`.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker or None
        The maker to use for the relaxation. Set to ``None`` to skip the relaxation.
    dielectric_maker : .BaseVaspMaker
        The maker to use for the dielectric calculation.
    """

    name: str = "relax and dielectric"
    relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )
    dielectric_maker: BaseVaspMaker = field(default_factory=DielectricMaker)

    @staticmethod
    def get_dielectric_tensors(output: OutputReference) -> dict[str, OutputReference]:
        """Get the tensors from the task document of the dielectric job."""
        properties = output.calcs_reversed[0].output.dielectric_properties
        return {
            "epsilon_electronic": properties.dielectric_tensor,
            "epsilon_ionic": properties.dielectric_ionic_tensor,
            "piezo_electronic": properties.piezo_tensor,
            "piezo_ionic": properties.piezo_ionic_tensor,
            "born": properties.born_charges,
        }
