"""Flows for calculating dielectric properties with force fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.common.flows.dielectric import BaseDielectricMaker
from atomate2.forcefields.jobs import ForceFieldDielectricMaker, ForceFieldRelaxMaker

if TYPE_CHECKING:
    from jobflow import OutputReference


@dataclass
class RelaxDielectricMaker(BaseDielectricMaker):
    """
    Maker to create a flow with a force field relaxation and dielectric calculation.

    By default, the structure is relaxed with MACE-OMAT and the dielectric properties
    are calculated with MACE-Field. MACE-Field only gives the electronic dielectric
    tensor and the Born effective charges, so the ionic contribution and the
    piezoelectric tensors are not calculated. It needs the path of a model file, see
    :obj:`.ForceFieldDielectricMaker`.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .ForceFieldRelaxMaker or None
        The maker to use for the relaxation. Set to ``None`` to skip the relaxation.
    dielectric_maker : .ForceFieldDielectricMaker
        The maker to use for the dielectric calculation.
    """

    name: str = "force field relax and dielectric"
    relax_maker: ForceFieldRelaxMaker | None = field(
        default_factory=lambda: ForceFieldRelaxMaker(
            force_field_name="MACE_MP_0",
            calculator_kwargs={"model": "medium-omat-0"},
            relax_kwargs={"fmax": 1e-5},
        )
    )
    dielectric_maker: ForceFieldDielectricMaker = field(
        default_factory=ForceFieldDielectricMaker
    )

    @staticmethod
    def get_dielectric_tensors(output: OutputReference) -> dict[str, OutputReference]:
        """Get the tensors from the output of the force field dielectric job."""
        return {"epsilon_electronic": output.epsilon_static, "born": output.born}
