""" Define schemas for NEB workflows. """
from __future__ import annotations
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from atomate2.common.jobs.neb import neb_spline_fit

if TYPE_CHECKING:

    from typing_extensions import Any, Self
    
    from pymatgen.core import Structure

class NebResult(BaseModel):
    """ Container class to store high-level NEB calculation info. """

    structures : list[Structure] = Field(
        None, description = "Relaxed structures along the reaction pathway."
    )

    energies : list[float] = Field(
        None, description = "Energies corresponding the structures in `structures`."
    )

    forward_barrier : float = Field(
        None, description = "Forward barrier for this reaction, i.e., the transition state energy minus the reactant / initial configuration energy."
    )

    reverse_barrier : float = Field(
        None, description = "Reverse barrier for this reaction, i.e., the transition state energy minus the product / final configuration energy."
    )

    ionic_steps : Optional[list] = Field(
        None, description = "List of calculations along the reaction pathway, including ionic relaxation data."
    )

    method : Optional[NebMethod] = Field(
        None, description = "Variety of NEB used in this calculation."
    )

    barrier_analysis : Optional[dict[str,Any]] = Field(
        None, description = "Analysis of the reaction barrier."
    )

    @model_validator(mode="after")
    def set_barriers(self) -> Self:
        if (
            not self.forward_barrier
            or not self.reverse_barrier
        ):
            self.barrier_analysis = neb_spline_fit(self.energies)
            for k in ("forward","reverse",):
                setattr(self,f"{k}_barrier", self.barrier_analysis[f"{k}_barrier"])               
        return self

class NebPathwayResult(BaseModel):
    """ Class for containing multiple NEB calculations, as along a reaction pathway."""

    hops : list[NebResult] = Field(
        None, description = "List of NEB calculations included in this calculation"
    )