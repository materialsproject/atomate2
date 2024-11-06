"""Define schemas for NEB workflows."""

from __future__ import annotations

from typing import Any, Optional

from emmet.core.neb import NebMethod
from pydantic import BaseModel, Field, model_validator
from pymatgen.core import Molecule, Structure
from typing_extensions import Self

from emmet.core.neb import neb_barrier_spline_fit


class NebResult(BaseModel, extra="allow"):  # type: ignore[call-arg]
    """Container class to store high-level NEB calculation info."""

    images: list[Structure | Molecule] = Field(
        None, description="Relaxed structures/molecules along the reaction pathway."
    )

    energies: list[float] = Field(
        None, description="Energies corresponding the structures in `images`."
    )

    forward_barrier: float = Field(
        None,
        description=(
            "Forward barrier for this reaction, "
            "i.e., the transition state energy minus "
            "the reactant / initial configuration energy."
        ),
    )

    reverse_barrier: float = Field(
        None,
        description=(
            "Reverse barrier for this reaction, "
            "i.e., the transition state energy minus "
            "the product / final configuration energy."
        ),
    )

    ionic_steps: Optional[list] = Field(
        None,
        description=(
            "List of calculations along the reaction pathway, "
            "including ionic relaxation data."
        ),
    )

    method: Optional[NebMethod] = Field(
        None, description="Variety of NEB used in this calculation."
    )

    barrier_analysis: Optional[dict[str, Any]] = Field(
        None, description="Analysis of the reaction barrier."
    )

    @model_validator(mode="after")
    def set_barriers(self) -> Self:
        """Perform analysis on barrier if needed."""
        if not self.forward_barrier or not self.reverse_barrier:
            self.barrier_analysis = neb_barrier_spline_fit(self.energies)
            for k in ("forward", "reverse"):
                setattr(self, f"{k}_barrier", self.barrier_analysis[f"{k}_barrier"])
        return self


class NebPathwayResult(BaseModel, extra="allow"):  # type: ignore[call-arg]
    """Class for containing multiple NEB calculations, as along a reaction pathway."""

    hops: dict[str, NebResult] = Field(
        None, description="Dict of NEB calculations included in this calculation"
    )

    forward_barriers: dict[str, float] = Field(
        None, description="Dict of the forward barriers computed here."
    )

    reverse_barriers: dict[str, float] = Field(
        None, description="Dict of the reverse barriers computed here."
    )

    @model_validator(mode="after")
    def set_barriers(self) -> Self:
        """Set barriers if needed."""
        for direction in ("forward", "reverse"):
            if getattr(self, f"{direction}_barriers", None) is None:
                setattr(
                    self,
                    f"{direction}_barriers",
                    {
                        idx: getattr(neb_calc, f"{direction}_barrier", None)
                        for idx, neb_calc in self.hops.items()
                    },
                )
        return self

    @property
    def max_barriers(self) -> dict[str, float]:
        """Retrieve the maximum barrier along each hop."""
        return {
            idx: max(self.forward_barriers[idx], self.reverse_barriers[idx])
            for idx in self.forward_barriers
        }
