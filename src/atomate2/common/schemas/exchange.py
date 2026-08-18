"""Schemas for magnetic exchange (Heisenberg + Vampire) calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure

if TYPE_CHECKING:
    from pymatgen.analysis.magnetism.heisenberg import HeisenbergModel

    from atomate2.vampire.schemas.vampire_output import VampireOutput


class ExchangeDocument(BaseModel):
    """Final document with fitted Heisenberg exchange parameters and Tc.

    This is the output of the ExchangeMaker workflow. The Heisenberg fields are
    always populated; the Vampire fields are populated only if the Monte-Carlo
    critical-temperature step was run.
    """

    formula: str | None = Field(
        None,
        description="Formula taken from pymatgen.core.structure.Structure.formula.",
    )
    formula_pretty: str | None = Field(
        None,
        description="Cleaned representation of the formula.",
    )
    parent_structure: Structure | None = Field(
        None,
        description="The ground-state (lowest-energy) structure used for the fit.",
    )
    heisenberg_settings: dict | None = Field(
        None,
        description="The {cutoff, tol} settings used by the HeisenbergMapper.",
    )
    vampire_settings: dict | None = Field(
        None,
        description="Keyword arguments passed to the Vampire Monte-Carlo run (e.g. "
        "mc_box_size, equil_timesteps, mc_timesteps), if run. Unset keys fall "
        "back to the VampireCaller defaults.",
    )
    residual: float | None = Field(
        None,
        description="Root-mean-square residual of the least-squares fit that "
        "produced ex_params, in meV per magnetic ion. Intensive in both the cell "
        "size and the number of orderings, so it is comparable between materials.",
    )
    ex_params: dict | None = Field(
        None, description="Fitted exchange parameters keyed by interaction label. The J_ij are in meV/muB^2 (they multiply the raw moments); the included 'E0' offset is in eV per magnetic ion."
    )
    ex_mat: dict | None = Field(
        None, description="Heisenberg Hamiltonian matrix used for the Heisenberg model fit."
    )
    heisenberg_model: dict | None = Field(
        None, description="Full HeisenbergModel as a serialized dict (as_dict())."
    )
    critical_temp: float | None = Field(
        None,
        description="Critical (Curie/Neel) temperature in Kelvin from Vampire, if run.",
    )
    vampire_output: dict | None = Field(
        None, description="Full VampireOutput as a serialized dict (as_dict()), if run."
    )

    @classmethod
    def from_model(
        cls,
        heisenberg_model: HeisenbergModel,
        parent_structure: Structure | None = None,
        vampire_output: VampireOutput | None = None,
        vampire_settings: dict | None = None,
    ) -> ExchangeDocument:
        """Construct an ExchangeDocument from a fitted model and optional Vampire run.

        Parameters
        ----------
        heisenberg_model : HeisenbergModel
            The fitted Heisenberg model from pymatgen's HeisenbergMapper.
        parent_structure : Structure or None
            The full ground-state structure (used for the parent_structure/formula
            fields). If None, falls back to ``heisenberg_model.structures[0]``, the
            full cell of the ground-state ordering (all ions retained).
        vampire_output : VampireOutput or None
            The Vampire Monte-Carlo result, if the critical-temperature step was run.
        vampire_settings : dict or None
            The keyword arguments the Vampire Monte-Carlo run was called with, if run.
        """
        if parent_structure is None:
            parent_structure = heisenberg_model.structures[0]

        return cls(
            formula=parent_structure.formula,
            formula_pretty=parent_structure.composition.reduced_formula,
            parent_structure=parent_structure,
            heisenberg_settings={
                "cutoff": heisenberg_model.cutoff,
                "tol": heisenberg_model.tol,
            },
            vampire_settings=vampire_settings if vampire_output else None,
            residual=heisenberg_model.residual,
            ex_params=heisenberg_model.ex_params,
            ex_mat=heisenberg_model.ex_mat.to_dict(),
            heisenberg_model=heisenberg_model.as_dict(),
            critical_temp=vampire_output.critical_temp if vampire_output else None,
            vampire_output=vampire_output.as_dict() if vampire_output else None,
        )
