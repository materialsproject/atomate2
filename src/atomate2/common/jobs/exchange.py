"""Jobs for fitting Heisenberg exchange parameters and building exchange docs.

These replace the atomate1 ``HeisenbergModelMapping``, ``HeisenbergModelToDb`` and
``VampireToDb`` firetasks. Outputs flow between jobs through jobflow references
rather than a MongoDB collection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jobflow import job
from pymatgen.analysis.magnetism.heisenberg import HeisenbergMapper

from atomate2.common.schemas.exchange import ExchangeDocument

if TYPE_CHECKING:
    from pymatgen.analysis.magnetism.heisenberg import HeisenbergModel
    from pymatgen.core.structure import Structure
    from atomate2.vampire.schemas.vampire_output import VampireOutput

logger = logging.getLogger(__name__)


@job(name="heisenberg mapping")
def heisenberg_mapping(
    structures: list[Structure],
    energies: list[float],
    parent: Structure | None = None,
    heisenberg_settings: dict | None = None,
) -> HeisenbergModel:
    """Fit a classical Heisenberg Hamiltonian to magnetic structures and energies.

    This wraps pymatgen's ``HeisenbergMapper`` to extract exchange parameters
    ``J_ij`` and the average exchange ``<J>`` (``javg``).

    Parameters
    ----------
    structures : list[Structure]
        Magnetic structures, each carrying a "magmom" site property. These should be
        ordered with the ground state first (index 0).
    energies : list[float]
        Energies **per atom** (eV) corresponding to each structure. These are
        converted to total energies internally, as required by HeisenbergMapper.
    heisenberg_settings : dict or None
        Keyword arguments for HeisenbergMapper, e.g. ``{"cutoff": 3.0, "tol": 0.04}``.

    Returns
    -------
    HeisenbergModel
        The fitted, MSONable Heisenberg model.
    """
    heisenberg_settings = heisenberg_settings or {}
    total_energies = [e * len(s) for s, e in zip(structures, energies, strict=True)]
    hmapper = HeisenbergMapper(structures, total_energies, parent, **heisenberg_settings)
    return hmapper.get_heisenberg_model()


@job(name="build exchange doc")
def build_exchange_doc(
    heisenberg_model: HeisenbergModel,
    parent_structure: Structure | None = None,
    vampire_output: VampireOutput | None = None,
) -> ExchangeDocument:
    """Assemble the final ExchangeDocument from a fitted model and optional Tc run.

    Parameters
    ----------
    heisenberg_model : HeisenbergModel
        The fitted Heisenberg model from :func:`heisenberg_mapping`.
    parent_structure : Structure or None
        The full parent structure from which the magnetic structures were derived. 
        This is used to store the final fitted exchange parameters in the context of the original structure.
    vampire_output : VampireOutput or None
        The Vampire Monte-Carlo result, if the critical-temperature step was run.

    Returns
    -------
    ExchangeDocument
        The final summary document.
    """
    return ExchangeDocument.from_model(
        heisenberg_model,
        parent_structure=parent_structure,
        vampire_output=vampire_output,
    )
