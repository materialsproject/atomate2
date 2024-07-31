"""Module defining response function Abinit input set generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from abipy.abio.factories import (
    ddepert_from_gsinput,
    ddkpert_from_gsinput,
    dtepert_from_gsinput,
)
from abipy.abio.input_tags import DDE, DDK, DTE, SCF

if TYPE_CHECKING:
    from pymatgen.io.abinit import PseudoTable

from atomate2.abinit.sets.core import NonSCFSetGenerator, StaticSetGenerator

__all__ = [
    "DdkSetGenerator",
    "DdeSetGenerator",
    "DteSetGenerator",
]


@dataclass
class DdkSetGenerator(NonSCFSetGenerator):
    """Class to generate Abinit DDK input sets."""

    calc_type: str = "DDK"
    factory: Callable = ddkpert_from_gsinput
    restart_from_deps: tuple = (f"{DDK}:1WF",)
    prev_outputs_deps: tuple = (f"{SCF}:WFK",)
    nbands_factor: float = 1.0  # TODO: to decide if nbdbuf or not


@dataclass
class DdeSetGenerator(StaticSetGenerator):
    """Class to generate Abinit DDE input sets."""

    calc_type: str = "DDE"
    factory: Callable = ddepert_from_gsinput
    pseudos: str | list[str] | PseudoTable | None = None
    restart_from_deps: tuple = (f"{DDE}:1WF|1DEN",)
    prev_outputs_deps: tuple = (f"{SCF}:WFK", f"{DDK}:1WF")
    factory_prev_inputs_kwargs: dict | None = field(
        default_factory=lambda: {"gs_input": (SCF,)}
    )
    user_abinit_settings: dict = field(
        default_factory=lambda: {"irdddk": 1, "ird1wf": 0}
    )


@dataclass
class DteSetGenerator(StaticSetGenerator):
    """Class to generate Abinit DTE input sets."""

    calc_type: str = "DTE"
    factory: Callable = dtepert_from_gsinput
    pseudos: str | list[str] | PseudoTable | None = None
    restart_from_deps: tuple = (f"{DTE}:1WF|1DEN",)
    prev_outputs_deps: tuple = (f"{SCF}:WFK", f"{DDE}:1WF", f"{DDE}:1DEN")
    factory_prev_inputs_kwargs: dict | None = field(
        default_factory=lambda: {"gs_input": (SCF,)}
    )
