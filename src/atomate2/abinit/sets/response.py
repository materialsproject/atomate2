"""Module defining response function Abinit input set generators."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from abipy.abio.factories import (
    ddepert_from_gsinput,
    ddkpert_from_gsinput,
    dtepert_from_gsinput,
)
from abipy.abio.input_tags import DDE, DDK, DTE, MOLECULAR_DYNAMICS, RELAX, SCF

from atomate2.abinit.sets.core import (
    NonSCFSetGenerator,
    StaticSetGenerator,
)

__all__ = [
    "DdkSetGenerator",
    "DdeSetGenerator",
    "DteSetGenerator",
]


GS_RESTART_FROM_DEPS: tuple = (f"{SCF}|{RELAX}|{MOLECULAR_DYNAMICS}:WFK|DEN",)


@dataclass
class DdkSetGenerator(NonSCFSetGenerator):
    """Class to generate Abinit DDK input sets."""

    calc_type: str = "DDK"
    factory: Callable = ddkpert_from_gsinput
    restart_from_deps: tuple = (f"{DDK}:DDK",)
    prev_outputs_deps: tuple = (f"{SCF}:WFK",)
    nbands_factor: float = 1.0  # TODO: to decide if nbdbuf or not


@dataclass
class DdeSetGenerator(StaticSetGenerator):
    """Class to generate Abinit DDE input sets."""

    calc_type: str = "DDE"
    factory: Callable = ddepert_from_gsinput
    restart_from_deps: tuple = (f"{DDE}:DDE",)
    prev_outputs_deps: tuple = (f"{SCF}:WFK", f"{DDK}:DDK")
    factory_prev_inputs_kwargs: dict | None = field(
        default_factory=lambda: {"gs_input": (SCF,)}
    )


@dataclass
class DteSetGenerator(StaticSetGenerator):
    """Class to generate Abinit DTE input sets."""

    calc_type: str = "DTE"
    factory: Callable = dtepert_from_gsinput
    restart_from_deps: tuple = (f"{DTE}:DTE",)
    prev_outputs_deps: tuple = (f"{SCF}:WFK", f"{DDE}:1WF|1DEN")
    factory_prev_inputs_kwargs: dict | None = field(
        default_factory=lambda: {"gs_input": (SCF,)}
    )
