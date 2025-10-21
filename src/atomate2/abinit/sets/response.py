"""Module defining response function Abinit input set generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from abipy.abio.factories import (
    ddepert_from_gsinput,
    ddkpert_from_gsinput,
    dtepert_from_gsinput,
    phononpert_from_gsinput,
)
from abipy.abio.input_tags import DDE, DDK, DTE, NSCF, PH_Q_PERT, SCF

if TYPE_CHECKING:
    from collections.abc import Callable

    from pymatgen.io.abinit import PseudoTable

from atomate2.abinit.sets.core import NonSCFSetGenerator, StaticSetGenerator

__all__ = [
    "DdeSetGenerator",
    "DdkSetGenerator",
    "DteSetGenerator",
    "PhononSetGenerator",
]


@dataclass
class DdkSetGenerator(NonSCFSetGenerator):
    """Class to generate Abinit DDK input sets."""

    calc_type: str = "DDK"
    factory: Callable = ddkpert_from_gsinput
    restart_from_deps: tuple = (f"{DDK}:1WF",)
    prev_outputs_deps: tuple = (f"{SCF}:WFK",)
    nbands_factor: float = 1.0


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
    )  # TODO: find common solution if more jobs will need to set irdddk=1


@dataclass
class DteSetGenerator(StaticSetGenerator):
    """Class to generate Abinit DTE input sets."""

    calc_type: str = "DTE"
    factory: Callable = dtepert_from_gsinput
    pseudos: str | list[str] | PseudoTable | None = None
    restart_from_deps: tuple = (f"{DTE}:1WF|1DEN",)
    prev_outputs_deps: tuple = (
        f"{SCF}:WFK",
        f"{DDE}:1WF",
        f"{DDE}:1DEN",
        f"{PH_Q_PERT}:1WF",
        f"{PH_Q_PERT}:1DEN",
    )
    factory_prev_inputs_kwargs: dict | None = field(
        default_factory=lambda: {"gs_input": (SCF,)}
    )


@dataclass
class PhononSetGenerator(StaticSetGenerator):
    """Class to generate Abinit Phonon input sets."""

    calc_type: str = "Phonon"
    factory: Callable = phononpert_from_gsinput
    pseudos: str | list[str] | PseudoTable | None = None
    restart_from_deps: tuple = (f"{SCF}:WFK",)
    prev_outputs_deps: tuple = (f"{SCF}:WFK", f"{NSCF}:WFQ")
    factory_prev_inputs_kwargs: dict | None = field(
        default_factory=lambda: {"gs_input": (SCF,)}
    )


@dataclass
class NscfWfqSetGenerator(NonSCFSetGenerator):
    """Class to generate a Non-SCF input set with a k point grid shifted by q."""

    calc_type: str = "wfq"
    user_abinit_settings: dict = field(
        default_factory=lambda: {
            "kptopt": 3,
            "nqpt": 1,
        }
    )
