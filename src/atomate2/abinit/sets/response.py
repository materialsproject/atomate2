"""Module defining response function ABINIT input set generators."""

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
    """
    Generator for ABINIT DDK (derivative of wavefunctions with respect to k) input sets.

    This class generates input sets for calculating the derivative of wavefunctions
    with respect to wavevector k, which is needed for electric field perturbations
    in DFPT calculations.

    Attributes
    ----------
    calc_type : str
        Type of calculation. Default is "DDK".
    factory : Callable
        Factory function for generating the input. Default is ddkpert_from_gsinput.
    restart_from_deps : tuple
        Dependencies for restarting calculations. Default is (f"{DDK}:1WF",).
    prev_outputs_deps : tuple
        Dependencies from previous calculations. Default is (f"{SCF}:WFK",).
    nbands_factor : float
        Factor to multiply the number of bands. Default is 1.0.
    """

    calc_type: str = "DDK"
    factory: Callable = ddkpert_from_gsinput
    restart_from_deps: tuple = (f"{DDK}:1WF",)
    prev_outputs_deps: tuple = (f"{SCF}:WFK",)
    nbands_factor: float = 1.0


@dataclass
class DdeSetGenerator(StaticSetGenerator):
    """
    Generator for ABINIT DDE (electric field perturbation) input sets.

    This class generates input sets for calculating the electronic dielectric
    tensor and Born effective charges using density functional perturbation
    theory (DFPT) with electric field perturbations.

    Attributes
    ----------
    calc_type : str
        Type of calculation. Default is "DDE".
    factory : Callable
        Factory function for generating the input. Default is ddepert_from_gsinput.
    pseudos : str or list[str] or PseudoTable or None
        Pseudopotentials specification. Default is None.
    restart_from_deps : tuple
        Dependencies for restarting calculations. Default is (f"{DDE}:1WF|1DEN",).
    prev_outputs_deps : tuple
        Dependencies from previous calculations (SCF and DDK).
        Default is (f"{SCF}:WFK", f"{DDK}:DDK").
    factory_prev_inputs_kwargs : dict or None
        Mapping of factory arguments to previous calculation types.
        Default is {"gs_input": (SCF,)}.
    """

    calc_type: str = "DDE"
    factory: Callable = ddepert_from_gsinput
    # Pseudos set to None to automatically recover them from the previous
    # SCF calculation
    pseudos: str | list[str] | PseudoTable | None = None
    restart_from_deps: tuple = (f"{DDE}:1WF|1DEN",)
    prev_outputs_deps: tuple = (
        f"{SCF}:WFK",
        f"{DDK}:DDK",
    )
    factory_prev_inputs_kwargs: dict | None = field(
        default_factory=lambda: {"gs_input": (SCF,)}
    )


@dataclass
class DteSetGenerator(StaticSetGenerator):
    """
    Generator for ABINIT DTE (mixed electric field-atomic displacement) input sets.

    This class generates input sets for calculating second-order derivatives
    with respect to both electric field and atomic displacements using DFPT.

    Attributes
    ----------
    calc_type : str
        Type of calculation. Default is "DTE".
    factory : Callable
        Factory function for generating the input. Default is dtepert_from_gsinput.
    pseudos : str or list[str] or PseudoTable or None
        Pseudopotentials specification. Default is None.
    restart_from_deps : tuple
        Dependencies for restarting calculations. Default is (f"{DTE}:1WF|1DEN",).
    prev_outputs_deps : tuple
        Dependencies from previous calculations (SCF, DDE, and phonon perturbations).
        Default is (f"{SCF}:WFK", f"{DDE}:1WF", f"{DDE}:1DEN",
        f"{PH_Q_PERT}:1WF", f"{PH_Q_PERT}:1DEN").
    factory_prev_inputs_kwargs : dict or None
        Mapping of factory arguments to previous calculation types.
        Default is {"gs_input": (SCF,)}.
    """

    calc_type: str = "DTE"
    factory: Callable = dtepert_from_gsinput
    # Pseudos set to None to automatically recover them from the previous
    # SCF calculation
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
    """
    Generator for ABINIT phonon perturbation input sets.

    This class generates input sets for calculating phonon properties using
    density functional perturbation theory (DFPT) with atomic displacement
    perturbations.

    Attributes
    ----------
    calc_type : str
        Type of calculation. Default is "Phonon".
    factory : Callable
        Factory function for generating the input. Default is phononpert_from_gsinput.
    pseudos : str or list[str] or PseudoTable or None
        Pseudopotentials specification. Default is None.
    restart_from_deps : tuple
        Dependencies for restarting calculations. Default is (f"{SCF}:WFK",).
    prev_outputs_deps : tuple
        Dependencies from previous calculations (SCF and NSCF with q-shifted grid).
        Default is (f"{SCF}:WFK", f"{NSCF}:WFQ").
    factory_prev_inputs_kwargs : dict or None
        Mapping of factory arguments to previous calculation types.
        Default is {"gs_input": (SCF,)}.
    """

    calc_type: str = "Phonon"
    factory: Callable = phononpert_from_gsinput
    # Pseudos set to None to automatically recover them from the previous
    # SCF calculation
    pseudos: str | list[str] | PseudoTable | None = None
    restart_from_deps: tuple = (f"{SCF}:WFK",)
    prev_outputs_deps: tuple = (f"{SCF}:WFK", f"{NSCF}:WFQ")
    factory_prev_inputs_kwargs: dict | None = field(
        default_factory=lambda: {"gs_input": (SCF,)}
    )
