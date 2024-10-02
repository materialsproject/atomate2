"""Module defining Abinit input set generators specific to GW calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from atomate2.abinit.sets.factories import bse_with_mdf_from_inputs 
from abipy.abio.input_tags import SCF, NSCF, SCREENING, SIGMA

from atomate2.abinit.files import load_abinit_input
from atomate2.abinit.sets.base import AbinitInputGenerator
from atomate2.abinit.sets.core import NonSCFSetGenerator 
from pymatgen.io.abinit.abiobjects import KSampling

if TYPE_CHECKING:
    from abipy.abio.inputs import AbinitInput
    from pymatgen.core import Structure
    from pymatgen.io.abinit import PseudoTable

__all__ = [
    "BSEmdfSetGenerator",
    "BSEscrSetGenerator",
]


@dataclass
class BSEmdfSetGenerator(AbinitInputGenerator):
    """Class to generate Abinit BSE with model dielectric function input sets."""

    calc_type: str = "bse_mdf"
    factory: Callable = bse_with_mdf_from_inputs 
    pseudos: str | list[str] | PseudoTable | None = None
    prev_outputs_deps: tuple = (f"{NSCF}:WFK",)
    factory_kwargs: dict = field(default_factory=dict)
    factory_prev_inputs_kwargs: dict | None = field(
        default_factory=lambda: {"nscf_input": (NSCF,),}
    )
    def get_abinit_input(
        self,
        structure: Structure | None = None,
        pseudos: PseudoTable | None = None,
        prev_outputs: list[str] | None = None,
        abinit_settings: dict | None = None,
        factory_kwargs: dict | None = None,
        kpoints_settings: dict | KSampling | None = None,
    ) -> AbinitInput:

        """Get AbinitInput object for BSE calculation."""
        if prev_outputs is None:
            raise RuntimeError("No previous_outputs. Cannot perform BSE calculation.")
        if len(prev_outputs) != 1:
            raise RuntimeError(
                "Should have exactly one previous outputs for mdf-BSE (one NSCF calculation)."
            )
        ab1 = load_abinit_input(prev_outputs[0])
        if NSCF not in ab1.runlevel:
            raise RuntimeError("Could not find one NSCF calculation.")

        return super().get_abinit_input(
            structure=structure,
            pseudos=pseudos,
            prev_outputs=prev_outputs,
            factory_kwargs=factory_kwargs,
            kpoints_settings=kpoints_settings,
        )

@dataclass
class BSEscrSetGenerator(AbinitInputGenerator):
    """Class to generate Abinit BSE with full dielectric function input sets."""

    calc_type: str = "bse_full"
    factory: Callable = bse_with_mdf_from_inputs 
    pseudos: str | list[str] | PseudoTable | None = None
    prev_outputs_deps: tuple = (f"{NSCF}:WFK", f"{SCREENING}:SCR",)
    factory_kwargs: dict = field(default_factory=dict)
    factory_prev_inputs_kwargs: dict | None = field(
        default_factory=lambda: {"nscf_input": (NSCF,), "scr_input": (SCREENING,),}
    )
    def get_abinit_input(
        self,
        structure: Structure | None = None,
        pseudos: PseudoTable | None = None,
        prev_outputs: list[str] | None = None,
        abinit_settings: dict | None = None,
        factory_kwargs: dict | None = None,
        kpoints_settings: dict | KSampling | None = None,
    ) -> AbinitInput:

        """Get AbinitInput object for BSE calculation."""
        if prev_outputs is None:
            raise RuntimeError("No previous_outputs. Cannot perform BSE calculation.")
        if len(prev_outputs) != 2:
            raise RuntimeError(
                "Should have exactly two previous outputs (one NSCF calculation "
                "and one SCREENING calculation)."
            )
        ab1 = load_abinit_input(prev_outputs[0])
        ab2 = load_abinit_input(prev_outputs[1])
        if NSCF in ab1.runlevel and SCREENING in ab2.runlevel:
            nscf_inp = ab1
            scr_inp = ab2
        elif SCREENING in ab1.runlevel and NSCF in ab2.runlevel:
            nscf_inp = ab2
            scr_inp = ab1
        else:
            raise RuntimeError("Could not find one NSCF and one SCREENING calculation.")

        if nscf_inp.vars["ngkpt"]!=scr_inp.vars["ngkpt"]:
            raise RuntimeError("Screening calculation k-grid is not compatible")

        return super().get_abinit_input(
            structure=structure,
            pseudos=pseudos,
            prev_outputs=prev_outputs,
            factory_kwargs=factory_kwargs,
            kpoints_settings=kpoints_settings,
        )
