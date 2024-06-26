"""Module defining Abinit input set generators specific to GW calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from abipy.abio.factories import scr_from_nscfinput, sigma_from_inputs, scf_input, nscf_from_gsinput
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
    "ScreeningSetGenerator",
    "SigmaSetGenerator",
]


@dataclass
class ScreeningSetGenerator(AbinitInputGenerator):
    """Class to generate Abinit Screening input sets."""


    calc_type: str = "scr"
    factory: Callable = scr_from_nscfinput 
    pseudos: str | list[str] | PseudoTable | None = None
    prev_outputs_deps: tuple = (f"{NSCF}:WFK",)
    factory_kwargs: dict = field(default_factory=dict)
    factory_prev_inputs_kwargs: dict | None = field(
        default_factory=lambda: {"nscf_input": (NSCF,)}
    )


    def get_abinit_input(
        self,
        structure: Structure | None = None,
        pseudos: PseudoTable | None = None,
        prev_outputs: list[str] | None = None,
        factory_kwargs: dict | None = None,
        kpoints_settings: dict | KSampling | None = None,
    ) -> AbinitInput:


        nscf_inp = load_abinit_input(prev_outputs[0])

        if factory_kwargs:
            factory_kwargs.update({"ecutwfn": nscf_inp["ecut"]})
        else:
            factory_kwargs={"ecutwfn": nscf_inp["ecut"]}

        return super().get_abinit_input(
            structure=structure,
            pseudos=pseudos,
            prev_outputs=prev_outputs,
            factory_kwargs=factory_kwargs,
            kpoints_settings=kpoints_settings,
        )


@dataclass
class SigmaSetGenerator(AbinitInputGenerator):
    """Class to generate Abinit Sigma input sets."""

    calc_type: str = "sigma"
    factory: Callable = sigma_from_inputs 
    pseudos: str | list[str] | PseudoTable | None = None
    prev_outputs_deps: tuple = (f"{NSCF}:WFK", f"{SCREENING}:SCR")
    factory_kwargs: dict = field(default_factory=dict)
    factory_prev_inputs_kwargs: dict | None = field(
        default_factory=lambda: {"nscf_input": (NSCF,), "scr_input": (SCREENING,)}
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

        """Get AbinitInput object for SCR calculation."""
        if prev_outputs is None:
            raise RuntimeError("No previous_outputs. Cannot perform Sigma calculation.")
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
        # TODO: do we need to check that the structures are the same in nscf and
        #  screening ?

        #previous_structure = get_final_structure(prev_outputs[0])
        # TODO: the structure in the previous abinit input may be slightly different
        #  from the one in the previous output (if abinit symmetrizes the structure)
        #  Should we set the structure in the previous_abinit_input ? Or should we
        #  assume that abinit will make the same symmetrization ?
        #  Or should we always symmetrize the structure before ?
        #  Or should we always set tolsym to 1.0e-8 ?
        #nscf_inp.set_structure(previous_structure)
        #scr_inp.set_structure(previous_structure)
        if structure is not None:
            if structure != previous_structure:
                raise RuntimeError(
                    "Structure is provided in non-SCF input set generator but "
                    "is not the same as the one from the previous (SCF) input set."
                )

        # Sigma.
        if factory_kwargs:
            factory_kwargs.update({"ecutsigx": scr_inp["ecut"]})
        else:
            factory_kwargs={"ecutsigx": scr_inp["ecut"]}
        return super().get_abinit_input(
            structure=structure,
            pseudos=pseudos,
            prev_outputs=prev_outputs,
            factory_kwargs=factory_kwargs,
            kpoints_settings=kpoints_settings,
        )

