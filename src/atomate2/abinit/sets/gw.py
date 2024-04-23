"""Module defining Abinit input set generators specific to GW calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from abipy.abio.factories import scr_from_nscfinput, sigma_from_inputs, scf_input, nscf_from_gsinput
from atomate2.abinit.sets.bse import bse_with_mdf_from_inputs 
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

@dataclass
class BSENonSCFSetGenerator(NonSCFSetGenerator):
    """Class to generate Abinit non-SCF input sets."""

    calc_type: str = "nscf_bse"
    factory: Callable = nscf_from_gsinput
    pseudos: str | list[str] | PseudoTable | None = None
    restart_from_deps: tuple = (f"{NSCF}:WFK",)
    prev_outputs_deps: tuple = (f"{SCF}:DEN",)
    nbands_factor: float = 1.2
    factory_kwargs: dict = field(default_factory=dict)

    factory_prev_inputs_kwargs: dict | None = field(
        default_factory=lambda: {"gs_input": (SCF,)}
    )

    def get_abinit_input(
        self,
        structure: Structure | None = None,
        pseudos: PseudoTable | None = None,
        prev_outputs: list[str] | None = None,
        abinit_settings: dict | None = None,
        factory_kwargs: dict | None = None,
        kpoints_settings: dict | KSampling | None = None,
        input_index: int | None = None,
        nscf_ngkpt: tuple | None = None,
        nscf_shiftk: tuple | None = None,
    ) -> AbinitInput:
        """Get AbinitInput object for Non-SCF calculation."""
        factory_kwargs = dict(factory_kwargs) if factory_kwargs else {}
        factory_kwargs["nband"] = self._get_nband(prev_outputs)
        kpoints_settings=KSampling.monkhorst(nscf_ngkpt, shiftk=nscf_shiftk, chksymbreak=0)
        return super().get_abinit_input(
            structure=structure,
            pseudos=pseudos,
            prev_outputs=prev_outputs,
            abinit_settings=abinit_settings,
            factory_kwargs=factory_kwargs,
            kpoints_settings=kpoints_settings,
        )

    def _get_nband(self, prev_outputs: list[str] | None) -> int:
        abinit_inputs = self.resolve_prev_inputs(
            prev_outputs, self.factory_prev_inputs_kwargs
        )
        if len(abinit_inputs) != 1:
            raise RuntimeError(
                f"Should have exactly one previous output. Found {len(abinit_inputs)}"
            )
        previous_abinit_input = next(iter(abinit_inputs.values()))
        n_band = previous_abinit_input.get(
            "nband",
            previous_abinit_input.structure.num_valence_electrons(
                previous_abinit_input.pseudos
            ),
        )

@dataclass
class BSEmdfSetGenerator(AbinitInputGenerator):
    """Class to generate Abinit non-SCF input sets."""

    calc_type: str = "bse_mdf"
    factory: Callable = bse_with_mdf_from_inputs 
    pseudos: str | list[str] | PseudoTable | None = None
    prev_outputs_deps: tuple = (f"{NSCF}:WFK",)
    factory_kwargs: dict = field(default_factory=dict)

    factory_prev_inputs_kwargs: dict | None = field(
        default_factory=lambda: {"nscf_input": (NSCF,), "sigma_input": (SIGMA,)}
    )
    def get_abinit_input(
        self,
        bs_loband: int | None = 1, 
        bs_nband: int | None = None, 
        mdf_epsinf: float | None = None, 
        mbpt_sciss: float | None = None, 
        prev_outputs: list[str] | None = None,
        abinit_settings: dict | None = None,
        factory_kwargs: dict | None = None,
    ) -> AbinitInput:

        factory_kwargs = dict(factory_kwargs) if factory_kwargs else {}
        factory_kwargs["bs_loband"] = bs_loband 
        factory_kwargs["bs_nband"] = bs_nband if bs_nband is not None else load_abinit_input(prev_outputs[0])["nband"]
        factory_kwargs["mdf_epsinf"] = mdf_epsinf 
        factory_kwargs["mbpt_sciss"] = mbpt_sciss 
        
        return super().get_abinit_input(
            prev_outputs=prev_outputs,
            abinit_settings=abinit_settings,
            factory_kwargs=factory_kwargs,
        )

