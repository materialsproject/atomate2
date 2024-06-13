"""Module defining core Abinit input set generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
from abipy.abio.factories import (
    dos_from_gsinput,
    ebands_from_gsinput,
    ion_ioncell_relax_input,
    nscf_from_gsinput,
    scf_input,
)
from abipy.abio.input_tags import MOLECULAR_DYNAMICS, NSCF, RELAX, SCF

from atomate2.abinit.sets.base import AbinitInputGenerator

if TYPE_CHECKING:
    from abipy.abio.inputs import AbinitInput
    from pymatgen.core import Structure
    from pymatgen.io.abinit import PseudoTable
    from pymatgen.io.abinit.abiobjects import KSampling


GS_RESTART_FROM_DEPS = (f"{SCF}|{RELAX}|{MOLECULAR_DYNAMICS}:WFK|DEN",)


@dataclass
class StaticSetGenerator(AbinitInputGenerator):
    """Common class for ground-state generators."""

    calc_type: str = "static"
    factory: Callable = scf_input
    restart_from_deps: tuple = GS_RESTART_FROM_DEPS

    def get_abinit_input(
        self,
        structure: Structure | None = None,
        pseudos: PseudoTable | None = None,
        prev_outputs: list[str] | None = None,
        abinit_settings: dict | None = None,
        factory_kwargs: dict | None = None,
        kpoints_settings: dict | KSampling | None = None,
        input_index: int | None = None,
    ) -> AbinitInput:
        """Generate the AbinitInput for the input set.

        Removes some standard variables related to relaxation.
        """
        # disable relax options in case they are present (from a restart)
        scf_abinit_settings = {
            "ionmov": None,
            "optcell": None,
            "ntime": None,
        }
        if abinit_settings:
            scf_abinit_settings.update(abinit_settings)

        return super().get_abinit_input(
            structure=structure,
            pseudos=pseudos,
            prev_outputs=prev_outputs,
            abinit_settings=scf_abinit_settings,
            factory_kwargs=factory_kwargs,
            kpoints_settings=kpoints_settings,
        )


@dataclass
class NonSCFSetGenerator(AbinitInputGenerator):
    """Class to generate Abinit non-SCF input sets."""

    calc_type: str = "nscf"
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
    ) -> AbinitInput:
        """Get AbinitInput object for Non-SCF calculation."""
        factory_kwargs = dict(factory_kwargs) if factory_kwargs else {}
        factory_kwargs["nband"] = self._get_nband(prev_outputs)

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
        return int(np.ceil(n_band * self.nbands_factor))


@dataclass
class LineNonSCFSetGenerator(NonSCFSetGenerator):
    """Class to generate Abinit non-SCF input sets."""

    calc_type: str = "nscf_line"
    factory: Callable = ebands_from_gsinput


@dataclass
class UniformNonSCFSetGenerator(NonSCFSetGenerator):
    """Class to generate Abinit non-SCF input sets."""

    calc_type: str = "nscf_uniform"
    factory: Callable = dos_from_gsinput


@dataclass
class NonScfWfqInputGenerator(AbinitInputGenerator):
    """Input set generator for Non-Scf Wfq calculations."""

    calc_type: str = "nscf_wfq"

    wfq_tol: dict = field(default_factory=lambda: {"tolwfr": 1e-18})

    restart_from_deps: tuple = (f"{NSCF}:WFQ",)
    prev_outputs_deps: tuple = (f"{SCF}:DEN",)

    def get_abinit_input(
        self,
        structure: Structure | None = None,
        pseudos: PseudoTable | None = None,
        prev_outputs: list[str] | None = None,
        abinit_settings: dict | None = None,
        factory_kwargs: dict | None = None,
        kpoints_settings: dict | KSampling | None = None,
        input_index: int | None = None,
    ) -> AbinitInput:
        """Get AbinitInput object for Non-SCF Wfq calculation."""
        raise NotImplementedError


@dataclass
class DdkInputGenerator(AbinitInputGenerator):
    """Input set generator for Non-Scf Wfq calculations."""

    calc_type: str = "ddk"

    def get_abinit_input(
        self,
        structure: Structure | None = None,
        pseudos: PseudoTable | None = None,
        prev_outputs: list[str] | None = None,
        abinit_settings: dict | None = None,
        factory_kwargs: dict | None = None,
        kpoints_settings: dict | KSampling | None = None,
        input_index: int | None = None,
    ) -> AbinitInput:
        """Get the abinit input for Ddk calculation."""
        raise NotImplementedError


@dataclass
class RelaxSetGenerator(AbinitInputGenerator):
    """Common class for ground-state generators."""

    calc_type: str = "relaxation"
    factory: Callable = ion_ioncell_relax_input
    restart_from_deps: tuple = GS_RESTART_FROM_DEPS
    relax_cell: bool = True
    tolmxf: float = 5e-5

    def get_abinit_input(
        self,
        structure: Structure | None = None,
        pseudos: PseudoTable | None = None,
        prev_outputs: list[str] | None = None,
        abinit_settings: dict | None = None,
        factory_kwargs: dict | None = None,
        kpoints_settings: dict | KSampling | None = None,
        input_index: int | None = None,
    ) -> AbinitInput:
        """Generate the AbinitInput for the input set.

        Sets tolmxf and determines the index of the MultiDataset.
        """
        abinit_settings = abinit_settings or {}
        # TODO move tolmxf to the factory?
        abinit_settings["tolmxf"] = self.tolmxf
        if input_index is None:
            input_index = 1 if self.relax_cell else 0

        return super().get_abinit_input(
            structure=structure,
            pseudos=pseudos,
            prev_outputs=prev_outputs,
            abinit_settings=abinit_settings,
            factory_kwargs=factory_kwargs,
            kpoints_settings=kpoints_settings,
            input_index=input_index,
        )
