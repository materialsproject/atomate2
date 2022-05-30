"""Module defining core Abinit input set generators."""

from dataclasses import dataclass, field
from typing import ClassVar, Optional

from abipy.abio.factories import ebands_from_gsinput, ion_ioncell_relax_input, scf_input
from abipy.abio.input_tags import MOLECULAR_DYNAMICS, NSCF, RELAX, SCF

from atomate2.abinit.files import load_abinit_input
from atomate2.abinit.sets.base import AbinitInputSetGenerator

__all__ = [
    "StaticSetGenerator",
    "NonSCFSetGenerator",
    "RelaxSetGenerator",
]


@dataclass
class StaticSetGenerator(AbinitInputSetGenerator):
    """Class to generate Abinit static input sets."""

    calc_type: str = "static"

    kppa: Optional[float] = None
    ecut: Optional[float] = None
    pawecutdg: Optional[float] = None
    nband: Optional[int] = None
    accuracy: str = "normal"
    spin_mode: str = "polarized"
    smearing: str = "fermi_dirac:0.1 eV"
    charge: float = 0.0
    scf_algorithm: Optional[str] = None
    shift_mode: str = "Monkhorst-Pack"

    restart_from_deps: tuple = (f"{SCF}|{RELAX}|{MOLECULAR_DYNAMICS}:WFK|DEN",)

    # class variables
    params: ClassVar[tuple] = (
        "kppa",
        "ecut",
        "pawecutdg",
        "nband",
        "accuracy",
        "spin_mode",
        "smearing",
        "charge",
        "scf_algorithm",
        "shift_mode",
    )

    def get_abinit_input(
        self, structure=None, pseudos=None, prev_outputs=None, **kwargs
    ):
        """Get AbinitInput object for static calculation."""
        if structure is None:
            raise RuntimeError("Structure is mandatory for StaticSet generation.")
        if prev_outputs is not None:
            raise RuntimeError(
                "Previous outputs not allowed for StaticSetGenerator. "
                "To restart from a previous static or otherwise scf "
                "(e.g. relaxation) calculation, use restart_from argument of "
                "get_input_set method instead."
            )

        return scf_input(
            structure=structure,
            pseudos=pseudos,
            **kwargs,
        )

    def on_restart(self, abinit_input):
        """Perform updates of AbinitInput upon restart.

        In this case, a static calculation can be started from a relaxation one.
        The relaxation-like variables need to be removed from the AbinitInput.
        """
        # Always remove relaxation-like variables so that if we make the SCF job
        # starting from a previous relaxation or molecular dynamics job, the
        # structure will indeed be static.
        abinit_input.pop_vars(["ionmov", "optcell", "ntime"])


@dataclass
class NonSCFSetGenerator(AbinitInputSetGenerator):
    """Class to generate Abinit non-SCF input sets."""

    calc_type: str = "nscf"

    nband: Optional[int] = None
    ndivsm: int = 15
    accuracy: str = "normal"

    restart_from_deps: tuple = (f"{NSCF}:WFK",)
    prev_outputs_deps: tuple = (f"{SCF}:DEN",)

    # class variables
    params: ClassVar[tuple] = (
        "nband",
        "ndivsm",
        "accuracy",
    )

    def get_abinit_input(
        self, structure=None, pseudos=None, prev_outputs=None, **kwargs
    ):
        """Get AbinitInput object for Non-SCF calculation."""
        if structure is not None:
            # TODO: maybe just check that the structure is the same as the one
            #  in the previous_input_set ?
            raise RuntimeError(
                "Structure should not be set in a non-SCF input set. "
                "It should come directly from the previous (SCF) input set."
            )
        if prev_outputs is None:
            raise RuntimeError(
                "No previous_outputs. Cannot perform non-SCF calculation."
            )
        if len(prev_outputs) != 1:
            raise RuntimeError(
                "Should have exactly one previous output (an SCF calculation)."
            )
        prev_output = prev_outputs[0]
        previous_abinit_input = load_abinit_input(prev_output)
        # if pseudos is not None:
        #     # TODO: maybe just check that the pseudos are the same as the one
        #      in the previous_input_set ?
        #     raise RuntimeError('Pseudos should not be set in a non-SCF input set. '
        #                        'It should come directly from the previous (SCF) '
        #                        'input set.')

        return ebands_from_gsinput(
            gsinput=previous_abinit_input,
            **kwargs,
        )


@dataclass
class NonScfWfqInputGenerator(AbinitInputSetGenerator):
    """Input set generator for Non-Scf Wfq calculations."""

    calc_type: str = "nscf_wfq"

    wfq_tol: dict = field(default_factory=lambda: {"tolwfr": 1.0e-18})

    restart_from_deps: tuple = (f"{NSCF}:WFQ",)
    prev_outputs_deps: tuple = (f"{SCF}:DEN",)

    def get_abinit_input(
        self, structure=None, pseudos=None, prev_outputs=None, qpt=None, **kwargs
    ):
        """Get AbinitInput object for Non-SCF Wfq calculation."""
        if qpt is None:
            raise RuntimeError(
                "Should provide q-point at which non-SCF Wfq calculation "
                "has to be done."
            )
        if structure is not None:
            # TODO: maybe just check that the structure is the same as the one
            #  in the previous_input_set ?
            raise RuntimeError(
                "Structure should not be set in a non-SCF Wfq input set. "
                "It should come directly from a previous (SCF) input set."
            )
        if prev_outputs is None:
            raise RuntimeError(
                "No previous_outputs. Cannot perform non-SCF Wfq calculation."
            )
        if len(prev_outputs) != 1:
            raise RuntimeError(
                "Should have exactly one previous output (an SCF calculation)."
            )
        prev_output = prev_outputs[0]
        wfq_input = load_abinit_input(prev_output)
        wfq_input.set_vars(kptopt=3, nqpt=1, iscf=-2, qpt=qpt, **self.wfq_tol)
        return wfq_input


@dataclass
class DdkInputGenerator(AbinitInputSetGenerator):
    """Input set generator for Non-Scf Wfq calculations."""

    calc_type: str = "ddk"

    def get_abinit_input(
        self, structure=None, pseudos=None, prev_outputs=None, **kwargs
    ):
        raise NotImplementedError()


@dataclass
class RelaxSetGenerator(StaticSetGenerator):
    """Class to generate Abinit relaxation input sets."""

    calc_type: str = "relaxation"

    relax_cell: bool = True
    tolmxf: float = 5.0e-5

    # class variables
    params: ClassVar[tuple] = (
        "kppa",
        "ecut",
        "pawecutdg",
        "nband",
        "accuracy",
        "spin_mode",
        "smearing",
        "charge",
        "scf_algorithm",
        "shift_mode",
        "relax_cell",
        "tolmxf",
    )

    def get_abinit_input(
        self, structure=None, pseudos=None, prev_outputs=None, **kwargs
    ):
        if structure is None:
            raise RuntimeError("Structure is mandatory for RelaxSet generation.")
        if prev_outputs is not None:
            raise RuntimeError(
                "Previous outputs not allowed for RelaxSetGenerator. "
                "To restart from a previous static or otherwise scf "
                "(e.g. relaxation) calculation, use restart_from argument of "
                "get_input_set method instead."
            )
        if kwargs.get("atoms_constraints", None) is not None:
            raise NotImplementedError("Atoms constraints not implemented.")

        try:
            tolmxf = kwargs.pop("tolmxf")
        except KeyError:
            tolmxf = self.tolmxf
        ind = 1 if self.relax_cell else 0
        relax_input = ion_ioncell_relax_input(structure, pseudos=pseudos, **kwargs)[ind]
        relax_input["tolmxf"] = tolmxf

        # try:
        #     atom_constraints = kwargs.pop("atoms_constraints")
        # except KeyError:
        #     atom_constraints = None
        #
        # try:
        #     relax_cell = kwargs.pop("relax_cell")
        # except KeyError:
        #     relax_cell = self.relax_cell
        #
        # if relax_cell:
        #     relax_method = aobj.RelaxationMethod.atoms_and_cell(
        #         atoms_constraints=atom_constraints
        #     )
        # else:
        #     relax_method = aobj.RelaxationMethod.atoms_only(
        #         atoms_constraints=atom_constraints
        #     )
        #
        # try:
        #     tolmxf = kwargs.pop("tolmxf")
        # except KeyError:
        #     tolmxf = self.tolmxf
        #
        # relax_method.abivars.update(tolmxf=tolmxf)
        #
        # relax_input = ion_ioncell_relax_input(structure, pseudos=pseudos, **kwargs)[0]
        # relax_input.set_vars(relax_method.to_abivars())

        return relax_input
