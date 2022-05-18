"""Module defining core Abinit input set generators."""

from dataclasses import dataclass
from typing import ClassVar, Optional

import pymatgen.io.abinit.abiobjects as aobj
from abipy.abio.factories import ebands_from_gsinput, ion_ioncell_relax_input, scf_input
from abipy.abio.input_tags import MOLECULAR_DYNAMICS, NSCF, RELAX, SCF

from atomate2.abinit.files import load_abinit_input
from atomate2.abinit.sets.base import UNSET, AbinitInputSetGenerator

__all__ = [
    "StaticSetGenerator",
    "NonSCFSetGenerator",
]


_DEFAULT_SCF_PARAMS = dict(
    kppa=None,
    ecut=None,
    pawecutdg=None,
    nband=None,
    accuracy="normal",
    spin_mode="polarized",
    smearing="fermi_dirac:0.1 eV",
    charge=0.0,
    scf_algorithm=None,
    shift_mode="Monkhorst-Pack",
)


@dataclass
class ScfSetMixin:

    kppa: float = UNSET
    ecut: float = UNSET
    pawecutdg: float = UNSET
    nband: int = UNSET
    accuracy: str = UNSET
    spin_mode: str = UNSET
    smearing: str = UNSET
    charge: float = UNSET
    scf_algorithm: str = UNSET
    shift_mode: str = UNSET

    # class variables
    DEFAULT_PARAMS: ClassVar[dict] = _DEFAULT_SCF_PARAMS
    ALLOW_RESTART_FROM: ClassVar[set] = {SCF, RELAX, MOLECULAR_DYNAMICS}

    def update_abinit_input(self, abinit_input, param, value):
        """Update AbinitInput for the specific parameter and its value."""
        if param == "ecut":
            # Set the cutoff energies.
            # TODO: make a check on pawecutdg ?
            # TODO: how to take accuracy into account ?
            if value is None:
                raise NotImplementedError("")
            # abinit_input.set_vars(_find_ecut_pawecutdg(
            #     value, None, abinit_input.pseudos, accuracy
            # ))
            abinit_input.set_vars({"ecut": value})
        else:
            raise RuntimeError(
                f'Cannot apply "{param}" input set generator update to '
                f"previous AbinitInput."
            )


@dataclass
class StaticSetGenerator(ScfSetMixin, AbinitInputSetGenerator):
    """Class to generate Abinit static input sets."""

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

    nband: Optional[int] = UNSET
    ndivsm: int = UNSET
    accuracy: str = UNSET

    prev_output_exts: tuple = tuple(["DEN"])

    # class variables
    DEFAULT_PARAMS: ClassVar[dict] = {
        "nband": None,
        "ndivsm": 15,
        "accuracy": "normal",
    }
    ALLOW_RESTART_FROM: ClassVar[set] = {NSCF}

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
        previous_abinit_input = load_abinit_input(prev_output.dirname)
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

    #
    #     previous_input_set: AbinitInputSet = None
    #     nband: Optional[int] = None
    #     ndivsm: int = 15
    #     accuracy: str = "normal"
    #
    #     # non-dataclass variables
    #     # DEFAULT_PARAMS: tuple = field(
    #     #     default=(
    #     #         'previous_input_set',
    #     #     ),
    #     #     init=False, repr=False, compare=False
    #     # )
    #
    def get_abinit_input(
        self, structure=None, pseudos=None, prev_outputs=None, **kwargs
    ):
        raise NotImplementedError()


@dataclass
class RelaxSetGenerator(ScfSetMixin, AbinitInputSetGenerator):
    """Class to generate Abinit relaxation input sets."""

    relax_cell: bool = True
    tolmxf: float = UNSET

    # class variables
    DEFAULT_PARAMS: ClassVar[dict] = {"relax_cell": True, "tolmxf": 5.0e-5}

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

        try:
            atom_constraints = kwargs.pop("atoms_constraints")
        except KeyError:
            atom_constraints = None

        try:
            relax_cell = kwargs.pop("relax_cell")
        except KeyError:
            relax_cell = self.relax_cell

        if relax_cell:
            relax_method = aobj.RelaxationMethod.atoms_and_cell(
                atoms_constraints=atom_constraints
            )
        else:
            relax_method = aobj.RelaxationMethod.atoms_only(
                atoms_constraints=atom_constraints
            )

        try:
            tolmxf = kwargs.pop("tolmxf")
        except KeyError:
            tolmxf = self.tolmxf

        relax_method.abivars.update(tolmxf=tolmxf)

        relax_input = ion_ioncell_relax_input(structure, pseudos=pseudos, **kwargs)[0]
        relax_input.set_vars(relax_method.to_abivars())

        return relax_input

    def update_abinit_input(self, abinit_input, param, value):
        if param == "relax_cell":
            if value:
                relax_method = aobj.RelaxationMethod.atoms_and_cell(
                    atoms_constraints=None
                )
            else:
                relax_method = aobj.RelaxationMethod.atoms_only(atoms_constraints=None)
            print(relax_method.to_abivars())
            abinit_input.set_vars(relax_method.to_abivars())
        else:
            super().update_abinit_input(abinit_input, param, value)
