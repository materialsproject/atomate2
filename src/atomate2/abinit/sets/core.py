"""Module defining core Abinit input set generators."""

from dataclasses import dataclass
from typing import ClassVar, Optional

import pymatgen.io.abinit.abiobjects as aobj
from abipy.abio.factories import ebands_from_gsinput, ion_ioncell_relax_input, scf_input
from abipy.abio.input_tags import MOLECULAR_DYNAMICS, NSCF, RELAX, SCF

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
            # abinit_input.set_vars(_find_ecut_pawecutdg(value, None, abinit_input.pseudos, accuracy))
            abinit_input.set_vars({"ecut": value})
        else:
            raise RuntimeError(
                f'Cannot apply "{param}" input set generator update to previous AbinitInput.'
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
                "To restart from a previous static or otherwise scf (e.g. relaxation) calculation, "
                "use restart_from argument of get_input_set method instead."
            )

        return scf_input(
            structure=structure,
            pseudos=pseudos,
            **kwargs,
        )

    def on_restart(self, abinit_input):
        """Perform updates of AbinitInput when the calculation is restarted from a previous one.

        In this case, a static calculation can be started from a relaxation one. The relaxation-like variables
        need to be removed from the AbinitInput.
        """
        # Always remove relaxation-like variables so that if we make the SCF job starting from a previous
        # relaxation or molecular dynamics job, the structure will indeed be static.
        abinit_input.pop_vars(["ionmov", "optcell", "ntime"])


@dataclass
class NonSCFSetGenerator(AbinitInputSetGenerator):
    """Class to generate Abinit non-SCF input sets."""

    nband: Optional[int] = UNSET
    ndivsm: int = UNSET
    accuracy: str = UNSET

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
            # TODO: maybe just check that the structure is the same as the one in the previous_input_set ?
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
        previous_abinit_input = prev_outputs[0].abinit_input_set.abinit_input
        # if pseudos is not None:
        #     # TODO: maybe just check that the pseudos are the same as the one in the previous_input_set ?
        #     raise RuntimeError('Pseudos should not be set in a non-SCF input set. '
        #                        'It should come directly from the previous (SCF) input set.')

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
                "To restart from a previous static or otherwise scf (e.g. relaxation) calculation, "
                "use restart_from argument of get_input_set method instead."
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

        print("KWARGS", kwargs)

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
        #
        # def update_abinit_input(self, abinit_input, param, value):
        #     if param == 'ecut':
        #         # Set the cutoff energies.
        #         # TODO: make a check on pawecutdg ?
        #         # TODO: how to take accuracy into account ?
        #         if value is None:
        #             raise NotImplementedError('')
        #         # abinit_input.set_vars(_find_ecut_pawecutdg(value, None, abinit_input.pseudos, accuracy))
        #         abinit_input.set_vars({'ecut': value})
        #     else:
        #         raise RuntimeError(f'Cannot apply "{param}" input set generator update to previous AbinitInput.')

    #
    #     previous_input_set = kwargs.get('previous_input_set', None)
    #     if previous_input_set is not None:
    #         previous_abinit_input = self.previous_input_set.abinit_input
    #         # Here we check here that the previous abinit input is a proper GS input (and not e.g. a screening ...)
    #         # Allow restarts from an scf, a relaxation or a molecular dynamics calculation.
    #         allowed_previous_levels = {SCF, RELAX, MOLECULAR_DYNAMICS}
    #         if (
    #                 len(
    #                     allowed_previous_levels.intersection(previous_abinit_input.runlevel)
    #                 )
    #                 == 0
    #         ):
    #             raise RuntimeError(
    #                 "Previous abinit input is not a proper Ground-State calculation. "
    #                 f'This is required for "{self.__class__.__name__}" input generator. '
    #                 f'Allowed previous calculations are: {" ".join(allowed_previous_levels)}'
    #             )
    #         relax_input = previous_abinit_input.deepcopy()
    #         relax_input.pop_irdvars()
    #         # relax_input.
    #         if structure is not None:
    #             # Update with the new structure
    #             # TODO: should we check something here about the structure in the
    #             #  previous_abinit_input and the new one ?
    #             #  e.g. at least that all the sites are the same ?
    #             #  maybe that they are not too far from each other ?
    #             relax_input.set_structure(structure=structure)
    #     elif structure is not None:
    #         relax_input = ion_ioncell_relax_input(
    #             structure, pseudos=pseudos, accuracy=self.accuracy, **kwargs
    #         )[0]
    #     else:
    #         raise RuntimeError("Both structure and previous_input_set are undefined.")
    #
    #     relax_input.set_vars(relax_method.to_abivars())

    #         relax_input.set_vars(_stopping_criterion("relax", accuracy))


# class RelaxInputGenerator(InputGenerator):
#     """Input generator for relaxation calculations."""
#
#     input_structure = True
#     input_previous_abinit_input = True
#     input_structure_and_previous_abinit_input = True
#     relax_cell = True
#
#     def factory_function(self, *args, pseudos=None, accuracy="normal", **kwargs):
#         """Create abinit input for relaxation.
#
#         This is a flexible factory function allowing to generate an abinit input from a previous
#         abinit input or from a structure or from a previous abinit input and a structure.
#         """
#         structure, previous_abinit_input = None, None
#         if len(args) == 1:
#             if isinstance(args[0], AbinitInput):
#                 previous_abinit_input = args[0]
#             elif isinstance(args[0], Structure):
#                 structure = args[0]
#             else:
#                 raise RuntimeError()
#         elif len(args) == 2:
#             structure, previous_abinit_input = args
#
#         try:
#             atom_constraints = kwargs.pop("atoms_constraints")
#         except KeyError:
#             atom_constraints = None
#
#         relax_cell = kwargs.get("relax_cell", self.relax_cell)
#         if relax_cell:
#             relax_method = aobj.RelaxationMethod.atoms_and_cell(
#                 atoms_constraints=atom_constraints
#             )
#         else:
#             relax_method = aobj.RelaxationMethod.atoms_only(
#                 atoms_constraints=atom_constraints
#             )
#
#         if previous_abinit_input is not None:
#             # Here we check here that the previous abinit input is a proper GS input (and not e.g. a screening ...)
#             # Allow restarts from an scf, a relaxation or a molecular dynamics calculation.
#             allowed_previous_levels = {SCF, RELAX, MOLECULAR_DYNAMICS}
#             if (
#                 len(
#                     allowed_previous_levels.intersection(previous_abinit_input.runlevel)
#                 )
#                 == 0
#             ):
#                 raise RuntimeError(
#                     "Previous abinit input is not a proper Ground-State calculation. "
#                     f'This is required for "{self.__class__.__name__}" input generator. '
#                     f'Allowed previous calculations are: {" ".join(allowed_previous_levels)}'
#                 )
#             relax_input = previous_abinit_input.deepcopy()
#             relax_input.pop_irdvars()
#             if structure is not None:
#                 # Update with the new structure
#                 # TODO: should we check something here about the structure in the
#                 #  previous_abinit_input and the new one ?
#                 #  e.g. at least that all the sites are the same ?
#                 #  maybe that they are not too far from each other ?
#                 relax_input.set_structure(structure=structure)
#         elif structure is not None:
#             relax_input = ion_ioncell_relax_input(
#                 structure, pseudos=pseudos, accuracy=accuracy, **kwargs
#             )[0]
#         else:
#             raise RuntimeError(
#                 "Both structure and previous_abinit_input are undefined."
#             )
#
#         relax_input.set_vars(relax_method.to_abivars())
#         relax_input.set_vars(_stopping_criterion("relax", accuracy))
#
#         return relax_input
