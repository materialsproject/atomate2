"""Factory objects and tools to generate abinit inputs.

Note this module is meant to be removed (replaced by InputSetGenerator/InputSet).
"""

from typing import Any, Optional

import pymatgen.io.abinit.abiobjects as aobj
from abipy.abio.factories import (
    _stopping_criterion,
    ebands_from_gsinput,
    ion_ioncell_relax_input,
    scf_input,
)
from abipy.abio.input_tags import MOLECULAR_DYNAMICS, RELAX, SCF
from abipy.abio.inputs import AbinitInput
from monty.json import MontyDecoder, MSONable, jsanitize
from pymatgen.core.structure import Structure
from pymatgen.util.serialization import pmg_serialize


# FIXME if the pseudos are passed as a PseudoTable the whole table will be serialized,
# it would be better to filter on the structure elements
class InputGenerator(MSONable):
    """Base input generator object.

    This serves as a base for input generators taking a structure and/or a previous AbinitInput as
    an input.
    """

    factory_function: Any = None
    input_structure = True
    input_previous_abinit_input = True
    input_structure_and_previous_abinit_input = True

    def __init__(self, **kwargs):
        """Construct InputGenerator object."""
        if self.factory_function is None:
            raise NotImplementedError("The factory function should be specified.")
        if not any(
            [
                self.input_structure,
                self.input_previous_abinit_input,
                self.input_structure_and_previous_abinit_input,
            ]
        ):
            raise NotImplementedError(
                "The input generator needs at least the structure or a previous AbinitInput."
            )
        self.kwargs = kwargs

    def generate_abinit_input(
        self,
        structure: Optional[Structure] = None,
        previous_abinit_input: Optional[AbinitInput] = None,
        pseudos=None,
        **kwargs,
    ):
        """Generate AbinitInput object from the structure or from a previous abinit input (e.g. GS AbinitInput)."""
        parameters = dict(self.kwargs)
        kwargs = dict(kwargs)
        extra_abivars = parameters.pop("extra_abivars", {})
        extra_abivars.update(kwargs.pop("extra_abivars", {}))

        # Here we can overwrite some of the general parameters if we want
        parameters.update(kwargs)

        # Generate abinit input and set extra abinit variables
        if previous_abinit_input is not None:
            if structure is not None:
                if not self.input_structure_and_previous_abinit_input:
                    raise RuntimeError(
                        "Both a previous AbinitInput and a Structure are provided. "
                        f'This is not allowed for "{self.__class__.__name__}" input generator.'
                    )
                abinit_input = self.factory_function(
                    structure, previous_abinit_input, **parameters
                )
            else:
                if not self.input_previous_abinit_input:
                    raise RuntimeError(
                        "A previous AbinitInput is provided. "
                        f'This is not allowed for "{self.__class__.__name__}" input generator.'
                    )
                abinit_input = self.factory_function(
                    previous_abinit_input, **parameters
                )
        elif structure is not None:
            if not self.input_structure:
                raise RuntimeError(
                    "A Structure is provided. "
                    f'This is not allowed for "{self.__class__.__name__}" input generator.'
                )
            # When only the structure is provided, the pseudos must be provided
            abinit_input = self.factory_function(
                structure, pseudos=pseudos, **parameters
            )
        else:
            raise RuntimeError(
                "Both structure and previous_abinit_input are undefined."
            )

        abinit_input.set_vars(extra_abivars)

        return abinit_input

    @pmg_serialize
    def as_dict(self):
        """Create dictionary representation of the input generator."""
        # sanitize to avoid numpy arrays and serialize MSONable objects
        return jsanitize(dict(kwargs=self.kwargs), strict=True)

    @classmethod
    def from_dict(cls, d):
        """Create instance of the input generator from its dictionary representation."""
        dec = MontyDecoder()
        return cls(**dec.process_decoded(d["kwargs"]))


class ScfInputGenerator(InputGenerator):
    """Input generator for Scf calculations."""

    factory_function = staticmethod(scf_input)
    input_structure = True
    input_previous_abinit_input = False
    input_structure_and_previous_abinit_input = False


class NScfInputGenerator(InputGenerator):
    """Input generator for Non-Scf calculations."""

    # TODO: try to do something similar to vasp ?
    #  i.e. with mode "uniform" or "line" respectively with a reciprocal_density and a line_density.
    factory_function = staticmethod(ebands_from_gsinput)
    input_structure = False
    input_previous_abinit_input = True
    input_structure_and_previous_abinit_input = False


class NScfWfqInputGenerator(InputGenerator):
    """Input generator for Non-Scf Wfq calculations."""

    # TODO: implement
    # factory_function = None
    # input_structure = False
    # input_previous_abinit_input = True
    # input_structure_and_previous_abinit_input = False

    @staticmethod
    def factory_function(*args):
        """Create abinit input for Nscf Wfq calculation."""
        print(args)


class RelaxInputGenerator(InputGenerator):
    """Input generator for relaxation calculations."""

    input_structure = True
    input_previous_abinit_input = True
    input_structure_and_previous_abinit_input = True
    relax_cell = True

    def factory_function(self, *args, pseudos=None, accuracy="normal", **kwargs):
        """Create abinit input for relaxation.

        This is a flexible factory function allowing to generate an abinit input from a previous
        abinit input or from a structure or from a previous abinit input and a structure.
        """
        structure, previous_abinit_input = None, None
        if len(args) == 1:
            if isinstance(args[0], AbinitInput):
                previous_abinit_input = args[0]
            elif isinstance(args[0], Structure):
                structure = args[0]
            else:
                raise RuntimeError()
        elif len(args) == 2:
            structure, previous_abinit_input = args

        try:
            atom_constraints = kwargs.pop("atoms_constraints")
        except KeyError:
            atom_constraints = None

        relax_cell = kwargs.get("relax_cell", self.relax_cell)
        if relax_cell:
            relax_method = aobj.RelaxationMethod.atoms_and_cell(
                atoms_constraints=atom_constraints
            )
        else:
            relax_method = aobj.RelaxationMethod.atoms_only(
                atoms_constraints=atom_constraints
            )

        if previous_abinit_input is not None:
            # Here we check here that the previous abinit input is a proper GS input (and not e.g. a screening ...)
            # Allow restarts from an scf, a relaxation or a molecular dynamics calculation.
            allowed_previous_levels = {SCF, RELAX, MOLECULAR_DYNAMICS}
            if (
                len(
                    allowed_previous_levels.intersection(previous_abinit_input.runlevel)
                )
                == 0
            ):
                raise RuntimeError(
                    "Previous abinit input is not a proper Ground-State calculation. "
                    f'This is required for "{self.__class__.__name__}" input generator. '
                    f'Allowed previous calculations are: {" ".join(allowed_previous_levels)}'
                )
            relax_input = previous_abinit_input.deepcopy()
            relax_input.pop_irdvars()
            if structure is not None:
                # Update with the new structure
                # TODO: should we check something here about the structure in the
                #  previous_abinit_input and the new one ?
                #  e.g. at least that all the sites are the same ?
                #  maybe that they are not too far from each other ?
                relax_input.set_structure(structure=structure)
        elif structure is not None:
            relax_input = ion_ioncell_relax_input(
                structure, pseudos=pseudos, accuracy=accuracy, **kwargs
            )[0]
        else:
            raise RuntimeError(
                "Both structure and previous_abinit_input are undefined."
            )

        relax_input.set_vars(relax_method.to_abivars())
        relax_input.set_vars(_stopping_criterion("relax", accuracy))

        return relax_input
