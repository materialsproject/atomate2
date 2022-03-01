"""Factory objects and tools to generate abinit inputs."""

from typing import Any

from abipy.abio.factories import ebands_from_gsinput, scf_input
from monty.json import MontyDecoder, MSONable, jsanitize
from pymatgen.util.serialization import pmg_serialize


# FIXME if the pseudos are passed as a PseudoTable the whole table will be serialized,
# it would be better to filter on the structure elements
class InputGenerator(MSONable):
    """Base input generator object."""

    factory_function: staticmethod[Any] = None
    # inputs = None
    structure_required = True
    gs_input_required = False

    def __init__(self, **kwargs):
        """Construct InputGenerator object."""
        if self.factory_function is None:
            raise NotImplementedError("The factory function should be specified.")
        # if self.inputs is None:
        #     raise NotImplementedError('The mandatory inputs to the factory function should be specified.')

        # These should contain "general" parameters (e.g. pseudos, ecut, kppa, ecuteps, ...)
        # TODO: Should we make a check on these ?
        self.kwargs = kwargs

    def generate_abinit_input(self, *args, **kwargs):
        """Generate AbinitInput object from the structure or from a previous abinit input (e.g. GS AbinitInput)."""
        parameters = dict(self.kwargs)
        kwargs = dict(kwargs)
        extra_abivars = parameters.pop("extra_abivars", {})
        extra_abivars.update(kwargs.pop("extra_abivars", {}))

        # Here we can overwrite some of the general parameters if we want
        parameters.update(kwargs)

        # Generate abinit input and set extra abinit variables
        abinit_input = self.factory_function(*args, **parameters)
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


class NScfInputGenerator(InputGenerator):
    """Input generator for Non-Scf calculations."""

    # TODO: try to do something similar to vasp ?
    # i.e. with mode "uniform" or "line" respectively with a reciprocal_density and a line_density.
    factory_function = staticmethod(ebands_from_gsinput)
    structure_required = False
    gs_input_required = True
