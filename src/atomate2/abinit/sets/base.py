"""Module defining base abinit input set and generator."""
import copy
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable, List, Union

import pseudo_dojo
from abipy.abio.inputs import AbinitInput
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.io.abinit.pseudos import PseudoTable

from atomate2.abinit.utils.common import (
    INDATA_PREFIX,
    INPUT_FILE_NAME,
    OUTDATA_PREFIX,
    TMPDATA_PREFIX,
)
from atomate2.common.sets import InputSet, InputSetGenerator

__all__ = ["AbinitInputSet", "AbinitInputSetGenerator"]

logger = logging.getLogger(__name__)


class _UNSET(MSONable):
    def __eq__(self, other):
        return self.__class__ == other.__class__


UNSET = _UNSET()


class AbinitInputSet(InputSet):
    """
    A class to represent a set of Abinit inputs.

    Parameters
    ----------
    abinit_input
        An AbinitInput object.
    """

    def __init__(
        self,
        abinit_input: AbinitInput,
    ):
        self.abinit_input = abinit_input
        # TODO: is this the place for this ? if we want to put the sets in abipy, do differently anyway (it's using
        #  atomate2.abinit.common)
        self.abinit_input["indata_prefix"] = (f'"{INDATA_PREFIX}"',)
        self.abinit_input["outdata_prefix"] = (f'"{OUTDATA_PREFIX}"',)
        self.abinit_input["tmpdata_prefix"] = (f'"{TMPDATA_PREFIX}"',)

    def write_input(
        self,
        directory: Union[str, Path],
        make_dir: bool = True,
        overwrite: bool = True,
    ):
        """Write Abinit input files to a directory."""
        directory = Path(directory)
        if make_dir and not directory.exists():
            os.makedirs(directory)

        if not overwrite and (directory / INPUT_FILE_NAME).exists():
            raise FileExistsError(f"{directory / INPUT_FILE_NAME} already exists.")

        with zopen(directory / INPUT_FILE_NAME, "wt") as f:

            input_string = str(self.abinit_input)
            # TODO: transfer and adapt pseudos management to abipy, this should be handled by AbinitInput.__str__()
            pseudos_string = '\npseudos "'
            pseudos_string += ",\n         ".join(
                [psp.filepath for psp in self.abinit_input.pseudos]
            )
            pseudos_string += '"'
            input_string += pseudos_string

            f.write(input_string)

    def set_vars(self, *args, **kwargs) -> dict:
        """Set the values of abinit variables.

        This sets the abinit variables in the abipy AbinitInput object.

        One can pass a dictionary mapping the abinit variables to their values or
        the abinit variables as keyword arguments. A combination of the two
        options is also allowed.

        Returns
        -------
        dict
            dictionary with the variables that have been added.
        """
        return self.abinit_input.set_vars(*args, **kwargs)

    def remove_vars(self, keys: Union[Iterable[str], str], strict: bool = True) -> dict:
        """Remove the abinit variables listed in keys.

        This removes the abinit variables from the abipy AbinitInput object.

        Parameters
        ----------
        keys
            string or list of strings with the names of the abinit variables to be removed.
        strict
            whether to raise a KeyError if one of the abinit variables to be removed is not present.

        Returns
        -------
        dict
            dictionary with the variables that have been removed.
        """
        return self.abinit_input.remove_vars(keys=keys, strict=strict)

    def set_structure(self, structure: Any) -> Structure:
        """Set the structure for this input set.

        This basically forwards the setting of the structure to the abipy AbinitInput object.
        """
        return self.abinit_input.set_structure(structure)

    def deepcopy(self):
        """Deep copy of the input set."""
        return copy.deepcopy(self)


@dataclass
class AbinitInputSetGenerator(InputSetGenerator):
    """A class to generate Abinit input sets."""

    # TODO: see how to deal with this
    #  could be one or several of:
    #   - a str representing a pseudo table (e.g. "ONCVPSP-PBE-PDv0.4") [+standard or stringent ? how ?]
    #     => the input set generator is easily serialized (not the whole table) and the pseudos objects
    #        are generated on the fly.
    #   - a str with a directory of where to find pseudos (how to distinguish from the previous one ?).
    #     => the directory should have a specific tree structure (e.g. each atom has its own directory).
    #   - a list of str corresponding to the explicit pseudos filepaths
    #     => the user needs to know exactly where the pseudos are in the server in which the jobs will be executed
    #   - a list of Pseudo objects
    #     => in this case the Pseudo objects are serialized/deserialized.
    #        one question might be that we may want to have
    #           a) "file-like" Pseudo objects, for which the pseudos files have to be present and discoverable
    #              somehow on the cluster, [this is the case currently I think]
    #           b) "full" Pseudo objects, for which the pseudo files do not need to be present on the cluster
    #              and they will be written at runtime. [is this something desirable ? means we have a lot of
    #              data in the database for storing the pseudos for each job]
    #
    pseudos: Union[
        List[str], PseudoTable
    ] = pseudo_dojo.OfficialDojoTable.from_djson_file(
        os.path.join(
            pseudo_dojo.dojotable_absdir("ONCVPSP-PBE-PDv0.4"), "standard.djson"
        )
    )

    extra_abivars: dict = field(default_factory=dict)

    # class variables
    DEFAULT_PARAMS: ClassVar[dict] = {}
    ALLOW_RESTART_FROM: ClassVar[set] = set()

    def __post_init__(self):
        """Perform post-initialization settings for AbinitInputSetGenerator."""
        params = {}
        # Here we assume the most
        for cls in self.__class__.mro()[::-1]:
            try:
                cls_params = getattr(cls, "DEFAULT_PARAMS")
                params.update(cls_params)
            except AttributeError:
                pass
        self.DEFAULT_PARAMS = dict(params)

    def get_input_set(  # type: ignore
        self,
        structure=None,
        restart_from=None,
        prev_outputs=None,
        update_params=True,
        **kwargs,
    ) -> AbinitInputSet:
        """Generate an AbinitInputSet object.

        Here we assume that restart_from is either an input set to restart from, or a directory (cannot be a job uuid
        or an OutputReference).
        """
        # TODO: Think about prev_outputs ? how do they enter the equation here ?
        #  One example is for GW, when doing sigma, we might need to know the input set for the screening so that
        #  we know which ecuteps, ecutsigx we can use ?
        # if prev_outputs is not None:
        #     raise NotImplementedError('get intput set with prev_outputs not yet implemented')
        pseudos = kwargs.get("pseudos", self.pseudos)

        if restart_from is None:
            # Take parameters from the kwargs or, if not present, from the input set generator
            # (default or explicitly set)
            logger.info("Getting parameters for the generation of the abinit input.")
            parameters = {
                param: kwargs.get(
                    param,
                    param_default
                    if getattr(self, param) == UNSET
                    else getattr(self, param),
                )
                for param, param_default in self.DEFAULT_PARAMS.items()
            }
            logging.debug("Parameters for the job:")
            logging.debug(
                "\n".join(
                    [f" - {param}: {str(value)}" for param, value in parameters.items()]
                )
            )
            abinit_input = self.get_abinit_input(
                structure=structure,
                pseudos=pseudos,
                prev_outputs=prev_outputs,
                # restart_from=None,
                **parameters,
            )
            # Always reset the ird variables
            # TODO: make sure this is ok. See if we add them here based on restart_from and prev_outputs ?
            abinit_input.pop_irdvars()
        else:
            if not isinstance(restart_from, AbinitInputSet):
                raise NotImplementedError(
                    "Restarting only allowed from a previous AbinitInputSet currently"
                )
            abinit_input = restart_from.abinit_input.deepcopy()
            # TODO: make sure this is ok
            if structure is not None:
                abinit_input.set_structure(structure)
            # Always reset the ird variables
            # TODO: see if we add them here based on restart_from and prev_outputs ?
            abinit_input.pop_irdvars()
            if len(self.ALLOW_RESTART_FROM.intersection(abinit_input.runlevel)) == 0:
                raise RuntimeError(
                    f"Restart is not allowed. "
                    f'For "{self.__class__.__name__}" input generator, the '
                    f'allowed previous calculations for restart are: {" ".join(self.ALLOW_RESTART_FROM)}'
                )

            if update_params:
                logger.info(
                    "Getting parameters to update for the generation of the abinit input."
                )
                self.on_restart(abinit_input)
                parameters = {}
                for param in self.DEFAULT_PARAMS:
                    if param in kwargs:
                        parameters[param] = kwargs[param]
                    elif getattr(self, param) != UNSET:
                        parameters[param] = getattr(self, param)
                logging.debug("Parameters updated for the job:")
                logging.debug(
                    "\n".join(
                        [
                            f" - {param}: {str(value)}"
                            for param, value in parameters.items()
                        ]
                    )
                )
                for param, value in parameters.items():
                    if value is UNSET:
                        continue
                    self.update_abinit_input(abinit_input, param, value)

        extra_abivars = dict(self.extra_abivars)
        extra_abivars.update(kwargs.get("extra_abivars", {}))
        # extra_abivars.update(kwargs.get('extra_abivars', {}))

        abinit_input.set_vars(**extra_abivars)

        return AbinitInputSet(
            abinit_input=abinit_input,
        )

    def get_abinit_input(
        self, structure=None, pseudos=None, prev_outputs=None, **kwargs
    ):
        """Get AbinitInput object."""
        raise NotImplementedError

    def update_abinit_input(self, abinit_input, param, value):
        """Update AbinitInput object."""
        raise RuntimeError(
            f'Cannot apply "{param}" input set generator update to previous AbinitInput.'
        )

    def on_restart(self, abinit_input):
        """Perform updates of AbinitInput when the calculation is restarted from a previous one."""
