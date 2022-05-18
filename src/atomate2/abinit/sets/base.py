"""Module defining base abinit input set and generator."""
import copy
import json
import logging
import os
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable, List, Optional, Union

import pseudo_dojo
from abipy.abio.inputs import AbinitInput
from abipy.flowtk.utils import Directory, irdvars_for_ext
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.io.abinit.pseudos import PseudoTable
from pymatgen.io.core import InputGenerator, InputSet

from atomate2.abinit.files import ALL_ABIEXTS, fname2ext, load_abinit_input, out_to_in
from atomate2.abinit.utils.common import (
    INDATA_PREFIX,
    INDIR_NAME,
    INPUT_FILE_NAME,
    OUTDATA_PREFIX,
    OUTDIR_NAME,
    TMPDATA_PREFIX,
    TMPDIR_NAME,
    InitializationError,
)

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
        input_files: Optional[Iterable[Union[str, Path]]] = None,
    ):
        self.input_files = input_files
        super().__init__(inputs={INPUT_FILE_NAME: abinit_input})

    def write_input(
        self,
        directory: Union[str, Path],
        make_dir: bool = True,
        overwrite: bool = True,
        zip_inputs: bool = False,
    ):
        """Write Abinit input files to a directory."""
        # TODO: do we allow zipping ? not sure if it really makes sense for abinit as
        #  the abinit input set also sets up links to previous files, sets up the
        #  indir, outdir and tmpdir, ...
        self.inputs["abinit_input.json"] = json.dumps(self.abinit_input.as_dict())
        super().write_input(
            directory=directory,
            make_dir=make_dir,
            overwrite=overwrite,
            zip_inputs=zip_inputs,
        )
        del self.inputs["abinit_input.json"]
        self.indir, self.outdir, self.tmpdir = self.set_workdir(workdir=directory)

        if self.input_files:
            out_to_in(
                out_files=self.input_files, indir=self.indir.path, link_files=True
            )

        # TODO: currently a hack to write down the pseudos ...
        #  transfer and adapt pseudos management to abipy, this should be handled
        #  by AbinitInput.__str__()
        with zopen(os.path.join(directory, INPUT_FILE_NAME), "wt") as f:
            abinit_input = self[INPUT_FILE_NAME]
            input_string = str(abinit_input)
            pseudos_string = '\npseudos "'
            pseudos_string += ",\n         ".join(
                [psp.filepath for psp in abinit_input.pseudos]
            )
            pseudos_string += '"'
            input_string += pseudos_string
            f.write(input_string)

        validation = True
        if validation:
            self.validate()

    def validate(self):
        # Check that all files in the input directory have their corresponding
        # ird variables.
        for filename in os.listdir(self.indir.path):
            ext = fname2ext(filename)
            if ext is None:
                raise ValueError(
                    f"'{filename}' file in input directory does not have a "
                    f"valid abinit extension."
                )
            irdvars = irdvars_for_ext(ext)
            for irdvar, irdval in irdvars.items():
                if irdvar not in self.abinit_input:
                    raise ValueError(
                        f"'{irdvar}' ird variable not set for '{filename}' file."
                    )
                if self.abinit_input[irdvar] != irdval:
                    raise ValueError(
                        f"'{irdvar} {irdval}' ird variable is wrong for "
                        f"'{filename}' file."
                    )

    @property
    def abinit_input(self):
        return self[INPUT_FILE_NAME]

    @staticmethod
    def set_workdir(workdir):
        """Set up the working directory.

        This also sets up and creates standard input, output and temporary
        directories, as well as standard file names for input and output.
        """
        workdir = os.path.abspath(workdir)

        # Directories with input|output|temporary data.
        indir = Directory(os.path.join(workdir, INDIR_NAME))
        outdir = Directory(os.path.join(workdir, OUTDIR_NAME))
        tmpdir = Directory(os.path.join(workdir, TMPDIR_NAME))

        # Create dirs for input, output and tmp data.
        indir.makedirs()
        outdir.makedirs()
        tmpdir.makedirs()

        return indir, outdir, tmpdir

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
            string or list of strings with the names of the abinit variables
            to be removed.
        strict
            whether to raise a KeyError if one of the abinit variables to be
            removed is not present.

        Returns
        -------
        dict
            dictionary with the variables that have been removed.
        """
        return self.abinit_input.remove_vars(keys=keys, strict=strict)

    def set_structure(self, structure: Any) -> Structure:
        """Set the structure for this input set.

        This basically forwards the setting of the structure to the abipy
        AbinitInput object.
        """
        return self.abinit_input.set_structure(structure)

    def deepcopy(self):
        """Deep copy of the input set."""
        return copy.deepcopy(self)


PrevOutput = namedtuple("PrevOutput", "dirname exts")


@dataclass
class AbinitInputSetGenerator(InputGenerator):
    """A class to generate Abinit input sets."""

    pseudos: Union[
        List[str], PseudoTable
    ] = pseudo_dojo.OfficialDojoTable.from_djson_file(
        os.path.join(
            pseudo_dojo.dojotable_absdir("ONCVPSP-PBE-PDv0.4"), "standard.djson"
        )
    )

    extra_abivars: dict = field(default_factory=dict)

    prev_output_exts: tuple = ("WFK", "DEN")

    # class variables
    DEFAULT_PARAMS: ClassVar[dict] = {}
    ALLOW_RESTART_FROM: ClassVar[set] = set()

    def __post_init__(self):
        """Perform post-initialization settings for AbinitInputSetGenerator."""
        params = {}
        # Here we assume the most
        for cls in self.__class__.mro()[::-1]:
            try:
                cls_params = getattr(cls, "DEFAULT_PARAMS")  # noqa: B009
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

        Here we assume that restart_from is a directory and prev_outputs is
        a list of directories. We also assume there is an abinit_input.json file
        in each of these directories containing the AbinitInput object used to
        execute abinit.
        """
        pseudos = kwargs.get("pseudos", self.pseudos)
        restart_from = self.check_format_prev_dirs(restart_from)
        prev_outputs = self.check_format_prev_dirs(prev_outputs)

        input_files = []

        if restart_from is None:
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
                **parameters,
            )
            # Always reset the ird variables
            abinit_input.pop_irdvars()
        else:
            abinit_input = load_abinit_input(restart_from)
            if len(self.ALLOW_RESTART_FROM.intersection(abinit_input.runlevel)) == 0:
                raise RuntimeError(
                    f"Restart is not allowed. "
                    f'For "{self.__class__.__name__}" input generator, the '
                    f"allowed previous calculations for restart are: "
                    f'{" ".join(self.ALLOW_RESTART_FROM)}'
                )

            # TODO: make sure this is ok
            if structure is not None:
                abinit_input.set_structure(structure)
            # Always reset the ird variables
            abinit_input.pop_irdvars()
            # Files for restart (e.g. continue a not yet converged
            # scf/nscf/relax calculation)
            irdvars, files = self.resolve_deps(restart_from)
            abinit_input.set_vars(irdvars)
            input_files.extend(files)

            if update_params:
                logger.info(
                    "Getting parameters to update for the generation of the "
                    "abinit input."
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

        # Files that are dependencies (e.g. band structure calculations
        # need the density)
        if prev_outputs:
            irdvars, files = self.resolve_deps(prev_outputs)
            abinit_input.set_vars(irdvars)
            input_files.extend(files)

        extra_abivars = dict(self.extra_abivars)
        extra_abivars.update(kwargs.get("extra_abivars", {}))

        abinit_input.set_vars(**extra_abivars)

        abinit_input["indata_prefix"] = (f'"{INDATA_PREFIX}"',)
        abinit_input["outdata_prefix"] = (f'"{OUTDATA_PREFIX}"',)
        abinit_input["tmpdata_prefix"] = (f'"{TMPDATA_PREFIX}"',)

        return AbinitInputSet(
            abinit_input=abinit_input,
            input_files=input_files,
        )

    def check_format_prev_dirs(self, prev_dirs):
        """Check and format the prev_dirs (restart or dependency)."""
        if prev_dirs is None:
            return None
        if isinstance(prev_dirs, (str, Path)):
            prev_dirs = [PrevOutput(prev_dirs, tuple(self.prev_output_exts))]
        elif isinstance(prev_dirs, (list, tuple)):
            if len(prev_dirs) == 2 and isinstance(prev_dirs[0], (str, Path)):
                if isinstance(prev_dirs[1], str) and prev_dirs[1] in ALL_ABIEXTS:
                    prev_dirs = [(prev_dirs[0], (prev_dirs[1],))]
                elif isinstance(prev_dirs[1], (list, tuple)):
                    prev_dirs = [prev_dirs]
            new_prev_dirs = []
            for prev_dir in prev_dirs:
                if isinstance(prev_dir, (str, Path)):
                    new_prev_dirs.append(
                        PrevOutput(prev_dir, tuple(self.prev_output_exts))
                    )
                elif isinstance(prev_dir, (list, tuple)):
                    if len(prev_dir) != 2:
                        raise ValueError(
                            "Wrong schema for single previous directory dependency. "
                            "Should be a list/tuple of directory and extension(s)."
                        )
                    if not isinstance(prev_dir[0], (str, Path)):
                        raise ValueError(
                            "Previous directory should be expressed as a str or Path."
                        )
                    if isinstance(prev_dir[1], str):
                        exts = [prev_dir[1]]
                    elif isinstance(prev_dir[1], (list, tuple)):
                        exts = prev_dir[1]
                    else:
                        raise ValueError("")
                    for ext in exts:
                        if ext not in ALL_ABIEXTS:
                            raise ValueError(
                                f"'{ext}' is not a valid Abinit file extension."
                            )
                    new_prev_dirs.append(PrevOutput(prev_dir[0], tuple(exts)))
                else:
                    raise ValueError("Wrong type for previous directory dependency.")
            prev_dirs = new_prev_dirs
        else:
            raise ValueError("Wrong type for previous directory dependency.")
        return prev_dirs

    def resolve_deps(self, prev_outputs):
        """Resolve dependencies.

        This method assumes that prev_outputs is in the correct format, i.e.
        a list of PrevOutput named tuples.
        """
        input_files = []
        deps_irdvars = {}
        for prev_output in prev_outputs:
            irdvars, inp_file = self.resolve_dep(prev_output=prev_output)
            input_files.append(inp_file)
            deps_irdvars.update(irdvars)

        return deps_irdvars, input_files

    @staticmethod
    def resolve_dep(prev_output):
        """Return irdvars and corresponding file for a given dependency.

        This method assumes that prev_output is in the correct format,
        i.e. a PrevOutput named tuple.
        """
        prev_outdir = Directory(os.path.join(prev_output.dirname, OUTDIR_NAME))

        for ext in prev_output.exts:
            restart_file = prev_outdir.has_abiext(ext)
            irdvars = irdvars_for_ext(ext)
            if restart_file:
                break
        else:
            msg = f"Cannot find {' or '.join(prev_output.exts)} file to restart from."
            logger.error(msg)
            raise InitializationError(msg)
        return irdvars, restart_file

    def get_abinit_input(
        self, structure=None, pseudos=None, prev_outputs=None, **kwargs
    ):
        """Get AbinitInput object."""
        raise NotImplementedError

    def update_abinit_input(self, abinit_input, param, value):
        """Update AbinitInput object."""
        raise RuntimeError(
            f'Cannot apply "{param}" input set generator update to'
            f"previous AbinitInput."
        )

    def on_restart(self, abinit_input):
        """Perform updates of AbinitInput upon restart of a previous calculation."""
