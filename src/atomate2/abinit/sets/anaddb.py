"""Module defining base anaddb input set and generator."""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from abipy.abio.inputs import AnaddbInput
from abipy.flowtk.utils import Directory
from monty.json import MontyEncoder, jsanitize
from pymatgen.io.core import InputSet

from atomate2.abinit.files import out_to_in
from atomate2.abinit.sets.base import AbinitMixinInputGenerator
from atomate2.abinit.utils.common import (
    ANADDB_INPUT_FILE_NAME,
    INDATA_PREFIX,
    INDIR_NAME,
    OUTDIR_NAME,
    TMPDIR_NAME,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from pymatgen.core.structure import Structure


logger = logging.getLogger(__name__)


class AnaddbInputSet(InputSet):
    """
    A class to represent a set of Anaddb inputs.

    Parameters
    ----------
    anaddb_input
        An AnaddbInput object.
    input_file
        A list with one input file (out_DDB) needed for the calculation.
    """

    def __init__(
        self,
        anaddb_input: AnaddbInput,
        input_files: Iterable[tuple[str, str]] | None = None,
        link_files: bool = True,
    ) -> None:
        self.input_files = input_files
        self.link_files = link_files
        super().__init__(
            inputs={
                ANADDB_INPUT_FILE_NAME: anaddb_input,
                "anaddb_input.json": json.dumps(
                    anaddb_input.as_dict(), cls=MontyEncoder
                ),
            }
        )

    def write_input(
        self,
        directory: str | Path,
        make_dir: bool = True,
        overwrite: bool = True,
        zip_inputs: bool = False,
    ) -> None:
        """Write Anaddb input files to a directory."""
        # TODO: do we allow zipping ? not sure if it really makes sense for abinit as
        #  the abinit input set also sets up links to previous files, sets up the
        #  indir, outdir and tmpdir, ...
        self.inputs["anaddb_input.json"] = json.dumps(
            jsanitize(self.anaddb_input.as_dict())
        )
        super().write_input(
            directory=directory,
            make_dir=make_dir,
            overwrite=overwrite,
            zip_inputs=zip_inputs,
        )
        del self.inputs["anaddb_input.json"]
        indir, _outdir, _tmpdir = self.set_workdir(workdir=directory)

        if self.input_files:
            out_to_in(
                out_files=self.input_files,
                indir=indir.path,
                link_files=self.link_files,
            )

    def validate(self) -> bool:
        """Validate the input set.

        Check that the input file exists and is a DDB file.
        """
        if not self.input_files:
            return False
        for _out_filepath, in_file in self.input_files:
            if not os.path.isfile(_out_filepath) or in_file != "in_DDB":
                return False
        return True

    @property
    def anaddb_input(self) -> AnaddbInput:
        """Get the AnaddbInput object."""
        return self[ANADDB_INPUT_FILE_NAME]

    @staticmethod
    def set_workdir(workdir: Path | str) -> tuple[Directory, Directory, Directory]:
        """Set up the working directory.

        This also sets up and creates standard input, output and temporary directories.
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
        """Set the values of anaddb variables.

        This sets the anaddb variables in the abipy AnaddbInput object.

        One can pass a dictionary mapping the anaddb variables to their values or
        the anaddb variables as keyword arguments. A combination of the two
        options is also allowed.

        Returns
        -------
        dict
            dictionary with the variables that have been added.
        """
        return self.anaddb_input.set_vars(*args, **kwargs)

    def remove_vars(self, keys: Iterable[str] | str, strict: bool = True) -> dict:
        """Remove the anaddb variables listed in keys.

        This removes the anaddb variables from the abipy AnaddbInput object.

        Parameters
        ----------
        keys
            string or list of strings with the names of the anaddb variables
            to be removed.
        strict
            whether to raise a KeyError if one of the anaddb variables to be
            removed is not present.

        Returns
        -------
        dict
            dictionary with the variables that have been removed.
        """
        return self.anaddb_input.remove_vars(keys=keys, strict=strict)

    def set_structure(self, structure: Any) -> Structure:
        """Set the structure for this input set.

        This basically forwards the setting of the structure to the abipy
        AnaddbInput object.
        """
        return self.anaddb_input.set_structure(structure)

    def deepcopy(self) -> AnaddbInputSet:
        """Deep copy of the input set."""
        return copy.deepcopy(self)


@dataclass
class AnaddbInputGenerator(AbinitMixinInputGenerator):
    """
    A class to generate Anaddb input sets.

    Parameters
    ----------
    calc_type
        A short description of the calculation type
    user_anaddb_settings
        A dictionary that allows to set any Abinit variable in the AbinitInput
        after it has been generated from the factory function. This will override
        any value or default previously set. Set a value to None to remove it
        from the input.
    prev_outputs_deps
        Defines the files that needs to be linked from previous calculations and
        are required for the execution of the current calculation.
        The format is a tuple where each element is a list of  "|" separated
        run levels (as defined in the AbinitInput object) followed by a colon and
        a list of "|" list of extensions of files that needs to be linked.
        The runlevel defines the type of calculations from which the file can
        be linked. An example is (f"{NSCF}:WFK",).
    force_gamma
        Force gamma centered kpoint generation.
    symprec
        Tolerance for symmetry finding, used for line mode band structure k-points.
    """

    factory: Callable | None = None
    calc_type: str = "anaddb"
    factory_kwargs: dict = field(default_factory=dict)
    user_abinit_settings: dict = field(default_factory=dict)
    prev_outputs_deps: str | tuple | None = field(
        default_factory=lambda: ("MRGDDB:DDB",)
    )

    def get_input_set(
        self,
        structure: Structure,
        prev_outputs: str | tuple | list | Path | None = None,
    ) -> AnaddbInputSet:
        """Generate an AnaddbInputSet object.

        Here we assume that and prev_outputs is
        a list of directories. We also assume there is an abinit_input.json file
        in each of these directories containing the AbinitInput object used to
        execute abinit.

        Parameters
        ----------
        structure : Structure
            Pymatgen Structure object.
        prev_outputs : str or Path or list or tuple
            Directory (as a str or Path) or list/tuple of directories (as a str
            or Path) needed as dependencies for the AbinitInputSet generated.
        """
        prev_outputs = self.check_format_prev_dirs(prev_outputs)

        input_files = []
        if prev_outputs is not None and not self.prev_outputs_deps:
            raise RuntimeError(
                f"Previous outputs not allowed for {self.__class__.__name__}."
            )
        irdvars, files = self.resolve_deps(
            prev_outputs, self.prev_outputs_deps, check_runlevel=False
        )
        input_files.extend(files)
        anaddb_input = self.get_anaddb_input(
            structure=structure,
            prev_outputs=prev_outputs,
        )

        anaddb_input.set_vars({"ddb_filepath": f'"{INDATA_PREFIX}_DDB"'})
        # anaddb_input["ddb_filepath"] = (f'"{INDATA_PREFIX}_DDB"',)
        # anaddb_input["outdata_prefix"] = (f'"{OUTDATA_PREFIX}"',)

        return AnaddbInputSet(
            anaddb_input=anaddb_input,
            input_files=input_files,
        )

    def get_anaddb_input(
        self,
        structure: Structure | None = None,
        prev_outputs: list[str] | None = None,
        abinit_settings: dict | None = None,
        factory_kwargs: dict | None = None,
    ) -> AnaddbInput:
        """
        Generate the AnaddbInput for the input set.

        Uses the defined factory function and additional parameters from user
        and subclasses.

        Parameters
        ----------
        structure
            A structure.
        prev_outputs
            A list of previous output directories.
        anaddb_settings
            A dictionary with additional abinit keywords to set.
        factory_kwargs
            A dictionary with additional factory keywords to set.

        Returns
        -------
            An AnaddbInput
        """
        total_factory_kwargs = dict(self.factory_kwargs) if self.factory_kwargs else {}

        if not self.prev_outputs_deps and prev_outputs:
            msg = f"Previous outputs not allowed for {self.__class__.__name__} "
            raise RuntimeError(msg)

        if structure:
            total_factory_kwargs["structure"] = structure
        if factory_kwargs:
            total_factory_kwargs.update(factory_kwargs)

        generated_input = self.factory(**total_factory_kwargs)

        if abinit_settings:
            generated_input.set_vars(abinit_settings)
        if self.user_abinit_settings:
            generated_input.set_vars(self.user_abinit_settings)

        return generated_input


def anaddbinp_dfpt_dte(
    structure: Structure, anaddb_kwargs: None | dict = None
) -> AnaddbInput:
    """
    Generate the AnaddbInput to retrieve information from the DTE.

    Parameters
    ----------
    structure
        A structure.
    anaddb_kwargs
        A dictionary with additional anaddb keywords to set.

    Returns
    -------
        An AnaddbInput

    """
    return AnaddbInput.dfpt(structure=structure, dte=True, anaddb_kwargs=anaddb_kwargs)


@dataclass
class AnaddbDfptDteInputGenerator(AnaddbInputGenerator):
    """
    A class to generate the AnaddbInput to retrieve information from the DTE.

    Parameters
    ----------
    factory
        A callable to generate the AnaddbInput for DTE DFPT.

    Returns
    -------
        An AnaddbInput

    """

    factory: Callable = anaddbinp_dfpt_dte
    # partial does not a __name__ so cannot jsanitize...
    # factory: Callable = partial(AnaddbInput.dfpt, dte=True)
