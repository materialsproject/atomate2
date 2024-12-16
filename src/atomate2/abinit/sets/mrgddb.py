"""Module defining base mrgddb input set and generator."""

from __future__ import annotations

import copy
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from abipy.abio.input_tags import DDE, DTE
from abipy.flowtk.utils import Directory
from pymatgen.io.core import InputSet

from atomate2.abinit.sets.base import AbinitMixinInputGenerator
from atomate2.abinit.utils.common import (
    INDIR_NAME,
    MRGDDB_INPUT_FILE_NAME,
    OUTDIR_NAME,
    TMPDIR_NAME,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

__all__ = ["MrgddbInputGenerator", "MrgddbInputSet"]

logger = logging.getLogger(__name__)


class MrgddbInputSet(InputSet):
    """
    A class to represent a set of Mrgddb inputs.

    Parameters
    ----------
    mrgddb_input
        An input (str) to mrgddb.
    input_files
        A list of input files needed for the calculation.
    """

    def __init__(
        self,
        mrgddb_input: str = None,
        input_files: Iterable[tuple[str, str]] | None = None,
    ) -> None:
        self.input_files = input_files
        super().__init__(
            inputs={
                MRGDDB_INPUT_FILE_NAME: mrgddb_input,
            }
        )

    def write_input(
        self,
        directory: str | Path,
        make_dir: bool = True,
        overwrite: bool = True,
        zip_inputs: bool = False,
    ) -> None:
        """Write Mrgddb input file to a directory."""
        # TODO: do we allow zipping ? not sure if it really makes sense for abinit as
        #  the abinit input set also sets up links to previous files, sets up the
        #  indir, outdir and tmpdir, ...
        super().write_input(
            directory=directory,
            make_dir=make_dir,
            overwrite=overwrite,
            zip_inputs=zip_inputs,
        )
        indir, outdir, tmpdir = self.set_workdir(workdir=directory)

    def validate(self) -> bool:
        """Validate the input set.

        Check that all input files exist and are DDB files.
        """
        if not self.input_files:
            return False
        for _out_filepath, in_file in self.input_files:
            if not os.path.isfile(_out_filepath) or in_file != "in_DDB":
                return False
        return True

    @property
    def mrgddb_input(self) -> str:
        """Get the Mrgddb input (str)."""
        return self[MRGDDB_INPUT_FILE_NAME]

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

    def deepcopy(self) -> MrgddbInputSet:
        """Deep copy of the input set."""
        return copy.deepcopy(self)


@dataclass
class MrgddbInputGenerator(AbinitMixinInputGenerator):
    """
    A class to generate Mrgddb input sets.

    Parameters
    ----------
    calc_type
        A short description of the calculation type
    prev_outputs_deps
        Defines the files that needs to be linked from previous calculations and
        are required for the execution of the current calculation.
        The format is a tuple where each element is a list of  "|" separated
        runlevels (as defined in the AbinitInput object) followed by a colon and
        a list of "|" list of extensions of files that needs to be linked.
        The runlevel defines the type of calculations from which the file can
        be linked. An example is (f"{NSCF}:WFK",).
    """

    calc_type: str = "mrgddb"
    prev_outputs_deps: tuple = (f"{DDE}:DDB", f"{DTE}:DDB")

    def get_input_set(
        self,
        prev_outputs: str | tuple | list | Path | None = None,
        workdir: str | Path | None = ".",
    ) -> MrgddbInputSet:
        """Generate an MrgddbInputSet object.

        Here we assume that prev_outputs is
        a list of directories.

        Parameters
        ----------
        prev_outputs : str or Path or list or tuple
            Directory (as a str or Path) or list/tuple of directories (as a str
            or Path) needed as dependencies for the MrgddbInputSet generated.
        """
        prev_outputs = self.check_format_prev_dirs(prev_outputs)

        input_files = []
        if prev_outputs is not None and not self.prev_outputs_deps:
            raise RuntimeError(
                f"Previous outputs not allowed for {self.__class__.__name__}."
            )
        irdvars, files = self.resolve_deps(prev_outputs, self.prev_outputs_deps)
        input_files.extend(files)
        mrgddb_input = self.get_mrgddb_input(
            prev_outputs=prev_outputs,
            workdir=workdir,
        )

        return MrgddbInputSet(
            mrgddb_input=mrgddb_input,
            input_files=input_files,
        )

    def get_mrgddb_input(
        self,
        prev_outputs: list[str] | None = None,
        workdir: str | Path | None = ".",
    ) -> str:
        """
        Generate the mrgddb input (str) for the input set.

        Parameters
        ----------
        prev_outputs
            A list of previous output directories.

        Returns
        -------
            A string
        """
        if not prev_outputs:
            raise RuntimeError(
                f"No previous_outputs. Required for {self.__class__.__name__}."
            )

        if not self.prev_outputs_deps and prev_outputs:
            msg = (
                f"Previous outputs not allowed for {self.__class__.__name__} "
                "Consider if get_input_set method "
                "can fit your needs instead."
            )
            raise RuntimeError(msg)

        irdvars, files = self.resolve_deps(prev_outputs, self.prev_outputs_deps)

        workdir = os.path.abspath(workdir)
        outdir = Directory(os.path.join(workdir, OUTDIR_NAME, "out_DDB"))

        generated_input = str(outdir)
        generated_input += "\n"
        generated_input += f"DDBs merged on {time.asctime()}"
        generated_input += "\n"
        generated_input += f"{len(files)}"
        for file_path, _ in files:
            generated_input += "\n"
            generated_input += f"{file_path}"

        return generated_input
