"""Module defining base MRGDVDB input set and generator."""

from __future__ import annotations

import copy
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from abipy.abio.input_tags import DDE, DTE, PH_Q_PERT
from abipy.flowtk.utils import Directory
from pymatgen.io.core import InputSet

from atomate2.abinit.sets.base import AbinitMixinInputGenerator, set_workdir
from atomate2.abinit.utils.common import MRGDV_INPUT_FILE_NAME, OUTDIR_NAME

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

__all__ = ["MrgdvInputGenerator", "MrgdvInputSet"]

logger = logging.getLogger(__name__)


class MrgdvInputSet(InputSet):
    """
    A class to represent a set of MRGDV inputs.

    MRGDV is an ABINIT utility for merging potential derivative database
    (POT) files from multiple perturbation calculations.

    Attributes
    ----------
    mrgdv_input : str or None
        An input string for the MRGDV utility.
    input_files : Iterable[tuple[str, str]] or None
        A list of (output_filepath, input_filename) tuples for POT files to merge.
    """

    def __init__(
        self,
        mrgdv_input: str | None = None,
        input_files: Iterable[tuple[str, str]] | None = None,
    ) -> None:
        """
        Initialize an MrgdvInputSet.

        Parameters
        ----------
        mrgdv_input : str or None
            An input string for the MRGDV utility. Default is None.
        input_files : Iterable[tuple[str, str]] or None
            List of (output_filepath, input_filename) tuples for files to merge.
            Default is None.
        """
        self.input_files = input_files
        super().__init__(
            inputs={
                MRGDV_INPUT_FILE_NAME: mrgdv_input,
            }
        )

    def write_input(
        self,
        directory: str | Path,
        make_dir: bool = True,
        overwrite: bool = True,
        zip_inputs: bool = False,
    ) -> None:
        """
        Write MRGDV input file to a directory.

        Creates the necessary directory structure including standard input,
        output, and temporary directories.

        Parameters
        ----------
        directory : str or Path
            The directory to write the input files to.
        make_dir : bool
            Whether to create the directory if it does not exist. Default is True.
        overwrite : bool
            Whether to overwrite existing files. Default is True.
        zip_inputs : bool
            Whether to zip the input files. Default is False.

        Notes
        -----
        The zip_inputs functionality may not be fully compatible with ABINIT
        workflows as the input set creates symbolic links to previous calculation
        files and sets up specific directory structures (indir, outdir, tmpdir).
        """
        super().write_input(
            directory=directory,
            make_dir=make_dir,
            overwrite=overwrite,
            zip_inputs=zip_inputs,
        )
        _indir, _outdir, _tmpdir = set_workdir(workdir=directory)

    def validate(self) -> bool:
        """
        Validate the input set.

        Checks that all input files exist and are POT files with the correct
        input filename prefix.

        Returns
        -------
        bool
            True if all input files are valid POT files, False otherwise.
        """
        if not self.input_files:
            return False
        for _out_filepath, in_file in self.input_files:
            if not os.path.isfile(_out_filepath) or not in_file.startswith("in_POT"):
                return False
        return True

    @property
    def mrgdv_input(self) -> str:
        """
        Get the MRGDV input string.

        Returns
        -------
        str
            The MRGDV input string.
        """
        return self[MRGDV_INPUT_FILE_NAME]

    def deepcopy(self) -> MrgdvInputSet:
        """
        Create a deep copy of the input set.

        Returns
        -------
        MrgdvInputSet
            A deep copy of this MrgdvInputSet object.
        """
        return copy.deepcopy(self)


@dataclass
class MrgdvInputGenerator(AbinitMixinInputGenerator):
    """
    A class to generate MRGDV input sets.

    MRGDV is an ABINIT utility that merges potential derivative database
    (POT) files from multiple perturbation calculations (DDE, DTE, phonons).

    Attributes
    ----------
    calc_type : str
        A short description of the calculation type. Default is "mrgdv".
    prev_outputs_deps : tuple
        Defines the files that need to be linked from previous calculations.
        The format is a tuple where each element is a list of "|" separated
        run levels (as defined in the AbinitInput object) followed by a colon and
        a list of "|" separated extensions of files that need to be linked.
        The run level defines the type of calculations from which the file can
        be linked.
        Default is (f"{DDE}:POT", f"{DTE}:POT", f"{PH_Q_PERT}:POT").
    """

    calc_type: str = "mrgdv"
    prev_outputs_deps: tuple = (f"{DDE}:POT", f"{DTE}:POT", f"{PH_Q_PERT}:POT")

    def get_input_set(
        self,
        prev_outputs: str | tuple | list | Path | None = None,
        workdir: str | Path | None = ".",
    ) -> MrgdvInputSet:
        """
        Generate an MrgdvInputSet object.

        Collects POT files from previous calculations and creates the input
        for the MRGDV utility to merge them.

        Parameters
        ----------
        prev_outputs : str or Path or list or tuple or None
            Directory (as a str or Path) or list/tuple of directories (as a str
            or Path) needed as dependencies for the MrgdvInputSet generated.
            Default is None.
        workdir : str or Path or None
            Working directory for the calculation. Default is ".".

        Returns
        -------
        MrgdvInputSet
            An MrgdvInputSet object ready to be written and executed.
        """
        prev_outputs = self.check_format_prev_dirs(prev_outputs)

        input_files = []
        if prev_outputs is not None and not self.prev_outputs_deps:
            raise RuntimeError(
                f"Previous outputs not allowed for {self.__class__.__name__}."
            )
        _irdvars, files = self.resolve_deps(prev_outputs, self.prev_outputs_deps)
        input_files.extend(files)
        mrgdv_input = self.get_mrgdv_input(
            prev_outputs=prev_outputs,
            workdir=workdir,
        )

        return MrgdvInputSet(
            mrgdv_input=mrgdv_input,
            input_files=input_files,
        )

    def get_mrgdv_input(
        self,
        prev_outputs: list[str] | None = None,
        workdir: str | Path | None = ".",
    ) -> str:
        """
        Generate the MRGDV input string for the input set.

        Creates an input string for the MRGDV utility that specifies the
        output DVDB file location, a timestamp, and the list of POT files
        to be merged.

        Parameters
        ----------
        prev_outputs : list[str] or None
            A list of previous output directories. Default is None.
        workdir : str or Path or None
            Working directory for the calculation. Default is ".".

        Returns
        -------
        str
            A string containing the MRGDV input specification.

        Raises
        ------
        RuntimeError
            If no previous outputs are provided or if previous outputs are
            not allowed for this generator.
        """
        if not prev_outputs:
            raise RuntimeError(
                f"No previous_outputs. Required for {self.__class__.__name__}."
            )

        if not self.prev_outputs_deps and prev_outputs:
            msg = (
                f"Previous outputs not allowed for {self.__class__.__name__}. "
                "Consider if get_input_set method "
                "can fit your needs instead."
            )
            raise RuntimeError(msg)

        _irdvars, files = self.resolve_deps(prev_outputs, self.prev_outputs_deps)

        workdir = os.path.abspath(workdir)
        outdir = Directory(os.path.join(workdir, OUTDIR_NAME, "out_dvdb"))

        generated_input = str(outdir)
        generated_input += "\n"
        generated_input += f"dvdbs merged on {time.asctime()}"
        generated_input += "\n"
        generated_input += f"{len(files)}"
        for file_path, _ in files:
            generated_input += "\n"
            generated_input += f"{file_path}"

        return generated_input
