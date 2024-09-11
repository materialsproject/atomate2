"""Module defining base JDFTx input set and generator."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib.resources import files as get_mod_path
from pathlib import Path
from typing import TYPE_CHECKING, Any

from monty.serialization import loadfn
from pymatgen.io.core import InputGenerator, InputSet
from pymatgen.util.typing import PathLike

from atomate2.jdftx.io.JDFTXInfile import (  # TODO update this to the pymatgen module
    JDFTXInfile,
    JDFTXStructure,
)

if TYPE_CHECKING:
    from pymatgen.core import Structure

_BASE_JDFTX_SET = loadfn(get_mod_path("atomate2.jdftx.sets") / "BaseJdftxSet.yaml")


class JdftxInputSet(InputSet):
    """
    A class to represent a JDFTx input file as a JDFTx InputSet.

    Parameters
    ----------
    jdftxinput
        A JdftxInput object
    """

    def __init__(self, jdftxinput: JDFTXInfile, jdftxstructure: JDFTXStructure) -> None:
        self.jdftxstructure = jdftxstructure
        self.jdftxinput = jdftxinput

    def write_input(
        self,
        directory: str | Path,
        infile: PathLike = "inputs.in",  # TODO I don't think this should be optional
        make_dir: bool = True,
        overwrite: bool = True,
    ) -> None:
        """Write JDFTx input file to a directory.

        Parameters
        ----------
        directory
            Directory to write input files to.
        make_dir
            Whether to create the directory if it does not already exist.
        overwrite
            Whether to overwrite an input file if it already exists.
        """
        directory = Path(directory)
        if make_dir:
            os.makedirs(directory, exist_ok=True)

        if not overwrite and (directory / infile).exists():
            raise FileExistsError(f"{directory / infile} already exists.")

        jdftxinput = condense_jdftxinputs(self.jdftxinput, self.jdftxstructure)

        jdftxinput.write_file(filename=(directory / infile))

    @staticmethod
    def from_directory(
        directory: str | Path,
    ) -> JdftxInputSet:
        """Load a set of JDFTx inputs from a directory.

        Parameters
        ----------
        directory
            Directory to read JDFTx inputs from.
        """
        directory = Path(directory)
        jdftxinput = JDFTXInfile.from_file(directory / "input.in")
        jdftxstructure = jdftxinput.to_JDFTXStructure(jdftxinput)
        return JdftxInputSet(jdftxinput=jdftxinput, jdftxstructure=jdftxstructure)


@dataclass
class JdftxInputGenerator(InputGenerator):
    """A class to generate JDFTx input sets."""

    # copy _BASE_JDFTX_SET to ensure each class instance has its own copy
    # otherwise in-place changes can affect other instances
    user_settings: dict = field(default_factory=dict)
    default_settings: dict = field(default_factory=lambda: _BASE_JDFTX_SET)

    def __post_init__(self) -> None:
        """Post init formatting of arguments."""
        self.settings = self.default_settings.copy()
        self.settings.update(self.user_settings)
        self._apply_settings(self.settings)

    def _apply_settings(
        self, settings: dict[str, Any]
    ) -> None:  # settings as attributes
        for key, value in settings.items():
            setattr(self, key, value)

    def get_input_set(
        self,
        structure: Structure = None,
    ) -> JdftxInputSet:
        """Get a JDFTx input set for a structure.

        Parameters
        ----------
        structure
            A Pymatgen Structure.

        Returns
        -------
        JdftxInputSet
            A JDFTx input set.
        """
        jdftx_structure = JDFTXStructure(structure)
        jdftxinputs = self.settings
        jdftxinput = JDFTXInfile.from_dict(jdftxinputs)

        return JdftxInputSet(jdftxinput=jdftxinput, jdftxstructure=jdftx_structure)


def condense_jdftxinputs(
    jdftxinput: JDFTXInfile, jdftxstructure: JDFTXStructure
) -> JDFTXInfile:
    """
    Function to combine a JDFTXInputs class with calculation
    settings and a JDFTxStructure that defines the structure
    into one JDFTXInputs instance.

    Parameters
    ----------
        jdftxinput: JDFTXInfile
            A JDFTXInfile object with calculation settings.

        jdftxstructure: JDFTXStructure
            A JDFTXStructure object that defines the structure.

    Returns
    -------
        JDFTXInfile
            A JDFTXInfile that includes the calculation
            parameters and input structure.
    """
    condensed_inputs = jdftxinput + JDFTXInfile.from_str(jdftxstructure.get_str())
    return condensed_inputs
