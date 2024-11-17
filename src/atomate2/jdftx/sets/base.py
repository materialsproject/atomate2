"""Module defining base JDFTx input set and generator."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib.resources import files as get_mod_path
from pathlib import Path
from typing import TYPE_CHECKING, Any

from monty.serialization import loadfn
from pymatgen.io.core import InputGenerator, InputSet
from pymatgen.io.jdftx.inputs import JDFTXInfile, JDFTXStructure
from pymatgen.io.vasp import Kpoints
from pymatgen.util.typing import Kpoint

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.util.typing import PathLike

_BASE_JDFTX_SET = loadfn(get_mod_path("atomate2.jdftx.sets") / "BaseJdftxSet.yaml")
_BEAST_CONFIG = loadfn(get_mod_path("atomate2.jdftx.sets") / "BeastConfig.yaml")
FILE_NAMES = {"in": "init.in", "out": "jdftx.out"}


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
        infile: PathLike = FILE_NAMES["in"],
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
    """A class to generate JDFTx input sets.
    
    Args:
        user_settings (dict): User JDFTx settings. This allows the user to
            override the default JDFTx settings loaded in the default_settings
            argument.
        user_kpoint_settings (dict): User settings for overriding the
            calculation of the k-point grid. The user can set the k-point
            density with "kpoint-density". 
        config_dict (dict): The config dictionary used to set input paremeters
            used in the calculation of JDFTx tags.
        default_settings: Default JDFTx settings.
    """

    # copy _BASE_JDFTX_SET to ensure each class instance has its own copy
    # otherwise in-place changes can affect other instances
    user_settings: dict = field(default_factory=dict)
    user_kpoint_settings: dict = field(default_factory=dict)
    config_dict: dict = field(default_factory=lambda: _BEAST_CONFIG)
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

    def get_kgrid(
        self, 
        structure:Structure
    ) -> Kpoint:
        """Get k-point grid.
        
        Parameters
        ----------
        structure
            A pymatgen structure.
        
        Returns
        -------
        Kpoints
            A tuple of integers specifying the k-point grid.
        """
        if self.user_kpoint_settings == {}:
            kpoints = Kpoints.automatic_density(
                structure=structure,
                kppa=self.config_dict.get("kpoint-density")
                )
        else:
            if "kpoint-density" in self.user_kpoint_settings.keys():
                kpoints = Kpoints.automatic_density(
                    structure=structure,
                    kppa=self.user_kpoint_settings["kpoint-density"]
                    )
        colomb_interaction = self.settings

def condense_jdftxinputs(
    jdftxinput: JDFTXInfile, jdftxstructure: JDFTXStructure
) -> JDFTXInfile:
    """
    Combine JDFTXInfile and JDFTxStructure into complete JDFTXInfile.

    Function combines a JDFTXInfile class with calculation
    settings and a JDFTxStructure that defines the structure
    into one JDFTXInfile instance.

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
    return jdftxinput + JDFTXInfile.from_str(jdftxstructure.get_str())
