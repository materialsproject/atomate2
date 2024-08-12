"""Module defining base VASP input set and generator."""

from __future__ import annotations

import glob
import os
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from importlib.resources import files as get_mod_path
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any

from monty.io import zopen
from monty.serialization import loadfn
from pymatgen.electronic_structure.core import Magmom
from pymatgen.io.core import InputGenerator, InputSet
from pymatgen.io.vasp import Incar, Kpoints, Outcar, Poscar, Potcar, Vasprun
from pymatgen.io.vasp.sets.base import (
    BadInputSetWarning,
    get_valid_magmom_struct,
    get_vasprun_outcar,
)
from atomate2.jdftx.io.JDFTXInfile import JDFTXInfile, JDFTXStructure #TODO update this to the pymatgen module
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath

from atomate2 import SETTINGS
from atomate2.jdftx.io.inputs import JdftxInput

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    def __init__( 
        self,
        jdftxinput: JDFTXInfile,
        jdftxstructure: JDFTXStructure
    ) -> None:
        self.jdftxstructure = jdftxstructure
        self.jdftxinput = jdftxinput

    def write_input(
        self,
        directory: str | Path,
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
        infile = "inputs.in"
        directory = Path(directory)
        if make_dir:
            os.makedirs(directory, exist_ok=True)

        if not overwrite and (directory / infile).exists():
            raise FileExistsError(f"{directory / infile} already exists.")

        self.jdftxinput.write_file(filename=(directory / infile))

    @staticmethod
    def from_directory(
        directory: str | Path, 
    ) -> JdftxInputSet:
        """Load a set of JDFTx inputs from a directory.

        Parameters
        ----------
        directory
            Directory to read VASP inputs from.
        optional_files
            Optional files to read in as well as a dict of {filename: Object class}.
            Object class must have a static/class method from_file.
        """
        directory = Path(directory)
        jdftxinput = JDFTXInfile.from_file(directory/"input.in") #jdftxinputs is a JDFTXInfile object
        jdftxstructure = jdftxinput.to_JDFTXStructure()
        return JdftxInputSet(jdftxinput=jdftxinput, jdftxstructure=jdftxstructure)
    


@dataclass
class JdftxInputGenerator(InputGenerator):
    """
    A class to generate JDFTx input sets.

   """

    # copy _BASE_JDFTX_SET to ensure each class instance has its own copy
    # otherwise in-place changes can affect other instances
    user_settings: dict = field(default_factory=dict)
    default_settings: dict = field(default_factory=lambda: _BASE_JDFTX_SET)

    def __post_init__(self) -> None:
        """Post init formatting of arguments."""
        self.settings = self.default_settings.copy()
        self.settings.update(self.user_settings)
        self._apply_settings(self.settings)

    def _apply_settings(self, settings: dict[str, Any]): #added this in case we want settings to be individual attributes
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

        return JdftxInputSet(
            jdftxinput=jdftxinput, jdftxstructure=jdftx_structure
        )

