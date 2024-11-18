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
from pymatgen.util.typing import Kpoint, Tuple3Floats

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
        coulomb_truncation (bool) = False: 
            Whether to use coulomb truncation and calculate the coulomb 
            truncation center. Only works for molecules and slabs.
        auto_kpoint_density:
            Reciprocal k-point density for automatic k-point calculation. If 
            k-points are specified in user_settings, they will not be 
            overridden.
        config_dict (dict): The config dictionary used to set input paremeters
            used in the calculation of JDFTx tags.
        default_settings: Default JDFTx settings.
    """

    # copy _BASE_JDFTX_SET to ensure each class instance has its own copy
    # otherwise in-place changes can affect other instances
    user_settings: dict = field(default_factory=dict)
    coulomb_truncation: bool = False
    auto_kpoint_density: int = 1000
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

        self.set_kgrid(structure=structure)
        self.set_coulomb_truncation(structure=structure)

        jdftxinputs = self.settings
        jdftxinput = JDFTXInfile.from_dict(jdftxinputs)
        return JdftxInputSet(jdftxinput=jdftxinput, jdftxstructure=jdftx_structure)

    def set_kgrid(
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
        # never override k grid definition in user settings
        if "kpoint-folding" in self.user_settings.keys():
            return 
        # calculate k-grid with k-point density
        else:
            kpoints = Kpoints.automatic_density(
                structure=structure,
                kppa=self.auto_kpoint_density
                )
            kpoints = kpoints.kpts
            if self._is_surface_calc():
                kpoints[-1] = 1
            elif self._is_molecule_calc():
                kpoints = [1, 1, 1]
            kpoint_update = {"kpoint-folding": 
                {
                "n0": kpoints[0], 
                "n1": kpoints[1],
                "n2": kpoints[2],
                }
            }
            self.settings.update(kpoint_update)
            return

    def set_coulomb_truncation(
            self, 
            structure:Structure, 
            jdftxinput:JDFTXInfile
        ) -> JDFTXInfile:
        """
        Set coulomb-truncation for JDFTXInfile.

        Check config_dict to determine whether to use coulomb-truncation. 
        Specify "coulomb-truncation": bool in self.user_settings to override 
        config_dict. Calculate center of mass of the unit cell, set the 
        coulomb-truncation tag in the JDFTXInfile, and return the JDFTXInfile.

        Parameters
        ----------
        structure
            A pymatgen structure
        jdftxinputs
            A pymatgen.io.jdftx.inputs.JDFTXInfile object

        Returns
        -------
        jdftxinputs
            A pymatgen.io.jdftx.inputs.JDFTXInfile object
        
        """
        if self._is_surface_calc():
            
        elif self._is_molecule_calc():
        
        else:
            jdftxinput["coulomb-interaction"] =  "Periodic"
        

    def _is_surface_calc(
            self
    ) -> bool:
        """
        Check if calculation is for surface.

        First set coulomb-truncation based on config_dict. Override with 
        self.settings for coulomb-interaction type Slab. 
        Override with self.user_coulomb_settings.

        Returns
        -------
            A boolean where True means this is a surface calculation
        """
        is_surface = False
        if "coulomb-interaction" in self.user_settings():
            if self.user_settings["coulomb-interaction"]["truncationType"] == "Slab":
                is_surface = True
        else:
            False
        
    def _is_molecule_calc(
            self
    ) -> bool:
         """
        Check if calculation is for surface.

        First set coulomb-truncation based on config_dict. Override with 
        self.settings for coulomb-interaction type Slab. 
        Override with self.user_coulomb_settings.

        Returns
        -------
            A boolean where True means this is a surface calculation
        """

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
