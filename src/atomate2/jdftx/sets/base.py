"""Module defining base JDFTx input set and generator."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib.resources import files as get_mod_path
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union
import numpy as np
from scipy import constants as const

from monty.serialization import loadfn
from pymatgen.core import Structure
from pymatgen.core.units import eV_to_Ha, ang_to_bohr
from pymatgen.io.core import InputGenerator, InputSet
from pymatgen.io.jdftx.inputs import JDFTXInfile, JDFTXStructure
from pymatgen.io.vasp import Kpoints
from pymatgen.util.typing import Kpoint, Tuple3Floats
from atomate2 import SETTINGS


if TYPE_CHECKING:
    from pymatgen.util.typing import PathLike

_BASE_JDFTX_SET = loadfn(get_mod_path("atomate2.jdftx.sets") / "BaseJdftxSet.yaml")
_BEAST_CONFIG = loadfn(get_mod_path("atomate2.jdftx.sets") / "BeastConfig.yaml")
_PSEUDO_CONFIG = loadfn(get_mod_path("atomate2.jdftx.sets") / "PseudosConfig.yaml")
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
        auto_kpoint_density (int) = 1000:
            Reciprocal k-point density for automatic k-point calculation. If 
            k-points are specified in user_settings, they will not be 
            overridden.
        potential (None, float) = None:
            Potential vs SHE for GC-DFT calculation.
        calc_type (str) = "bulk":
            Type of calculation used for setting input parameters. Options are:
            ["bulk", "surface", "molecule"].
        pseudopotentials (str) = "GBRV"
        config_dict (dict): The config dictionary used to set input paremeters
            used in the calculation of JDFTx tags.
        default_settings: Default JDFTx settings.
    """

    # copy _BASE_JDFTX_SET to ensure each class instance has its own copy
    # otherwise in-place changes can affect other instances
    user_settings: dict = field(default_factory=dict)
    coulomb_truncation: bool = False
    auto_kpoint_density: int = 1000
    potential: Union[None, float] = None
    calc_type: str = "bulk"
    pseudopotentials: str = "GBRV"
    config_dict: dict = field(default_factory=lambda: _BEAST_CONFIG)
    default_settings: dict = field(default_factory=lambda: _BASE_JDFTX_SET)

    def __post_init__(self) -> None:
        """Post init formatting of arguments."""
        calc_type_options = ["bulk", "surface", "molecule"]
        if self.calc_type not in calc_type_options:
            raise ValueError(
                f"calc type f{self.calc_type} not in list of supported calc "
                "types: {calc_type_options}."
            )
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
        print(self.settings)
        self.set_kgrid(structure=structure)
        self.set_coulomb_interaction(structure=structure)
        self.set_nbands(structure=structure)
        self.set_mu()
        self.set_pseudos()
        self._apply_settings(self.settings)

        jdftx_structure = JDFTXStructure(structure)
        jdftxinputs = self.settings
        jdftxinput = JDFTXInfile.from_dict(jdftxinputs)

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
            kpoints = kpoints.kpts[0]
            if self.calc_type == "surface":
                kpoints = (kpoints[0], kpoints[1], 1)
            elif self.calc_type == "molecule":
                kpoints = (1,1,1)
            kpoint_update = {"kpoint-folding": 
                {
                "n0": kpoints[0], 
                "n1": kpoints[1],
                "n2": kpoints[2],
                }
            }
            self.settings.update(kpoint_update)
            return

    def set_coulomb_interaction(
            self, 
            structure:Structure, 
    ) -> JDFTXInfile:
        """
        Set coulomb-interaction and coulomb-truncation for JDFTXInfile.

        Description

        Parameters
        ----------
        structure
            A pymatgen structure

        Returns
        -------
        jdftxinputs
            A pymatgen.io.jdftx.inputs.JDFTXInfile object
        
        """
        if "coulomb-interaction" in self.settings.keys():
            return
        if self.calc_type == "bulk":
            self.settings["coulomb-interaction"] = {
                "truncationType": "Periodic",
            }
            return
        elif self.calc_type == "surface":
            self.settings["coulomb-interaction"] = {
                "truncationType": "Slab",
                "dir": "001",
            }
        elif self.calc_type == "molecule":
            self.settings["coulomb-interaction"] = {
                "truncationType": "Isolated",
            }
        com = center_of_mass(structure=structure)
        com = com.T @ structure.lattice.matrix * ang_to_bohr
        self.settings["coulomb-truncation-embed"] = {
            "c0": com[0][0],
            "c1": com[0][1],
            "c2": com[0][2],
        }
        return
    
    def set_nbands(
            self, 
            structure:Structure
    ) -> None:
        """
        Set number of bands in DFT calculation.
        """
        nelec = 0
        for atom in structure.species:
            nelec += _PSEUDO_CONFIG[self.pseudopotentials][str(atom)]
        nbands_add = int(nelec / 2) + 10
        nbands_mult = int((nelec/2)) * _BEAST_CONFIG["bands_multiplier"]
        self.settings["nbands"] = max(nbands_add, nbands_mult)
        return
             
    def set_pseudos(
            self,
    ) -> None:
        """
        Set ion-species tag corresponding to pseudopotentials
        """
        if SETTINGS.JDFTX_PSEUDOS_DIR != None:
            psuedos_str = str(
                Path(SETTINGS.JDFTX_PSEUDOS_DIR) / Path(self.pseudopotentials)
            )
        else:
            pseudos_str = self.pseudopotentials
        add_tags = []
        for suffix in _PSEUDO_CONFIG[self.pseudopotentials]["suffixes"]:
            add_tags.append(pseudos_str+"/$ID"+suffix)
        # do not override pseudopotentials in settings
        if "ion-species" in self.settings.keys():
            return 
        else:
            self.settings["ion-species"] = add_tags
            return
        
    def set_mu(self) -> None:
        """
        Set absolute electron chemical potential (fermi level) for GC-DFT.
        """
        # never override mu in settings
        if "target-mu" in self.settings.keys():
            return
        elif self.potential == None:
            return
        else:
            solvent_model = self.settings["pcm-variant"]
            ashep = _BEAST_CONFIG["ASHEP"][solvent_model]
            # calculate absolute potential in Hartree
            mu = -(ashep - self.potential) / eV_to_Ha 
            self.settings["target-mu"] = {"mu": mu}
        return

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

def center_of_mass(structure:Structure) -> np.ndarray: 
    weights = [site.species.weight for site in structure]
    com = np.average(structure.frac_coords, weights=weights, axis=0)
    return com[..., np.newaxis]