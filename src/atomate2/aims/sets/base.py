"""Module defining base FHI-aims input set and generator."""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Sequence
from warnings import warn

import numpy as np
from ase.calculators.aims import Aims
from ase.constraints import FixScaledParametricRelations
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.io.core import InputFile, InputGenerator, InputSet

from atomate2.aims.io.parsers import AimsParseError, read_aims_output
from atomate2.aims.utils.common import (
    CONTROL_FILE_NAME,
    GEOMETRY_FILE_NAME,
    PARAMS_JSON_FILE_NAME,
    TMPDIR_NAME,
    cwd,
)
from atomate2.aims.utils.msonable_atoms import MSONableAtoms

if TYPE_CHECKING:
    from pathlib import Path

    from ase.cell import Cell

DEFAULT_AIMS_PROPERTIES = [
    "energy",
    "free_energy",
    "forces",
    "stress",
    "stresses",
    "dipole",
    "magmom",
]

logger = logging.getLogger(__name__)


@dataclass
class AimsInputFile(InputFile):
    """The input file for an FHI-aims calculation.

    Parameters
    ----------
    _content_str: str
        The contents of the input file as a string
    """

    _content_str: str = ""

    def get_string(self) -> str:
        """Get the contents of the input file.

        Returns
        -------
        The contents of the input file
        """
        return self._content_str

    def get_str(self) -> str:
        """Get the contents of the input file.

        Returns
        -------
        The contents of the input file
        """
        return self._content_str

    @classmethod
    def from_string(cls, contents: str):
        """Create an input file from the contents string.

        Parameters
        ----------
        contents: str
            The contents of the input file
        """
        return cls(contents)

    @classmethod
    def from_str(cls, contents: str):
        """Create an input file from the contents string.

        Parameters
        ----------
        contents: str
            The contents of the input file
        """
        return cls(contents)


class AimsInputSet(InputSet):
    """A class to represent a set of Aims inputs."""

    def __init__(
        self,
        parameters: dict[str, Any],
        atoms: MSONableAtoms,
        properties: Sequence[str] = ("energy", "free_energy"),
    ):
        """Construct the AimsInputSet.

        Parameters
        ----------
        parameters: Dict[str, Any]
            The ASE parameters object for the calculation
        atoms: .MSONableAtoms
            The atoms objects to create the inputs for
        properties: Sequence[str]
            The properties to calculate for the calculation
        """
        self._parameters = parameters
        self._atoms = MSONableAtoms(atoms)
        self._properties = properties

        aims_control_in, aims_geometry_in = self.get_input_files()
        super().__init__(
            inputs={
                CONTROL_FILE_NAME: aims_control_in,
                GEOMETRY_FILE_NAME: aims_geometry_in,
                PARAMS_JSON_FILE_NAME: json.dumps(
                    self._parameters, indent=2, cls=MontyEncoder
                ),
            }
        )

    def get_input_files(self) -> tuple[str, str]:
        """Get the input file contents for the calculation.

        Returns
        -------
        The contents of the control.in and geometry.in file
        """
        property_flags = {
            "forces": "compute_forces",
            "stress": "compute_analytical_stress",
            "stresses": "compute_heat_flux",
        }
        updated_params = dict(**self._parameters)
        for prop in self._properties:
            aims_name = property_flags.get(prop, None)
            if aims_name is not None:
                updated_params[aims_name] = True

        with cwd(TMPDIR_NAME, mkdir=True, rmdir=True):
            aims_calc = Aims(atoms=self._atoms, **updated_params)
            scaled = np.any(self._atoms.pbc)
            geo_constrain = any(
                [False]
                + [
                    isinstance(const, FixScaledParametricRelations)
                    for const in self._atoms.constraints
                ]
            )
            aims_calc.write_input(
                atoms=self._atoms,
                properties=self._properties,
                scaled=scaled,
                geo_constrain=geo_constrain,
            )
            aims_control_in = AimsInputFile.from_file("control.in")
            aims_geometry_in = AimsInputFile.from_file("geometry.in")
        return aims_control_in, aims_geometry_in

    @property
    def control_in(self) -> str:
        """Get the control.in file contents."""
        return self[CONTROL_FILE_NAME]

    @property
    def geometry_in(self) -> str:
        """Get the geometry.in file contents."""
        return self[GEOMETRY_FILE_NAME]

    @property
    def parameters_json(self) -> str:
        """Get the JSON representation of the parameters dict."""
        return self[PARAMS_JSON_FILE_NAME]

    def set_parameters(self, *args, **kwargs) -> dict[str, Any]:
        """Set the parameters object for the AimsTemplate.

        This sets the parameters object that is passed to an AimsTempalte and
        resets the control.in file

        One can pass a dictionary mapping the aims variables to their values or
        the aims variables as keyword arguments. A combination of the two
        options is also allowed.

        Returns
        -------
        dict[str, Any]
            dictionary with the variables that have been added.
        """
        self._parameters.clear()
        for arg in args:
            self._parameters.update(arg)

        self._parameters.update(kwargs)

        aims_control_in, _ = self.get_input_files()
        self.inputs[CONTROL_FILE_NAME] = aims_control_in
        self.inputs[PARAMS_JSON_FILE_NAME] = json.dumps(
            self._parameters, indent=2, cls=MontyEncoder
        )
        self.__dict__.update(self.inputs)

        return self._parameters

    def remove_parameters(
        self, keys: Iterable[str] | str, strict: bool = True
    ) -> dict[str, Any]:
        """Remove the aims parameters listed in keys.

        This removes the aims variables from the parameters object.

        Parameters
        ----------
        keys: Iterable[str] or str
            string or list of strings with the names of the aims parameters
            to be removed.
        strict: bool
            whether to raise a KeyError if one of the aims parameters to be
            removed is not present.

        Returns
        -------
        Dictionary with the variables that have been removed.
        """
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            if strict and key not in self._parameters:
                raise ValueError(f"The key ({key}) is not in self._parameters")

            if key not in self._parameters:
                continue

            del self._parameters[key]

        return self.set_parameters(**self._parameters)

    def set_atoms(self, atoms: MSONableAtoms):
        """Set the atoms object for this input set.

        Parameters
        ----------
        atoms: .MSONableAtoms
            The new atoms for the calculation
        """
        self._atoms = MSONableAtoms(atoms)

        _, aims_geometry_in = self.get_input_files()
        self.inputs[GEOMETRY_FILE_NAME] = aims_geometry_in
        self.__dict__.update(self.inputs)

    def deepcopy(self):
        """Deep copy of the input set."""
        return copy.deepcopy(self)


@dataclass
class AimsInputGenerator(InputGenerator):
    """
    A class to generate Aims input sets.

    Parameters
    ----------
    user_parameters: Dict[str, Any]
        Updates the default parameters for the FHI-aims calculator
    user_kpoints_settings: Dict[str, Any]
        The settings used to create the k-grid parameters for FHI-aims
    """

    user_parameters: dict[str, Any] = field(default_factory=dict)
    user_kpoints_settings: dict[str, Any] = field(default_factory=dict)

    def get_input_set(  # type: ignore
        self,
        atoms: MSONableAtoms = None,
        prev_dir: str | Path = None,
        properties: list[str] = None,
    ) -> AimsInputSet:
        """Generate an AimsInputSet object.

        Parameters
        ----------
        atoms : .MSONableAtoms
            ASE Atoms object to generate the input set for.
        prev_dir: str or Path
            Path to the previous working directory
        properties: list[str]
            System properties that are being calculated

        Returns
        -------
        The input set for the calculation of atoms
        """
        prev_atoms, prev_parameters, prev_results = self._read_previous(prev_dir)
        atoms = atoms if atoms is not None else prev_atoms

        parameters = self._get_input_parameters(atoms, prev_parameters)
        properties = self._get_properties(properties, parameters, prev_results)

        return AimsInputSet(parameters=parameters, atoms=atoms, properties=properties)

    def _read_previous(
        self, prev_dir: str | Path = None
    ) -> tuple[MSONableAtoms, dict[str, Any], dict[str, list[float]]]:
        """Read in previous results.

        Parameters
        ----------
        prev_dir: str or Path
            The previous directory for the calculation
        """
        prev_atoms = None
        prev_parameters = {}
        prev_results = {}

        if prev_dir:
            # strip hostname from the directory (not good, works only with run_locally.
            # Should be checked with Fireworks, will not for sure work with
            # jobflow_remote)
            split_prev_dir = str(prev_dir).split(":")[-1]
            with open(f"{split_prev_dir}/parameters.json") as param_file:
                prev_parameters = json.load(param_file, cls=MontyDecoder)

            try:
                prev_atoms = read_aims_output(f"{split_prev_dir}/aims.out")
                prev_results = prev_atoms.calc.results
            except (IndexError, AimsParseError):
                pass

        return prev_atoms, prev_parameters, prev_results

    def _get_properties(
        self,
        properties: list[str] = None,
        parameters: dict[str, Any] = None,
        prev_results: dict[str, list[float]] = None,
    ) -> list[str]:
        """Get the properties to calculate.

        Parameters
        ----------
        properties: list[str]
            The currently requested properties
        parameters: dict[str, Any]
            The parameters for this calculation
        prev_results: dict[str, list[float]]
            The previous calculation results

        Returns
        -------
        The list of properties to calculate
        """
        if properties is None:
            properties = ["energy", "free_energy"]

        for key in prev_results:
            if key not in properties and key in DEFAULT_AIMS_PROPERTIES:
                properties.append(key)

        if "compute_forces" in parameters and "forces" not in properties:
            properties.append("forces")
        if "compute_heat_flux" in parameters and "stresses" not in properties:
            properties.append("stress")
            properties.append("stresses")
        if "stress" not in properties and (
            ("compute_analytical_stress" in parameters)
            or ("compute_numerical_stress" in parameters)
            or ("compute_heat_flux" in parameters)
        ):
            properties.append("stress")

        return properties

    def _get_input_parameters(
        self, atoms: MSONableAtoms, prev_parameters: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Create the input parameters.

        Parameters
        ----------
        atoms: .MSONableAtoms
            The atoms object for the structures
        prev_parameters: dict[str, Any]
            The previous calculation's calculation parameters

        Returns
        -------
        The input object
        """
        # Get the default configuration
        # FHI-aims recommends using their defaults so bare-bones default parameters
        parameters: dict[str, Any] = {
            "xc": "pbe",
            "relativistic": "atomic_zora scalar",
        }

        # Override default parameters with previous parameters
        prev_parameters = (
            {} if prev_parameters is None else copy.deepcopy(prev_parameters)
        )
        prev_parameters.pop("relax_geometry", None)
        prev_parameters.pop("relax_unit_cell", None)

        kpt_settings = copy.deepcopy(self.user_kpoints_settings)
        if "k_grid" in prev_parameters:
            density = self.k2d(atoms, prev_parameters.pop("k_grid"))
            if "density" not in kpt_settings:
                kpt_settings["density"] = density

        parameter_updates = self.get_parameter_updates(atoms, prev_parameters)
        parameters = recursive_update(parameters, parameter_updates)

        # Override default parameters with user_parameters
        parameters = recursive_update(parameters, self.user_parameters)
        if ("k_grid" in parameters) and ("density" in kpt_settings):
            warn(
                "WARNING: the k_grid is set in user_parameters and in the kpt_settings,"
                " using the one passed in user_parameters.",
                stacklevel=1,
            )
        elif np.any(atoms.pbc) and ("k_grid" not in parameters):
            density = kpt_settings.get("density", 5.0)
            even = kpt_settings.get("even", True)
            parameters["k_grid"] = self.d2k(atoms, density, even)
        elif not np.any(atoms.pbc) and "k_grid" in parameters:
            warn("WARNING: removing unnecessary k_grid information", stacklevel=1)
            del parameters["k_grid"]

        return parameters

    def get_parameter_updates(
        self, atoms: MSONableAtoms, prev_parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Update the parameters for a given calculation type.

        Parameters
        ----------
        atoms : .MSONableAtoms
            ASE Atoms object.
        prev_parameters: Dict[str, Any]
            Previous calculation parameters.

        Returns
        -------
        A dictionary of updates to apply.
        """
        return prev_parameters

    def d2k(
        self,
        atoms: MSONableAtoms,
        kptdensity: float | list[float] = 5.0,
        even: bool = True,
    ) -> Iterable[float]:
        """Convert k-point density to Monkhorst-Pack grid size.

        inspired by [ase.calculators.calculator.kptdensity2monkhorstpack]

        Parameters
        ----------
        atoms: .MSONableAtoms
            Contains unit cell and information about boundary conditions.
        kptdensity: float or list of floats
            Required k-point density.  Default value is 5.0 point per Ang^-1.
        even: bool
            Round up to even numbers.

        Returns
        -------
        Monkhorst-Pack grid size in all directions
        """
        recipcell = atoms.cell.reciprocal()
        return self.d2k_recipcell(recipcell, atoms.pbc, kptdensity, even)

    def k2d(self, atoms: MSONableAtoms, k_grid: Iterable[int]):
        """Generate the kpoint density in each direction from given k_grid.

        Parameters
        ----------
        atoms: .MSONableAtoms
            Atoms object of interest.
        k_grid: Iterable[int]
            k_grid that was used.

        Returns
        -------
        Density of kpoints in each direction. result.mean() computes average density
        """
        recipcell = atoms.cell.reciprocal()
        densities = k_grid / (2 * np.pi * np.sqrt((recipcell**2).sum(axis=1)))
        return np.array(densities)

    @staticmethod
    def d2k_recipcell(
        recipcell: Cell,
        pbc: list[bool],
        kptdensity: float | Sequence[float] = 5.0,
        even: bool = True,
    ) -> Sequence[int]:
        """Convert k-point density to Monkhorst-Pack grid size.

        Parameters
        ----------
        recipcell: Cell
            The reciprocal cell
        pbc: list[bools]
            If element of pbc is True then system is periodic in that direction
        kptdensity: float or list[floats]
            Required k-point density.  Default value is 3.5 point per Ang^-1.
        even: bool
            Round up to even numbers.

        Returns
        -------
        Monkhorst-Pack grid size in all directions
        """
        if not isinstance(kptdensity, Iterable):
            kptdensity = 3 * [float(kptdensity)]
        kpts: list[int] = []
        for i in range(3):
            if pbc[i]:
                k = (
                    2
                    * np.pi
                    * np.sqrt((recipcell[i] ** 2).sum())
                    * float(kptdensity[i])
                )
                if even:
                    kpts.append(2 * int(np.ceil(k / 2)))
                else:
                    kpts.append(int(np.ceil(k)))
            else:
                kpts.append(1)
        return kpts


def recursive_update(d: dict, u: dict) -> dict:
    """
    Update a dictionary recursively and return it.

    Parameters
    ----------
    d: Dict
        Input dictionary to modify
    u: Dict
        Dictionary of updates to apply

    Returns
    -------
    The updated dictionary.

    Example
    -------
        d = {'activate_hybrid': {"hybrid_functional": "HSE06"}}
        u = {'activate_hybrid': {"cutoff_radius": 8}}

        yields {'activate_hybrid': {"hybrid_functional": "HSE06", "cutoff_radius": 8}}}
    """
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        elif k == "output" and isinstance(
            v, list
        ):  # for all other keys the list addition is not needed (I guess)
            old_v = d.get(k, [])
            d[k] = old_v + v
        else:
            d[k] = v
    return d
