"""Module defining base CP2K input set and generator."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import W
from typing import Any, Iterable, Dict

import numpy as np
from monty.io import zopen
from monty.serialization import loadfn
from pkg_resources import resource_filename
from pymatgen.core.structure import Structure, Molecule
from pymatgen.io.core import InputGenerator, InputSet
from pymatgen.io.cp2k.inputs import Cp2kInput
from pymatgen.io.cp2k.outputs import Cp2kOutput
from pymatgen.io.cp2k.sets import DftSet

from pymatgen.io.vasp import Kpoints # TODO continues to be an issue that kpoint functionality is kept in vasp io module

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath

from atomate2 import SETTINGS

_BASE_CP2K_SET = loadfn(resource_filename("atomate2.cp2k.sets", "BaseCp2kSet.yaml"))

__all__ = ["Cp2kInputSet", "Cp2kInputGenerator"]


class Cp2kInputSet(InputSet):
    """
    A class to represent a set of CP2K inputs.
    """

    def __init__(
        self,
        cp2k_input: Cp2kInput,
        optional_files: dict | None = None,
    ):
        """
        Initialize the set

        Parameters
        ----------
        cp2k_input: Cp2kInput object for representing the main cp2k input file
        optional_files: If CP2K contains external file links, then they can be stored here.

        """
        self.cp2k_input = cp2k_input
        self.optional_files = {} if optional_files is None else optional_files

    def write_input(
        self,
        directory: str | Path,
        make_dir: bool = True,
        overwrite: bool = True,
    ):
        """
        Write Cp2k input file to a directory.

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
        if make_dir and not directory.exists():
            os.makedirs(directory)

        inputs = {
            "input": {
                "filename": "cp2k.inp",
                "object": self.cp2k_input
            },
        }
        inputs.update(self.optional_files)

        for k, v in inputs.items():
            fn = v.get("filename")
            obj = v.get("object")
            if v is not None and (overwrite or not (directory / k).exists()):
                with zopen(directory / fn, "wt") as f:
                    f.write(str(obj))
            elif not overwrite and (directory / fn).exists():
                raise FileExistsError(f"{directory / fn} already exists.")

    @staticmethod
    def from_directory(directory: str | Path, optional_files: dict = None):
        """
        Load a set of CP2K inputs from a directory.

        Parameters
        ----------
        directory
            Directory to read CP2K inputs from.
        optional_files
            Optional files to read in as well as a dict of {filename: Object class}.
            Object class must have a static/class method from_file.
        """
        directory = Path(directory)
        if (directory / "cp2k.inp").exists():
            return Cp2kInput.from_file(directory / "cp2k.inp")

    # TODO Validation
    @property
    def is_valid(self) -> bool:
        """
        Whether the input set is valid.

        Returns
        -------
        bool
            Whether the input set is valid.
        """
        return True


@dataclass
class Cp2kInputGenerator(InputGenerator):
    """
    A class to generate Cp2k input sets.

    Parameters
    ----------
    user_input_settings:

    user_kpoints_settings:

    use_structure_charge
        If set to True, then the overall charge of the structure (``structure.charge``)
        is  used to set NELECT.
    sort_structure
        Whether to sort the structure (using the default sort order of
        electronegativity) before generating input files. Defaults to True, the behavior
        you would want most of the time. This ensures that similar atomic species are
        grouped together.
    vdw
        Adds default parameters for van-der-Waals functionals supported by VASP to
        INCAR. Supported functionals are: DFT-D2, undamped DFT-D3, DFT-D3 with
        Becke-Jonson damping, Tkatchenko-Scheffler, Tkatchenko-Scheffler with iterative
        Hirshfeld partitioning, MBD@rSC, dDsC, Dion's vdW-DF, DF2, optPBE, optB88,
        optB86b and rVV10.
    symprec
        Tolerance for symmetry finding, used for line mode band structure k-points.
    auto_ispin
        If generating input set from a previous calculation, this controls whether
        to disable magnetisation (ISPIN = 1) if the absolute value of all magnetic
        moments are less than 0.02.
    config_dict
        The config dictionary to use containing the base input set settings.
    """

    user_input_settings: dict = field(default_factory=dict)
    user_kpoints_settings: dict | Kpoints = field(default_factory=dict)
    auto_kspacing: bool = True
    use_structure_charge: bool = False
    sort_structure: bool = True
    symprec: float = SETTINGS.SYMPREC
    force_gamma: bool = False
    config_dict: dict = field(default_factory=lambda: _BASE_CP2K_SET)

    def __post_init__(self):
        """Post init formatting of arguments."""
        pass

    def get_input_set(  # type: ignore
        self,
        structure: Structure | Molecule = None,
        prev_dir: str | Path = None,
        optional_files: dict | None = None,
    ) -> Cp2kInputSet:
        """
        Get a CP2K input set.

        Note, if both ``structure`` and ``prev_dir`` are set, then the structure
        specified will be preferred over the final structure from the last CP2K run.

        Parameters
        ----------
        structure
            A structure.
        prev_dir
            A previous directory to generate the input set from.
        Returns
        -------
        Cp2kInput
            A Cp2k input set.
        """
        structure, prev_input, cp2k_output = self._get_previous(structure, prev_dir)

        input_updates = self.get_input_updates(
            structure,
            prev_input=prev_input,
        )
        if isinstance(structure, Structure):
            kpoints_updates = self.get_kpoints_updates(
                structure,
                prev_input=prev_input,
            )
            kpoints = self._get_kpoints(structure, kpoints_updates)
        else:
            kpoints = None

        cp2k_input = self._get_input(
            structure,
            kpoints,
            prev_input,
            input_updates,
        )
        return Cp2kInputSet(
            cp2k_input=cp2k_input,
            optional_files=optional_files
        )

    def get_input_updates(self, structure, prev_input) -> dict:
        """
        Get updates to the cp2k input for this calculation type.

        Parameters
        ----------
        structure
            A structure.
        prev_input
            A Cp2kInput from a previous calculation.
        bandgap
            The band gap.
        cp2k_output
            A Cp2kOutput from a previous calculation.

        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        raise NotImplementedError 

    def get_kpoints_updates(
        self,
        structure: Structure,
        prev_input: Cp2kInput = None,
    ) -> dict:
        """
        Get updates to the kpoints configuration for this calculation type.

        Note, these updates will be ignored if the user has set user_kpoint_settings.

        Parameters
        ----------
        structure
            A structure.
        prev_input
            A Cp2kInput from a previous calculation.
        bandgap
            The band gap.
        cp2k_output
            A Cp2kOutput from a previous calculation.

        Returns
        -------
        dict
            A dictionary of updates to apply to the KPOINTS config.
        """
        return {}

    def _get_previous(self, structure: Structure = None, prev_dir: str | Path = None):
        """Load previous calculation outputs and decide which structure to use."""
        if structure is None and prev_dir is None:
            raise ValueError("Either structure or prev_dir must be set.")

        prev_input = {}
        prev_structure = None
        cp2k_output = None
        if prev_dir:
            cp2k_output = Cp2kOutput(Path(prev_dir) / "cp2k.out")
            prev_input = cp2k_output.input
            prev_structure = cp2k_output.final_structure

        structure = structure if structure is not None else prev_structure
        structure = self._get_structure(structure)

        return structure, prev_input, cp2k_output

    def _get_structure(self, structure):
        """Get the standardized structure."""

        if self.sort_structure and hasattr(structure, "get_sorted_structure"):
            structure = structure.get_sorted_structure()

        return structure

    def _get_input(
        self,
        structure: Structure | Molecule,
        kpoints: Kpoints | None = None,
        previous_input: Cp2kInput = None,
        input_updates: dict = None,
    ):
        """Get the input."""
        previous_input = {} if previous_input is None else previous_input
        input_updates = {} if input_updates is None else input_updates
        input_settings = dict(self.config_dict["cp2k_input"])

        # Generate base input but override with user input settings
        input_settings = recursive_update(input_settings, input_updates)
        input_settings = recursive_update(input_settings, self.user_input_settings)
        overrides = input_settings.pop("override_default_params") if "override_default_params" in input_settings else {}
        cp2k_input = DftSet(structure=structure, kpoints=kpoints, **input_settings)

        for setting in input_settings:
            if hasattr(cp2k_input, setting) and input_settings[setting]:
                if callable(getattr(cp2k_input, setting)):
                    subsettings = input_settings.get(setting)
                    getattr(cp2k_input, setting)(**subsettings if isinstance(subsettings, dict) else {})

        cp2k_input.update(overrides)
        return cp2k_input
    
    def _get_kpoints(
        self,
        structure: Structure,
        kpoints_updates: dict[str, Any] | None,
    ) -> Kpoints | None:
        """Get the kpoints file."""
        kpoints_updates = {} if kpoints_updates is None else kpoints_updates

        # use user setting if set otherwise default to base config settings
        if self.user_kpoints_settings != {}:
            kconfig = deepcopy(self.user_kpoints_settings)
        else:
            # apply updates to k-points config
            kconfig = deepcopy(self.config_dict.get("KPOINTS", {}))
            kconfig.update(kpoints_updates)

        if isinstance(kconfig, Kpoints):
            return kconfig

        explicit = (
            kconfig.get("explicit", False)
            or len(kconfig.get("added_kpoints", [])) > 0
            or "zero_weighted_reciprocal_density" in kconfig
            or "zero_weighted_line_density" in kconfig
        )
        # handle length generation first as this doesn't support any additional options
        if kconfig.get("length"):
            if explicit:
                raise ValueError(
                    "length option cannot be used with explicit k-point generation, "
                    "added_kpoints, or zero weighted k-points."
                )
            # If length is in kpoints settings use Kpoints.automatic
            return Kpoints.automatic(kconfig["length"])

        base_kpoints = None
        if kconfig.get("line_density"):
            # handle line density generation
            kpath = HighSymmKpath(structure, **kconfig.get("kpath_kwargs", {}))
            frac_k_points, k_points_labels = kpath.get_kpoints(
                line_density=kconfig["line_density"], coords_are_cartesian=False
            )
            base_kpoints = Kpoints(
                comment="Non SCF run along symmetry lines",
                style=Kpoints.supported_modes.Line_mode,
                num_kpts=len(frac_k_points),
                kpts=frac_k_points,
                labels=k_points_labels,
                kpts_weights=[1] * len(frac_k_points),
            )
        elif kconfig.get("grid_density") or kconfig.get("reciprocal_density"):
            # handle regular weighted k-point grid generation
            if kconfig.get("grid_density"):
                base_kpoints = Kpoints.automatic_density(
                    structure, int(kconfig["grid_density"]), self.force_gamma
                )
            if kconfig.get("reciprocal_density"):
                base_kpoints = Kpoints.automatic_density_by_vol(
                    structure, kconfig["reciprocal_density"], self.force_gamma
                )
            if explicit:
                sga = SpacegroupAnalyzer(structure, symprec=self.symprec)
                mesh = sga.get_ir_reciprocal_mesh(base_kpoints.kpts[0])
                base_kpoints = Kpoints(
                    comment="Uniform grid",
                    style=Kpoints.supported_modes.Reciprocal,
                    num_kpts=len(mesh),
                    kpts=[i[0] for i in mesh],
                    kpts_weights=[i[1] for i in mesh],
                )
            else:
                # if not explicit that means no other options have been specified
                # so we can return the k-points as is
                return base_kpoints

        zero_weighted_kpoints = None
        if kconfig.get("zero_weighted_line_density"):
            # zero_weighted k-points along line mode path
            kpath = HighSymmKpath(structure)
            frac_k_points, k_points_labels = kpath.get_kpoints(
                line_density=kconfig["zero_weighted_line_density"],
                coords_are_cartesian=False,
            )
            zero_weighted_kpoints = Kpoints(
                comment="Hybrid run along symmetry lines",
                style=Kpoints.supported_modes.Reciprocal,
                num_kpts=len(frac_k_points),
                kpts=frac_k_points,
                labels=k_points_labels,
                kpts_weights=[0] * len(frac_k_points),
            )
        elif kconfig.get("zero_weighted_reciprocal_density"):
            zero_weighted_kpoints = Kpoints.automatic_density_by_vol(
                structure, kconfig["zero_weighted_reciprocal_density"], self.force_gamma
            )
            sga = SpacegroupAnalyzer(structure, symprec=self.symprec)
            mesh = sga.get_ir_reciprocal_mesh(zero_weighted_kpoints.kpts[0])
            zero_weighted_kpoints = Kpoints(
                comment="Uniform grid",
                style=Kpoints.supported_modes.Reciprocal,
                num_kpts=len(mesh),
                kpts=[i[0] for i in mesh],
                kpts_weights=[0 for i in mesh],
            )

        added_kpoints = None
        if kconfig.get("added_kpoints"):
            added_kpoints = Kpoints(
                comment="Specified k-points only",
                style=Kpoints.supported_modes.Reciprocal,
                num_kpts=len(kconfig.get("added_kpoints")),
                kpts=kconfig.get("added_kpoints"),
                labels=["user-defined"] * len(kconfig.get("added_kpoints")),
                kpts_weights=[0] * len(kconfig.get("added_kpoints")),
            )

        if base_kpoints and not (added_kpoints or zero_weighted_kpoints):
            return base_kpoints
        elif added_kpoints and not (base_kpoints or zero_weighted_kpoints):
            return added_kpoints
        
        # do some sanity checking
        if "line_density" in kconfig and zero_weighted_kpoints:
            raise ValueError(
                "Cannot combined line_density and zero weighted k-points options"
            )
        elif zero_weighted_kpoints and not base_kpoints:
            raise ValueError(
                "Zero weighted k-points must be used with reciprocal_density or "
                "grid_density options"
            )
        elif not (base_kpoints or zero_weighted_kpoints or added_kpoints):
            return None

        return _combine_kpoints(base_kpoints, zero_weighted_kpoints, added_kpoints)


def multiple_input_updators():
    """
    This utility function acts to decorate child classes of Cp2kInputGenerator so that multiple sets can
    combine to produce more complex ones.

    For example, the HybridRelaxSetGenerator is a combination of the RelaxSet and the HybridSet. This decorator
    allows HybridRelaxSetGenerator to be defined consisely as:

    @dataclass
    @multiple_input_updators()
    class HybridRelaxSetGenerator(HybridSetGenerator, RelaxSetGenerator):
        pass

    Where multiple_input_updators() will joing the get_input_updates functions from HybridSetGenerator and
    RelaxSetGenerator to produce a combined effect. 
    """
    def decorate(myclass):
        def multi(foo):
            def get_input_updates(self, *args, **kwargs):
                updates = {}
                for parent in (self.__class__.__bases__ if isinstance(self, Cp2kInputGenerator) else self.__bases__):
                    if parent.__bases__ == (Cp2kInputGenerator, ):
                        updates.update(getattr(parent, foo.__name__)(self, *args, **kwargs))
                    else:
                        updates.update(get_input_updates(parent, *args, **kwargs))
                return updates
            return get_input_updates

        if callable(getattr(myclass, "get_input_updates")):
            setattr(myclass, "get_input_updates", multi(getattr(myclass, "get_input_updates")))
        return myclass 
    return decorate


def recursive_update(d: Dict, u: Dict):
    """
    Update a dictionary recursively.

    Args:
        d: Input dictionary
        u: Update dictionary
    
    Example:
        d = {'activate_hybrid': {"hybrid_functional": "HSE06"}}
        u = {'activate_hybrid': {"cutoff_radius": 8}}
    
        yields {'activate_hybrid': {"hybrid_functional": "HSE06", "cutoff_radius": 8}}}
    """
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d