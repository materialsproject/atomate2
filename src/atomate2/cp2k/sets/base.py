"""Module defining base CP2K input set and generator."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from importlib.resources import files as get_mod_path
from pathlib import Path
from typing import Any

import numpy as np
from monty.io import zopen
from monty.serialization import loadfn
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.core import InputGenerator, InputSet
from pymatgen.io.cp2k.inputs import (
    BasisFile,
    Cp2kInput,
    GaussianTypeOrbitalBasisSet,
    GthPotential,
    PotentialFile,
)
from pymatgen.io.cp2k.outputs import Cp2kOutput
from pymatgen.io.cp2k.sets import DftSet
from pymatgen.io.vasp import Kpoints  # TODO Currently uses the VASP implementation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath

from atomate2 import SETTINGS

_BASE_CP2K_SET = loadfn(get_mod_path("atomate2.cp2k.sets") / "BaseCp2kSet.yaml")
_BASE_GAPW_SET = loadfn(get_mod_path("atomate2.cp2k.sets") / "BaseAllSet.yaml")


class Cp2kInputSet(InputSet):
    """A class to represent a set of CP2K inputs."""

    def __init__(
        self,
        cp2k_input: Cp2kInput,
        optional_files: dict | None = None,
    ) -> None:
        """Initialize the set.

        Parameters
        ----------
        cp2k_input
            Cp2kInput object for representing the main cp2k input file

        optional_files
            Any additional files needed for running the calculations. Most common
            use is to make data files available which are not guaranteed to be available
            at runtime.

            Format pseudocode:
                {
                    "name of this optional data": {
                        "filename": filename to write to,
                        "object": object with a __str__ method to write the file
                    }
                }

            Some examples uses:

                (1) Cp2kInputGenerator below will try to put the basis and potential
                    info into their own optional files. This allows them to run when the
                    cp2k executable cannot find this info due to version mismatch,
                    custom data, etc.
                (2) Include files. CP2K preprocessor link input sections like the
                    structure definition to an external file in order to keep the main
                    input file neat. This use case requires "@include" parameters (see
                    pymatgen.io.cp2k or the cp2k manual)
                (3) Other custom data files like vdw kernel tables, truncated coulomb.
        """
        self.cp2k_input = cp2k_input
        self.optional_files = optional_files or {}

    def write_input(
        self,
        directory: str | Path,
        make_dir: bool = True,
        overwrite: bool = True,
    ) -> None:
        """Write Cp2k input file to a directory.

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

        inputs = {
            "input": {"filename": "cp2k.inp", "object": self.cp2k_input},
        }
        inputs.update(self.optional_files)

        for key, val in inputs.items():
            filename = val.get("filename")
            obj = val.get("object")
            if val is not None and (overwrite or not (directory / key).exists()):
                with zopen(directory / filename, "wt") as file:
                    file.write(str(obj))
            elif not overwrite and (directory / filename).exists():
                raise FileExistsError(f"{directory / filename} already exists.")

    @staticmethod
    def from_directory(
        directory: str | Path, optional_files: dict = None
    ) -> Cp2kInputSet:
        """Load a set of CP2K inputs from a directory.

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
            cp2k_input = Cp2kInput.from_file(directory / "cp2k.inp")
        else:
            raise FileNotFoundError
        optional_files = optional_files or {}
        optional = {}
        for filename, obj in optional_files.items():
            optional[filename] = {
                "filename": filename,
                "object": obj.from_file(filename),
            }

        return Cp2kInputSet(cp2k_input=cp2k_input, optional_files=optional)

    # TODO Validation
    @property
    def is_valid(self) -> bool:
        """Whether the input set is valid."""
        return True


@dataclass
class Cp2kInputGenerator(InputGenerator):
    """
    A class to generate Cp2k input sets.

    Parameters
    ----------
    user_input_settings:
        Updates to the inputs in the base config dict.
    user_kpoints_settings:
        Updates to the kpoint settings in the base config dict
    sort_structure
        Whether to sort the structure (using the default sort order of
        electronegativity) before generating input files. Defaults to True, the behavior
        you would want most of the time. This ensures that similar atomic species are
        grouped together.
    symprec
        Tolerance for symmetry finding, used for line mode band structure k-points.
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

    def get_input_set(
        self,
        structure: Structure | Molecule = None,
        prev_dir: str | Path = None,
        optional_files: dict | None = None,
    ) -> Cp2kInputSet:
        """Get a CP2K input set.

        Note, if both ``structure`` and ``prev_dir`` are set, then the structure
        specified will be preferred over the final structure from the last CP2K run.

        Parameters
        ----------
        structure
            A structure.
        prev_dir
            A previous directory to generate the input set from.
        optional_files
            Additional files (e.g. vdw kernel file) to be included in the input set.

        Returns
        -------
        Cp2kInput
            A Cp2k input set.
        """
        structure, prev_input, _cp2k_output = self._get_previous(structure, prev_dir)

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
        optional_files = optional_files or {}
        optional_files["basis"] = {
            "filename": "BASIS",
            "object": self._get_basis_file(cp2k_input=cp2k_input),
        }
        optional_files["potential"] = {
            "filename": "POTENTIAL",
            "object": self._get_potential_file(cp2k_input=cp2k_input),
        }

        return Cp2kInputSet(cp2k_input=cp2k_input, optional_files=optional_files)

    def get_input_updates(self, structure: Structure, prev_input: Cp2kInput) -> dict:
        """Get updates to the cp2k input for this calculation type.

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
        self, structure: Structure, prev_input: Cp2kInput = None
    ) -> dict:
        """Get updates to the kpoints configuration for this calculation type.

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

    def _get_previous(
        self, structure: Structure = None, prev_dir: str | Path = None
    ) -> tuple[Structure, Cp2kInput, Cp2kOutput]:
        """Load previous calculation outputs and decide which structure to use."""
        if structure is None and prev_dir is None:
            raise ValueError("Either structure or prev_dir must be set")

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

    def _get_structure(self, structure: Structure) -> Structure:
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
    ) -> Cp2kInput:
        """Get the input."""
        previous_input = previous_input or {}
        input_updates = input_updates or {}
        input_settings = dict(self.config_dict["cp2k_input"])

        # Generate base input but override with user input settings
        input_settings = recursive_update(input_settings, input_updates)
        input_settings = recursive_update(input_settings, self.user_input_settings)
        overrides = (
            input_settings.pop("override_default_params")
            if "override_default_params" in input_settings
            else {}
        )
        cp2k_input = DftSet(structure=structure, kpoints=kpoints, **input_settings)

        for setting in input_settings:
            if (
                hasattr(cp2k_input, setting)
                and input_settings[setting]
                and callable(getattr(cp2k_input, setting))
            ):
                sub_settings = input_settings.get(setting)
                getattr(cp2k_input, setting)(
                    **sub_settings if isinstance(sub_settings, dict) else {}
                )

        cp2k_input.update(overrides)
        return cp2k_input

    def _get_basis_file(self, cp2k_input: Cp2kInput) -> BasisFile:
        """Get input object's basis sets and convert them to a basis file object.

        Allows calculation to execute if the basis sets are not available on the
        execution resource.
        """
        basis_sets = []
        for el in cp2k_input.structure.symbol_set:
            for data in cp2k_input.basis_and_potential[el].values():
                if isinstance(data, GaussianTypeOrbitalBasisSet):
                    basis_sets.append(data)  # noqa: PERF401
        if not basis_sets:
            return None
        cp2k_input.safeset({"force_eval": {"dft": {"BASIS_SET_FILE_NAME": "BASIS"}}})
        return BasisFile(objects=basis_sets)

    def _get_potential_file(self, cp2k_input: Cp2kInput) -> PotentialFile:
        """Get the potentials and convert them to a potential file object.

        Allows calculation to execute if the potentials are not available on the
        execution resource.
        """
        potentials = []
        for el in cp2k_input.structure.symbol_set:
            for data in cp2k_input.basis_and_potential[el].values():
                if isinstance(data, GthPotential):
                    potentials.append(data)  # noqa: PERF401
        if not potentials:
            return None
        cp2k_input.safeset(
            {"force_eval": {"dft": {"POTENTIAL_FILE_NAME": "POTENTIAL"}}}
        )
        return PotentialFile(objects=potentials)

    def _get_kpoints(
        self,
        structure: Structure,
        kpoints_updates: dict[str, Any] | None,
    ) -> Kpoints | None:
        """Get the kpoints object."""
        kpoints_updates = kpoints_updates or {}

        # use user setting if set otherwise default to base config settings
        if self.user_kpoints_settings != {}:
            kpt_config = deepcopy(self.user_kpoints_settings)
        else:
            # apply updates to k-points config
            kpt_config = deepcopy(self.config_dict.get("KPOINTS", {}))
            kpt_config.update(kpoints_updates)

        if isinstance(kpt_config, Kpoints):
            return kpt_config

        explicit = (
            kpt_config.get("explicit")
            or len(kpt_config.get("added_kpoints", [])) > 0
            or "zero_weighted_reciprocal_density" in kpt_config
            or "zero_weighted_line_density" in kpt_config
        )
        # handle length generation first as this doesn't support any additional options
        if kpt_config.get("length"):
            if explicit:
                raise ValueError(
                    "length option cannot be used with explicit k-point generation, "
                    "added_kpoints, or zero weighted k-points."
                )
            # If length is in kpoints settings use Kpoints.automatic
            return Kpoints.automatic(kpt_config["length"])

        base_kpoints = None
        if kpt_config.get("line_density"):
            # handle line density generation
            kpath = HighSymmKpath(structure, **kpt_config.get("kpath_kwargs", {}))
            frac_k_points, k_points_labels = kpath.get_kpoints(
                line_density=kpt_config["line_density"], coords_are_cartesian=False
            )
            base_kpoints = Kpoints(
                comment="Non SCF run along symmetry lines",
                style=Kpoints.supported_modes.Line_mode,
                num_kpts=len(frac_k_points),
                kpts=frac_k_points,
                labels=k_points_labels,
                kpts_weights=[1] * len(frac_k_points),
            )
        elif kpt_config.get("grid_density") or kpt_config.get("reciprocal_density"):
            # handle regular weighted k-point grid generation
            if kpt_config.get("grid_density"):
                base_kpoints = Kpoints.automatic_density(
                    structure, int(kpt_config["grid_density"]), self.force_gamma
                )
            if kpt_config.get("reciprocal_density"):
                base_kpoints = Kpoints.automatic_density_by_vol(
                    structure, kpt_config["reciprocal_density"], self.force_gamma
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
        if kpt_config.get("zero_weighted_line_density"):
            # zero_weighted k-points along line mode path
            kpath = HighSymmKpath(structure)
            frac_k_points, k_points_labels = kpath.get_kpoints(
                line_density=kpt_config["zero_weighted_line_density"],
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
        elif kpt_config.get("zero_weighted_reciprocal_density"):
            zero_weighted_kpoints = Kpoints.automatic_density_by_vol(
                structure,
                kpt_config["zero_weighted_reciprocal_density"],
                self.force_gamma,
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
        if kpt_config.get("added_kpoints"):
            added_kpoints = Kpoints(
                comment="Specified k-points only",
                style=Kpoints.supported_modes.Reciprocal,
                num_kpts=len(kpt_config.get("added_kpoints")),
                kpts=kpt_config.get("added_kpoints"),
                labels=["user-defined"] * len(kpt_config.get("added_kpoints")),
                kpts_weights=[0] * len(kpt_config.get("added_kpoints")),
            )

        if base_kpoints and not (added_kpoints or zero_weighted_kpoints):
            return base_kpoints
        if added_kpoints and not (base_kpoints or zero_weighted_kpoints):
            return added_kpoints

        # do some sanity checking
        if "line_density" in kpt_config and zero_weighted_kpoints:
            raise ValueError(
                "Cannot combined line_density and zero weighted k-points options"
            )
        if zero_weighted_kpoints and not base_kpoints:
            raise ValueError(
                "Zero weighted k-points must be used with reciprocal_density or "
                "grid_density options"
            )
        if not (base_kpoints or zero_weighted_kpoints or added_kpoints):
            return None

        return _combine_kpoints(base_kpoints, zero_weighted_kpoints, added_kpoints)


# TODO From `atomate2.vasp.sets.base`. Should possibly go in common.
# only reservation is if, eventually, CP2K gets it own kpoint object version
# instead of using the vasp kpoint objects.
def _combine_kpoints(*kpoints_objects: Kpoints) -> Kpoints:
    """Combine k-points files together."""
    labels = []
    kpoints = []
    weights = []

    for kpoints_object in filter(None, kpoints_objects):
        if kpoints_object.style != Kpoints.supported_modes.Reciprocal:
            raise ValueError(
                "Can only combine kpoints with style=Kpoints.supported_modes.Reciprocal"
            )
        if kpoints_object.labels is None:
            labels.append([""] * len(kpoints_object.kpts))
        else:
            labels.append(kpoints_object.labels)

        weights.append(kpoints_object.kpts_weights)
        kpoints.append(kpoints_object.kpts)

    labels = np.concatenate(labels).tolist()
    weights = np.concatenate(weights).tolist()
    kpoints = np.concatenate(kpoints)
    return Kpoints(
        comment="Combined k-points",
        style=Kpoints.supported_modes.Reciprocal,
        num_kpts=len(kpoints),
        kpts=kpoints,
        labels=labels,
        kpts_weights=weights,
    )


@dataclass
class Cp2kAllElectronInputGenerator(Cp2kInputGenerator):
    """
    A class to generate Cp2k input sets for all electron calculations.

    Parameters
    ----------
    user_input_settings:
        Updates to the inputs in the base config dict.
    sort_structure
        Whether to sort the structure (using the default sort order of
        electronegativity) before generating input files. Defaults to True, the behavior
        you would want most of the time. This ensures that similar atomic species are
        grouped together.
    symprec
        Tolerance for symmetry finding, used for line mode band structure k-points.
    config_dict
        The config dictionary to use containing the base input set settings.
    """

    user_input_settings: dict = field(default_factory=dict)
    sort_structure: bool = True
    symprec: float = SETTINGS.SYMPREC
    config_dict: dict = field(default_factory=lambda: _BASE_GAPW_SET)

    def _get_kpoints(
        self, structure: Structure, kpoints_updates: dict[str, Any] | None
    ) -> Kpoints | None:
        """No Kpoints possible."""
        return None


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
    Dict
        The updated dictionary.

    Example
    ----------
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
