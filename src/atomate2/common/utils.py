"""Common utilities for atomate2."""

from __future__ import annotations

import re
from importlib import import_module
from typing import TYPE_CHECKING, Any

from monty.serialization import loadfn
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure


def get_supercell_matrix(
    structure: Structure,
    min_length: float,
    max_length: float,
    prefer_90_degrees: bool,
    allow_orthorhombic: bool = False,
    **kwargs,
) -> list[list[float]]:
    """
    Determine supercell size with given min_length and max_length.

    Parameters
    ----------
    structure: Structure Object
        Input structure that will be used to determine supercell
    min_length: float
        minimum length of cell in Angstrom
    max_length: float
        maximum length of cell in Angstrom
    prefer_90_degrees: bool
        if True, the algorithm will try to find a cell with 90 degree angles first
    allow_orthorhombic: bool
        if True, orthorhombic supercells are allowed
    **kwargs:
        Additional parameters that can be set.
    """
    kwargs.setdefault("force_diagonal", False)
    common_kwds = dict(
        min_length=min_length,
        max_length=max_length,
        min_atoms=kwargs.get("min_atoms"),
        max_atoms=kwargs.get("max_atoms"),
        step_size=kwargs.get("step_size", 0.1),
        force_diagonal=kwargs["force_diagonal"],
    )
    if not prefer_90_degrees:
        transformation = CubicSupercellTransformation(
            **common_kwds,
            force_90_degrees=False,
            allow_orthorhombic=allow_orthorhombic,
        )
        transformation.apply_transformation(structure=structure)
    else:
        try:
            common_kwds.update({"max_atoms": kwargs.get("max_atoms", 1200)})
            transformation = CubicSupercellTransformation(
                **common_kwds,
                force_90_degrees=True,
                angle_tolerance=kwargs.get("angle_tolerance", 1e-2),
                allow_orthorhombic=allow_orthorhombic,
            )
            transformation.apply_transformation(structure=structure)

        except AttributeError:
            transformation = CubicSupercellTransformation(
                **common_kwds,
                force_90_degrees=False,
                allow_orthorhombic=allow_orthorhombic,
            )
            transformation.apply_transformation(structure=structure)
    # matrix from pymatgen has to be transposed
    return transformation.transformation_matrix.transpose().tolist()


def get_transformations(
    transformations: tuple[str, ...], params: tuple[dict, ...] | None
) -> list:
    """Get instantiated transformation objects from their names and parameters.

    Parameters
    ----------
    transformations
        name of the transformations
    params
        parameters to pass to the transformation objects

    Returns
    -------
    A list of initiated transformation objects
    """
    params = ({},) * len(transformations) if params is None else params

    if len(params) != len(transformations):
        raise ValueError("Number of transformations and parameters must be the same.")

    transformation_objects = []
    for transformation, transformation_params in zip(
        transformations, params, strict=True
    ):
        found = False
        for module in (
            "advanced_transformations",
            "site_transformations",
            "standard_transformations",
        ):
            mod = import_module(f"pymatgen.transformations.{module}")

            try:
                t_cls = getattr(mod, transformation)
                found = True
                continue
            except AttributeError:
                pass

        if not found:
            raise ValueError(f"Could not find transformation: {transformation}")

        t_obj = t_cls(**transformation_params)
        transformation_objects.append(t_obj)
    return transformation_objects


def parse_custodian(dir_name: Path) -> dict | None:
    """
    Parse custodian.json file.

    Calculations done using custodian have a custodian.json file which tracks the makers
    performed and any errors detected and fixed.

    Parameters
    ----------
    dir_name
        Path to calculation directory.

    Returns
    -------
    Optional[dict]
        The information parsed from custodian.json file.
    """
    filenames = tuple(dir_name.glob("custodian.json*"))
    if len(filenames) >= 1:
        return loadfn(filenames[0], cls=None)
    return None


def parse_transformations(
    dir_name: Path,
) -> tuple[dict, int | None, list[str] | None, str | None]:
    """Parse transformations.json file."""
    transformations = {}
    filenames = tuple(dir_name.glob("transformations.json*"))
    icsd_id = None
    if len(filenames) >= 1:
        transformations = loadfn(filenames[0], cls=None)
        try:
            match = re.match(r"(\d+)-ICSD", transformations["history"][0]["source"])
            if match:
                icsd_id = int(match.group(1))
        except (KeyError, IndexError):
            pass

    # We don't want to leave tags or authors in the
    # transformations file because they'd be copied into
    # every structure generated after this one.
    other_parameters = transformations.get("other_parameters", {})
    new_tags = other_parameters.pop("tags", None)
    new_author = other_parameters.pop("author", None)

    if "other_parameters" in transformations and not other_parameters:
        # if dict is now empty remove it
        transformations.pop("other_parameters")

    return transformations, icsd_id, new_tags, new_author


def parse_additional_json(dir_name: Path) -> dict[str, Any]:
    """Parse additional JSON files in the directory."""
    additional_json = {}
    for filename in dir_name.glob("*.json*"):
        key = filename.name.split(".")[0]
        # ignore FW.json(.gz) so jobflow doesn't try to parse prev_dir
        # OutputReferences was causing atomate2 MP workflows to fail with ValueError:
        # Could not resolve reference 7f5a7f14-464c-4a5b-85f9-8d11b595be3b not in store
        # or cache contact @janosh in case of questions
        if key not in ("custodian", "transformations", "FW"):
            additional_json[key] = loadfn(filename, cls=None)
    return additional_json


def _recursive_get_dir_names(jobs: list, dir_names: list) -> None:
    """Recursively get all `output.dir_name` from a list of jobs.

    Parameters
    ----------
    jobs : list of jobs, Makers, Flows, etc.
    dir_names : a list to add the `dir_name`'s to.
    """
    for a_job in jobs:
        if (sub_jobs := getattr(a_job, "jobs", None)) is not None:
            _recursive_get_dir_names(sub_jobs, dir_names)
        else:
            dir_names.append(a_job.output.dir_name)
