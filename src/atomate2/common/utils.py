"""Common utilities for atomate2."""

from __future__ import annotations

import re
from importlib import import_module
from typing import TYPE_CHECKING, Any

from monty.serialization import loadfn

if TYPE_CHECKING:
    from pathlib import Path


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
