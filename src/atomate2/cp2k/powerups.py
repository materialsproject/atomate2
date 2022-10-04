"""Powerups for performing common modifications on CP2K jobs and flows."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from jobflow import Flow, Job, Maker
from pymatgen.io.vasp import Kpoints

from atomate2.cp2k.jobs.base import BaseCp2kMaker


def update_user_input_settings(
    flow: Job | Flow | Maker,
    input_updates: dict[str, Any],
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseCp2kMaker,
) -> Job | Flow | Maker:
    """
    Update the user_cp2k_settings of any Cp2kInputGenerators in the flow.

    Alternatively, if a Maker is supplied, the user_cp2k_settings of the maker will
    be updated.

    Note, this returns a copy of the original Job/Flow/Maker. I.e., the update does not
    happen in place.

    Parameters
    ----------
    flow : .Job or .Flow or .Maker
        A job, flow or Maker.
    input_updates : dict
        The updates to apply. Existing keys in user_input_settings will not be modified
        unless explicitly specified in ``input_updates``.
    name_filter : str or None
        A filter for the name of the jobs.
    class_filter : Maker or None
        A filter for the Cp2kMaker class used to generate the flows. Note the class
        filter will match any subclasses.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input flow/job/maker modified to use the updated input settings.
    """
    dict_mod_updates = {
        f"input_set_generator->user_input_settings->{k}": v
        for k, v in input_updates.items()
    }
    updated_flow = deepcopy(flow)
    if isinstance(updated_flow, Maker):
        updated_flow = updated_flow.update_kwargs(
            {"_set": dict_mod_updates},
            name_filter=name_filter,
            class_filter=class_filter,
            dict_mod=True,
        )
    else:
        updated_flow.update_maker_kwargs(
            {"_set": dict_mod_updates},
            name_filter=name_filter,
            class_filter=class_filter,
            dict_mod=True,
        )

    return updated_flow


def update_user_kpoints_settings(
    flow: Job | Flow | Maker,
    kpoints_updates: dict[str, Any] | Kpoints,
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseCp2kMaker,
) -> Job | Flow | Maker:
    """
    Update the user_kpoints_settings of any Cp2kInputGenerators in the flow.

    Alternatively, if a Maker is supplied, the user_kpoints_settings of the maker will
    be updated.

    Note, this returns a copy of the original Job/Flow/Maker. I.e., the update does not
    happen in place.

    Parameters
    ----------
    flow : .Job or .Flow or .Maker
        A job, flow or Maker.
    kpoints_updates : dict
        The updates to apply. Can be specified as a dictionary or as a Kpoints object.
        If a dictionary is supplied, existing keys in user_kpoints_settings will not be
        modified unless explicitly specified in ``kpoints_updates``.
    name_filter : str or None
        A filter for the name of the jobs.
    class_filter : Maker or None
        A filter for the Cp2kMaker class used to generate the flows. Note the class
        filter will match any subclasses.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input flow/job/maker modified to use the updated kpoints settings.
    """
    if isinstance(kpoints_updates, Kpoints):
        dict_mod_updates = {
            "input_set_generator->user_kpoints_settings": kpoints_updates
        }
    else:
        dict_mod_updates = {
            f"input_set_generator->user_kpoints_settings->{k}": v
            for k, v in kpoints_updates.items()
        }

    updated_flow = deepcopy(flow)
    if isinstance(updated_flow, Maker):
        updated_flow = updated_flow.update_kwargs(
            {"_set": dict_mod_updates},
            name_filter=name_filter,
            class_filter=class_filter,
            dict_mod=True,
        )
    else:
        updated_flow.update_maker_kwargs(
            {"_set": dict_mod_updates},
            name_filter=name_filter,
            class_filter=class_filter,
            dict_mod=True,
        )
    return updated_flow

