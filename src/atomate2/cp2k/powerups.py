"""Powerups for performing common modifications on CP2K jobs and flows."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from jobflow import Flow, Job, Maker
from pymatgen.io.vasp import Kpoints

from atomate2.common.powerups import add_metadata_to_flow as base_add_metadata_to_flow
from atomate2.common.powerups import update_custodian_handlers as base_custodian_handler
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

    # Convert nested dictionary updates for cp2k input settings
    # into dict_mod update format
    def nested_to_dictmod(
        dct: dict, kk: str = "input_set_generator->user_input_settings"
    ) -> dict:
        d2 = {}
        for k, v in dct.items():
            k2 = f"{kk}->{k}"
            if isinstance(v, dict):
                d2.update(nested_to_dictmod(v, kk=k2))
            else:
                d2[k2] = v
        return d2

    dict_mod_updates = nested_to_dictmod(input_updates)

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


def add_metadata_to_flow(
    flow: Flow, additional_fields: dict, class_filter: Maker = BaseCp2kMaker
) -> Flow:
    """
    Return the Cp2k flow with additional field(metadata) to the task doc.

    This allows adding metadata to the task-docs, could be useful
    to query results from DB.

    Parameters
    ----------
    flow:
    additional_fields : dict
        A dict with metadata.
    class_filter: .BaseCp2kMaker
        The Maker to which additional metadata needs to be added

    Returns
    -------
    Flow
        Flow with added metadata to the task-doc.
    """
    return base_add_metadata_to_flow(
        flow=flow, class_filter=class_filter, additional_fields=additional_fields
    )


def update_cp2k_custodian_handlers(
    flow: Flow, custom_handlers: tuple, class_filter: Maker = BaseCp2kMaker
) -> Flow:
    """
    Return the flow with custom custodian handlers for Cp2k jobs.

    This allows user to selectively set error correcting handlers for Cp2k jobs
    or completely unset error handlers.

    Parameters
    ----------
    flow:
    custom_handlers : tuple
        A tuple with custodian handlers.
    class_filter: .BaseCp2kMaker
        The Maker to which custom custodian handler needs to be added

    Returns
    -------
    Flow
        Flow with modified custodian handlers.
    """
    return base_custodian_handler(
        flow=flow, custom_handlers=custom_handlers, class_filter=class_filter
    )
