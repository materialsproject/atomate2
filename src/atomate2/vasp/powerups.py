"""Powerups for performing common modifications on VASP jobs and flows."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from jobflow import Flow, Job, Maker
from pymatgen.io.vasp import Kpoints

from atomate2.common.powerups import add_metadata_to_flow as base_add_metadata_to_flow
from atomate2.common.powerups import update_custodian_handlers as base_custodian_handler
from atomate2.vasp.jobs.base import BaseVaspMaker


def update_vasp_input_generators(
    flow: Job | Flow | Maker,
    dict_mod_updates: dict[str, Any],
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseVaspMaker,
) -> Job | Flow | Maker:
    """
    Update any VaspInputGenerators or Makers in the flow.

    Note, this returns a copy of the original Job/Flow/Maker. I.e., the update does not
    happen in place.

    Parameters
    ----------
    flow : .Job or .Flow or .Maker
        A job, flow or Maker.
    dict_mod_updates : dict
        The updates to apply. Existing keys will not be modified unless explicitly
        specified in ``dict_mod_updates``.
    name_filter : str or None
        A filter for the name of the jobs.
    class_filter : Maker or None
        A filter for the VaspMaker class used to generate the flows. Note the class
        filter will match any subclasses.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input flow/job/maker modified to use the updated incar settings.
    """
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


def update_user_incar_settings(
    flow: Job | Flow | Maker,
    incar_updates: dict[str, Any],
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseVaspMaker,
) -> Job | Flow | Maker:
    """
    Update the user_incar_settings of any VaspInputGenerators in the flow.

    Alternatively, if a Maker is supplied, the user_incar_settings of the maker will
    be updated.

    Note, this returns a copy of the original Job/Flow/Maker. I.e., the update does not
    happen in place.

    Parameters
    ----------
    flow : .Job or .Flow or .Maker
        A job, flow or Maker.
    incar_updates : dict
        The updates to apply. Existing keys in user_incar_settings will not be modified
        unless explicitly specified in ``incar_updates``.
    name_filter : str or None
        A filter for the name of the jobs.
    class_filter : Maker or None
        A filter for the VaspMaker class used to generate the flows. Note the class
        filter will match any subclasses.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input flow/job/maker modified to use the updated incar settings.
    """
    return update_vasp_input_generators(
        flow=flow,
        dict_mod_updates={
            f"input_set_generator->user_incar_settings->{k}": v
            for k, v in incar_updates.items()
        },
        name_filter=name_filter,
        class_filter=class_filter,
    )


def update_user_potcar_settings(
    flow: Job | Flow | Maker,
    potcar_updates: dict[str, Any],
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseVaspMaker,
) -> Job | Flow | Maker:
    """
    Update the user_potcar_settings of any VaspInputGenerators in the flow.

    Alternatively, if a Maker is supplied, the user_potcar_settings of the maker will
    be updated.

    Note, this returns a copy of the original Job/Flow/Maker. I.e., the update does not
    happen in place.

    Parameters
    ----------
    flow : .Job or .Flow or .Maker
        A job, flow or Maker.
    potcar_updates : dict
        The updates to apply. Existing keys in user_potcar_settings will not be modified
        unless explicitly specified in ``potcar_updates``.
    name_filter : str or None
        A filter for the name of the jobs.
    class_filter : Maker or None
        A filter for the VaspMaker class used to generate the flows. Note the class
        filter will match any subclasses.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input flow/job/maker modified to use the updated potcar settings.
    """
    return update_vasp_input_generators(
        flow=flow,
        dict_mod_updates={
            f"input_set_generator->user_potcar_settings->{k}": v
            for k, v in potcar_updates.items()
        },
        name_filter=name_filter,
        class_filter=class_filter,
    )


def update_user_potcar_functional(
    flow: Job | Flow | Maker,
    potcar_functional: str,
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseVaspMaker,
) -> Job | Flow | Maker:
    """
    Update the user_potcar_functional of any VaspInputGenerators in the flow.

    Alternatively, if a Maker is supplied, the user_potcar_functional of the maker will
    be updated.

    Note, this returns a copy of the original Job/Flow/Maker. I.e., the update does not
    happen in place.

    Parameters
    ----------
    flow : .Job or .Flow or .Maker
        A job, flow or Maker.
    potcar_functional : str
        The new potcar functional to use.
    name_filter : str or None
        A filter for the name of the jobs.
    class_filter : Maker or None
        A filter for the VaspMaker class used to generate the flows. Note the class
        filter will match any subclasses.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input flow/job/maker modified to use the updated potcar settings.
    """
    return update_vasp_input_generators(
        flow=flow,
        dict_mod_updates={
            "input_set_generator->user_potcar_functional": potcar_functional
        },
        name_filter=name_filter,
        class_filter=class_filter,
    )


def update_user_kpoints_settings(
    flow: Job | Flow | Maker,
    kpoints_updates: dict[str, Any] | Kpoints,
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseVaspMaker,
) -> Job | Flow | Maker:
    """
    Update the user_kpoints_settings of any VaspInputGenerators in the flow.

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
        A filter for the VaspMaker class used to generate the flows. Note the class
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
    return update_vasp_input_generators(
        flow=flow,
        dict_mod_updates=dict_mod_updates,
        name_filter=name_filter,
        class_filter=class_filter,
    )


def use_auto_ispin(
    flow: Job | Flow | Maker,
    value: bool = True,
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseVaspMaker,
) -> Job | Flow | Maker:
    """
    Update the auto_ispin setting of any VaspInputGenerators in the flow.

    Alternatively, if a Maker is supplied, the auto_ispin of the maker will be updated.

    Note, this returns a copy of the original Job/Flow/Maker. I.e., the update does not
    happen in place.

    Parameters
    ----------
    flow : .Job or .Flow or .Maker
        A job, flow or Maker.
    value : bool
        The value of auto_ispin to set.
    name_filter : str or None
        A filter for the name of the jobs.
    class_filter : Maker or None
        A filter for the VaspMaker class used to generate the flows. Note the class
        filter will match any subclasses.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input flow/job/maker but with auto_ispin set.
    """
    return update_vasp_input_generators(
        flow=flow,
        dict_mod_updates={"input_set_generator->auto_ispin": value},
        name_filter=name_filter,
        class_filter=class_filter,
    )


def add_metadata_to_flow(
    flow: Flow, additional_fields: dict, class_filter: Maker = BaseVaspMaker
) -> Flow:
    """
    Return the VASP flow with additional field(metadata) to the task doc.

    This allows adding metadata to the task-docs, could be useful
    to query results from DB.

    Parameters
    ----------
    flow : Flow
        The flow to which to add metadata.
    additional_fields : dict
        A dict with metadata.
    class_filter: .BaseVaspMaker
        The Maker to which additional metadata needs to be added

    Returns
    -------
    Flow
        Flow with added metadata to the task-doc.
    """
    return base_add_metadata_to_flow(
        flow=flow, class_filter=class_filter, additional_fields=additional_fields
    )


def update_vasp_custodian_handlers(
    flow: Flow, custom_handlers: tuple, class_filter: Maker = BaseVaspMaker
) -> Flow:
    """
    Return the flow with custom custodian handlers for VASP jobs.

    This allows user to selectively set error correcting handlers for VASP jobs
    or completely unset error handlers.

    Parameters
    ----------
    flow:
    custom_handlers : tuple
        A tuple with custodian handlers.
    class_filter: .Maker
        The Maker to which custom custodian handler needs to be added

    Returns
    -------
    Flow
        Flow with modified custodian handlers.
    """
    return base_custodian_handler(
        flow=flow, custom_handlers=custom_handlers, class_filter=class_filter
    )
