"""Powerups for performing common modifications on ABINIT jobs and flows."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from jobflow import Flow, Job, Maker
from pymatgen.io.abinit.abiobjects import KSampling

from atomate2.abinit.jobs.base import BaseAbinitMaker


def update_maker_kwargs(
    class_filter: Maker | None,
    dict_mod_updates: dict,
    flow: Job | Flow | Maker,
    name_filter: str | None,
) -> Job | Flow | Maker:
    """
    Update an object inside a Job, a Flow or a Maker.

    A generic method to be shared for more specific updates that will
    build the dict_mod_updates.

    Parameters
    ----------
    flow : .Job or .Flow or .Maker
        A job, flow or Maker.
    dict_mod_updates : dict
        The updates to apply.
    name_filter : str or None
        A filter for the name of the jobs.
    class_filter : Maker or None
        A filter for the class used to generate the flows. Note the class
        filter will match any subclasses.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input modified flow/job/maker.
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


def update_user_abinit_settings(
    flow: Job | Flow | Maker,
    abinit_updates: dict[str, Any],
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseAbinitMaker,
) -> Job | Flow | Maker:
    """
    Update the user_abinit_settings of any AbinitInputGenerator in the flow.

    Alternatively, if a Maker is supplied, the user_abinit_settings of the maker will
    be updated.

    Note, this returns a copy of the original Job/Flow/Maker. I.e., the update does not
    happen in place.

    Parameters
    ----------
    flow : .Job or .Flow or .Maker
        A job, flow or Maker.
    abinit_updates : dict
        The updates to apply. Existing keys in user_abinit_settings will not be modified
        unless explicitly specified in ``abinit_updates``.
    name_filter : str or None
        A filter for the name of the jobs.
    class_filter : Maker or None
        A filter for the BaseAbinitMaker class used to generate the flows. Note the
        class filter will match any subclasses.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input flow/job/maker modified to use the updated abinit settings.
    """
    dict_mod_updates = {
        f"input_set_generator->user_abinit_settings->{k}": v
        for k, v in abinit_updates.items()
    }

    return update_maker_kwargs(class_filter, dict_mod_updates, flow, name_filter)


def update_factory_kwargs(
    flow: Job | Flow | Maker,
    factory_updates: dict[str, Any],
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseAbinitMaker,
) -> Job | Flow | Maker:
    """
    Update the factory_kwargs of any AbinitInputGenerator in the flow.

    Alternatively, if a Maker is supplied, the factory_kwargs of the maker will
    be updated.

    Note, this returns a copy of the original Job/Flow/Maker. I.e., the update does not
    happen in place.

    Parameters
    ----------
    flow : .Job or .Flow or .Maker
        A job, flow or Maker.
    factory_updates : dict
        The updates to apply. Existing keys in factory_kwargs will not be modified
        unless explicitly specified in ``factory_updates``.
    name_filter : str or None
        A filter for the name of the jobs.
    class_filter : Maker or None
        A filter for the BaseAbinitMaker class used to generate the flows. Note the
        class filter will match any subclasses.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input flow/job/maker modified to use the updated factory settings.
    """
    dict_mod_updates = {
        f"input_set_generator->factory_kwargs->{k}": v
        for k, v in factory_updates.items()
    }

    return update_maker_kwargs(class_filter, dict_mod_updates, flow, name_filter)


def update_user_kpoints_settings(
    flow: Job | Flow | Maker,
    kpoints_updates: dict[str, Any] | KSampling,
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseAbinitMaker,
) -> Job | Flow | Maker:
    """
    Update the user_kpoints_settings of any AbinitInputGenerator in the flow.

    Alternatively, if a Maker is supplied, the user_kpoints_settings of the maker will
    be updated.

    Note, this returns a copy of the original Job/Flow/Maker. I.e., the update does not
    happen in place.

    Parameters
    ----------
    flow : .Job or .Flow or .Maker
        A job, flow or Maker.
    kpoints_updates : dict
        The updates to apply. Can be specified as a dictionary or as a KSampling object.
        If a dictionary is supplied, existing keys in user_kpoints_settings will not be
        modified unless explicitly specified in ``kpoints_updates``.
    name_filter : str or None
        A filter for the name of the jobs.
    class_filter : Maker or None
        A filter for the BaseAbinitMaker class used to generate the flows. Note the
        class filter will match any subclasses.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input flow/job/maker modified to use the updated kpoints settings.
    """
    if isinstance(kpoints_updates, KSampling):
        dict_mod_updates = {
            "input_set_generator->user_kpoints_settings": kpoints_updates
        }
    else:
        dict_mod_updates = {
            f"input_set_generator->user_kpoints_settings->{k}": v
            for k, v in kpoints_updates.items()
        }

    return update_maker_kwargs(class_filter, dict_mod_updates, flow, name_filter)


def update_generator_attributes(
    flow: Job | Flow | Maker,
    generator_updates: dict[str, Any],
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseAbinitMaker,
) -> Job | Flow | Maker:
    """
    Update any attribute of any AbinitInputGenerator in the flow.

    Alternatively, if a Maker is supplied, the attributes of the maker will
    be updated.

    Note, this returns a copy of the original Job/Flow/Maker. I.e., the update does not
    happen in place.

    Parameters
    ----------
    flow : .Job or .Flow or .Maker
        A job, flow or Maker.
    generator_updates : dict
        The updates to apply to the input generator.
    name_filter : str or None
        A filter for the name of the jobs.
    class_filter : Maker or None
        A filter for the BaseAbinitMaker class used to generate the flows. Note the
        class filter will match any subclasses.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input flow/job/maker modified to use the updated factory settings.
    """
    dict_mod_updates = {
        f"input_set_generator->{k}": v for k, v in generator_updates.items()
    }

    return update_maker_kwargs(class_filter, dict_mod_updates, flow, name_filter)
