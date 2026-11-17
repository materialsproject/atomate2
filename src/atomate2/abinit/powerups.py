"""Powerups for performing common modifications on ABINIT jobs and flows.

Powerups are utility functions that modify Job, Flow, or Maker objects to
update their configuration, such as ABINIT settings, k-points, or factory
parameters. All powerup functions return modified copies without altering
the original objects.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from jobflow import Flow, Job, Maker
from pymatgen.io.abinit.abiobjects import KSampling

from atomate2.abinit.files import del_gzip_files
from atomate2.abinit.jobs.base import BaseAbinitMaker

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "append_clean_flow",
    "update_factory_kwargs",
    "update_generator_attributes",
    "update_taskdoc_kwargs",
    "update_user_abinit_settings",
    "update_user_kpoints_settings",
]


def update_maker_kwargs(
    class_filter: Maker | None,
    dict_mod_updates: dict,
    flow: Job | Flow | Maker,
    name_filter: str | None,
) -> Job | Flow | Maker:
    """
    Update an object inside a Job, Flow, or Maker.

    A generic method to be shared for more specific update functions that
    will build the dict_mod_updates. Returns a copy of the original
    Job/Flow/Maker without modifying the input in place.

    Parameters
    ----------
    class_filter : Maker or None
        A filter for the class used to generate the flows. The class filter
        will match any subclasses. Default is None.
    dict_mod_updates : dict
        The updates to apply.
    flow : Job or Flow or Maker
        A job, flow, or Maker to update.
    name_filter : str or None
        A filter for the name of the jobs. Default is None.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input flow/job/maker with the applied updates.
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

    Alternatively, if a Maker is supplied, the user_abinit_settings of the maker
    will be updated. Returns a copy of the original Job/Flow/Maker without
    modifying the input in place.

    Parameters
    ----------
    flow : Job or Flow or Maker
        A job, flow, or Maker to update.
    abinit_updates : dict[str, Any]
        The updates to apply. Existing keys in user_abinit_settings will not
        be modified unless explicitly specified in ``abinit_updates``.
    name_filter : str or None
        A filter for the name of the jobs. Default is None.
    class_filter : type[Maker] or None
        A filter for the BaseAbinitMaker class used to generate the flows.
        The class filter will match any subclasses. Default is BaseAbinitMaker.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input flow/job/maker modified to use the updated
        ABINIT settings.
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

    Alternatively, if a Maker is supplied, the factory_kwargs of the maker
    will be updated. Returns a copy of the original Job/Flow/Maker without
    modifying the input in place.

    Parameters
    ----------
    flow : Job or Flow or Maker
        A job, flow, or Maker to update.
    factory_updates : dict[str, Any]
        The updates to apply. Existing keys in factory_kwargs will not be
        modified unless explicitly specified in ``factory_updates``.
    name_filter : str or None
        A filter for the name of the jobs. Default is None.
    class_filter : type[Maker] or None
        A filter for the BaseAbinitMaker class used to generate the flows.
        The class filter will match any subclasses. Default is BaseAbinitMaker.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input flow/job/maker modified to use the updated
        factory settings.
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

    Alternatively, if a Maker is supplied, the user_kpoints_settings of the maker
    will be updated. Returns a copy of the original Job/Flow/Maker without
    modifying the input in place.

    Parameters
    ----------
    flow : Job or Flow or Maker
        A job, flow, or Maker to update.
    kpoints_updates : dict[str, Any] or KSampling
        The updates to apply. Can be specified as a dictionary or as a
        KSampling object. If a dictionary is supplied, existing keys in
        user_kpoints_settings will not be modified unless explicitly
        specified in ``kpoints_updates``.
    name_filter : str or None
        A filter for the name of the jobs. Default is None.
    class_filter : type[Maker] or None
        A filter for the BaseAbinitMaker class used to generate the flows.
        The class filter will match any subclasses. Default is BaseAbinitMaker.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input flow/job/maker modified to use the updated
        k-points settings.
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

    Alternatively, if a Maker is supplied, the attributes of the maker
    will be updated. Returns a copy of the original Job/Flow/Maker without
    modifying the input in place.

    Parameters
    ----------
    flow : Job or Flow or Maker
        A job, flow, or Maker to update.
    generator_updates : dict[str, Any]
        The updates to apply to the input generator.
    name_filter : str or None
        A filter for the name of the jobs. Default is None.
    class_filter : type[Maker] or None
        A filter for the BaseAbinitMaker class used to generate the flows.
        The class filter will match any subclasses. Default is BaseAbinitMaker.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input flow/job/maker modified to use the updated
        generator attributes.
    """
    dict_mod_updates = {
        f"input_set_generator->{k}": v for k, v in generator_updates.items()
    }

    return update_maker_kwargs(class_filter, dict_mod_updates, flow, name_filter)


def update_taskdoc_kwargs(
    flow: Job | Flow | Maker,
    taskdoc_updates: dict[str, Any],
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseAbinitMaker,
) -> Job | Flow | Maker:
    """
    Update any attribute of any AbinitTaskDoc in the flow.

    Alternatively, if a Maker is supplied, the attributes of the maker
    will be updated. Returns a copy of the original Job/Flow/Maker without
    modifying the input in place.

    Parameters
    ----------
    flow : Job or Flow or Maker
        A job, flow, or Maker to update.
    taskdoc_updates : dict[str, Any]
        The updates to apply to the task document kwargs.
    name_filter : str or None
        A filter for the name of the jobs. Default is None.
    class_filter : type[Maker] or None
        A filter for the BaseAbinitMaker class used to generate the flows.
        The class filter will match any subclasses. Default is BaseAbinitMaker.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input flow/job/maker modified to use the updated
        task document kwargs.
    """
    dict_mod_updates = {
        f"task_document_kwargs->{k}": v for k, v in taskdoc_updates.items()
    }

    return update_maker_kwargs(class_filter, dict_mod_updates, flow, name_filter)


def append_clean_flow(
    flow: Job | Flow,
    exclude_files_from_zip: list[str | Path] | None = None,
    delete: bool = True,
    exclude_files_from_del: list[str | Path] | None = None,
    include_files_to_del: list[str | Path] | None = None,
) -> Flow:
    r"""
    Append a job to delete and/or compress files of the flow.

    Returns a copy of the original Job/Flow without modifying the input
    in place.

    Parameters
    ----------
    flow : Job or Flow
        A job or flow to append cleanup to.
    exclude_files_from_zip : list[str or Path] or None
        Filenames to exclude from compression. Supports glob file matching,
        e.g., "\*.dat". Default is None.
    delete : bool
        Whether to activate deletion of files. Default is True.
    exclude_files_from_del : list[str or Path] or None
        Filenames to exclude from deletion. Supports glob file matching,
        e.g., "\*.dat". Default is None.
    include_files_to_del : list[str or Path] or None
        Filenames to include for deletion as a list of str or Path objects
        given relative to directory. Glob file paths are supported,
        e.g., "\*.dat". If None, all files in the directory will be deleted.
        Default is None.

    Returns
    -------
    Flow
        A copy of the input flow/job modified to delete/compress files
        once completed.
    """
    copied_flow = deepcopy(flow)
    outputs_to_clean = []
    if isinstance(copied_flow, Job):
        outputs_to_clean.append(copied_flow.output)
    elif isinstance(copied_flow, Flow):
        for job, _ in copied_flow.iterflow():
            outputs_to_clean.append(job.output)
    else:
        raise TypeError(
            f"The function 'del_gzip_files' accepts Job or Flow as input, "
            f"but {type(copied_flow)} was passed."
        )

    return Flow(
        [
            copied_flow,
            del_gzip_files(
                outputs=outputs_to_clean,
                exclude_files_from_zip=exclude_files_from_zip,
                delete=delete,
                exclude_files_from_del=exclude_files_from_del,
                include_files_to_del=include_files_to_del,
            ),
        ],
        output=copied_flow.output,
    )
