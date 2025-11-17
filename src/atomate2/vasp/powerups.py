"""Powerups for performing common modifications on VASP jobs and flows.

This module provides utility functions (powerups) to modify VASP computational
workflows, including updating INCAR settings, POTCAR configurations, k-points,
and custodian handlers. All powerup functions return modified copies without
altering the original objects.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TypeVar

from jobflow.core.flow import Flow
from jobflow.core.job import Job
from jobflow.core.maker import Maker
from pymatgen.io.vasp import Kpoints

from atomate2.common.powerups import add_metadata_to_flow as base_add_metadata_to_flow
from atomate2.common.powerups import update_custodian_handlers as base_custodian_handler
from atomate2.vasp.jobs.base import BaseVaspMaker

JobType = TypeVar("JobType", Job, Flow, Maker)
"""TypeVar for generic job, flow, or maker types. Used to ensure powerup functions
return the same type as their input."""


def update_vasp_input_generators(
    flow: JobType,
    dict_mod_updates: dict[str, Any],
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseVaspMaker,
) -> JobType:
    """Update VaspInputGenerators or Makers in a job, flow, or maker.

    This function applies modifications to VASP input generators throughout a
    workflow. It creates a deep copy of the input, so the original remains unchanged.

    Parameters
    ----------
    flow : Job | Flow | Maker
        A job, flow, or maker to update.
    dict_mod_updates : dict[str, Any]
        Dictionary of updates to apply using arrow notation (e.g.,
        'input_set_generator->user_incar_settings->ENCUT'). Existing keys are
        preserved unless explicitly overridden.
    name_filter : str | None, optional
        Filter to apply updates only to jobs matching this name pattern.
        Default is None (no filtering).
    class_filter : type[Maker] | None, optional
        Filter to apply updates only to makers of this class or its subclasses.
        Default is BaseVaspMaker.

    Returns
    -------
    Job | Flow | Maker
        A deep copy of the input with updated VASP input generator settings.
    """
    updated_flow = deepcopy(flow)

    tmp_dict = {
        "update": {"_set": dict_mod_updates},
        "name_filter": name_filter,
        "class_filter": class_filter,
        "dict_mod": True,
    }

    if isinstance(updated_flow, Maker):
        updated_flow = updated_flow.update_kwargs(**tmp_dict)
    else:
        updated_flow.update_maker_kwargs(**tmp_dict)

    return updated_flow


def update_user_incar_settings(
    flow: JobType,
    incar_updates: dict[str, Any],
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseVaspMaker,
) -> JobType:
    """Update user INCAR settings in VaspInputGenerators.

    Modifies the user_incar_settings attribute of VASP input generators within
    jobs, flows, or makers. Creates a copy of the input.

    Parameters
    ----------
    flow : Job | Flow | Maker
        A job, flow, or maker to update.
    incar_updates : dict[str, Any]
        Dictionary mapping INCAR tags to their new values (e.g.,
        {'ENCUT': 520, 'EDIFF': 1e-5}). Only specified keys are modified.
    name_filter : str | None, optional
        Filter to apply updates only to jobs matching this name pattern.
        Default is None (no filtering).
    class_filter : type[Maker] | None, optional
        Filter to apply updates only to makers of this class or its subclasses.
        Default is BaseVaspMaker.

    Returns
    -------
    Job | Flow | Maker
        A deep copy of the input with updated INCAR settings.

    Examples
    --------
    >>> flow = update_user_incar_settings(flow, {"ENCUT": 520, "ISMEAR": 0})
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
    flow: JobType,
    potcar_updates: dict[str, Any],
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseVaspMaker,
) -> JobType:
    """Update user POTCAR settings in VaspInputGenerators.

    Modifies the user_potcar_settings attribute of VASP input generators within
    jobs, flows, or makers. Creates a copy of the input.

    Parameters
    ----------
    flow : Job | Flow | Maker
        A job, flow, or maker to update.
    potcar_updates : dict[str, Any]
        Dictionary mapping element symbols to POTCAR specifications (e.g.,
        {'Fe': 'Fe_pv', 'O': 'O'}). Only specified elements are modified.
    name_filter : str | None, optional
        Filter to apply updates only to jobs matching this name pattern.
        Default is None (no filtering).
    class_filter : type[Maker] | None, optional
        Filter to apply updates only to makers of this class or its subclasses.
        Default is BaseVaspMaker.

    Returns
    -------
    Job | Flow | Maker
        A deep copy of the input with updated POTCAR settings.

    Examples
    --------
    >>> flow = update_user_potcar_settings(flow, {"Fe": "Fe_pv", "O": "O_s"})
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
    flow: JobType,
    potcar_functional: str,
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseVaspMaker,
) -> JobType:
    """Update POTCAR functional in VaspInputGenerators.

    Modifies the user_potcar_functional attribute of VASP input generators within
    jobs, flows, or makers. Creates a copy of the input.

    Parameters
    ----------
    flow : Job | Flow | Maker
        A job, flow, or maker to update.
    potcar_functional : str
        The POTCAR functional to use (e.g., 'PBE', 'PBE_52', 'PBE_54', 'LDA').
    name_filter : str | None, optional
        Filter to apply updates only to jobs matching this name pattern.
        Default is None (no filtering).
    class_filter : type[Maker] | None, optional
        Filter to apply updates only to makers of this class or its subclasses.
        Default is BaseVaspMaker.

    Returns
    -------
    Job | Flow | Maker
        A deep copy of the input with updated POTCAR functional.

    Examples
    --------
    >>> flow = update_user_potcar_functional(flow, "PBE_54")
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
    flow: JobType,
    kpoints_updates: dict[str, Any] | Kpoints,
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseVaspMaker,
) -> JobType:
    """Update user k-points settings in VaspInputGenerators.

    Modifies the user_kpoints_settings attribute of VASP input generators within
    jobs, flows, or makers. Creates a copy of the input.

    Parameters
    ----------
    flow : Job | Flow | Maker
        A job, flow, or maker to update.
    kpoints_updates : dict[str, Any] | Kpoints
        K-points updates to apply. Can be either:
        - A dictionary with k-points settings (e.g., {'reciprocal_density': 100})
        - A Kpoints object that replaces the entire user_kpoints_settings
    name_filter : str | None, optional
        Filter to apply updates only to jobs matching this name pattern.
        Default is None (no filtering).
    class_filter : type[Maker] | None, optional
        Filter to apply updates only to makers of this class or its subclasses.
        Default is BaseVaspMaker.

    Returns
    -------
    Job | Flow | Maker
        A deep copy of the input with updated k-points settings.

    Examples
    --------
    >>> flow = update_user_kpoints_settings(flow, {"reciprocal_density": 200})
    >>> # Or with a Kpoints object
    >>> kpts = Kpoints.gamma_automatic((4, 4, 4))
    >>> flow = update_user_kpoints_settings(flow, kpts)
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
    flow: JobType,
    value: bool = True,
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseVaspMaker,
) -> JobType:
    """Update automatic ISPIN setting in VaspInputGenerators.

    Controls whether ISPIN is automatically determined based on the magnetic
    moments in the structure. Creates a copy of the input.

    Parameters
    ----------
    flow : Job | Flow | Maker
        A job, flow, or maker to update.
    value : bool, optional
        Whether to enable automatic ISPIN determination. Default is True.
    name_filter : str | None, optional
        Filter to apply updates only to jobs matching this name pattern.
        Default is None (no filtering).
    class_filter : type[Maker] | None, optional
        Filter to apply updates only to makers of this class or its subclasses.
        Default is BaseVaspMaker.

    Returns
    -------
    Job | Flow | Maker
        A deep copy of the input with updated auto_ispin setting.

    Notes
    -----
    When auto_ispin is True, ISPIN=2 is used if the structure has non-zero
    magnetic moments, otherwise ISPIN=1 is used.
    """
    return update_vasp_input_generators(
        flow=flow,
        dict_mod_updates={"input_set_generator->auto_ispin": value},
        name_filter=name_filter,
        class_filter=class_filter,
    )


def add_metadata_to_flow(
    flow: Flow, additional_fields: dict, class_filter: type[Maker] = BaseVaspMaker
) -> Flow:
    """Add custom metadata fields to VASP task documents in a flow.

    Adds user-defined metadata to the task documents generated by VASP jobs,
    which is useful for organizing and querying results in databases.

    Parameters
    ----------
    flow : Flow
        The flow to which metadata will be added.
    additional_fields : dict
        Dictionary of metadata fields to add to task documents. Keys are field
        names, values are the metadata values.
    class_filter : Maker, optional
        The maker class to which metadata will be added. Only jobs created by
        this maker class or its subclasses will have metadata added.
        Default is BaseVaspMaker.

    Returns
    -------
    Flow
        A copy of the flow with metadata added to matching task documents.

    Examples
    --------
    >>> metadata = {"project": "battery_materials", "batch": "exp_001"}
    >>> flow = add_metadata_to_flow(flow, metadata)
    """
    return base_add_metadata_to_flow(
        flow=flow, class_filter=class_filter, additional_fields=additional_fields
    )


def update_vasp_custodian_handlers(
    flow: Flow, custom_handlers: tuple, class_filter: type[Maker] = BaseVaspMaker
) -> Flow:
    """Update custodian error handlers for VASP jobs in a flow.

    Replaces the default custodian error handlers with custom handlers,
    allowing users to customize error handling and recovery behavior or
    disable error handling entirely.

    Parameters
    ----------
    flow : Flow
        The flow whose custodian handlers will be updated.
    custom_handlers : tuple
        Tuple of custodian handler objects to use for error handling.
        Pass an empty tuple () to disable error handling.
    class_filter : Maker, optional
        The maker class for which handlers will be updated. Only jobs created
        by this maker class or its subclasses will have their handlers modified.
        Default is BaseVaspMaker.

    Returns
    -------
    Flow
        A copy of the flow with updated custodian handlers.

    Notes
    -----
    Custodian handlers are executed in the order they appear in the tuple.
    Common handlers include VaspErrorHandler, MeshSymmetryErrorHandler, etc.

    Examples
    --------
    >>> from custodian.vasp.handlers import VaspErrorHandler
    >>> handlers = (VaspErrorHandler(),)
    >>> flow = update_vasp_custodian_handlers(flow, handlers)
    """
    return base_custodian_handler(
        flow=flow, custom_handlers=custom_handlers, class_filter=class_filter
    )
