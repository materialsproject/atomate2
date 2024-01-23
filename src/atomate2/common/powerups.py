"""Utilities for modifying workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jobflow import Flow, Maker


def add_metadata_to_flow(
    flow: Flow, additional_fields: dict, class_filter: Maker
) -> Flow:
    """
    Return the flow with additional field(metadata) to the task doc.

    This allows adding metadata to the task-docs, could be useful
    to query results from DB.

    Parameters
    ----------
    flow:
    additional_fields : dict
        A dict with metadata.
    class_filter: .Maker
        The Maker to which additional metadata needs to be added

    Returns
    -------
    Flow
        Flow with added metadata to the task-doc.
    """
    flow.update_maker_kwargs(
        {
            "_set": {
                f"task_document_kwargs->additional_fields->{field}": value
                for field, value in additional_fields.items()
            }
        },
        dict_mod=True,
        class_filter=class_filter,
    )

    return flow


def update_custodian_handlers(
    flow: Flow, custom_handlers: tuple, class_filter: Maker
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
    code = class_filter.name.split(" ")[1]
    flow.update_maker_kwargs(
        {"_set": {f"run_{code}_kwargs->handlers": custom_handlers}},
        dict_mod=True,
        class_filter=class_filter,
    )

    return flow
