"""Utilities for modifying workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jobflow.core.flow import Flow
    from jobflow.core.maker import Maker


def add_metadata_to_flow(
    flow: Flow, additional_fields: dict, class_filter: type[Maker]
) -> Flow:
    """
    Add additional metadata fields to task documents in a flow.

    This function updates the task document kwargs for jobs in the flow,
    allowing metadata to be added for easier querying of results from a database.

    This function does not add metadata to the job themselves, only to the output
    generated upon job completion.

    Parameters
    ----------
    flow : Flow
        The jobflow Flow object to modify.
    additional_fields : dict
        Dictionary containing metadata fields and their values to add to task documents.
    class_filter : type[Maker]
        The Maker class type to which additional metadata should be added.
        Only jobs created by this Maker type will be modified.

    Returns
    -------
    Flow
        The modified flow with added metadata in the task documents.

    Examples
    --------
    >>> from atomate2.vasp.flows.core import RelaxBandStructureMaker
    >>> flow = RelaxBandStructureMaker().make(structure)
    >>> metadata = {"project": "battery_materials", "version": "2.0"}
    >>> flow = add_metadata_to_flow(flow, metadata, RelaxBandStructureMaker)
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
    flow: Flow, custom_handlers: tuple, class_filter: type[Maker]
) -> Flow:
    """
    Update custodian error handlers for VASP jobs in a flow.

    This function allows selective configuration of error-correcting handlers
    for VASP jobs or complete removal of error handlers.

    Parameters
    ----------
    flow : Flow
        The jobflow Flow object to modify.
    custom_handlers : tuple
        Tuple of custodian handler objects to use for error correction.
        Pass an empty tuple to disable all error handlers.
    class_filter : type[Maker]
        The Maker class type to which custom custodian handlers should be applied.
        Only jobs created by this Maker type will be modified.

    Returns
    -------
    Flow
        The modified flow with updated custodian handlers.

    Examples
    --------
    >>> from custodian.vasp.handlers import VaspErrorHandler, MeshSymmetryErrorHandler
    >>> from atomate2.vasp.flows.core import RelaxBandStructureMaker
    >>> flow = RelaxBandStructureMaker().make(structure)
    >>> handlers = (VaspErrorHandler(), MeshSymmetryErrorHandler())
    >>> flow = update_custodian_handlers(flow, handlers, RelaxBandStructureMaker)
    """
    code = class_filter.name.split(" ")[1]

    flow.update_maker_kwargs(
        {"_set": {f"run_{code}_kwargs->handlers": custom_handlers}},
        dict_mod=True,
        class_filter=class_filter,
    )

    return flow
