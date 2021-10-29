"""Powerups for performing common modifications on VASP jobs and flows."""

from typing import Any, Dict, Optional, Type, Union

from jobflow import Flow, Job, Maker
from pymatgen.io.vasp import Kpoints

from atomate2.vasp.jobs.base import BaseVaspMaker


def update_user_incar_settings(
    flow: Union[Job, Flow],
    incar_updates: Dict[str, Any],
    name_filter: Optional[str] = None,
    class_filter: Optional[Type[Maker]] = BaseVaspMaker,
):
    """
    Update the user_incar_settings of any VaspInputSetGenerators in the flow.

    Parameters
    ----------
    flow
        A job or flow.
    incar_updates
        The updates to apply. Existing keys in user_incar_settings will not be modified
        unless explicitly specified in ``incar_updates``.
    name_filter
        A filter for the name of the jobs.
    class_filter
        A filter for the VaspMaker class used to generate the flows. Note the class
        filter will match any subclasses.
    """
    dict_mod_updates = {
        f"input_set_generator->user_incar_settings->{k}": v
        for k, v in incar_updates.items()
    }
    flow.update_maker_kwargs(
        {"_set": dict_mod_updates},
        name_filter=name_filter,
        class_filter=class_filter,
        dict_mod=True,
    )


def update_user_potcar_settings(
    flow: Union[Job, Flow],
    potcar_updates: Dict[str, Any],
    name_filter: Optional[str] = None,
    class_filter: Optional[Type[Maker]] = BaseVaspMaker,
):
    """
    Update the user_potcar_settings of any VaspInputSetGenerators in the flow.

    Parameters
    ----------
    flow
        A job or flow.
    potcar_updates
        The updates to apply. Existing keys in user_potcar_settings will not be modified
        unless explicitly specified in ``potcar_updates``.
    name_filter
        A filter for the name of the jobs.
    class_filter
        A filter for the VaspMaker class used to generate the flows. Note the class
        filter will match any subclasses.
    """
    dict_mod_updates = {
        f"input_set_generator->user_potcar_settings->{k}": v
        for k, v in potcar_updates.items()
    }
    flow.update_maker_kwargs(
        {"_set": dict_mod_updates},
        name_filter=name_filter,
        class_filter=class_filter,
        dict_mod=True,
    )


def update_user_kpoints_settings(
    flow: Union[Job, Flow],
    kpoints_updates: Union[Dict[str, Any], Kpoints],
    name_filter: Optional[str] = None,
    class_filter: Optional[Type[Maker]] = BaseVaspMaker,
):
    """
    Update the user_kpoints_settings of any VaspInputSetGenerators in the flow.

    Parameters
    ----------
    flow
        A job or flow.
    kpoints_updates
        The updates to apply. Can be specified as a dictionary or as a Kpoints object.
        If a dictionary is supplied, existing keys in user_kpoints_settings will not be
        modified unless explicitly specified in ``kpoints_updates``.
    name_filter
        A filter for the name of the jobs.
    class_filter
        A filter for the VaspMaker class used to generate the flows. Note the class
        filter will match any subclasses.
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

    flow.update_maker_kwargs(
        {"_set": dict_mod_updates},
        name_filter=name_filter,
        class_filter=class_filter,
        dict_mod=True,
    )
