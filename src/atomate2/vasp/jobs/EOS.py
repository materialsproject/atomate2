"""
Module to define EOS jobs

For consistency with atomate implementation, define eos_relax_maker
and eos_deformation_maker with updated parameters

Also define MP-compatible jobs mp_gga_eos_relax_maker and 
mp_gga_eos_deformation_maker.
Using snake case so the names are legible
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import TransmuterMaker
from atomate2.vasp.sets.core import eos_set_generator
from atomate2.vasp.sets.mp import mp_gga_eos_set_generator

if TYPE_CHECKING:
    from atomate2.vasp.sets.base import VaspInputGenerator


@dataclass
class eos_relax_maker(BaseVaspMaker):
    """
    Maker to create VASP relaxation job using EOS parameters

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "EOS GGA relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory= lambda : eos_set_generator(
            user_incar_settings = {"ISIF": 3}
        )
    )

@dataclass
class eos_deformation_maker(TransmuterMaker):
    """
    Maker to create a deformed input struc and relax using EOS parameters

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "EOS GGA deform and relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=eos_set_generator
    )

@dataclass
class mp_gga_eos_relax_maker(BaseVaspMaker):
    """
    Maker to create VASP relaxation job using EOS parameters

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "MP EOS GGA relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory= lambda : mp_gga_eos_set_generator(
            user_incar_settings = {"ISIF": 3}
        )
    )

@dataclass
class mp_gga_eos_deformation_maker(TransmuterMaker):
    """
    Maker to create a deformed input struc and relax using EOS parameters

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "MP EOS GGA deform and relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=mp_gga_eos_set_generator
    )
