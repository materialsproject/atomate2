""" Jobs that compose MPMorph flows. """

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.vasp.jobs.md import MDMaker
from atomate2.vasp.sets.mpmorph import MPMorphMDSetGenerator

if TYPE_CHECKING:
    from atomate2.vasp.sets.base import VaspInputGenerator
    from atomate2.vasp.jobs.base import BaseVaspMaker

@dataclass
class BaseMPMorphMDMaker(MDMaker):
    """
    Maker to create VASP molecular dynamics jobs for amorphous materials.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputSetGenerator
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

    name : str = "MPMorph MD Maker"
    input_set_generator : VaspInputGenerator = field(
        default_factory = MPMorphMDSetGenerator
    )

