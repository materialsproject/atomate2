"""Jobs used in the calculation of elastic tensors."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.sets.core import StaticSetGenerator

if TYPE_CHECKING:
    from atomate2.vasp.sets.base import VaspInputGenerator

logger = logging.getLogger(__name__)


@dataclass
class ElasticRelaxMaker(BaseVaspMaker):
    """
    Maker to perform an elastic relaxation.

    The input set is for a tight relaxation, where only the atomic positions are
    allowed to relax (ISIF=2). Both the k-point mesh density and convergence parameters
    are stricter than a normal relaxation.

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

    name: str = "elastic relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings={"grid_density": 7_000},
            user_incar_settings={
                "IBRION": 2,
                "ISIF": 2,
                "ENCUT": 700,
                "EDIFF": 1e-7,
                "LAECHG": False,
                "EDIFFG": -0.001,
                "LREAL": False,
                "ALGO": "Normal",
                "NSW": 99,
                "LCHARG": False,
            },
        )
    )
