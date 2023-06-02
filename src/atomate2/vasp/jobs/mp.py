"""
Module defining jobs for Materials Project r2SCAN workflows

Reference: https://doi.org/10.1103/PhysRevMaterials.6.013801
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from monty.serialization import loadfn
from pkg_resources import resource_filename

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import StaticSetGenerator

if TYPE_CHECKING:
    from atomate2.vasp.sets.base import VaspInputGenerator


__all__ = ["MPPreRelaxMaker", "MPRelaxMaker", "MPStaticMaker"]

_BASE_MP_R2SCAN_RELAX_SET = loadfn(
    resource_filename("atomate2.vasp.sets", "BaseMPr2SCANRelaxSet.yaml")
)


class MPRelaxR2SCANGenerator(VaspInputGenerator):
    config_dict: dict = field(lambda: _BASE_MP_R2SCAN_RELAX_SET)


@dataclass
class MPPreRelaxMaker(BaseVaspMaker):
    """
    Maker to create VASP pre-relaxation job using PBEsol by default.

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

    name: str = "MP-PreRelax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPRelaxR2SCANGenerator(
            {"user_incar_settings": {"EDIFFG": -0.05, "METAGGA": None, "GGA": "PS"}}
        )
    )


@dataclass
class MPRelaxMaker(BaseVaspMaker):
    """
    Maker to create VASP relaxation job using r2SCAN by default.

    Parameters
    ----------
    name : str
        The job name.
    bandgap : float
        The bandgap of the material in eV. Used to determine the k-point density.
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

    name: str = "MP-Relax"
    bandgap: float = 0.0
    input_set_generator: VaspInputGenerator = field(
        default_factory=MPRelaxR2SCANGenerator
    )

    def __post_init__(self):
        if self.bandgap < 1e-4:
            kspacing = 0.22
            ismear = 2
            sigma = 0.2
        else:
            rmin = 25.22 - 2.87 * self.bandgap
            kspacing = 2 * np.pi * 1.0265 / (rmin - 1.0183)
            ismear = -5
            sigma = 0.05

        self.input_set_generator.config_dict["user_incar_settings"].update(
            KSPACING=kspacing if 0.22 < kspacing < 0.44 else 0.44,
            ISMEAR=ismear,
            SIGMA=sigma,
        )


@dataclass
class MPStaticMaker(MPRelaxMaker):
    """
    Maker to create VASP static job using r2SCAN by default.

    Parameters
    ----------
    name : str
        The job name.
    bandgap : float
        The bandgap of the material in eV. Used to determine the k-point density.
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

    name: str = "MP-Static"
    input_set_generator: VaspInputGenerator = field(StaticSetGenerator)
