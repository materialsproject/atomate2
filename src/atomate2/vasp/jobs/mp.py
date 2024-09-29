"""
Module defining Materials Project job makers.

Reference: https://doi.org/10.1103/PhysRevMaterials.6.013801

In case of questions, consult @Andrew-S-Rosen, @esoteric-ephemera or @janosh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pymatgen.io.vasp.sets import (
    MPRelaxSet,
    MPScanRelaxSet,
    MPScanStaticSet,
    MPStaticSet,
)

from atomate2.vasp.jobs.base import BaseVaspMaker

if TYPE_CHECKING:
    from atomate2.vasp.sets.base import VaspInputGenerator


@dataclass
class MPGGARelaxMaker(BaseVaspMaker):
    """
    Maker to create VASP relaxation job using PBE GGA by default.

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

    name: str = "MP GGA relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPRelaxSet(
            force_gamma=True, auto_metal_kpoints=True, inherit_incar=False
        )
    )


@dataclass
class MPGGAStaticMaker(BaseVaspMaker):
    """
    Maker to create VASP static job using PBE GGA by default.

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

    name: str = "MP GGA static"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPStaticSet(
            force_gamma=True, auto_metal_kpoints=True, inherit_incar=False
        )
    )


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

    name: str = "MP pre-relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPScanRelaxSet(
            auto_ismear=False,
            inherit_incar=False,
            user_incar_settings={
                "EDIFFG": -0.05,
                "GGA": "PS",
                "LWAVE": True,
                "LCHARG": True,
                "LELF": False,  # prevents KPAR > 1
                "METAGGA": None,
            },
        )
    )


@dataclass
class MPMetaGGARelaxMaker(BaseVaspMaker):
    """
    Maker to create VASP relaxation job using r2SCAN by default.

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

    name: str = "MP meta-GGA relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPScanRelaxSet(
            auto_ismear=False,
            inherit_incar=False,
            user_incar_settings={
                "GGA": None,  # unset GGA, shouldn't be set anyway but best be sure
                "LCHARG": True,
                "LWAVE": True,
                "LELF": False,  # prevents KPAR > 1
            },
        )
    )


@dataclass
class MPMetaGGAStaticMaker(BaseVaspMaker):
    """
    Maker to create VASP static job using r2SCAN by default.

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

    name: str = "MP meta-GGA static"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPScanStaticSet(
            auto_ismear=False,
            inherit_incar=False,
            user_incar_settings={
                "ALGO": "FAST",
                "GGA": None,  # unset GGA, shouldn't be set anyway but best be sure
                "LCHARG": True,
                "LWAVE": False,
                "LVHAR": None,  # not needed, unset
                "LELF": False,  # prevents KPAR > 1
            },
        )
    )
