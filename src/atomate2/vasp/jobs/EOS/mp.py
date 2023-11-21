"""
Module to create MP-compatible EOS jobs using updated parameters.

MPGGA prefix: EOS jobs using MP's default PBE-GGA parameters

MPMetaGGA prefix: EOS jobs using MP's default r2SCAN Meta-GGA parameters

These jobs use the highest k-point density in standard MP jobs,
KSPACING = 0.22, which is comparable to
    KPPRA = k-points per reciprocal atom = 3000

This is justified by the SI of

    K. Latimer, S. Dwaraknath, K. Mathew, D. Winston, and K.A. Persson,
    npj Comput. Materials 4, 40 (2018),
    DOI: 10.1038/s41524-018-0091-x

which shows in Fig. S12 that all EOS parameters (E0, V0, B0, B1)
do not deviate by more than 1.5%, and typically less than 0.1%,
from a well-converged value when KPPRA = 3,000.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.EOS.base import DeformationMaker
from atomate2.vasp.sets.EOS import (
    MPGGAEosRelaxSetGenerator,
    MPGGAEosStaticSetGenerator,
    MPMetaGGAEosPreRelaxSetGenerator,
    MPMetaGGAEosRelaxSetGenerator,
    MPMetaGGAEosStaticSetGenerator,
)

if TYPE_CHECKING:
    from atomate2.vasp.sets.base import VaspInputGenerator


@dataclass
class MPGGAEosRelaxMaker(BaseVaspMaker):
    """
    Maker to create MP-compatible GGA relax job in VASP using EOS parameters.

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

    name: str = "EOS MP GGA relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPGGAEosRelaxSetGenerator(
            user_incar_settings={"ISIF": 3}
        )
    )


@dataclass
class MPGGAEosStaticMaker(BaseVaspMaker):
    """
    Maker to create MP-compatible GGA static job in VASP using EOS parameters.

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

    name: str = "EOS MP GGA static"
    input_set_generator: VaspInputGenerator = field(
        default_factory=MPGGAEosStaticSetGenerator
    )


@dataclass
class MPGGADeformationMaker(DeformationMaker):
    """
    Maker to deform input structure and relax using MP-PBE-GGA EOS parameters.

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

    name: str = "EOS MP GGA deform and relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=MPGGAEosRelaxSetGenerator
    )


@dataclass
class MPMetaGGAEosPreRelaxMaker(BaseVaspMaker):
    """
    Maker to create MP-compatible r2SCAN relax job in VASP using EOS parameters.

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

    name: str = "EOS MP meta-GGA pre-relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPMetaGGAEosPreRelaxSetGenerator(
            user_incar_settings={"ISIF": 3}
        )
    )


@dataclass
class MPMetaGGAEosRelaxMaker(BaseVaspMaker):
    """
    Maker to create MP-compatible r2SCAN relax job in VASP using EOS parameters.

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

    name: str = "EOS MP meta-GGA relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPMetaGGAEosRelaxSetGenerator(
            user_incar_settings={"ISIF": 3}
        )
    )


@dataclass
class MPMetaGGAEosStaticMaker(BaseVaspMaker):
    """
    Maker to create MP-compatible meta-GGA static job in VASP using EOS parameters.

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

    name: str = "EOS MP meta-GGA relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=MPMetaGGAEosStaticSetGenerator
    )


@dataclass
class MPMetaGGADeformationMaker(DeformationMaker):
    """
    Maker to deform input structure and relax using MP-meta-GGA EOS parameters.

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

    name: str = "EOS MP meta-GGA deform and relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=MPMetaGGAEosRelaxSetGenerator
    )
