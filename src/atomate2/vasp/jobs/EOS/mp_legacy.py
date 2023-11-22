"""
EOS jobs using atomate1-compatible PBE-GGA params.

This implementation is consistent with
K. Latimer, S. Dwaraknath, K. Mathew, D. Winston, and K.A. Persson,
npj Comput. Materials 4, 40 (2018),
DOI: 10.1038/s41524-018-0091-x

Also see the original atomate workflows

- atomate.vasp.workflows.base.wf_bulk_modulus:
https://github.com/hackingmaterials/atomate/blob/main/atomate/vasp/workflows/presets/core.py#L564
- atomate.vasp.workflows.base.bulk_modulus.get_wf_bulk_modulus:
https://github.com/hackingmaterials/atomate/blob/main/atomate/vasp/workflows/base/bulk_modulus.py#L21

These WFs are interesting that the k-point density is **extremely** high
despite the convergence tests in the SI of the Latimer et al. paper not
showing strong sensitivity when
    "number k-points per reciprocal atom" >= 3,000
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.eos.base import DeformationMaker
from atomate2.vasp.sets.eos import (
    MPLegacyEosRelaxSetGenerator,
    MPLegacyEosStaticSetGenerator,
)

if TYPE_CHECKING:
    from atomate2.vasp.sets.base import VaspInputGenerator


@dataclass
class MPLegacyEosRelaxMaker(BaseVaspMaker):
    """
    Maker to create MP atomate1-compatible GGA relax job in VASP.

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

    name: str = "EOS MP legacy GGA relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPLegacyEosRelaxSetGenerator(
            user_incar_settings={"ISIF": 3}
        )
    )


@dataclass
class MPLegacyEosStaticMaker(BaseVaspMaker):
    """
    Maker to create MP atomate1-compatible GGA static job in VASP.

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

    name: str = "EOS MP legacy GGA static"
    input_set_generator: VaspInputGenerator = field(
        default_factory=MPLegacyEosStaticSetGenerator
    )


@dataclass
class MPLegacyDeformationMaker(DeformationMaker):
    """
    Maker to deform input structure and relax using MP-atomate1-PBE-GGA params.

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

    name: str = "EOS MP legacy GGA deform and relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=MPLegacyEosRelaxSetGenerator
    )
