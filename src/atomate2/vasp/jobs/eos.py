"""
Module to define EOS jobs using the default atomate2 parameters.

Some jobs have prefixes to indicate their purpose:
- MPLegacy: for consistency with the atomate 1 implementation, or
[LDMWP] K. Latimer, S. Dwaraknath, K. Mathew, D. Winston, and K.A. Persson,
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
    "number of k-points per reciprocal atom" >= 3,000

- MPGGA: MP-compatible PBE-GGA jobs
- MPMetaGGA: MP-compatible r2SCAN meta-GGA jobs

MPGGA and MPMetaGGA jobs use the highest k-point density in standard MP jobs,
KSPACING = 0.22, which is comparable to
KPPRA = k-points per reciprocal atom = 3000

This is justified by the SI of [LDMWP], which shows in Fig. S12
that all EOS parameters (E0, V0, B0, B1) do not deviate by more than 1.5%,
and typically less than 0.1%, from a well-converged value when KPPRA = 3,000.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.sets.eos import (
    EosSetGenerator,
    MPGGAEosRelaxSetGenerator,
    MPGGAEosStaticSetGenerator,
    MPLegacyEosRelaxSetGenerator,
    MPLegacyEosStaticSetGenerator,
    MPMetaGGAEosPreRelaxSetGenerator,
    MPMetaGGAEosRelaxSetGenerator,
    MPMetaGGAEosStaticSetGenerator,
)

if TYPE_CHECKING:
    from atomate2.vasp.sets.base import VaspInputGenerator


copy_wavecar = lambda: {"additional_vasp_files": ("WAVECAR",)}  # noqa: E731


# No prefix, base atomate 2 parameters
@dataclass
class EosRelaxMaker(BaseVaspMaker):
    """
    Maker to create VASP relaxation job using EOS parameters.

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
    input_set_generator: VaspInputGenerator = field(default_factory=EosSetGenerator)
    copy_vasp_kwargs: dict = field(default_factory=copy_wavecar)


# MPLegacy prefix, legacy MP PBE-GGA
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
        default_factory=MPLegacyEosRelaxSetGenerator
    )
    copy_vasp_kwargs: dict = field(default_factory=copy_wavecar)


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
    copy_vasp_kwargs: dict = field(default_factory=copy_wavecar)


# MPGGA prefix, MP PBE-GGA compatible parameters
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
        default_factory=MPGGAEosRelaxSetGenerator
    )
    copy_vasp_kwargs: dict = field(default_factory=copy_wavecar)


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
    copy_vasp_kwargs: dict = field(default_factory=copy_wavecar)


# MPMetaGGA prefix, MP r2SCAN-meta-GGA compatible
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
        default_factory=MPMetaGGAEosPreRelaxSetGenerator
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
        default_factory=MPMetaGGAEosRelaxSetGenerator
    )
    copy_vasp_kwargs: dict = field(default_factory=copy_wavecar)


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

    name: str = "EOS MP meta-GGA static"
    input_set_generator: VaspInputGenerator = field(
        default_factory=MPMetaGGAEosStaticSetGenerator
    )
    copy_vasp_kwargs: dict = field(default_factory=copy_wavecar)
