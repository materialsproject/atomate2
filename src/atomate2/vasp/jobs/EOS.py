"""
Module to define EOS jobs.

For consistency with atomate implementation, define eos_relax_maker
and eos_deformation_maker with updated parameters

Also define MP-compatible jobs mp_gga_eos_relax_maker and
mp_gga_eos_deformation_maker.
Using snake case so the names are legible
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)

from atomate2.vasp.jobs.base import BaseVaspMaker, vasp_job
from atomate2.vasp.sets.core import eos_set_generator
from atomate2.vasp.sets.mp import mp_gga_eos_set_generator

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Structure

    from atomate2.vasp.sets.base import VaspInputGenerator


@dataclass
class deformation_maker(BaseVaspMaker):
    """
    A maker to apply deformations to a structure before writing the input sets.

    Modified version of vasp.jobs.core.TransmuterMaker,
    allows calling deformation on the fly rather than as class attr.

    Note that if a transformation yields many structures, only the last structure in the
    list is used.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : StaticSetGenerator
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

    name: str = "deformation and relaxation"
    input_set_generator: VaspInputGenerator = field(default_factory=eos_set_generator)

    @vasp_job
    def make(
        self,
        structure: Structure,
        deformation_matrix: list | tuple,
        prev_vasp_dir: str | Path | None = None,
    ):
        """
        Run a deformation and relaxation VASP job.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_vasp_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.
        deformation_matrix : list or tuple
            The deformation matrix to apply.
            Should be a 3x3 square matrix in list or tuple form
        transformation_params : tuple of dict or None
            The parameters used to instantiate each transformation class.
            Given as a list of dicts.

        """
        deformation = DeformStructureTransformation(deformation=deformation_matrix)
        structure = deformation.apply_transformation(structure)
        return super().make.original(self, structure, prev_vasp_dir)


@dataclass
class eos_relax_maker(BaseVaspMaker):
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
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: eos_set_generator(user_incar_settings={"ISIF": 3})
    )


@dataclass
class mp_gga_eos_relax_maker(BaseVaspMaker):
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

    name: str = "MP EOS GGA relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: mp_gga_eos_set_generator(
            user_incar_settings={"ISIF": 3}
        )
    )


@dataclass
class mp_gga_deformation_maker(deformation_maker):
    """
    Maker to create a deformed input structure and relax using EOS parameters.

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
