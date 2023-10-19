"""
Module to define EOS jobs.

For consistency with atomate implementation, define EosRelaxMaker
and EosDeformationMaker with updated parameters

Also define MP-compatible PBE-GGA jobs:
    MPGGAEosRelaxMaker, MPGGADeformationMaker, and MPGGAEosStaticMaker;
and MP-compatible r2SCAN meta-GGA jobs:
    MPMetaGGAEosRelaxMaker, MPMetaGGADeformationMaker,
    and MPMetaGGAEosStaticMaker
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from jobflow import job
from pymatgen.analysis.eos import EOS
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)

from atomate2.vasp.jobs.base import BaseVaspMaker, vasp_job
from atomate2.vasp.sets.EOS import (
    EosSetGenerator,
    MPGGAEosRelaxSetGenerator,
    MPGGAEosStaticSetGenerator,
    MPMetaGGAEosRelaxSetGenerator,
    MPMetaGGAEosStaticSetGenerator,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Structure

    from atomate2.vasp.sets.base import VaspInputGenerator


@job
def postprocess_EOS(data_dict: dict[str, list], EOS_models: list | None = None):
    """Take dict of E(V) data and fit to each equation of state in EOS_models."""
    if EOS_models is None:
        EOS_models = [
            "murnaghan",
            "birch",
            "birch_murnaghan",
            "pourier_tarantola",
            "vinet",
        ]

    output = {"EV": data_dict.copy(), "EOS": {}}
    for key in data_dict:
        output["EOS"][key] = {}  # type: ignore[index]
        data = np.asarray(data_dict[key])
        for eos_name in EOS_models:
            eos = EOS(eos_name=eos_name).fit(data[:, 0], data[:, 1])
            eos_d = {**eos.results, "b0 GPa": float(eos.b0_GPa)}
            output["EOS"][key][eos_name] = eos_d  # type: ignore[index]
    return output


"""
No prefix: job definitoons using the default atomate2 parameter sets
"""


@dataclass
class DeformationMaker(BaseVaspMaker):
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

    name: str = "EOS deform and relax"
    input_set_generator: VaspInputGenerator = field(default_factory=EosSetGenerator)

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
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: EosSetGenerator(user_incar_settings={"ISIF": 3})
    )


"""
MPGGA prefix: Same jobs but using MP's default PBE-GGA parameters
"""


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


"""
MPMetaGGA prefix: jobs using MP's default r2SCAN Meta-GGA parameters
"""


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
