"""Core jobs for running VASP calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.alchemy.transmuters import StandardTransmuter

from atomate2.common.utils import get_transformations
from atomate2.vasp.jobs.base import BaseVaspMaker, vasp_job
from atomate2.vasp.sets.core import (
    HSEBSSetGenerator,
    HSERelaxSetGenerator,
    HSEStaticSetGenerator,
    HSETightRelaxSetGenerator,
    NonSCFSetGenerator,
    RelaxSetGenerator,
    StaticSetGenerator,
    TightRelaxSetGenerator,
)

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Response
    from pymatgen.core.structure import Structure

    from atomate2.vasp.sets.base import VaspInputGenerator


logger = logging.getLogger(__name__)


@dataclass
class StaticMaker(BaseVaspMaker):
    """
    Maker to create VASP static jobs.

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

    name: str = "static"
    input_set_generator: VaspInputGenerator = field(default_factory=StaticSetGenerator)


@dataclass
class RelaxMaker(BaseVaspMaker):
    """
    Maker to create VASP relaxation jobs.

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

    name: str = "relax"
    input_set_generator: VaspInputGenerator = field(default_factory=RelaxSetGenerator)


@dataclass
class TightRelaxMaker(BaseVaspMaker):
    """
    Maker to create tight VASP relaxation jobs.

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

    name: str = "tight relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=TightRelaxSetGenerator
    )


@dataclass
class NonSCFMaker(BaseVaspMaker):
    """
    Maker to create non self consistent field VASP jobs.

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

    name: str = "non-scf"
    input_set_generator: VaspInputGenerator = field(default_factory=NonSCFSetGenerator)

    @vasp_job
    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None,
        mode: str = "uniform",
    ) -> Response:
        """Run a non-scf VASP job.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.
        mode : str
            Type of band structure calculation. Options are:
            - "line": Full band structure along symmetry lines.
            - "uniform": Uniform mesh band structure.
        """
        self.input_set_generator.mode = mode

        # parse DOS only for uniform band structure
        self.task_document_kwargs.setdefault("parse_dos", mode == "uniform")
        self.task_document_kwargs.setdefault("parse_bandstructure", mode)
        # copy previous inputs
        self.copy_vasp_kwargs.setdefault("additional_vasp_files", ("CHGCAR",))

        return super().make.original(self, structure, prev_dir)


@dataclass
class HSERelaxMaker(BaseVaspMaker):
    """
    Maker to create HSE06 relaxation jobs.

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

    name: str = "hse relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=HSERelaxSetGenerator
    )


@dataclass
class HSETightRelaxMaker(BaseVaspMaker):
    """
    Maker to create tight VASP relaxation jobs.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator
        A generator used to make the input set.
    write_input_set_kwargs
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    run_vasp_kwargs
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "hse tight relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=HSETightRelaxSetGenerator
    )


@dataclass
class HSEStaticMaker(BaseVaspMaker):
    """
    Maker to create HSE06 static jobs.

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

    name: str = "hse static"
    input_set_generator: VaspInputGenerator = field(
        default_factory=HSEStaticSetGenerator
    )


@dataclass
class HSEBSMaker(BaseVaspMaker):
    """
    Maker to create HSE06 band structure jobs.

    .. warning::
        The number of bands will automatically be adjusted based on the number of bands
        in the previous calculation. Therefore, if starting from a previous structure
        ensure you are starting from a static/relaxation calculation that has the same
        number of atoms (i.e., not a smaller/larger cell), as otherwise the number of
        bands may be set incorrectly.

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

    name: str = "hse band structure"
    input_set_generator: VaspInputGenerator = field(default_factory=HSEBSSetGenerator)

    @vasp_job
    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
        mode: Literal["line", "uniform", "gap"] = "uniform",
    ) -> Response:
        """Run an HSE06 band structure VASP job.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.
        mode : str = "uniform"
            Type of band structure calculation. Options are:
            - "line": Full band structure along symmetry lines.
            - "uniform": Uniform mesh band structure.
            - "gap": Get the energy at the CBM and VBM.
        """
        self.input_set_generator.mode = mode

        if mode == "gap" and prev_dir is None:
            logger.warning(
                "HSE band structure in 'gap' mode requires a previous VASP calculation "
                "directory from which to extract the VBM and CBM k-points. This "
                "calculation will instead be a standard uniform calculation."
            )
            mode = "uniform"

        # parse DOS only for uniform band structure
        self.task_document_kwargs.setdefault("parse_dos", "uniform" in mode)

        parse_bandstructure = "uniform" if mode == "gap" else mode
        self.task_document_kwargs.setdefault("parse_bandstructure", parse_bandstructure)

        # copy previous inputs
        if prev_dir is not None:
            self.copy_vasp_kwargs.setdefault("additional_vasp_files", ("CHGCAR",))

        return super().make.original(self, structure, prev_dir)


@dataclass
class DielectricMaker(BaseVaspMaker):
    """
    Maker to create dielectric calculation VASP jobs.

    .. Note::
        The input structure should be well relaxed to avoid imaginary modes. For
        example, using :obj:`TightRelaxMaker`.

    .. Note::
        If starting from a previous calculation, magnetism will be disabled if all
        MAGMOMs are less than 0.02.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .StaticSetGenerator
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

    name: str = "dielectric"
    input_set_generator: StaticSetGenerator = field(
        default_factory=lambda: StaticSetGenerator(lepsilon=True, auto_ispin=True)
    )


@dataclass
class TransmuterMaker(BaseVaspMaker):
    """
    A maker to apply transformations to a structure before writing the input sets.

    Note that if a transformation yields many structures, only the last structure in the
    list is used.

    Parameters
    ----------
    name : str
        The job name.
    transformations : tuple of str
        The transformations to apply. Given as a list of names of transformation classes
        as defined in the modules in pymatgen.transformations. For example,
        ``['DeformStructureTransformation', 'SupercellTransformation']``.
    transformation_params : tuple of dict or None
        The parameters used to instantiate each transformation class. Given as a list of
        dicts.
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

    name: str = "transmuter"
    transformations: tuple[str, ...] = field(default_factory=tuple)
    transformation_params: tuple[dict, ...] | None = None
    input_set_generator: VaspInputGenerator = field(default_factory=StaticSetGenerator)

    @vasp_job
    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ) -> Response:
        """Run a transmuter VASP job.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.
        """
        transformations = get_transformations(
            self.transformations, self.transformation_params
        )
        ts = TransformedStructure(structure)
        transmuter = StandardTransmuter([ts], transformations)
        structure = transmuter.transformed_structures[-1].final_structure

        # to avoid MongoDB errors, ":" is automatically converted to "."
        tjson = transmuter.transformed_structures[-1]
        self.write_additional_data.setdefault("transformations:json", tjson)

        return super().make.original(self, structure, prev_dir)
