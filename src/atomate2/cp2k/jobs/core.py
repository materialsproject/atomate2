"""Core jobs for running CP2K calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from custodian.cp2k.handlers import (
    AbortHandler,
    FrozenJobErrorHandler,
    NumericalPrecisionHandler,
    StdErrHandler,
    WalltimeHandler,
)
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.alchemy.transmuters import StandardTransmuter

from atomate2.common.utils import get_transformations
from atomate2.cp2k.jobs.base import BaseCp2kMaker, cp2k_job
from atomate2.cp2k.sets.core import (
    CellOptSetGenerator,
    HybridCellOptSetGenerator,
    HybridRelaxSetGenerator,
    HybridStaticSetGenerator,
    MDSetGenerator,
    NonSCFSetGenerator,
    RelaxSetGenerator,
    StaticSetGenerator,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure

    from atomate2.cp2k.sets.base import Cp2kInputGenerator


logger = logging.getLogger(__name__)


@dataclass
class StaticMaker(BaseCp2kMaker):
    """
    Maker to create CP2K static jobs.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .Cp2kInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_cp2k_input_set`.
    copy_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_cp2k_outputs`.
    run_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_cp2k`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
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
    input_set_generator: Cp2kInputGenerator = field(default_factory=StaticSetGenerator)


@dataclass
class RelaxMaker(BaseCp2kMaker):
    """
    Maker to create CP2K relaxation jobs.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .Cp2kInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_cp2k_input_set`.
    copy_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_cp2k_outputs`.
    run_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_cp2k`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
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
    input_set_generator: Cp2kInputGenerator = field(default_factory=RelaxSetGenerator)


@dataclass
class CellOptMaker(BaseCp2kMaker):
    """
    Maker to create CP2K cell optimization jobs.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .Cp2kInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_cp2k_input_set`.
    copy_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_cp2k_outputs`.
    run_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_cp2k`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
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
    input_set_generator: Cp2kInputGenerator = field(default_factory=CellOptSetGenerator)


@dataclass
class HybridStaticMaker(BaseCp2kMaker):
    """
    Maker for static hybrid jobs.

    Parameters
    ----------
    name : str
        The job name.
    hybrid_functional : str
        Built-in hybrid functional to use.
    input_set_generator : .Cp2kInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_cp2k_input_set`.
    copy_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_cp2k_outputs`.
    run_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_cp2k`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "hybrid static"
    input_set_generator: Cp2kInputGenerator = field(
        default_factory=HybridStaticSetGenerator
    )


@dataclass
class HybridRelaxMaker(BaseCp2kMaker):
    """
    Maker for relax hybrid jobs.

    Parameters
    ----------
    name : str
        The job name.
    hybrid_functional : str
        Built-in hybrid functional to use.
    input_set_generator : .Cp2kInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_cp2k_input_set`.
    copy_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_cp2k_outputs`.
    run_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_cp2k`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "hybrid relax"
    input_set_generator: Cp2kInputGenerator = field(
        default_factory=HybridRelaxSetGenerator
    )


@dataclass
class HybridCellOptMaker(BaseCp2kMaker):
    """
    Maker for cell opt. hybrid jobs.

    Parameters
    ----------
    name : str
        The job name.
    hybrid_functional : str
        Built-in hybrid functional to use.
    input_set_generator : .Cp2kInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_cp2k_input_set`.
    copy_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_cp2k_outputs`.
    run_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_cp2k`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "hybrid cell opt"
    input_set_generator: Cp2kInputGenerator = field(
        default_factory=HybridCellOptSetGenerator
    )


@dataclass
class NonSCFMaker(BaseCp2kMaker):
    """
    Maker to create non self consistent field CP2K jobs.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .Cp2kInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_cp2k_input_set`.
    copy_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_cp2k_outputs`.
    run_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_cp2k`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
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
    input_set_generator: Cp2kInputGenerator = field(default_factory=NonSCFSetGenerator)

    # Explicitly pass the handlers to avoid the unconverged scf handler
    run_cp2k_kwargs: dict = field(
        default_factory=lambda: {
            "handlers": (
                StdErrHandler(),
                FrozenJobErrorHandler(),
                AbortHandler(),
                NumericalPrecisionHandler(),
                WalltimeHandler(),
            ),
            "validators": (),
        }
    )

    @cp2k_job
    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None,
        mode: str = "uniform",
    ) -> None:
        """Run a non-scf CP2K job.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous CP2K calculation directory to copy output files from.
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
        self.copy_cp2k_kwargs.setdefault("additional_cp2k_files", ("wfn",))

        return super().make.original(self, structure, prev_dir)


@dataclass
class TransmuterMaker(BaseCp2kMaker):
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
        Keyword arguments that will get passed to :obj:`.write_cp2k_input_set`.
    copy_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_cp2k_outputs`.
    run_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_cp2k`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
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
    input_set_generator: Cp2kInputGenerator = field(default_factory=StaticSetGenerator)

    @cp2k_job
    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ) -> None:
        """Run a transmuter Cp2k job.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous Cp2k calculation directory to copy output files from.
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


@dataclass
class MDMaker(BaseCp2kMaker):
    """
    Maker for creating MD jobs.

    Parameters
    ----------
    name
        The job name.
    input_set_generator
        A generator used to make the input set.
    task_document_kwargs
        Task document kwargs to pass to the base maker. By default
        this maker will turn-on the storing of a trajectory.
    """

    name: str = "md"
    input_set_generator: Cp2kInputGenerator = field(default_factory=MDSetGenerator)
    task_document_kwargs: dict = field(
        default_factory=lambda: {"store_trajectory": True}
    )
