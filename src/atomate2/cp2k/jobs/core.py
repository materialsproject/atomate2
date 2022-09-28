"""Core jobs for running CP2K calculations."""

from __future__ import annotations
from email.policy import default
from lib2to3.pytree import Base

import logging
from dataclasses import dataclass, field
from pathlib import Path

from custodian.cp2k.jobs import Cp2kJob
from custodian.cp2k.validators import Cp2kOutputValidator

from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.alchemy.transmuters import StandardTransmuter
from pymatgen.core.structure import Structure

from atomate2.cp2k.jobs.base import BaseCp2kMaker, cp2k_job
from atomate2.cp2k.sets.base import Cp2kInputGenerator
from atomate2.cp2k.sets.core import (
    NonSCFSetGenerator,
    HybridStaticSetGenerator, HybridRelaxSetGenerator, HybridCellOptSetGenerator,
    StaticSetGenerator, RelaxSetGenerator, CellOptSetGenerator,
    MDSetGenerator,
)

logger = logging.getLogger(__name__)

__all__ = [
    "StaticMaker",
    "RelaxMaker",
    "NonSCFMaker",
]


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

    name: str = "relax"
    input_set_generator: Cp2kInputGenerator = field(default_factory=RelaxSetGenerator)

@dataclass
class HybridStaticMaker(BaseCp2kMaker):

    name: str = "static"
    input_set_generator: Cp2kInputGenerator = field(default_factory=HybridStaticSetGenerator)

@dataclass
class HybridRelaxMaker(BaseCp2kMaker):

    name: str = "hybrid relax"
    input_set_generator: Cp2kInputGenerator = field(default_factory=HybridRelaxSetGenerator)

@dataclass
class HybridCellOptMaker(BaseCp2kMaker):

    name: str = "hybrid relax"
    input_set_generator: Cp2kInputGenerator = field(default_factory=HybridCellOptSetGenerator)

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

    @cp2k_job
    def make(
        self,
        structure: Structure,
        prev_cp2k_dir: str | Path | None,
        mode: str = "uniform",
    ):
        """
        Run a non-scf CP2K job.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_vasp_dir : str or Path or None
            A previous CP2K calculation directory to copy output files from.
        mode : str
            Type of band structure calculation. Options are:
            - "line": Full band structure along symmetry lines.
            - "uniform": Uniform mesh band structure.
        """
        self.input_set_generator.mode = mode

        if "parse_dos" not in self.task_document_kwargs:
            # parse DOS only for uniform band structure
            self.task_document_kwargs["parse_dos"] = mode == "uniform"

        if "parse_bandstructure" not in self.task_document_kwargs:
            self.task_document_kwargs["parse_bandstructure"] = mode

        # copy previous inputs
        if "additional_cp2k_files" not in self.copy_cp2k_kwargs:
            self.copy_cp2k_kwargs["additional_cp2k_files"] = ("wfn",)

        return super().make.original(self, structure, prev_cp2k_dir)


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
        prev_cp2k_dir: str | Path | None = None,
    ):
        """
        Run a transmuter Cp2k job.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_vasp_dir : str or Path or None
            A previous Cp2k calculation directory to copy output files from.
        """
        transformations = _get_transformations(
            self.transformations, self.transformation_params
        )
        ts = TransformedStructure(structure)
        transmuter = StandardTransmuter([ts], transformations)
        structure = transmuter.transformed_structures[-1].final_structure

        # to avoid mongoDB errors, ":" is automatically converted to "."
        if "transformations:json" not in self.write_additional_data:
            tjson = transmuter.transformed_structures[-1]
            self.write_additional_data["transformations:json"] = tjson

        return super().make.original(self, structure, prev_cp2k_dir)


@dataclass
class MDMaker(BaseCp2kMaker):

    name: str = "md"
    input_set_generator: Cp2kInputGenerator = field(default_factory=MDSetGenerator)
    task_document_kwargs: dict = field(
        default_factory=lambda: {
            "store_trajectory": True
        }
    )


# TODO THis should go in common
def _get_transformations(
    transformations: tuple[str, ...], params: tuple[dict, ...] | None
):
    """Get instantiated transformation objects from their names and parameters."""
    params = ({},) * len(transformations) if params is None else params

    if len(params) != len(transformations):
        raise ValueError("Number of transformations and parameters must be the same.")

    transformation_objects = []
    for transformation, transformation_params in zip(transformations, params):
        found = False
        for m in (
            "advanced_transformations",
            "site_transformations",
            "standard_transformations",
        ):
            from importlib import import_module

            mod = import_module(f"pymatgen.transformations.{m}")

            try:
                t_cls = getattr(mod, transformation)
                found = True
                continue
            except AttributeError:
                pass

        if not found:
            raise ValueError(f"Could not find transformation: {transformation}")

        t_obj = t_cls(**transformation_params)
        transformation_objects.append(t_obj)
    return transformation_objects
