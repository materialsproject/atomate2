"""Definition of base CP2K job maker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from jobflow import Maker, Response, job
from monty.serialization import dumpfn
from monty.shutil import gzip_dir
from pymatgen.core import Structure
from pymatgen.core.trajectory import Trajectory
from pymatgen.electronic_structure.bandstructure import (
    BandStructure,
    BandStructureSymmLine,
)
from pymatgen.electronic_structure.dos import DOS, CompleteDos, Dos
from pymatgen.io.cube import Cube

from atomate2.cp2k.files import copy_cp2k_outputs, write_cp2k_input_set
from atomate2.cp2k.run import run_cp2k, should_stop_children
from atomate2.cp2k.schemas.task import TaskDocument
from atomate2.cp2k.sets.base import Cp2kInputGenerator

__all__ = ["BaseCp2kMaker", "cp2k_job"]


_DATA_OBJECTS = [
    BandStructure,
    BandStructureSymmLine,
    DOS,
    Dos,
    CompleteDos,
    Cube,
    Trajectory,
]


def cp2k_job(method: Callable):
    """
    Decorate the ``make`` method of CP2K job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.Job` that configures common
    settings for all CP2K jobs. For example, it ensures that large data objects
    (band structures, density of states, Cubes, etc) are all stored in the
    atomate2 data store. It also configures the output schema to be a CP2K
    :obj:`.TaskDocument`.

    Any makers that return CP2K jobs (not flows) should decorate the ``make`` method
    with @cp2k_job. For example:

    .. code-block:: python

        class MyCp2kMaker(BaseCp2kMaker):
            @cp2k_job
            def make(structure):
                # code to run Cp2k job.
                pass

    Parameters
    ----------
    method : callable
        A BaseCp2kMaker.make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate Cp2k jobs.
    """
    return job(method, data=_DATA_OBJECTS, output_schema=TaskDocument)


@dataclass
class BaseCp2kMaker(Maker):
    """
    Base CP2K job maker.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_cp2k_input_set`.
    copy_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_cp2k_outputs`.
    copy_cp2k_kwargs : dict
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

    name: str = "base cp2k job"
    input_set_generator: Cp2kInputGenerator = field(default_factory=Cp2kInputGenerator)
    write_input_set_kwargs: dict = field(default_factory=dict)
    copy_cp2k_kwargs: dict = field(default_factory=dict)
    run_cp2k_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
    stop_children_kwargs: dict = field(default_factory=dict)
    write_additional_data: dict = field(default_factory=dict)

    @cp2k_job
    def make(self, structure: Structure, prev_cp2k_dir: str | Path | None = None):
        """
        Run a CP2K calculation.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_vasp_dir : str or Path or None
            A previous CP2K calculation directory to copy output files from.
        """
        # copy previous inputs
        from_prev = prev_cp2k_dir is not None
        if prev_cp2k_dir is not None:
            copy_cp2k_outputs(prev_cp2k_dir, **self.copy_cp2k_kwargs)

        # write cp2k input files
        self.write_input_set_kwargs["from_prev"] = from_prev
        write_cp2k_input_set(
            structure, self.input_set_generator, **self.write_input_set_kwargs
        )

        # write any additional data
        for filename, data in self.write_additional_data.items():
            dumpfn(data, filename.replace(":", "."))

        # run cp2k
        run_cp2k(**self.run_cp2k_kwargs)

        # parse cp2k outputs
        task_doc = TaskDocument.from_directory(Path.cwd(), **self.task_document_kwargs)
        task_doc.task_label = self.name

        # decide whether child jobs should proceed
        stop_children = should_stop_children(task_doc, **self.stop_children_kwargs)

        # gzip folder
        gzip_dir(".")

        return Response(
            stop_children=stop_children,
            stored_data={"custodian": task_doc.custodian},
            output=task_doc,
        )
