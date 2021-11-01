"""Definition of base VASP job maker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from jobflow import Maker, Response, job
from monty.serialization import dumpfn
from monty.shutil import gzip_dir
from pymatgen.core import Structure

from atomate2.vasp.file import copy_vasp_outputs, write_vasp_input_set
from atomate2.vasp.run import run_vasp, should_stop_children
from atomate2.vasp.schemas.task import TaskDocument
from atomate2.vasp.sets.base import VaspInputSetGenerator

__all__ = ["BaseVaspMaker"]


@dataclass
class BaseVaspMaker(Maker):
    """
    Base VASP job maker.

    Parameters
    ----------
    name
        The job name.
    input_set_generator
        A generator used to make the input set.
    write_input_set_kwargs
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
    stop_children_kwargs
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data
        Additional data to write to the current directory. Given as a dict of
        {filename: data}.
    """

    name: str = "base vasp job"
    input_set_generator: VaspInputSetGenerator = field(
        default_factory=VaspInputSetGenerator
    )
    write_input_set_kwargs: dict = field(default_factory=dict)
    copy_vasp_kwargs: dict = field(default_factory=dict)
    run_vasp_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
    stop_children_kwargs: dict = field(default_factory=dict)
    write_additional_data: dict = field(default_factory=dict)

    @job(output_schema=TaskDocument)
    def make(self, structure: Structure, prev_vasp_dir: Union[str, Path] = None):
        """
        Run a VASP calculation.

        Parameters
        ----------
        structure
            A pymatgen structure object.
        prev_vasp_dir
            A previous VASP calculation directory to copy output files from.
        """
        # copy previous inputs
        from_prev = prev_vasp_dir is not None
        if prev_vasp_dir is not None:
            copy_vasp_outputs(prev_vasp_dir, **self.copy_vasp_kwargs)

        if "from_prev" not in self.write_input_set_kwargs:
            self.write_input_set_kwargs["from_prev"] = from_prev

        # write vasp input files
        write_vasp_input_set(
            structure, self.input_set_generator, **self.write_input_set_kwargs
        )

        # write any additional data
        for filename, data in self.write_additional_data.items():
            dumpfn(data, filename)

        # run vasp
        run_vasp(**self.run_vasp_kwargs)

        # parse vasp outputs
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
