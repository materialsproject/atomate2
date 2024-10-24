from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List

from jobflow import Maker, Response, job
from monty.serialization import dumpfn
from monty.shutil import gzip_dir
from pymatgen.core import Structure

from ..files import write_lammps_input_set
from ..run import run_lammps
from ..schemas.task import TaskDocument
from ..sets.base import BaseLammpsGenerator

_DATA_OBJECTS: List[Any] = []  # populate with calc output data objects here

__all__ = ("BaseLammpsMaker", "lammps_job")


def lammps_job(method: Callable):
    return job(method, data=_DATA_OBJECTS, output_schema=TaskDocument)


@dataclass
class BaseLammpsMaker(Maker):
    name: str = "Base LAMMPS job"
    input_set_generator: BaseLammpsGenerator = field(
        default_factory=BaseLammpsGenerator
    )
    write_input_set_kwargs: dict = field(default_factory=dict)
    run_lammps_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
    write_additional_data: dict = field(default_factory=dict)

    @lammps_job
    def make(self, input_structure: Structure | Path):
        """Run a LAMMPS calculation."""

        write_lammps_input_set(
            input_structure, self.input_set_generator, **self.write_input_set_kwargs
        )

        for filename, data in self.write_additional_data.items():
            dumpfn(data, filename.replace(":", "."))

        run_lammps(**self.run_lammps_kwargs)

        task_doc = TaskDocument.from_directory(
            Path.cwd(), task_label=self.name, **self.task_document_kwargs
        )
        task_doc.task_label = self.name

        gzip_dir(".")

        return Response(output=task_doc)


@dataclass
class LammpsMaker(BaseLammpsMaker):
    name: str = "Simple LAMMPS job"
    input_set_generator: BaseLammpsGenerator = field(
        default_factory=BaseLammpsGenerator
    )