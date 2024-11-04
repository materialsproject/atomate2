from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List
import os

from jobflow import Maker, Response, job
from monty.serialization import dumpfn
from monty.shutil import gzip_dir
from pymatgen.core import Structure

from ..files import write_lammps_input_set
from ..run import run_lammps
from ..schemas.task import LammpsTaskDocument
from ..sets.base import BaseLammpsSet

_DATA_OBJECTS: List[str] = ["raw_log_file", "inputs", "metadata", "trajectory"]

__all__ = ("BaseLammpsMaker", "lammps_job")


def lammps_job(method: Callable):
    return job(method, data=_DATA_OBJECTS, output_schema=LammpsTaskDocument)


@dataclass
class BaseLammpsMaker(Maker):
    '''
    Basic Maker class for LAMMPS jobs. 
    
    name: str
        Name of the job
    input_set_generator: BaseLammpsGenerator
        Input set generator for the job, default is the BaseLammpsGenerator
    write_input_set_kwargs: dict
        Additional kwargs to write_lammps_input_set
    run_lammps_kwargs: dict
        Additional kwargs to run_lammps
    task_document_kwargs: dict
        Additional kwargs to TaskDocument.from_directory
    write_additional_data: dict
        Additional data to write to the job directory
    '''
    name: str = "Base LAMMPS job"
    input_set_generator: BaseLammpsSet = field(
        default_factory=BaseLammpsSet
    )
    write_input_set_kwargs: dict = field(default_factory=dict)
    force_field: str = field(default_factory=str)
    run_lammps_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
    write_additional_data: dict = field(default_factory=dict)

    @lammps_job
    def make(self, input_structure: Structure | Path, prev_dir : Path = None) -> Response:
        """Run a LAMMPS calculation."""
        
        if prev_dir:
            if os.path.exists(os.path.join(prev_dir, "md.restart")):
                self.input_set_generator.settings.update({'read_restart': os.path.join(prev_dir, 'md.restart'),
                                                         'restart_flag': 'read_restart',
                                                         'read_data_flag': '#'})
            else:
                raise FileNotFoundError("No restart file found in the previous directory. If present, it should be named 'md.restart'")
        
        write_lammps_input_set(
            input_structure, self.input_set_generator, **self.write_input_set_kwargs
        )

        for filename, data in self.write_additional_data.items():
            dumpfn(data, filename.replace(":", "."))

        run_lammps(**self.run_lammps_kwargs)

        task_doc = LammpsTaskDocument.from_directory(
            os.getcwd(), task_label=self.name, **self.task_document_kwargs
        )
        
        task_doc.composition = input_structure.composition
        task_doc.reduced_formula = input_structure.composition.reduced_formula
        task_doc.task_label = self.name
        task_doc.inputs = self.input_set_generator.settings
        
        #gzip_dir(".")

        return Response(output=task_doc)