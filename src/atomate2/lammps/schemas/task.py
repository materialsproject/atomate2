from pathlib import Path
from typing import Type

from atomate2.common.schemas.structure import StructureMetadata
from pydantic import Field
from pymatgen.io.lammps.outputs import parse_lammps_dumps, parse_lammps_log


class TaskDocument(StructureMetadata):

    dir_name: str = Field()

    task_label: str = Field()
    
    log_file : str = Field()
    
    dump_files : str = Field() #needs to reimplement to handle different types of dump files/outputs

    @classmethod
    def from_directory(
        cls: Type["TaskDocument"],
        dir_name: str | Path,
        task_label: str,
    ) -> "TaskDocument":
        
        log = parse_lammps_log(dir_name)
        dumps = parse_lammps_dumps(dir_name)
        
        return TaskDocument(dir_name=str(dir_name), task_label=task_label) 

    class Config:
        extras = "allow"
        
''' References to implement here: https://github.com/materialsproject/emmet/blob/main/emmet-core/emmet/core/openmm/calculations.py 
https://github.com/materialsproject/emmet/blob/main/emmet-core/emmet/core/openmm/tasks.py
This might take a lot of work considering the sheer number of possible lammps outputs there can be. 
'''