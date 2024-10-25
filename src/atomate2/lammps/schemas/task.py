from pathlib import Path
from typing import Type, Optional

from emmet.core.structure import StructureMetadata
from pydantic import Field
from pymatgen.io.lammps.outputs import parse_lammps_dumps, parse_lammps_log
import os
from glob import glob

class TaskDocument(StructureMetadata):

    dir_name: str = Field()

    task_label: str = Field()
    
    log_file : Optional[list] = Field(None, description="Log file output from lammps run")
    
    dump_files : Optional[list] = Field(None, description="All dump files generated during lammps run") #needs to reimplement to handle different types of dump files/outputs

    @classmethod
    def from_directory(
        cls: Type["TaskDocument"],
        dir_name: str | Path,
        task_label: str,
    ) -> "TaskDocument":
        

        log_file = os.path.join(dir_name, "log.lammps")
        if os.path.exists(log_file):
            log = parse_lammps_log(log_file)
            
        dump_files = []
        for file in glob("*dump*", root_dir=dir_name):
            dump_files.append(parse_lammps_dumps(file))
        
        return TaskDocument(dir_name=str(dir_name), task_label=task_label, log_file=log, dump_files=dump_files) 

    class Config:
        extras = "allow"
        
''' References to implement here: https://github.com/materialsproject/emmet/blob/main/emmet-core/emmet/core/openmm/calculations.py 
https://github.com/materialsproject/emmet/blob/main/emmet-core/emmet/core/openmm/tasks.py
This might take a lot of work considering the sheer number of possible lammps outputs there can be. 
'''