from pathlib import Path
from typing import Type, Optional, List

from emmet.core.structure import StructureMetadata
from pymatgen.core.trajectory import Trajectory
from pymatgen.core import Composition
from pydantic import Field
from pymatgen.io.lammps.outputs import parse_lammps_dumps, parse_lammps_log
from atomate2.lammps.files import trajectory_from_lammps_dump
import os
from glob import glob
from atomate2.utils.datetime import datetime_str

class LammpsTaskDocument(StructureMetadata):

    dir_name: str = Field(None, description="Directory where the task was run")

    task_label: str = Field(None, description="Label for the task")
    
    last_updated : Optional[str] = Field(datetime_str(), description="Timestamp for the last time the task was updated")
    
    trajectory : Optional[List[Trajectory]] = Field(None, description="Pymatgen trajectory output from lammps run")
    
    composition : Optional[Composition] = Field(None, description="Composition of the system")
    
    reduced_formula : Optional[str] = Field(None, description="Reduced formula of the system")
    
    dump_files : Optional[list] = Field(None, description="Path to dump files")
    
    metadata : dict = Field(None, description="Metadata for the task")
    
    raw_log_file : Optional[str] = Field(None, description="Log file output from lammps run")
    
    thermo_log : Optional[list] = Field(None, description="Parsed log output from lammps run, with a focus on thermo data")
    
    inputs : dict = Field(None, description="Input files for the task")
    
    @classmethod
    def from_directory(
        cls: Type["LammpsTaskDocument"],
        dir_name: str | Path,
        task_label: str,
        store_trajectory: bool = True,
    ) -> "LammpsTaskDocument":

        log_file = os.path.join(dir_name, "log.lammps")
        try:
            with open(log_file) as f:
                raw_log = f.read()
            thermo_log = parse_lammps_log(log_file)
        except FileNotFoundError:
            Warning(f"Log file not found for {dir_name}")
            raw_log = None
            thermo_log = None
            
        dump_files = [os.path.join(dir_name, file) for file in glob("*.dump*", root_dir=dir_name)]
        print(dump_files)
        if store_trajectory:
            trajectories = [trajectory_from_lammps_dump(os.path.join(dir_name, dump_file)) for dump_file in dump_files]
            
        return LammpsTaskDocument(dir_name=str(dir_name), 
                                  task_label=task_label, 
                                  raw_log_file=raw_log, 
                                  thermo_log=thermo_log, 
                                  dump_files=dump_files,
                                  trajectory=trajectories,
                                  )
        
''' References to implement here: https://github.com/materialsproject/emmet/blob/main/emmet-core/emmet/core/openmm/calculations.py 
https://github.com/materialsproject/emmet/blob/main/emmet-core/emmet/core/openmm/tasks.py
This might take a lot of work considering the sheer number of possible lammps outputs there can be. 
'''