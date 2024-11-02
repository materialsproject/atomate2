from pathlib import Path
from typing import Type, Optional, List, Literal

from emmet.core.structure import StructureMetadata
from pymatgen.core.trajectory import Trajectory
from pymatgen.core import Composition, Structure
from pydantic import Field
from pymatgen.io.lammps.outputs import parse_lammps_dumps, parse_lammps_log
from atomate2.lammps.files import DumpConvertor
import os
from glob import glob
from atomate2.utils.datetime import datetime_str

class LammpsTaskDocument(StructureMetadata):

    dir_name: str = Field(None, description="Directory where the task was run")

    task_label: str = Field(None, description="Label for the task")
    
    last_updated : str = Field(datetime_str(), description="Timestamp for the last time the task was updated")
    
    trajectory : Optional[List[Trajectory]] = Field(None, description="Pymatgen trajectory output from lammps run")
    
    composition : Composition = Field(None, description="Composition of the system")
    
    reduced_formula : str = Field(None, description="Reduced formula of the system")
    
    dump_files : Optional[list] = Field(None, description="Path to dump files")
    
    structure : Optional[Structure] = Field(None, description="Final structure of the system, taken from the last dump file")
    
    metadata : dict = Field(None, description="Metadata for the task")
    
    raw_log_file : str = Field(None, description="Log file output from lammps run")
    
    thermo_log : list = Field(None, description="Parsed log output from lammps run, with a focus on thermo data")
    
    inputs : dict = Field(None, description="Input files for the task")
    
    @classmethod
    def from_directory(
        cls: Type["LammpsTaskDocument"],
        dir_name: str | Path,
        task_label: str,
        store_trajectory: Literal["no", "partial", "full"] = "partial",
        trajectory_format : Literal["pmg", "ase"] = "pmg",
        output_file_pattern: str | None = None,
    ) -> "LammpsTaskDocument":
        """
        Create a LammpsTaskDocument from a directory containing the output of a LAMMPS run.
        
        dir_name: str | Path
            Directory where the task was run
        task_label: str
            Label for the task
        store_trajectory: Literal["no", "partial", "full"]
            Whether to store the trajectory output from the lammps run. Defualt is 'partial', which stores only the positions of the atoms
        trajectory_format: Literal["pmg", "ase"]
            Format of the trajectory output. Default is 'pmg'
        output_file_pattern: str
            Pattern for the output file, written to disk in dir_name. Default is None.  
        """
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
        if store_trajectory in ["partial", "full"]:
            trajectories = [DumpConvertor(store_md_outputs=store_trajectory, 
                                          dumpfile=os.path.join(dir_name, dump_file)).save(filename=f'{output_file_pattern}{i}',
                                                                                           fmt=trajectory_format)
                            for i, dump_file in enumerate(dump_files)]
            structure = trajectories[-1][-1] if trajectories and trajectory_format == 'pmg' else None
        
        try:
            with open(os.path.join(dir_name, "in.lammps")) as f:
                input_file = f.read()
            with open(os.path.join(dir_name, "system.data")) as f:
                data_file = f.read()
        except FileNotFoundError:
            Warning(f"Input or data file not found for {dir_name}")
            input_file = None
            data_file = None
        
        inputs = {"in.lammps": input_file, "system.data": data_file}
            
        return LammpsTaskDocument(dir_name=str(dir_name), 
                                  task_label=task_label, 
                                  raw_log_file=raw_log, 
                                  thermo_log=thermo_log, 
                                  dump_files=dump_files,
                                  trajectory=trajectories,
                                  structure=structure,
                                  inputs=inputs
                                  )
        
''' References to implement here: https://github.com/materialsproject/emmet/blob/main/emmet-core/emmet/core/openmm/calculations.py 
https://github.com/materialsproject/emmet/blob/main/emmet-core/emmet/core/openmm/tasks.py
This might take a lot of work considering the sheer number of possible lammps outputs there can be. 
'''