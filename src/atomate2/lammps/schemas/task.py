from pathlib import Path
from typing import Type, Optional, List, Literal
import warnings
from emmet.core.structure import StructureMetadata
from emmet.core.vasp.task_valid import TaskState
from emmet.core.vasp.calculation import StoreTrajectoryOption
from pymatgen.core.trajectory import Trajectory
from pymatgen.core import Composition, Structure
from pydantic import Field
from pymatgen.io.lammps.outputs import parse_lammps_log
from pymatgen.io.lammps.generators import LammpsData, LammpsInputFile
from atomate2.lammps.files import DumpConvertor
import os
from glob import glob
from atomate2.utils.datetime import datetime_str

class LammpsTaskDocument(StructureMetadata):

    dir_name: str = Field(None, description="Directory where the task was run")

    task_label: str = Field(None, description="Label for the task")
    
    last_updated : str = Field(datetime_str(), description="Timestamp for the last time the task was updated")
    
    trajectories : Optional[List[Trajectory]] = Field(None, description="Pymatgen trajectories output from lammps run")
    
    composition : Composition | None = Field(None, description="Composition of the system")
    
    state : TaskState = Field(None, description="State of the calculation")
    
    reduced_formula : str | None = Field(None, description="Reduced formula of the system")
    
    dump_files : Optional[dict] = Field(None, description="Dump files produced by lammps run")
    
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
        store_trajectory: StoreTrajectoryOption = StoreTrajectoryOption.NO,
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
            state = TaskState.ERROR if "ERROR" in raw_log else TaskState.SUCCESS
        except ValueError:
            Warning(f"Error parsing log file for {dir_name}, incomplete job")
            raw_log = ''
            thermo_log = []
            state = TaskState.ERROR
            
        dump_file_keys = glob("*dump*", root_dir=dir_name)
        dump_files = {}
        for dump_file in dump_file_keys:
            with open(os.path.join(dir_name, dump_file), 'rt') as f:
                dump_files[dump_file] = f.read()

        if store_trajectory != StoreTrajectoryOption.NO:
            warnings.warn("Trajectory data might be large, only store if absolutely necessary. Consider manually parsing the dump files instead.")
            if output_file_pattern is None:
                output_file_pattern = "trajectory"
            trajectories = [DumpConvertor(store_md_outputs=store_trajectory, 
                                          dumpfile=os.path.join(dir_name, dump_file)).save(filename=f'{output_file_pattern}{i}.traj',
                                                                                           fmt=trajectory_format)
                            for i, dump_file in enumerate(dump_files)]
            structure = trajectories[-1][-1] if trajectories and trajectory_format == 'pmg' else None
        
        try:
            input_file = LammpsInputFile.from_file(os.path.join(dir_name, "in.lammps"), ignore_comments=True)
            data_files = [LammpsData.from_file(os.path.join(dir_name, file), atom_style=input_file.get_args("atom_style")) for file in glob("*.data*", root_dir=dir_name)]
   
        except FileNotFoundError:
            Warning(f"Input or data file not found for {dir_name}")
            input_file = None
            data_files = None
        
        inputs = {"in.lammps": input_file, "data_files": data_files}
        composition = data_files[0].structure.composition if data_files else None
        reduced_formula = composition.reduced_formula if composition else None
            
        return LammpsTaskDocument(dir_name=str(dir_name), 
                                  task_label=task_label, 
                                  raw_log_file=raw_log, 
                                  thermo_log=thermo_log, 
                                  dump_files=dump_files,
                                  trajectories=trajectories if store_trajectory != StoreTrajectoryOption.NO else None,
                                  structure=structure if store_trajectory != StoreTrajectoryOption.NO else None,
                                  composition=composition,
                                  reduced_formula=reduced_formula,
                                  inputs=inputs,
                                  state=state,
                                  )