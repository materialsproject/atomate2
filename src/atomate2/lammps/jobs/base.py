from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List
import os
import glob

from jobflow import Maker, Response, job
from pymatgen.core import Structure
from pymatgen.io.lammps.generators import LammpsData, CombinedData

from emmet.core.vasp.task_valid import TaskState

from atomate2.common.files import gzip_files
from atomate2.lammps.files import write_lammps_input_set
from atomate2.lammps.run import run_lammps
from atomate2.lammps.schemas.task import LammpsTaskDocument, StoreTrajectoryOption
from atomate2.lammps.sets.base import BaseLammpsSetGenerator
import warnings

_DATA_OBJECTS: List[str] = ["raw_log_file", "inputs", "metadata", "trajectory", "dump_files"]

__all__ = ("BaseLammpsMaker", "lammps_job")


def lammps_job(method: Callable):
    return job(method, data=_DATA_OBJECTS, output_schema=LammpsTaskDocument)


FF_STYLE_KEYS = ["pair_style", "bond_style", "angle_style", "dihedral_style", "improper_style"]
FF_COEFF_KEYS = ["pair_coeff", "bond_coeff", "angle_coeff", "dihedral_coeff", "improper_coeff"]


@dataclass
class BaseLammpsMaker(Maker):
    '''
    Basic Maker class for LAMMPS jobs. 
    
    name: str
        Name of the job
    input_set_generator: BaseLammpsGenerator
        Input set generator for the job, default is the BaseLammpsSet
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
    input_set_generator: BaseLammpsSetGenerator = field(
        default_factory = BaseLammpsSetGenerator
    )
    write_input_set_kwargs: dict = field(default_factory=dict)
    force_field: str | dict = field(default_factory=str)
    run_lammps_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
    write_additional_data: LammpsData | CombinedData = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.force_field:
            raise ValueError("Force field not specified")
        
        if isinstance(self.force_field, Path):
            try:
                with open(self.force_field, 'rt') as f:
                    self.force_field = f.read()
            except FileNotFoundError:
                warnings.warn("Force field file could not be read, given path will be fed as is to LAMMPS. Make sure the path is correct.")
            
        # Validate force field here!
        
        if self.task_document_kwargs.get('store_trajectory', None) != StoreTrajectoryOption.NO:
            warnings.warn("Trajectory data might be large, only store if absolutely necessary. Consider manually parsing the dump files instead.")

    @lammps_job
    def make(self, input_structure: Structure | Path | LammpsData = None, prev_dir : Path = None) -> Response:
        """Run a LAMMPS calculation."""
        
        if prev_dir:
            restart_files = glob.glob(os.path.join(prev_dir, "*restart*"))
            if len(restart_files) != 1:
                raise FileNotFoundError("No/More than one restart file found in the previous directory. If present, it should have the extension '.restart'!")
            self.input_set_generator.settings.input_settings['AtomDefinition'].update({'read_restart': os.path.join(prev_dir, restart_files[0])})

        if isinstance(input_structure, Path):
            input_structure = LammpsData.from_file(input_structure, atom_style=self.input_set_generator.settings.get('atom_style', 'full'))
        
        if isinstance(self.force_field, str):
            self.input_set_generator.settings.input_settings['ForceField'] = {}
            
        if isinstance(self.force_field, dict):
            force_field_coeffs = ''
            for key, value in self.force_field.items():
                if key in FF_STYLE_KEYS:
                    self.input_set_generator.settings.input_settings['Initialization'].update({key: value})
                if key in FF_COEFF_KEYS:
                   force_field_coeffs += f"{key} {value}\n"
                else:
                    warnings.warn(f"Force field key {key} not recognized, will be ignored.")
            self.force_field = force_field_coeffs

        write_lammps_input_set(
            data=input_structure, 
            input_set_generator=self.input_set_generator, 
            force_field=self.force_field, 
            additional_data=self.write_additional_data,
            **self.write_input_set_kwargs
        )

        run_lammps(**self.run_lammps_kwargs)

        task_doc = LammpsTaskDocument.from_directory(
            os.getcwd(), task_label=self.name, **self.task_document_kwargs
        )
        
        if task_doc.state == TaskState.ERROR:
            try: 
                error = ""
                for index, line in enumerate(task_doc.raw_log_file.split('\n')):
                    if "ERROR" in line:
                        error=error.join(task_doc.raw_log_file.split('\n')[index:])
                        break
            except ValueError:
                error = "could not parse log file"
            raise Exception(f"Task {task_doc.task_label} failed, error: {error}") 
        
        task_doc.composition = input_structure.composition
        task_doc.reduced_formula = input_structure.composition.reduced_formula
        task_doc.task_label = self.name
        task_doc.inputs = self.input_set_generator.settings.settings
        
        #gzip_files(".")

        return Response(output=task_doc)