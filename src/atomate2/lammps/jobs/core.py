from dataclasses import dataclass, field
from atomate2.lammps.sets.core import BaseLammpsSetGenerator, LammpsNVTSet, LammpsNPTSet, LammpsMinimizeSet
from atomate2.lammps.jobs.base import BaseLammpsMaker
from pymatgen.io.lammps.inputs import LammpsInputFile
from string import Template
from pathlib import Path

@dataclass
class LammpsNVTMaker(BaseLammpsMaker):
    name: str = "nvt"
    input_set_generator: BaseLammpsSetGenerator = field(default_factory=LammpsNVTSet)    

@dataclass
class LammpsNPTMaker(BaseLammpsMaker):
    name: str = "npt"
    input_set_generator: BaseLammpsSetGenerator = field(default_factory=LammpsNPTSet)

@dataclass
class MinimizationMaker(BaseLammpsMaker):
    name: str = "minimization"
    input_set_generator: BaseLammpsSetGenerator = field(default_factory=LammpsMinimizeSet)

@dataclass
class CustomLammpsMaker(BaseLammpsMaker):
    '''
    Custom LAMMPS job maker. This maker exists if using a custom LAMMPS input file, 
    which might end up being a very popular use case (i.e., when you have a more 
    complex job that cannot be achieved with a combination of minimization, NVT, and NPT jobs). 
    
    args:
        name: str
            Name of the job
        inputfile: str | LammpsInputFile
            Path to the LAMMPS input file or a LammpsInputFile object, can be read with pmg.io.lammps.inputs.LammpsInputFile
            (Note: make sure pymatgen can read the file correctly before passing it to the job here)
            If you want to modify settings in this, pass the file as a string and have $variables in the file and specify
            "variables" in the settings dict.
        settings: dict
            Additional settings to pass to the input set generator. 
            If you have variables in the input file, pass them here as a dict.
            Commonly used variables such as units, timestep, etc. are validated and set automatically if not provided.
        keep_stages: bool
            Whether to keep the stages of the input file (default is True). 
            Check the LammpsInputFile class for more info on what this means.
        include_defaults: bool
            Whether to use the default settings for the input set generator (default is False). 
            (Check the _BASE_LAMMPS_SETTINGS dict in pymatgen.io.lammps.generators for the default settings)
        validate_params: bool
            Whether to validate the parameters in the input file (default is True). 
            (Only common inputs args such as units, timestep, etc. are validated,)
    '''
    name: str = "custom_lammps_job"
    inputfile : str | LammpsInputFile | Path = field(default=None)
    settings : dict = field(default_factory=dict)
    keep_stages : bool = field(default=True)
    include_defaults : bool = field(default=False)
    validate_params: bool = field(default=True)
    
    def __post_init__(self):
        if not self.inputfile:
            raise ValueError("Input file not specified. Use this maker only if you have a custom LAMMPS input file!")
        
        self.input_set_generator = BaseLammpsSetGenerator(inputfile=self.inputfile,
                                                          include_defaults=self.include_defaults,
                                                          settings=self.settings,
                                                          validate_params=False)