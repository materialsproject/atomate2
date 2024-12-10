from dataclasses import dataclass, field
from atomate2.lammps.sets.core import BaseLammpsSetGenerator, LammpsNVTSet, LammpsNPTSet, LammpsMinimizeSet
from atomate2.lammps.jobs.base import BaseLammpsMaker
from pymatgen.io.lammps.inputs import LammpsInputFile

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
    Custom LAMMPS job maker. Use this if you want to use a custom LAMMPS input file.
    This is esp. useful if you have a more complex job that cannot be achieved with a 
    combination of minimization, NVT, and NPT jobs. 
    
    args:
        name: str
            Name of the job
        inputfile: str | LammpsInputFile
            Path to the LAMMPS input file or a LammpsInputFile object, can be read with pmg.io.lammps.inputs.LammpsInputFile
            (Note: make sure pymatgen can read the file correctly before passing it to the job here)
        settings: dict
            Additional settings to pass to the input set generator 
            (here essentially acts as metadata, not actually passed to input set generator cause of how flexible input files can be!)
        keep_stages: bool
            Whether to keep the stages of the input file (default is True). 
            Check the LammpsInputFile class for more info on what this means.
    '''
    name: str = "custom_lammps_job"
    inputfile : str | LammpsInputFile = field(default=None)
    settings : dict = field(default_factory=dict)
    keep_stages : bool = field(default=True)
    
    def __post_init__(self):
        if not self.inputfile:
            raise ValueError("Input file not specified. Use this maker only if you have a custom LAMMPS input file!")
        self.input_set_generator = BaseLammpsSetGenerator(inputfile=self.inputfile,
                                                          settings=self.settings,
                                                          calc_type=self.name,
                                                          override_updates=True,
                                                          keep_stages=self.keep_stages)