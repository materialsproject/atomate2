from dataclasses import dataclass, field
from atomate2.lammps.sets.core import BaseLammpsSetGenerator, LammpsNVTSet, LammpsNPTSet, LammpsMinimizeSet
from atomate2.lammps.jobs.base import BaseLammpsMaker

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
    name: str = "custom_lammps_job"
    template : str = None
    settings : dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.input_set_generator = BaseLammpsSetGenerator(template=self.template, settings=self.settings)