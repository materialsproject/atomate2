import abc
from dataclasses import dataclass, field
from typing import Callable, Union

from pymatgen.core import Structure

from atomate2.lammps.sets.base import (
    BaseLammpsGenerator,
    LammpsMelt,
    LammpsMinimization,
    LammpsQuench,
    LammpsThermalize,
)
from atomate2.lammps.sets.core import CustomLammpsSet
from ..jobs.base import BaseLammpsMaker


class StoppingCriteria(abc.ABC):
    """Abstract class for encapsulating criteria for ending a LAMMPS dynamics run."""


class AutocorrelationCriteria(StoppingCriteria):
    pass


@dataclass
class MeltMaker(BaseLammpsMaker):
    melt_temperature: float 
    stopping_criteria: str | Callable[[Structure, List[Structure]]] | StoppingCriteria  #recheck type inside Callable here
    name: str = "melt"
    input_set_generator: BaseLammpsGenerator = field(default_factory=LammpsMelt)


@dataclass
class ThermalizeMaker(BaseLammpsMaker):
    thermalization_temperature: float
    stopping_criteria: str | Callable[[Union[Structure, List[Structure]]]] | StoppingCriteria
    name: str = "thermalize"
    input_set_generator: BaseLammpsGenerator = field(default_factory=LammpsThermalize)


@dataclass
class QuenchMaker(BaseLammpsMaker):
    name: str = "quench"
    input_set_generator: BaseLammpsGenerator = field(default_factory=LammpsQuench)


@dataclass
class MinimizationMaker(BaseLammpsMaker):
    name: str = "minimization"
    input_set_generator: BaseLammpsGenerator = field(default_factory=LammpsMinimization)
    
    
class CustomLammpsMaker(BaseLammpsMaker):
    name: str = "custom_lammps_job"
    template : str = None
    input_set_generator: BaseLammpsGenerator = field(default_factory=CustomLammpsSet)