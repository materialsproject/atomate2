"""
Input sets for LAMMPS, initially developed inside
pymatgen by Ryan Kingsbury & Guillaume Brunin.
"""

import logging
import os
from pathlib import Path
from string import Template

from monty.io import zopen
from pymatgen.core import Structure
from pymatgen.io.core import InputGenerator, InputSet
from pymatgen.io.lammps.data import CombinedData, LammpsData
from pymatgen.io.lammps.inputs import LammpsInputFile
from pymatgen.io.lammps.sets import LammpsInputSet
from pymatgen.io.lammps.generators import BaseLammpsGenerator, LammpsMinimization
from typing import Union

__author__ = "Ryan Kingsbury, Guillaume Brunin (Matgenix)"

logger = logging.getLogger(__name__)
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

_BASE_LAMMPS_SETTINGS = {"atom_style": "atomic",
                 "start_temp": 300,
                 "end_temp": 300,
                 "pressure": 0,
                 "units": "metal",
                 "nsteps": 1000,
                 "timestep": 0.001,
                 "log_interval": 100,
                 "traj_interval": 100,
}

class BaseLammpsSet(BaseLammpsGenerator):
    """
    Basic LAMMPS input set generator.
    """

    atom_style = "atomic"
    start_temp = 300
    end_temp = 300
    pressure = 0
    units = "metal"
    nsteps = 1000
    timestep = 0.001
    force_field_def = None
    template = os.path.join(template_dir, "md.template")
            
    def __init__(self, 
                 atom_style : str = "atomic",
                 start_temp : float = 300,
                 end_temp : float = 300,
                 pressure : float = 0,
                 units : str = "metal",
                 nsteps : int = 1000,
                 timestep : float = 0.001,
                 log_interval : int = 100,
                 traj_interval : int = 100,
                 force_field : Union[str, dict] = None,
                 template : str = None,
                 **kwargs):
        
        self.atom_style = atom_style
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.pressure = pressure
        self.units = units
        self.nsteps = nsteps
        self.timestep = timestep
        self.log_interval = log_interval
        self.traj_interval = traj_interval
        self.force_field = force_field
        self.species = None
        
        if template is None:
            self.template = os.path.join(template_dir, "md.template")
        else:
            self.template = template
        
        self.settings = _BASE_LAMMPS_SETTINGS.copy()
        self.settings.update({'atom_style': atom_style,
                         'start_temp': self.start_temp,
                         'end_temp': self.end_temp,
                         'pressure': self.pressure,
                         'units': self.units,
                         'nsteps': self.nsteps,
                         'timestep': self.timestep,
                         'log_interval': self.log_interval,
                         'traj_interval': self.traj_interval})
        
        if isinstance(force_field, dict):
            try:
                pair_style = force_field.pop('pair_style', None)
                pair_coeff = force_field.pop('pair_coeff', None)
                species = force_field.pop('species', None)
                species_str = '' if species is None else ' '.join(species)
                if not isinstance(pair_style, str) or not isinstance(pair_coeff, str) or not isinstance(species, list):
                    raise KeyError
            
                self.force_field_def = f'pair_style {pair_style}\n pair_coeff {pair_coeff} {species_str}'
                
            except KeyError:
                logger.error(f"Force field parameters (pair_style and pair_coeff) not found in {force_field}, check input format!")
                raise KeyError
        else:
            raise TypeError(f"Force field should be a dictionary, got {type(force_field)}")
            
        if self.force_field_def is not None:
            self.settings.update({'force_field': self.force_field_def, 'species': species_str})
        super().__init__(template = self.template, settings = self.settings, **kwargs)
        
    @classmethod
    def generate_sets(cls, settings, **kwargs):
        return super().__init__(template = cls.template, settings = settings, **kwargs)