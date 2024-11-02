"""
Input sets for LAMMPS, initially developed inside
pymatgen by Ryan Kingsbury & Guillaume Brunin.
"""

import logging
import os
from pathlib import Path
from string import Template

from pymatgen.core import Structure
from pymatgen.io.core import InputGenerator, InputSet
from pymatgen.io.lammps.data import CombinedData, LammpsData
from pymatgen.io.lammps.inputs import LammpsInputFile
from pymatgen.io.lammps.sets import LammpsInputSet
from pymatgen.io.lammps.generators import BaseLammpsGenerator, LammpsMinimization
from typing import Union, Literal
from monty.serialization import loadfn
from atomate2.lammps.sets.utils import process_ensemble_conditions, update_settings

__author__ = "Ryan Kingsbury, Guillaume Brunin (Matgenix)"

logger = logging.getLogger(__name__)
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

class BaseLammpsSet(BaseLammpsGenerator):
    """
    Basic LAMMPS input set generator.
    """
            
    def __init__(self, 
                 atom_style : str = "atomic",
                 ensemble : Literal["nve", "nvt", "npt"] = "nve",
                 start_temp : float = 300,
                 end_temp : float = 300,
                 start_pressure : float = 0,
                 end_pressure : float = 0,
                 units : str = "metal",
                 nsteps : int = 1000,
                 timestep : float = 0.001,
                 log_interval : int = 100,
                 traj_interval : int = 100,
                 force_field : Union[str, dict] = None,
                 template : str = None,
                 settings : dict = None,
                 **kwargs):
        
        template = os.path.join(template_dir, "md.template") if template is None else template
        self.settings = settings
        self.atom_style = atom_style
        self.ensemble = ensemble
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.start_pressure = start_pressure
        self.end_pressure = end_pressure
        self.units = units
        self.nsteps = nsteps
        self.timestep = timestep
        self.log_interval = log_interval
        self.traj_interval = traj_interval
        self.force_field = force_field
        self.species = None
        
        process_kwargs = kwargs.copy()
        process_kwargs.update({'atom_style': self.atom_style, 'ensemble': self.ensemble, 'start_temp': self.start_temp,
                       'end_temp': self.end_temp, 'start_pressure': self.start_pressure, 'end_pressure': self.end_pressure,
                       'units': self.units, 'nsteps': self.nsteps, 'timestep': self.timestep, 'log_interval': self.log_interval,
                       'traj_interval': self.traj_interval})
                
        self.settings = update_settings(settings = self.settings, **process_kwargs)
        self.settings = process_ensemble_conditions(self.settings)
        
        if isinstance(force_field, dict):
            try:
                pair_style = force_field.pop('pair_style', None)
                pair_coeff = force_field.pop('pair_coeff', None)
                species = force_field.pop('species', None)
                species_str = '' if species is None else ' '.join(species)
                if not isinstance(pair_style, str) or not isinstance(pair_coeff, str) or not isinstance(species, list):
                    raise KeyError
            
                self.force_field = f'pair_style {pair_style}\n pair_coeff {pair_coeff} {species_str}'
                
            except KeyError:
                logger.error(f"Force field parameters (pair_style and pair_coeff) not found in {force_field}, check input format!")
                raise KeyError
        else:
            raise TypeError(f"Force field should be a dictionary, got {type(force_field)}")
            
        if self.force_field is not None:
            self.settings.update({'force_field': self.force_field, 'species': species_str})
                
    
        super().__init__(template = template, settings = self.settings, **kwargs)