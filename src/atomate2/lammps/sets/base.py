"""
Input sets for LAMMPS, initially developed inside
pymatgen by Ryan Kingsbury & Guillaume Brunin.
"""

import logging
import os
from pymatgen.io.lammps.generators import BaseLammpsGenerator
from typing import Union, Literal
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
                 ensemble : Literal["nve", "nvt", "npt", "nph"] = "nve",
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
                 pressure_symmetry : Literal["iso", "aniso"] = "iso",
                 thermostat : Literal["langevin", "nose-hoover"] = "nose-hoover",
                 barostat : Literal["berendsen", "nose-hoover"] = "nose-hoover",
                 template : str = None,
                 settings : dict = None,
                 **kwargs):
        
        template = os.path.join(template_dir, "md.template") if template is None else template
        self.atom_style = atom_style
        self.ensemble = ensemble
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.start_pressure = start_pressure
        self.end_pressure = end_pressure
        self.pressure_symmetry = pressure_symmetry
        self.units = units
        self.nsteps = nsteps
        self.timestep = timestep
        self.log_interval = log_interval
        self.traj_interval = traj_interval
        self.thermostat = thermostat
        self.barostat = barostat
        self.force_field = force_field.copy() if isinstance(force_field, dict) else force_field
        self.species = None
        
        process_kwargs = kwargs.copy()
        process_kwargs.update({'atom_style': self.atom_style, 'ensemble': self.ensemble, 'start_temp': self.start_temp,
                       'end_temp': self.end_temp, 'start_pressure': self.start_pressure, 'end_pressure': self.end_pressure,
                       'psymm': self.pressure_symmetry, 'units': self.units, 'nsteps': self.nsteps, 'timestep': self.timestep, 
                       'thermostat': self.thermostat, 'barostat': self.barostat,
                       'log_interval': self.log_interval, 'traj_interval': self.traj_interval})
        
        self.settings = update_settings(settings = settings, **process_kwargs)
        self.settings = process_ensemble_conditions(self.settings)
        
        if isinstance(force_field, dict):
            try:
                pair_style = force_field.pop('pair_style')
                pair_coeff = force_field.pop('pair_coeff')
                species = force_field.pop('species')
                self.species = '' if species is None else ' '.join(species)
                if not isinstance(pair_style, str) or not isinstance(pair_coeff, str) or not isinstance(species, list):
                    raise KeyError
            
                self.force_field = f'pair_style {pair_style}\n pair_coeff {pair_coeff} {self.species}'
                
            except KeyError:
                logger.error(f"Force field parameters (pair_style, pair_coeff and species) not found in {force_field}, check input format!")
                raise KeyError
        
        if isinstance(force_field, str):
            self.force_field = force_field
            self.species = ' '.join(force_field.split(' ')[6:]) #check if logic holds for general FF
            
        else:
            Warning(f"Force field should be a dictionary, got {type(force_field)}")
            
        if self.force_field is not None:
            self.settings.update({'force_field': self.force_field, 'species': self.species})
                
    
        super().__init__(template = template, settings = self.settings, **kwargs)