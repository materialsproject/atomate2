"""
Input sets for LAMMPS, initially developed inside
pymatgen by Ryan Kingsbury & Guillaume Brunin.
"""

import logging
import os
import numpy as np
from pymatgen.io.lammps.generators import BaseLammpsGenerator
from typing import Literal, Sequence
from atomate2.lammps.sets.utils import process_ensemble_conditions, update_settings

from atomate2.ase.md import MDEnsemble
from atomate2.lammps.sets.utils import LammpsInterchange

logger = logging.getLogger(__name__)
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

class BaseLammpsSet(BaseLammpsGenerator):
    """
    Basic LAMMPS MD input set generator.
    Note: This class is based on templates and is not intended to be very flexible. 

    Attributes:
    
    atom_style : str 
        Atom style for the LAMMPS input file.
    dimension : int
        Dimension of the simulation.
    boundary : str
        Boundary conditions for the simulation. Defualts to PBCs in all directions.
    ensemble : MDEnsemble | str
        Ensemble for the simulation. Defaults to NVT.
    temperature : float | Sequence | np.ndarray | None
        Temperature for the simulation. Defaults to 300 K. If a sequence is provided, 
        the first value is the starting temperature and the last value is the final temperature, 
        with a linear interpolation during the simulation.
    pressure : float | Sequence | np.ndarray | None
        Pressure for the simulation. Defaults to 0 bar. If a sequence is provided, 
        the first value is the starting pressure and the last value is the final pressure, 
        with a linear interpolation during the simulation. A non-zero pressure requires a barostat.
        A 2D pressure input (e.g. [[0, 100, 0], [100, 100, 0]]) will be interpreted as anisotropic pressure, 
        with the first list being the starting pressure and the second list being the final pressure.
    units : str
        Units for the simulation. Defaults to metal.
    n_steps : int
        Number of steps for the simulation. Defaults to 1000.
    timestep : float
        Timestep for the simulation. Defaults to 0.001 (=ps in metal units).
    log_interval : int
        Interval for logging the simulation. Defaults to 100.
    traj_interval : int
        Interval for writing the trajectory. Defaults to 100.
    force_field : Union[str, dict]
        Force field for the simulation. Can be a string or a dictionary with the pair_style, pair_coeff and species.
        If a dictionary is provided, the species should be a list of the atomic species in the simulation. 
        If a string is provided, the string should be of the form "pair_style {pair_style}\n pair_coeff {pair_coeff} {species}",
        as would be written in a lammps input file.
    pressure_symmetry : Literal["iso", "aniso"]
        Symmetry of the pressure. Defaults to isotropic. If anisotropic pressure is provided,
        the pressure should be a 2D array with the starting and final pressures.
    thermostat : Literal["langevin", "nose-hoover"]
        Thermostat for the simulation. Defaults to Langevin.
    barostat : Literal["berendsen", "nose-hoover"]
        Barostat for the simulation. Defaults to nose-hoover.
    friction : float
        Friction coefficient for the thermostat and barostat. Defaults to 100*timestep.
    template : str
        Path to the template file used to create the LAMMPS input file. Provide this for a more flexible input set.
        All variables in the template file should be prefixed with a $ sign, e.g. $nsteps.
    settings : dict
        Additional settings for the LAMMPS input file. These will be added to the default settings. 
        Provide this for a more flexible input set. E.x., for a custom template file with variable $nsteps,
        provide {"nsteps":1000} to control the variable.
    **kwargs : dict
        Additional keyword arguments to pass to the BaseLammpsGenerator from pymatgen.
    """
            
    def __init__(self, 
                 atom_style : str = "atomic",
                 dimension : int = 3,
                 boundary : str = "p p p",
                 ensemble : MDEnsemble = MDEnsemble.nvt,
                 temperature : float | Sequence | np.ndarray | None = 300,
                 pressure : float | Sequence | np.ndarray | None = 0,
                 units : str = "metal",
                 n_steps : int = 1000,
                 timestep : float = 0.001,
                 log_interval : int = 100,
                 traj_interval : int = 100,
                 force_field : str | dict = None,
                 pressure_symmetry : Literal["iso", "aniso"] = "iso",
                 thermostat : Literal["langevin", "nose-hoover"] = "nose-hoover",
                 barostat : Literal["berendsen", "nose-hoover"] = "nose-hoover",
                 friction : float | None = None,
                 template : str = None,
                 settings : dict | None = None,
                 interchange : LammpsInterchange = None,
                 **kwargs):
        
        template = os.path.join(template_dir, "md.template") if template is None else template
        
        self.atom_style = atom_style
        self.dimension = dimension
        self.boundary = boundary
        self.ensemble = ensemble
        
        if isinstance(temperature, (int, float)):
            self.start_temp = temperature
            self.end_temp = temperature
        elif isinstance(temperature, (list, np.ndarray)):
            self.start_temp = temperature[0]
            self.end_temp = temperature[-1]
        
        self.temperature = temperature
        self.pressure = pressure
        
        if isinstance(pressure, (int, float)):
            self.start_pressure = pressure
            self.end_pressure = pressure
        elif isinstance(pressure, (list, np.ndarray)):                
            self.start_pressure = pressure[0]
            self.end_pressure = pressure[-1]

        self.pressure_symmetry = pressure_symmetry
        self.units = units
        self.n_steps = n_steps
        self.timestep = timestep
        self.log_interval = log_interval
        self.traj_interval = traj_interval
        self.thermostat = thermostat
        self.barostat = barostat
        self.friction = 100*timestep if friction is None else friction
        self.force_field = force_field.copy() if isinstance(force_field, dict) else force_field
        self.species = None
        self.interchange = interchange
        
        process_kwargs = kwargs.copy()
        if settings is None:
            settings = {}
        settings.update({
                        'atom_style': self.atom_style, 
                        'dimension': self.dimension, 
                        'boundary': self.boundary,
                        'ensemble': self.ensemble.value, 
                        'start_temp': self.start_temp, 
                        'end_temp': self.end_temp, 
                        'start_pressure': self.start_pressure, 
                        'end_pressure': self.end_pressure, 
                        'psymm': self.pressure_symmetry, 
                        'units': self.units, 
                        'nsteps': self.n_steps, 
                        'timestep': self.timestep, 
                        'thermostat': self.thermostat, 
                        'barostat': self.barostat,
                        'tfriction': self.friction,
                        'pfriction': self.friction,
                        'log_interval': self.log_interval, 
                        'traj_interval': self.traj_interval
                        })
        
        self.settings = update_settings(settings = settings, **process_kwargs)
        self.settings = process_ensemble_conditions(self.settings)
        
        self.set_force_field(force_field)        
    
        super().__init__(template = template, settings = self.settings, **kwargs)
    

    def set_force_field(self, force_field : str | dict | None = None):
        
        if isinstance(force_field, dict):
            try:
                pair_style = force_field.get('pair_style')
                pair_coeff = force_field.get('pair_coeff')
                species = force_field.get('species')
                self.species = '' if species is None else ' '.join(species)
                if not isinstance(pair_style, str) or not isinstance(pair_coeff, str) or not isinstance(species, list):
                    raise KeyError
            
                self.force_field = f'pair_style {pair_style}\n pair_coeff {pair_coeff} {self.species}'
                
            except KeyError:
                logger.error("Force field parameters (pair_style, pair_coeff and species) not found in force_field, check input format!")
                raise KeyError
        
        if isinstance(force_field, str):
            self.force_field = force_field
            self.species = ' '.join(force_field.split(' ')[6:]) #check if logic holds for general FF
            
        else:
            Warning(f"Force field should be a dictionary, got {type(force_field)}")
            
        if self.force_field:
            self.settings.update({'force_field': self.force_field, 'species': self.species, 'dump_modify_flag': 'dump_modify'}) 
        
        if not self.force_field and self.interchange:
            self.settings.update({'force_field': 'pair_style lj/cut/coul/cut 10.0'}) #Assumes the pair_style is lj/cut if no force field is provided (which should be true for openFF forcefields?)