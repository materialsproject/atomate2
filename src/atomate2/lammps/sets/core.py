from pymatgen.io.lammps.generators import BaseLammpsSetGenerator, LammpsSettings
from typing import Literal
from atomate2.ase.md import MDEnsemble
import numpy as np


class LammpsNVESet(BaseLammpsSetGenerator):
    """
    Lammps input set for NVE MD simulations. 
    """
    ensemble : MDEnsemble = MDEnsemble.nve
    settings : dict = {}
    
    def __init__(self, **kwargs):
        self.settings.update({'ensemble': self.ensemble.value})
        self.settings.update({k:v for k,v in kwargs.items() if k in vars(LammpsSettings)})
        kwargs = {k:v for k,v in kwargs.items() if k not in vars(LammpsSettings)}
        
        super().__init__(calc_type=f'lammps_{self.ensemble.value}', settings=self.settings, **kwargs)


class LammpsNVTSet(BaseLammpsSetGenerator):
    """
    Lammps input set for NVT MD simulations. 
    """
    ensemble : MDEnsemble = MDEnsemble.nvt
    friction : float = None
    thermostat : Literal['langevin', 'nose-hoover'] = "langevin"
    start_temp : float = 300.0
    end_temp : float = 300.0
    settings : dict = {}
    
    def __init__(self, 
                 thermostat : Literal['langevin', 'nose-hoover'] = 'langevin',
                 start_temp : float = 300.0,
                 end_temp : float = 300.0,
                 friction : float = None,
                 **kwargs):
        self.friction = friction
        self.thermostat = thermostat                     
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.settings.update({'ensemble': self.ensemble.value,
                              'thermostat': self.thermostat,
                              'start_temp': start_temp,
                              'end_temp': end_temp})
        
        self.settings.update({k:v for k,v in kwargs.items() if k in vars(LammpsSettings)})
        kwargs = {k:v for k,v in kwargs.items() if k not in vars(LammpsSettings)}
        
        super().__init__(calc_type=f'lammps_{self.ensemble.value}', settings=self.settings, **kwargs)

class LammpsNPTSet(BaseLammpsSetGenerator):
    """
    Lammps input set for NPT MD simulations.
    """
    ensemble : MDEnsemble = MDEnsemble.npt
    friction : float = None
    barostat : Literal['berendsen', 'nose-hoover', 'nph'] = "nose-hoover"
    start_pressure : float = 1.0
    end_pressure : float = 1.0
    start_temp : float = 300
    end_temp : float = 300
    settings : dict = {}
    
    def __init__(self, 
                 barostat : Literal['berendsen', 'nose-hoover', 'nph'] = 'nose-hoover',
                 friction : float = None,
                 start_pressure : float = 1.0,
                 end_pressure : float = 1.0,
                 start_temp : float = 300,
                 end_temp : float = 300,
                 **kwargs):
        
        self.barostat = barostat
        self.friction = friction
        self.start_pressure = start_pressure
        self.end_pressure = end_pressure
        self.start_temp = start_temp
        self.end_temp = end_temp
        
        self.settings = {'ensemble': self.ensemble.value,
                         'barostat': self.barostat,
                            'start_pressure': start_pressure,
                            'end_pressure': end_pressure,
                            'start_temp': start_temp,
                            'end_temp': end_temp
        }        
        self.settings.update({k:v for k,v in kwargs.items() if k in vars(LammpsSettings)})
        kwargs = {k:v for k,v in kwargs.items() if k not in vars(LammpsSettings)}
        
        super().__init__(calc_type=f'lammps_{self.ensemble.value}', settings=self.settings, **kwargs)
        
        
class LammpsMinimizeSet(BaseLammpsSetGenerator):
    """
    Input set for minimization simulations.
    """
    settings : dict = None
    max_steps : int = 10000
    pressure : float = 0
    tol : float = 1.0e-6
    
    def __init__(self,
                 max_steps : int = 10000,
                 pressure : float = 0,
                 tol : float = 1.0e-6,
                 **kwargs):
        
        self.max_steps = max_steps
        self.tol = tol
        self.pressure = pressure
        
        self.settings = {'ensemble': 'minimize',
                         'nsteps': self.max_steps,
                         'start_pressure': self.pressure,
                         'end_pressure': self.pressure,
                         'tol': self.tol}
        self.settings.update({k:v for k,v in kwargs.items() if k in vars(LammpsSettings)})
        kwargs = {k:v for k,v in kwargs.items() if k not in vars(LammpsSettings)}
            
        super().__init__(calc_type='lammps_minimization', settings=self.settings, **kwargs)
    