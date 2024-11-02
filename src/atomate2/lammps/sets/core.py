from .base import BaseLammpsSet
from atomate2.lammps.sets.utils import update_settings, _BASE_LAMMPS_SETTINGS
from typing import Literal

class LammpsNVTSet(BaseLammpsSet):
    """
    Input set for NVT simulations.
    """
    ensemble : str = "nvt"
    thermostat : str = "lengevin"
    friction : float = 0.1
    
    def __init__(self, 
                 thermostat : Literal['langevin', 'nose-hoover'] = 'nose-hoover',
                 friction : float = None,
                 **kwargs):
        
        if thermostat == 'nose-hoover':
            thermostat = 'temp'
        if friction is None:
            friction = 100*kwargs.get('timestep', _BASE_LAMMPS_SETTINGS['timestep'])
        if friction < kwargs.get('timestep', 0.001):
            raise ValueError("Friction should be more than the timestep")
        
        self.settings = {'thermostat': thermostat, 
                         'tfriction': friction,
                         'ensemble': 'nvt'
                         }
        
        self.settings = update_settings(settings=self.settings, **kwargs)
        super().__init__(**kwargs)

class LammpsNPTSet(BaseLammpsSet):
    """
    Input set for NPT simulations.
    """
    ensemble : str = "npt"
    barostat : str = "berendsen"
    thermostat : str = "nose-hoover"
    friction : float = 0.1
    
    def __init__(self, 
                 barostat : Literal['berendsen', 'nose-hoover'] = 'berendsen',
                 thermostat : Literal['langevin', 'nose-hoover'] = 'nose-hoover',
                 friction : float = None,
                 **kwargs):
        
        if barostat == 'nose-hoover':
            barostat = 'temp'
        if thermostat == 'nose-hoover':
            thermostat = 'temp'
        if friction is None:
            friction = 100*kwargs.get('timestep', _BASE_LAMMPS_SETTINGS['timestep'])
        if friction < kwargs.get('timestep', 0.001):
            raise ValueError("Friction should be more than the timestep")
        
        self.settings = {'barostat': barostat, 
                         'thermostat': thermostat, 
                         'pfriction': friction,
                         'tfriction': friction,
                         'ensemble': 'npt'
                         }
        
        self.settings = update_settings(settings=self.settings, **kwargs)
        super().__init__(**kwargs)
