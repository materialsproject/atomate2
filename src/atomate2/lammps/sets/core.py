import os
from .base import BaseLammpsSet, template_dir
from atomate2.lammps.sets.utils import update_settings, _BASE_LAMMPS_SETTINGS
from typing import Literal, Union

class LammpsNVTSet(BaseLammpsSet):
    """
    Input set for NVT simulations.
    """
    ensemble : str = "nvt"
    thermostat : str = "lengevin"
    force_field : Union[str, dict] = None
    friction : float = 0.1
    settings : dict = None
    
    def __init__(self, 
                 thermostat : Literal['langevin', 'nose-hoover'] = 'nose-hoover',
                 friction : float = None,
                 force_field : Union[str, dict] = None,
                 **kwargs):
        
        self.thermostat = thermostat
        if friction is None:
            friction = 100*kwargs.get('timestep', _BASE_LAMMPS_SETTINGS['timestep'])
        if friction < kwargs.get('timestep', 0.001):
            raise ValueError("Friction should be more than the timestep")
        
        self.settings = {'thermostat': self.thermostat, 
                         'tfriction': friction,
                         'ensemble': 'nvt'
                         }
        
        self.settings = update_settings(settings=self.settings, **kwargs)
        super().__init__(ensemble=self.ensemble, thermostat=self.thermostat, force_field=force_field, settings=self.settings, **kwargs)

class LammpsNPTSet(BaseLammpsSet):
    """
    Input set for NPT simulations.
    """
    ensemble : str = "npt"
    barostat : str = "berendsen"
    force_field : Union[str, dict] = None
    friction : float = 0.1
    settings : dict = None
    
    def __init__(self, 
                 barostat : Literal['berendsen', 'nose-hoover'] = 'berendsen',
                 friction : float = None,
                 force_field : Union[str, dict] = None,
                 **kwargs):
        
        self.barostat = barostat
 
        if friction is None:
            friction = 100*kwargs.get('timestep', _BASE_LAMMPS_SETTINGS['timestep'])
        if friction < kwargs.get('timestep', 0.001):
            raise ValueError("Friction should be more than the timestep")
        
        self.settings = {'barostat': barostat, 
                         'pfriction': friction,
                         'ensemble': 'npt'
                         }
        
        self.settings = update_settings(settings=self.settings,**kwargs)
        super().__init__(ensemble=self.ensemble, barostat=self.barostat, force_field=force_field, settings=self.settings, **kwargs)
        
        
class LammpsMinimizeSet(BaseLammpsSet):
    """
    Input set for minimization simulations.
    """
    setting : dict = {'ensemble': 'minimize'}
    
    def __init__(self, **kwargs):
        template = os.path.join(template_dir, "minimize.template")
        super().__init__(ensemble='minimize', template=template, settings=self.settings, **kwargs)


class CustomLammpsSet(BaseLammpsSet):
    """
    Input set for custom simulations.
    """
    def __init__(self, template : str = None, settings : dict = None, **kwargs):
        super().__init__(settings=settings, template=template, **kwargs)