from .base import BaseLammpsSet

class LammpsNVTSet(BaseLammpsSet):
    """
    Input set for NVT simulations.
    """
    thermostat : str = "lengevin"
    friction : float = 0.1
    
    def __init__(self, 
                 thermostat : str = "lengevin",
                friction : float = 0.1,
                 **kwargs):
        self.settings.update({'thermostat': self.thermostat, 'friction': self.friction})
        super().__init__(**kwargs)
        #TODO: fix this logic
        #super().__init__(input_set_updates=input_set_updates, **kwargs)
