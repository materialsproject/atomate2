from dataclasses import dataclass
from typing import Callable, Optional, Union

@dataclass
class JMinSettings():
    '''
    A class for storing generic minimization settings read from a JDFTx out file
    '''
    dirUpdateScheme: str = None
    linminMethod: str = None
    nIterations: int = None
    history: int = None
    knormThreshold: float = None
    energyDiffThreshold: float = None
    nEnergyDiff: int = None
    alphaTstart: float = None
    alphaTmin: float = None
    updateTestStepSize: bool = None
    alphaTreduceFactor: float = None
    alphaTincreaseFactor: float = None
    nAlphaAdjustMax: int = None
    wolfeEnergy: float = None
    wolfeGradient: float = None
    fdTest: bool = None
    #
    start_flag: str = None

    def __init__(self, dirUpdateScheme: str = None, linminMethod: str = None,
                 nIterations: int = None, history: int = None, knormThreshold: float = None,
                 energyDiffThreshold: float = None, nEnergyDiff: int = None, alphaTstart: float = None,
                 alphaTmin: float = None, updateTestStepSize: bool = None, alphaTreduceFactor: float = None,
                 alphaTincreaseFactor: float = None, nAlphaAdjustMax: int = None, wolfeEnergy: float = None,
                 wolfeGradient: float = None, fdTest: bool = None):
        self.dirUpdateScheme = self._assign_type(dirUpdateScheme, str)
        self.linminMethod = self._assign_type(linminMethod, str)
        self.nIterations = self._assign_type(nIterations, int)
        self.history = self._assign_type(history, int)
        self.knormThreshold = self._assign_type(knormThreshold, float)
        self.energyDiffThreshold = self._assign_type(energyDiffThreshold, float)
        self.nEnergyDiff = self._assign_type(nEnergyDiff, int)
        self.alphaTstart = self._assign_type(alphaTstart, float)
        self.alphaTmin = self._assign_type(alphaTmin, float)
        self.updateTestStepSize = self._assign_type(updateTestStepSize, bool)
        self.alphaTreduceFactor = self._assign_type(alphaTreduceFactor, float)
        self.alphaTincreaseFactor = self._assign_type(alphaTincreaseFactor, float)
        self.nAlphaAdjustMax = self._assign_type(nAlphaAdjustMax, int)
        self.wolfeEnergy = self._assign_type(wolfeEnergy, float)
        self.wolfeGradient = self._assign_type(wolfeGradient, float)
        self.fdTest = self._assign_type(fdTest, bool)

    def _assign_type(self, val: Optional[str], val_type: Callable[[str], Union[float, int, str]]) -> Optional[Union[float, int, str]]:
        if val is None:
            return None
        else:
            return val_type(val)


@dataclass
class JMinSettingsElectronic(JMinSettings):
    '''
    A class for storing lattice minimization settings read from a JDFTx out file
    '''

    start_flag: str = "electronic-minimize"


    def __init__(self, dirUpdateScheme: str = None, linminMethod: str = None,
                 nIterations: int = None, history: int = None, knormThreshold: float = None,
                 energyDiffThreshold: float = None, nEnergyDiff: int = None, alphaTstart: float = None,
                 alphaTmin: float = None, updateTestStepSize: bool = None, alphaTreduceFactor: float = None,
                 alphaTincreaseFactor: float = None, nAlphaAdjustMax: int = None, wolfeEnergy: float = None,
                 wolfeGradient: float = None, fdTest: bool = None):
        super().__init__(dirUpdateScheme=dirUpdateScheme, linminMethod=linminMethod,
                         nIterations=nIterations, history=history, knormThreshold=knormThreshold,
                         energyDiffThreshold=energyDiffThreshold, nEnergyDiff=nEnergyDiff, alphaTstart=alphaTstart,
                         alphaTmin=alphaTmin, updateTestStepSize=updateTestStepSize, alphaTreduceFactor=alphaTreduceFactor,
                         alphaTincreaseFactor=alphaTincreaseFactor, nAlphaAdjustMax=nAlphaAdjustMax,
                         wolfeEnergy=wolfeEnergy, wolfeGradient=wolfeGradient,
                         fdTest=fdTest)


@dataclass
class JMinSettingsFluid(JMinSettings):
    '''
    A class for storing lattice minimization settings read from a JDFTx out file
    '''

    start_flag: str = "fluid-minimize"


    def __init__(self, dirUpdateScheme: str = None, linminMethod: str = None,
                 nIterations: int = None, history: int = None, knormThreshold: float = None,
                 energyDiffThreshold: float = None, nEnergyDiff: int = None, alphaTstart: float = None,
                 alphaTmin: float = None, updateTestStepSize: bool = None, alphaTreduceFactor: float = None,
                 alphaTincreaseFactor: float = None, nAlphaAdjustMax: int = None, wolfeEnergy: float = None,
                 wolfeGradient: float = None, fdTest: bool = None):
        super().__init__(dirUpdateScheme=dirUpdateScheme, linminMethod=linminMethod,
                         nIterations=nIterations, history=history, knormThreshold=knormThreshold,
                         energyDiffThreshold=energyDiffThreshold, nEnergyDiff=nEnergyDiff, alphaTstart=alphaTstart,
                         alphaTmin=alphaTmin, updateTestStepSize=updateTestStepSize, alphaTreduceFactor=alphaTreduceFactor,
                         alphaTincreaseFactor=alphaTincreaseFactor, nAlphaAdjustMax=nAlphaAdjustMax,
                         wolfeEnergy=wolfeEnergy, wolfeGradient=wolfeGradient,
                         fdTest=fdTest)


@dataclass
class JMinSettingsLattice(JMinSettings):
    '''
    A class for storing lattice minimization settings read from a JDFTx out file
    '''

    start_flag: str = "lattice-minimize"


    def __init__(self, dirUpdateScheme: str = None, linminMethod: str = None,
                 nIterations: int = None, history: int = None, knormThreshold: float = None,
                 energyDiffThreshold: float = None, nEnergyDiff: int = None, alphaTstart: float = None,
                 alphaTmin: float = None, updateTestStepSize: bool = None, alphaTreduceFactor: float = None,
                 alphaTincreaseFactor: float = None, nAlphaAdjustMax: int = None, wolfeEnergy: float = None,
                 wolfeGradient: float = None, fdTest: bool = None):
        super().__init__(dirUpdateScheme=dirUpdateScheme, linminMethod=linminMethod,
                         nIterations=nIterations, history=history, knormThreshold=knormThreshold,
                         energyDiffThreshold=energyDiffThreshold, nEnergyDiff=nEnergyDiff, alphaTstart=alphaTstart,
                         alphaTmin=alphaTmin, updateTestStepSize=updateTestStepSize, alphaTreduceFactor=alphaTreduceFactor,
                         alphaTincreaseFactor=alphaTincreaseFactor, nAlphaAdjustMax=nAlphaAdjustMax,
                         wolfeEnergy=wolfeEnergy, wolfeGradient=wolfeGradient,
                         fdTest=fdTest)


@dataclass
class JMinSettingsIonic(JMinSettings):
    '''
    A class for storing ionic minimization settings read from a JDFTx out file
    '''

    start_flag: str = "ionic-minimize"


    def __init__(self, dirUpdateScheme: str = None, linminMethod: str = None,
                 nIterations: int = None, history: int = None, knormThreshold: float = None,
                 energyDiffThreshold: float = None, nEnergyDiff: int = None, alphaTstart: float = None,
                 alphaTmin: float = None, updateTestStepSize: bool = None, alphaTreduceFactor: float = None,
                 alphaTincreaseFactor: float = None, nAlphaAdjustMax: int = None, wolfeEnergy: float = None,
                 wolfeGradient: float = None, fdTest: bool = None):
        super().__init__(dirUpdateScheme=dirUpdateScheme, linminMethod=linminMethod,
                         nIterations=nIterations, history=history, knormThreshold=knormThreshold,
                         energyDiffThreshold=energyDiffThreshold, nEnergyDiff=nEnergyDiff, alphaTstart=alphaTstart,
                         alphaTmin=alphaTmin, updateTestStepSize=updateTestStepSize, alphaTreduceFactor=alphaTreduceFactor,
                         alphaTincreaseFactor=alphaTincreaseFactor, nAlphaAdjustMax=nAlphaAdjustMax,
                         wolfeEnergy=wolfeEnergy, wolfeGradient=wolfeGradient,
                         fdTest=fdTest)