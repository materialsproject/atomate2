from dataclasses import dataclass


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
    updateTestStepSize: float = None
    alphaTreduceFactor: float = None
    alphaTincreaseFactor: float = None
    nAlphaAdjustMax: int = None
    wolfeEnergyThreshold: float = None
    wolfeGradientThreshold: float = None
    fdTest: bool = None
    #
    start_flag: str = None

    def __init__(self, dirUpdateScheme: str = None, linminMethod: str = None,
                 nIterations: int = None, history: int = None, knormThreshold: float = None,
                 energyDiffThreshold: float = None, nEnergyDiff: int = None, alphaTstart: float = None,
                 alphaTmin: float = None, updateTestStepSize: float = None, alphaTreduceFactor: float = None,
                 alphaTincreaseFactor: float = None, nAlphaAdjustMax: int = None, wolfeEnergyThreshold: float = None,
                 wolfeGradientThreshold: float = None, fdTest: bool = None):
        self.dirUpdateScheme = dirUpdateScheme
        self.linminMethod = linminMethod
        self.nIterations = nIterations
        self.history = history
        self.knormThreshold = knormThreshold
        self.energyDiffThreshold = energyDiffThreshold
        self.nEnergyDiff = nEnergyDiff
        self.alphaTstart = alphaTstart
        self.alphaTmin = alphaTmin
        self.updateTestStepSize = updateTestStepSize
        self.alphaTreduceFactor = alphaTreduceFactor
        self.alphaTincreaseFactor = alphaTincreaseFactor
        self.nAlphaAdjustMax = nAlphaAdjustMax
        self.wolfeEnergyThreshold = wolfeEnergyThreshold
        self.wolfeGradientThreshold = wolfeGradientThreshold
        self.fdTest = fdTest


@dataclass
class JMinSettingsElectronic(JMinSettings):
    '''
    A class for storing lattice minimization settings read from a JDFTx out file
    '''

    start_flag: str = "electronic-minimize"


    def __init__(self, dirUpdateScheme: str = None, linminMethod: str = None,
                 nIterations: int = None, history: int = None, knormThreshold: float = None,
                 energyDiffThreshold: float = None, nEnergyDiff: int = None, alphaTstart: float = None,
                 alphaTmin: float = None, updateTestStepSize: float = None, alphaTreduceFactor: float = None,
                 alphaTincreaseFactor: float = None, nAlphaAdjustMax: int = None, wolfeEnergyThreshold: float = None,
                 wolfeGradientThreshold: float = None, fdTest: bool = None):
        super().__init__(dirUpdateScheme=dirUpdateScheme, linminMethod=linminMethod,
                         nIterations=nIterations, history=history, knormThreshold=knormThreshold,
                         energyDiffThreshold=energyDiffThreshold, nEnergyDiff=nEnergyDiff, alphaTstart=alphaTstart,
                         alphaTmin=alphaTmin, updateTestStepSize=updateTestStepSize, alphaTreduceFactor=alphaTreduceFactor,
                         alphaTincreaseFactor=alphaTincreaseFactor, nAlphaAdjustMax=nAlphaAdjustMax,
                         wolfeEnergyThreshold=wolfeEnergyThreshold, wolfeGradientThreshold=wolfeGradientThreshold,
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
                 alphaTmin: float = None, updateTestStepSize: float = None, alphaTreduceFactor: float = None,
                 alphaTincreaseFactor: float = None, nAlphaAdjustMax: int = None, wolfeEnergyThreshold: float = None,
                 wolfeGradientThreshold: float = None, fdTest: bool = None):
        super().__init__(dirUpdateScheme=dirUpdateScheme, linminMethod=linminMethod,
                         nIterations=nIterations, history=history, knormThreshold=knormThreshold,
                         energyDiffThreshold=energyDiffThreshold, nEnergyDiff=nEnergyDiff, alphaTstart=alphaTstart,
                         alphaTmin=alphaTmin, updateTestStepSize=updateTestStepSize, alphaTreduceFactor=alphaTreduceFactor,
                         alphaTincreaseFactor=alphaTincreaseFactor, nAlphaAdjustMax=nAlphaAdjustMax,
                         wolfeEnergyThreshold=wolfeEnergyThreshold, wolfeGradientThreshold=wolfeGradientThreshold,
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
                 alphaTmin: float = None, updateTestStepSize: float = None, alphaTreduceFactor: float = None,
                 alphaTincreaseFactor: float = None, nAlphaAdjustMax: int = None, wolfeEnergyThreshold: float = None,
                 wolfeGradientThreshold: float = None, fdTest: bool = None):
        super().__init__(dirUpdateScheme=dirUpdateScheme, linminMethod=linminMethod,
                         nIterations=nIterations, history=history, knormThreshold=knormThreshold,
                         energyDiffThreshold=energyDiffThreshold, nEnergyDiff=nEnergyDiff, alphaTstart=alphaTstart,
                         alphaTmin=alphaTmin, updateTestStepSize=updateTestStepSize, alphaTreduceFactor=alphaTreduceFactor,
                         alphaTincreaseFactor=alphaTincreaseFactor, nAlphaAdjustMax=nAlphaAdjustMax,
                         wolfeEnergyThreshold=wolfeEnergyThreshold, wolfeGradientThreshold=wolfeGradientThreshold,
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
                 alphaTmin: float = None, updateTestStepSize: float = None, alphaTreduceFactor: float = None,
                 alphaTincreaseFactor: float = None, nAlphaAdjustMax: int = None, wolfeEnergyThreshold: float = None,
                 wolfeGradientThreshold: float = None, fdTest: bool = None):
        super().__init__(dirUpdateScheme=dirUpdateScheme, linminMethod=linminMethod,
                         nIterations=nIterations, history=history, knormThreshold=knormThreshold,
                         energyDiffThreshold=energyDiffThreshold, nEnergyDiff=nEnergyDiff, alphaTstart=alphaTstart,
                         alphaTmin=alphaTmin, updateTestStepSize=updateTestStepSize, alphaTreduceFactor=alphaTreduceFactor,
                         alphaTincreaseFactor=alphaTincreaseFactor, nAlphaAdjustMax=nAlphaAdjustMax,
                         wolfeEnergyThreshold=wolfeEnergyThreshold, wolfeGradientThreshold=wolfeGradientThreshold,
                         fdTest=fdTest)