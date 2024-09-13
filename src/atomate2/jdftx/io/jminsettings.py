""" Store generic minimization settings read from a JDFTx out file.

This module contains the JMinSettings class for storing generic minimization
and mutants for storing specific minimization settings read from a JDFTx out
file.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Union


@dataclass
class JMinSettings:
    """ Store generic minimization settings read from a JDFTx out file.

    Store generic minimization settings read from a JDFTx out file.
    """

    dirupdatescheme: str = None
    linminmethod: str = None
    niterations: int = None
    history: int = None
    knormthreshold: float = None
    energydiffthreshold: float = None
    nenergydiff: int = None
    alphatstart: float = None
    alphatmin: float = None
    updateteststepsize: bool = None
    alphatreducefactor: float = None
    alphatincreasefactor: float = None
    nalphaadjustmax: int = None
    wolfeenergy: float = None
    wolfegradient: float = None
    fdtest: bool = None
    maxthreshold: bool = None
    start_flag: str = None

    def __init__(
        self,
        dirupdatescheme: str = None,
        linminmethod: str = None,
        niterations: int = None,
        history: int = None,
        knormthreshold: float = None,
        energydiffthreshold: float = None,
        nenergydiff: int = None,
        alphatstart: float = None,
        alphatmin: float = None,
        updateteststepsize: bool = None,
        alphatreducefactor: float = None,
        alphatincreasefactor: float = None,
        nalphaadjustmax: int = None,
        wolfeenergy: float = None,
        wolfegradient: float = None,
        fdtest: bool = None,
        maxthreshold: bool = None,
    ):
        self.dirupdatescheme = self._assign_type(dirupdatescheme, str)
        self.linminmethod = self._assign_type(linminmethod, str)
        self.niterations = self._assign_type(niterations, int)
        self.history = self._assign_type(history, int)
        self.knormthreshold = self._assign_type(knormthreshold, float)
        self.energydiffthreshold = self._assign_type(energydiffthreshold, float)
        self.nenergydiff = self._assign_type(nenergydiff, int)
        self.alphatstart = self._assign_type(alphatstart, float)
        self.alphatmin = self._assign_type(alphatmin, float)
        self.updateteststepsize = self._assign_type(updateteststepsize, bool)
        self.alphatreducefactor = self._assign_type(alphatreducefactor, float)
        self.alphatincreasefactor = self._assign_type(alphatincreasefactor, float)
        self.nalphaadjustmax = self._assign_type(nalphaadjustmax, int)
        self.wolfeenergy = self._assign_type(wolfeenergy, float)
        self.wolfegradient = self._assign_type(wolfegradient, float)
        self.fdtest = self._assign_type(fdtest, bool)
        self.maxthreshold = self._assign_type(maxthreshold, bool)

    def _assign_type(
        self, val: Optional[str], val_type: Callable[[str], Union[float, int, str]]
    ) -> Optional[Union[float, int, str]]:
        """ Assign the type of the value.

        Assign the type of the value.

        Parameters
        ----------
        val: Optional[str]
            The value to assign the type to
        val_type: Callable[[str], Union[float, int, str]]
            The type to assign to the value

        Returns
        -------
        Optional[Union[float, int, str]]
            The value with the assigned type
        """
        if val is None:
            return None
        return val_type(val)


@dataclass
class JMinSettingsElectronic(JMinSettings):
    """ JMInSettings mutant for electronic minimization settings.

    A class for storing electronic minimization settings read from a
    JDFTx out file.
    """

    start_flag: str = "electronic-minimize"

    def __init__(
        self,
        dirupdatescheme: str = None,
        linminmethod: str = None,
        niterations: int = None,
        history: int = None,
        knormthreshold: float = None,
        energydiffthreshold: float = None,
        nenergydiff: int = None,
        alphatstart: float = None,
        alphatmin: float = None,
        updateteststepsize: bool = None,
        alphatreducefactor: float = None,
        alphatincreasefactor: float = None,
        nalphaadjustmax: int = None,
        wolfeenergy: float = None,
        wolfegradient: float = None,
        fdtest: bool = None,
        maxthreshold: bool = None,
    ):
        super().__init__(
            dirupdatescheme=dirupdatescheme,
            linminmethod=linminmethod,
            niterations=niterations,
            history=history,
            knormthreshold=knormthreshold,
            energydiffthreshold=energydiffthreshold,
            nenergydiff=nenergydiff,
            alphatstart=alphatstart,
            alphatmin=alphatmin,
            updateteststepsize=updateteststepsize,
            alphatreducefactor=alphatreducefactor,
            alphatincreasefactor=alphatincreasefactor,
            nalphaadjustmax=nalphaadjustmax,
            wolfeenergy=wolfeenergy,
            wolfegradient=wolfegradient,
            fdtest=fdtest,
            maxthreshold=maxthreshold,
        )


@dataclass
class JMinSettingsFluid(JMinSettings):
    """ JMInSettings mutant for fluid minimization settings.

    A class for storing fluid minimization settings read from a
    JDFTx out file.
    """

    start_flag: str = "fluid-minimize"

    def __init__(
        self,
        dirupdatescheme: str = None,
        linminmethod: str = None,
        niterations: int = None,
        history: int = None,
        knormthreshold: float = None,
        energydiffthreshold: float = None,
        nenergydiff: int = None,
        alphatstart: float = None,
        alphatmin: float = None,
        updateteststepsize: bool = None,
        alphatreducefactor: float = None,
        alphatincreasefactor: float = None,
        nalphaadjustmax: int = None,
        wolfeenergy: float = None,
        wolfegradient: float = None,
        fdtest: bool = None,
        maxthreshold: bool = None,
    ):
        super().__init__(
            dirupdatescheme=dirupdatescheme,
            linminmethod=linminmethod,
            niterations=niterations,
            history=history,
            knormthreshold=knormthreshold,
            energydiffthreshold=energydiffthreshold,
            nenergydiff=nenergydiff,
            alphatstart=alphatstart,
            alphatmin=alphatmin,
            updateteststepsize=updateteststepsize,
            alphatreducefactor=alphatreducefactor,
            alphatincreasefactor=alphatincreasefactor,
            nalphaadjustmax=nalphaadjustmax,
            wolfeenergy=wolfeenergy,
            wolfegradient=wolfegradient,
            fdtest=fdtest,
            maxthreshold=maxthreshold,
        )


@dataclass
class JMinSettingsLattice(JMinSettings):
    """ JMInSettings mutant for lattice minimization settings.

    A class for storing lattice minimization settings read from a
    JDFTx out file.
    """

    start_flag: str = "lattice-minimize"

    def __init__(
        self,
        dirupdatescheme: str = None,
        linminmethod: str = None,
        niterations: int = None,
        history: int = None,
        knormthreshold: float = None,
        energydiffthreshold: float = None,
        nenergydiff: int = None,
        alphatstart: float = None,
        alphatmin: float = None,
        updateteststepsize: bool = None,
        alphatreducefactor: float = None,
        alphatincreasefactor: float = None,
        nalphaadjustmax: int = None,
        wolfeenergy: float = None,
        wolfegradient: float = None,
        fdtest: bool = None,
        maxthreshold: bool = None,
    ):
        super().__init__(
            dirupdatescheme=dirupdatescheme,
            linminmethod=linminmethod,
            niterations=niterations,
            history=history,
            knormthreshold=knormthreshold,
            energydiffthreshold=energydiffthreshold,
            nenergydiff=nenergydiff,
            alphatstart=alphatstart,
            alphatmin=alphatmin,
            updateteststepsize=updateteststepsize,
            alphatreducefactor=alphatreducefactor,
            alphatincreasefactor=alphatincreasefactor,
            nalphaadjustmax=nalphaadjustmax,
            wolfeenergy=wolfeenergy,
            wolfegradient=wolfegradient,
            fdtest=fdtest,
            maxthreshold=maxthreshold,
        )


@dataclass
class JMinSettingsIonic(JMinSettings):
    """ JMInSettings mutant for ionic minimization settings.

    A class for storing ionic minimization settings read from a
    JDFTx out file.
    """

    start_flag: str = "ionic-minimize"

    def __init__(
        self,
        dirupdatescheme: str = None,
        linminmethod: str = None,
        niterations: int = None,
        history: int = None,
        knormthreshold: float = None,
        energydiffthreshold: float = None,
        nenergydiff: int = None,
        alphatstart: float = None,
        alphatmin: float = None,
        updateteststepsize: bool = None,
        alphatreducefactor: float = None,
        alphatincreasefactor: float = None,
        nalphaadjustmax: int = None,
        wolfeenergy: float = None,
        wolfegradient: float = None,
        fdtest: bool = None,
        maxthreshold: bool = None,
    ):
        super().__init__(
            dirupdatescheme=dirupdatescheme,
            linminmethod=linminmethod,
            niterations=niterations,
            history=history,
            knormthreshold=knormthreshold,
            energydiffthreshold=energydiffthreshold,
            nenergydiff=nenergydiff,
            alphatstart=alphatstart,
            alphatmin=alphatmin,
            updateteststepsize=updateteststepsize,
            alphatreducefactor=alphatreducefactor,
            alphatincreasefactor=alphatincreasefactor,
            nalphaadjustmax=nalphaadjustmax,
            wolfeenergy=wolfeenergy,
            wolfegradient=wolfegradient,
            fdtest=fdtest,
            maxthreshold=maxthreshold,
        )
