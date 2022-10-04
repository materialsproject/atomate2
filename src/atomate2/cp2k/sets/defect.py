"""Module defining core VASP input set generators."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cp2k.inputs import Cp2kInput
from pymatgen.io.cp2k.outputs import Cp2kOutput

from atomate2.cp2k.sets.base import Cp2kInputGenerator, multi, multiple_updators
from atomate2.cp2k.sets.core import (
    StaticSetGenerator, RelaxSetGenerator, CellOptSetGenerator, HybridSetGenerator,
    HybridStaticSetGenerator, HybridRelaxSetGenerator, HybridCellOptSetGenerator
) 
logger = logging.getLogger(__name__)

@dataclass
class DefectSetGenerator(Cp2kInputGenerator):
    """
    """

    def get_input_updates(self, structure: Structure, *args, **kwargs) -> dict:
        """
        """
        return {'print_v_hartree': True}

@dataclass
@multiple_updators(multi)
class DefectStaticSetGenerator(DefectSetGenerator, StaticSetGenerator):
    pass    

@dataclass
@multiple_updators(multi)
class DefectRelaxSetGenerator(DefectSetGenerator, RelaxSetGenerator):
    pass

@dataclass
@multiple_updators(multi)
class DefectCellOptSetGenerator(DefectSetGenerator, CellOptSetGenerator):
    pass

@dataclass
@multiple_updators(multi)
class DefectHybridStaticSetGenerator(DefectSetGenerator, HybridSetGenerator, StaticSetGenerator):
    pass   

@dataclass
@multiple_updators(multi)
class DefectHybridRelaxSetGenerator(DefectSetGenerator, HybridSetGenerator, RelaxSetGenerator):
    pass   

@dataclass
@multiple_updators(multi)
class DefectHybridCellOptSetGenerator(DefectSetGenerator, HybridSetGenerator, CellOptSetGenerator):
    pass   