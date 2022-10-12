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

from atomate2.cp2k.sets.base import Cp2kInputGenerator, multiple_input_updators
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
@multiple_input_updators()
class DefectStaticSetGenerator(DefectSetGenerator, StaticSetGenerator):
    pass    

@dataclass
@multiple_input_updators()
class DefectRelaxSetGenerator(DefectSetGenerator, RelaxSetGenerator):
    pass

@dataclass
@multiple_input_updators()
class DefectCellOptSetGenerator(DefectSetGenerator, CellOptSetGenerator):
    pass

@dataclass
@multiple_input_updators()
class DefectHybridStaticSetGenerator(DefectSetGenerator, HybridStaticSetGenerator):
    pass   

@dataclass
@multiple_input_updators()
class DefectHybridRelaxSetGenerator(DefectSetGenerator, HybridRelaxSetGenerator):
    pass   

@dataclass
@multiple_input_updators()
class DefectHybridCellOptSetGenerator(DefectSetGenerator, HybridCellOptSetGenerator):
    pass   