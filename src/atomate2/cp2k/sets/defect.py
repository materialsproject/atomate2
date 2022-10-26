"""Module defining defect input set generators."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pymatgen.core import Structure

from atomate2.cp2k.sets.base import Cp2kInputGenerator, multiple_input_updators
from atomate2.cp2k.sets.core import (
    StaticSetGenerator, RelaxSetGenerator, CellOptSetGenerator,
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