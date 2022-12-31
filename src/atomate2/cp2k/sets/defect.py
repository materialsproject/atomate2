"""Module defining defect input set generators."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pymatgen.core import Structure

from atomate2.cp2k.sets.base import Cp2kInputGenerator, multiple_input_updators
from atomate2.cp2k.sets.core import (
    HybridSetGenerator, StaticSetGenerator, RelaxSetGenerator, CellOptSetGenerator,
) 
logger = logging.getLogger(__name__)

@dataclass
class DefectSetGenerator(Cp2kInputGenerator):
    """
    Base input set generator for defect calculations. Adds printing of the 
    partial density of states and the electrostatic potential.
    """

    def get_input_updates(self, structure: Structure, *args, **kwargs) -> dict:
        """Get input updates"""
        return {'print_v_hartree': True, "print_pdos": True, "print_dos": True}

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
class DefectHybridStaticSetGenerator(DefectSetGenerator, StaticSetGenerator, HybridSetGenerator):
    pass   

@dataclass
@multiple_input_updators()
class DefectHybridRelaxSetGenerator(DefectSetGenerator, RelaxSetGenerator, HybridSetGenerator):
    pass

@dataclass
@multiple_input_updators()
class DefectHybridCellOptSetGenerator(DefectSetGenerator, CellOptSetGenerator, HybridSetGenerator):
    pass 