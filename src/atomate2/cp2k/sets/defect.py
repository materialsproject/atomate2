"""Module defining defect input set generators."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pymatgen.core import Structure

from atomate2.cp2k.sets.base import Cp2kInputGenerator
from atomate2.cp2k.sets.core import (
    StaticSetGenerator, RelaxSetGenerator, CellOptSetGenerator,
    HybridStaticSetGenerator, HybridRelaxSetGenerator, HybridCellOptSetGenerator
)
logger = logging.getLogger(__name__)

DEFECT_SET_UPDATES = {'print_v_hartree': True, "print_pdos": True, "print_dos": True}

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
class DefectStaticSetGenerator(StaticSetGenerator):

    def __post_init__(self):
        self.user_input_settings.update(DEFECT_SET_UPDATES)

@dataclass
class DefectRelaxSetGenerator(RelaxSetGenerator):

    def __post_init__(self):
        self.user_input_settings.update(DEFECT_SET_UPDATES)

@dataclass
class DefectCellOptSetGenerator(CellOptSetGenerator):

    def __post_init__(self):
        self.user_input_settings.update(DEFECT_SET_UPDATES)

@dataclass
class DefectHybridStaticSetGenerator(HybridStaticSetGenerator):

    def __post_init__(self):
        self.user_input_settings.update(DEFECT_SET_UPDATES)

@dataclass
class DefectHybridRelaxSetGenerator(HybridRelaxSetGenerator):

    def __post_init__(self):
        self.user_input_settings.update(DEFECT_SET_UPDATES)

@dataclass
class DefectHybridCellOptSetGenerator(HybridCellOptSetGenerator):

    def __post_init__(self):
        self.user_input_settings.update(DEFECT_SET_UPDATES)