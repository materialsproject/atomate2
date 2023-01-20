"""Module defining defect input set generators."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pymatgen.core import Structure
from pymatgen.io.cp2k.utils import get_truncated_coulomb_cutoff

from atomate2.cp2k.sets.base import Cp2kInputGenerator
from atomate2.cp2k.sets.core import (
    StaticSetGenerator, RelaxSetGenerator, CellOptSetGenerator,
    HybridStaticSetGenerator, HybridRelaxSetGenerator, HybridCellOptSetGenerator
)
logger = logging.getLogger(__name__)

DEFECT_SET_UPDATES = {'print_v_hartree': True, "print_pdos": True, "print_dos": True}

@dataclass
class DefectStaticSetGenerator(StaticSetGenerator):

    def get_input_updates(self, *args, **kwargs) -> dict:
        updates = super().get_input_updates(*args, **kwargs)
        updates.update(DEFECT_SET_UPDATES)
        return updates

@dataclass
class DefectRelaxSetGenerator(RelaxSetGenerator):

    def get_input_updates(self, *args, **kwargs) -> dict:
        updates = super().get_input_updates(*args, **kwargs)
        updates.update(DEFECT_SET_UPDATES)
        return updates

@dataclass
class DefectCellOptSetGenerator(CellOptSetGenerator):

    def get_input_updates(self, *args, **kwargs) -> dict:
        updates = super().get_input_updates(*args, **kwargs)
        updates.update(DEFECT_SET_UPDATES)
        return updates

@dataclass
class DefectHybridStaticSetGenerator(HybridStaticSetGenerator):

    def get_input_updates(self, *args, **kwargs) -> dict:
        updates = super().get_input_updates(*args, **kwargs)
        updates.update(DEFECT_SET_UPDATES)
        return updates

@dataclass
class DefectHybridRelaxSetGenerator(HybridRelaxSetGenerator):

    def get_input_updates(self, *args, **kwargs) -> dict:
        updates = super().get_input_updates(*args, **kwargs)
        updates.update(DEFECT_SET_UPDATES)
        return updates

@dataclass
class DefectHybridCellOptSetGenerator(HybridCellOptSetGenerator):

    def get_input_updates(self, *args, **kwargs) -> dict:
        updates = super().get_input_updates(*args, **kwargs)
        updates.update(DEFECT_SET_UPDATES)
        return updates