"""Module defining defect input set generators."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from atomate2.cp2k.sets.core import (
    CellOptSetGenerator,
    HybridCellOptSetGenerator,
    HybridRelaxSetGenerator,
    HybridStaticSetGenerator,
    RelaxSetGenerator,
    StaticSetGenerator,
)

logger = logging.getLogger(__name__)

DEFECT_SET_UPDATES = {"print_v_hartree": True, "print_pdos": True, "print_dos": True}


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
