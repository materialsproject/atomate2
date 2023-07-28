"""Module defining qchem calculation types."""
from pathlib import Path

try:
    import atomate2.qchem.schemas.calc_types.enums
except ImportError:
    import atomate2.qchem.schemas.calc_types.generate

from atomate2.qchem.schemas.calc_types.enums import CalcType, LevelOfTheory, TaskType
from atomate2.qchem.schemas.calc_types.utils import calc_type, level_of_theory, task_type, solvent, lot_solvent_string