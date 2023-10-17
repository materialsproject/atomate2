"""Flow for calculating surface adsorption energies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.vasp.jobs.adsorption import ()

from atomate2.common.jobs.utils import structure_to_conventional, structure_to_primitive
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import DielectricMaker, StaticMaker, TightRelaxMaker
from atomate2.vasp.sets.core import StaticSetGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure
    from atomate2.vasp.jobs.base import BaseVaspMaker