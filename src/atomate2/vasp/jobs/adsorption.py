"""Jobs for running adsorption calculations."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Response, job

from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.surface import Slab, generate_all_slabs
from pymatgen.io.vasp.sets import MVLSlabSet
from pymatgen.transformations.advanced_transformations import SlabTransformation
from pymatgen.transformations.standard_transformations import SupercellTransformation

