"""Module defining core QChem input set generators"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core.structure import Molecule

from atomate2.qchem.sets.base import Q 