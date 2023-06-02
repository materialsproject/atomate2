"""
Module defining flows for Materials Project r2SCAN workflows

Reference: https://doi.org/10.1103/PhysRevMaterials.6.013801
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from jobflow import Flow, Maker
