"""Module defining core QChem input set generators."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from atomate2.qchem.sets.base import QCInputGenerator

logger = logging.getLogger(__name__)


@dataclass
class SinglePointSetGenerator(QCInputGenerator):
    """Generate QChem Single Point input sets."""

    job_type: str = "sp"
    scf_algorithm: str = "diis"
    basis_set: str = "def2-tzvppd"


@dataclass
class OptSetGenerator(QCInputGenerator):
    """Generate QChem Optimization input sets."""

    job_type: str = "opt"
    scf_algorithm: str = "diis"
    basis_set: str = "def2-tzvppd"


@dataclass
class TransitionStateSetGenerator(QCInputGenerator):
    """Generate QChem Transition State calculation input sets."""

    job_type: str = "ts"
    scf_algorithm: str = "diis"
    basis_set: str = "def2-tzvppd"


@dataclass
class ForceSetGenerator(QCInputGenerator):
    """Generate QChem force input sets."""

    job_type: str = "force"
    scf_algorithm: str = "diis"
    basis_set: str = "def2-tzvppd"


@dataclass
class FreqSetGenerator(QCInputGenerator):
    """Generate QChem frequency calculation input sets."""

    job_type: str = "freq"
    scf_algorithm: str = "diis"
    basis_set: str = "def2-tzvppd"


@dataclass
class PESScanSetGenerator(QCInputGenerator):
    """Generate QChem PES scan input sets."""

    job_type: str = "pes_scan"
    scf_algorithm: str = "diis"
    basis_set: str = "def2-tzvppd"
