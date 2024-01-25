"""Module defining core QChem input set generators."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from atomate2.qchem.sets.base import QCInputGenerator

logger = logging.getLogger(__name__)


@dataclass
class SinglePointSetGenerator(QCInputGenerator):
    """Generate QChem Single Point input sets."""

    def __init__(self) -> None:
        super().__init__(job_type="sp", scf_algorithm="diis", basis_set="def2-tzvppd")


@dataclass
class OptSetGenerator(QCInputGenerator):
    """Generate QChem Optimization input sets."""

    def __init__(self) -> None:
        super().__init__(job_type="opt", scf_algorithm="diis", basis_set="def2-tzvppd")


@dataclass
class TransitionStateSetGenerator(QCInputGenerator):
    """Generate QChem Transition State calculation input sets."""

    def __init__(self) -> None:
        super().__init__(job_type="ts", scf_algorithm="diis", basis_set="def2-tzvppd")


@dataclass
class ForceSetGenerator(QCInputGenerator):
    """Generate QChem force input sets."""

    def __init__(self) -> None:
        super().__init__(
            job_type="force", scf_algorithm="diis", basis_set="def2-tzvppd"
        )


@dataclass
class FreqSetGenerator(QCInputGenerator):
    """Generate QChem frequency calculation input sets."""

    def __init__(self) -> None:
        super().__init__(job_type="freq", scf_algorithm="diis", basis_set="def2-tzvppd")


@dataclass
class PESScanSetGenerator(QCInputGenerator):
    """Generate QChem PES scan input sets."""

    def __init__(self) -> None:
        super().__init__(
            job_type="pes_scan", scf_algorithm="diis", basis_set="def2-tzvppd"
        )
