"""Module defining core QChem input set generators."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from atomate2.qchem.sets.base import QCInputGenerator

logger = logging.getLogger(__name__)

__all__ = [
    "SinglePointSetGenerator",
    "OptSetGenerator",
    "TransitionStateSetGenerator",
    "ForceSetGenerator",
    "FreqSetGenerator",
    "PESScanSetGenerator",
]


@dataclass
class SinglePointSetGenerator(QCInputGenerator):
    """Class to generate QChem Single Point input sets."""

    # def get_input_set_updates(self) -> dict:
    #     """Get updates to the input dict for a single point calculation."""
    def __init__(self):
        super().__init__(job_type="sp", scf_algorithm="diis", basis_set="def2-tzvppd")


@dataclass
class OptSetGenerator(QCInputGenerator):
    """Class to generate QChem Optimization input sets."""

    # def get_input_set_updates(self) -> dict:
    #     """Get updates to the input dict for a geometry optimization calculation."""
    def __init__(self):
        super().__init__(job_type="opt", scf_algorithm="diis", basis_set="def2-tzvppd")


@dataclass
class TransitionStateSetGenerator(QCInputGenerator):
    """Class to generate QChem Transition State calculation input sets."""

    # def get_basis_set_updates(self) -> dict:
    #     """Get updates to the input dict for a transition state calculation."""
    def __init__(self):
        super().__init__(job_type="ts", scf_algorithm="diis", basis_set="def2-tzvppd")


@dataclass
class ForceSetGenerator(QCInputGenerator):
    """Class to generate QChem force input sets."""

    # def get_basis_set_updates(self) -> dict:
    #     """Get updates to the input dict for a force(gradient) calculation."""
    def __init__(self):
        super().__init__(
            job_type="force", scf_algorithm="diis", basis_set="def2-tzvppd"
        )


@dataclass
class FreqSetGenerator(QCInputGenerator):
    """Class to generate QChem frequency calculation input sets."""

    # def get_basis_set_updates(self) -> dict:
    #     """Get updates to the input dict for a frequency calculation."""
    def __init__(self):
        super().__init__(job_type="freq", scf_algorithm="diis", basis_set="def2-tzvppd")


@dataclass
class PESScanSetGenerator(QCInputGenerator):
    """Class to generate QChem PES scan input sets."""

    # def get_basis_set_updates(self) -> dict:
    #     """Get updates to the input dict for a pes scan calculation."""
    def __init__(self):
        super().__init__(
            job_type="pes_scan", scf_algorithm="diis", basis_set="def2-tzvppd"
        )
