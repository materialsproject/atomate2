"""Module defining core QChem input set generators."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from atomate2.qchem.sets.base import QChemInputGenerator

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
class SinglePointSetGenerator(QChemInputGenerator):
    """Class to generate QChem Single Point input sets."""

    def get_input_set_updates(self) -> dict:
        """Get updates to the input dict for a single point calculation."""
        return {"basis_set": "def2-tzvppd", "scf_algorithm": "diis", "job_type": "sp"}


@dataclass
class OptSetGenerator(QChemInputGenerator):
    """Class to generate QChem Optimization input sets."""

    def get_input_set_updates(self) -> dict:
        """Get updates to the input dict for a geometry optimization calculation."""
        return {"basis_set": "def2-tzvppd", "scf_algorithm": "diis", "job_type": "opt"}


@dataclass
class TransitionStateSetGenerator(QChemInputGenerator):
    """Class to generate QChem Transition State calculation input sets."""

    def get_basis_set_updates(self) -> dict:
        """Get updates to the input dict for a transition state calculation."""
        return {"basis_set": "def2-tzvppd", "scf_algorithm": "diis", "job_type": "ts"}


@dataclass
class ForceSetGenerator(QChemInputGenerator):
    """Class to generate QChem force input sets."""

    def get_basis_set_updates(self) -> dict:
        """Get updates to the input dict for a force(gradient) calculation."""
        return {
            "basis_set": "def2-tzvppd",
            "scf_algorithm": "diis",
            "job_type": "force",
        }


@dataclass
class FreqSetGenerator(QChemInputGenerator):
    """Class to generate QChem frequency calculation input sets."""

    def get_basis_set_updates(self) -> dict:
        """Get updates to the input dict for a frequency calculation."""
        return {"basis_set": "def2-tzvppd", "scf_algorithm": "diis", "job_type": "freq"}


@dataclass
class PESScanSetGenerator(QChemInputGenerator):
    """Class to generate QChem PES scan input sets."""

    def get_basis_set_updates(self) -> dict:
        """Get updates to the input dict for a pes scan calculation."""
        return {
            "basis_set": "def2-tzvppd",
            "scf_algorithm": "diis",
            "job_type": "pes_scan",
        }
