"""Define a Flow to test the convergence of a calculation.

Checks to see if the calculations are converged with respect to a particular parameter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.aims.jobs.base import BaseAimsMaker
from atomate2.aims.jobs.convergence import convergence_iteration

if TYPE_CHECKING:
    from pymatgen.core import Molecule, Structure

    from atomate2.aims.utils.msonable_atoms import MSONableAtoms


@dataclass
class ConvergenceMaker(Maker):
    """Defines a convergence workflow with a maximum number of steps.

    A job that performs convergence run for a given number of steps. Stops either
    when all steps are done, or when the convergence criterion is reached, that is when
    the absolute difference between the subsequent values of the convergence field is
    less than a given epsilon.

    Parameters
    ----------
    name : str
        A name for the job
    maker: .BaseAimsMaker
        A maker for the run
    criterion_name: str
        A name for the convergence criterion. Must be in the run results
    epsilon: float
        A difference in criterion value for subsequent runs
    convergence_field: str
        An input parameter that changes to achieve convergence
    convergence_steps: Iterable
        An iterable of the possible values for the convergence field.
        If the iterable is depleted and the convergence is not reached,
        that the job is failed
    last_idx: int
        The index of the last allowed convergence step (max number of steps)
    """

    name: str = "convergence"
    maker: BaseAimsMaker = field(default_factory=BaseAimsMaker)
    criterion_name: str = "energy_per_atom"
    epsilon: float = 0.001
    convergence_field: str = field(default_factory=str)
    convergence_steps: list = field(default_factory=list)
    last_idx: int = None

    def __post_init__(self):
        """Set the value of the last index."""
        self.last_idx = len(self.convergence_steps)

    def make(self, structure: MSONableAtoms | Structure | Molecule) -> Flow:
        """Create a top-level flow controlling convergence iteration.

        Parameters
        ----------
        structure : .MSONableAtoms or Structure or Molecule
            a structure to run a job
        """
        convergence_job = convergence_iteration(
            structure,
            self.last_idx,
            self.maker,
            self.criterion_name,
            self.epsilon,
            self.convergence_field,
            self.convergence_steps,
        )
        return Flow([convergence_job], output=convergence_job.output)
