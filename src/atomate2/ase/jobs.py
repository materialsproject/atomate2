"""Define general ASE-calculator jobs."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ase.io import Trajectory as AseTrajectory
from jobflow import Maker, job
from pymatgen.core.trajectory import Trajectory as PmgTrajectory

from atomate2.ase.schemas import AseResult, AseTaskDocument
from atomate2.ase.utils import AseRelaxer

logger = logging.getLogger(__name__)

_ASE_DATA_OBJECTS = [PmgTrajectory, AseTrajectory]

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Callable

    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure


def ase_job(method: Callable) -> job:
    """
    Decorate the ``make`` method of ASE job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.Job` that configures common
    settings for all ASE jobs. For example, it ensures that large data objects
    (currently only trajectories) are all stored in the atomate2 data store.
    It also configures the output schema to be a AseTaskDocument :obj:`.TaskDoc`.

    Any makers that return ASE jobs (not flows) should decorate the
    ``make`` method with @ase_job. For example:

    .. code-block:: python

        class MyAseMaker(Maker):
            @ase_job
            def make(structure):
                # code to run ase job.
                pass

    Parameters
    ----------
    method : callable
        A Maker.make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate forcefield jobs.
    """
    return job(method, data=_ASE_DATA_OBJECTS, output_schema=AseTaskDocument)


@dataclass
class AseRelaxMaker(Maker):
    """
    Base Maker to calculate forces and stresses using any ASE calculator.

    Should be subclassed to use a specific ASE. The user should
    define `self.calculator` when subclassing.

    Parameters
    ----------
    name : str
        The job name.
    relax_cell : bool = True
        Whether to allow the cell shape/volume to change during relaxation.
    fix_symmetry : bool = False
        Whether to fix the symmetry during relaxation.
        Refines the symmetry of the initial structure.
    symprec : float = 1e-2
        Tolerance for symmetry finding in case of fix_symmetry.
    steps : int
        Maximum number of ionic steps allowed during relaxation.
    relax_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.AseTaskDocument()`.
    """

    name: str = "ASE relaxation"
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    @ase_job
    def make(
        self, structure: Structure, prev_dir: str | Path | None = None
    ) -> AseTaskDocument:
        """
        Relax a structure using ASE as a job.

        Parameters
        ----------
        structure: .Structure
            pymatgen structure.
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from. Unused, just
                added to match the method signature of other makers.
        """
        return AseTaskDocument.from_ase_compatible_result(
            getattr(self.calculator, "name", self.calculator.__class__),
            self._make(structure, prev_dir=prev_dir),
            self.relax_cell,
            self.steps,
            self.relax_kwargs,
            self.optimizer_kwargs,
            self.fix_symmetry,
            self.symprec,
            **self.task_document_kwargs,
        )

    def _make(
        self, structure: Structure, prev_dir: str | Path | None = None
    ) -> AseResult:
        """
        Relax a structure using ASE, not as a job.

        This method exists to permit child classes to redefine `make`
        for different output schemas.

        Parameters
        ----------
        structure: .Structure
            pymatgen structure.
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from. Unused, just
                added to match the method signature of other makers.
        """
        if self.steps < 0:
            logger.warning(
                "WARNING: A negative number of steps is not possible. "
                "Behavior may vary..."
            )
        self.task_document_kwargs.setdefault("dir_name", os.getcwd())

        relaxer = AseRelaxer(
            self.calculator,
            relax_cell=self.relax_cell,
            fix_symmetry=self.fix_symmetry,
            symprec=self.symprec,
            **self.optimizer_kwargs,
        )
        return relaxer.relax(structure, steps=self.steps, **self.relax_kwargs)

    @property
    def calculator(self) -> Calculator:
        """ASE calculator, can be overwritten by user."""
        return NotImplemented


@dataclass
class LennardJonesRelaxMaker(AseRelaxMaker):
    """
    Relax a structure with a Lennard-Jones 6-12 potential.

    This serves mostly as an example of how to create atomate2
    jobs with existing ASE calculators, and test purposes.

    Parameters
    ----------
    name : str
        The job name.
    relax_cell : bool = True
        Whether to allow the cell shape/volume to change during relaxation.
    fix_symmetry : bool = False
        Whether to fix the symmetry during relaxation.
        Refines the symmetry of the initial structure.
    symprec : float = 1e-2
        Tolerance for symmetry finding in case of fix_symmetry.
    steps : int
        Maximum number of ionic steps allowed during relaxation.
    relax_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.AseTaskDocument()`.
    """

    name: str = "Lennard-Jones 6-12 relaxation"

    @property
    def calculator(self) -> Calculator:
        """Lennard-Jones calculator."""
        from ase.calculators.lj import LennardJones

        return LennardJones(**self.calculator_kwargs)


@dataclass
class LennardJonesStaticMaker(LennardJonesRelaxMaker):
    """
    Single-point Lennard-Jones 6-12 potential calculation.

    Parameters
    ----------
    name : str
        The job name.
    relax_cell : bool = True
        Whether to allow the cell shape/volume to change during relaxation.
    fix_symmetry : bool = False
        Whether to fix the symmetry during relaxation.
        Refines the symmetry of the initial structure.
    symprec : float = 1e-2
        Tolerance for symmetry finding in case of fix_symmetry.
    steps : int
        Maximum number of ionic steps allowed during relaxation.
    relax_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.AseTaskDocument()`.
    """

    name: str = "Lennard-Jones 6-12 static"
    steps: int = 1


@dataclass
class GFNxTBRelaxMaker(AseRelaxMaker):
    """
    Relax a structure with TBLite (GFN-xTB).

    If you use TBLite in your work, consider citing:
    H. Neugebauer, B. Bädorf, S. Ehlert, A. Hansen, and S. Grimme,
    J. Comput. Chem. 44, 2120 (2023).

    If you use GFN1-xTB, consider citing:
    S. Grimme, C. Bannwarth, and P. Shushkov,
    J. Chem. Theory Comput. 13, 1989 (2017).

    If you use GFN2-xTB, consider citing:
    C. Bannwarth, S. Ehlert, and S. Grimme
    J. Chem. Theory Comput. 15, 1652 (2019)

    Parameters
    ----------
    name : str
        The job name.
    relax_cell : bool = True
        Whether to allow the cell shape/volume to change during relaxation.
    fix_symmetry : bool = False
        Whether to fix the symmetry during relaxation.
        Refines the symmetry of the initial structure.
    symprec : float = 1e-2
        Tolerance for symmetry finding in case of fix_symmetry.
    steps : int
        Maximum number of ionic steps allowed during relaxation.
    relax_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.AseTaskDocument()`.
    """

    name: str = "GFN-xTB relaxation"
    calculator_kwargs: dict = field(
        default_factory=lambda: {
            "method": "GFN1-xTB",
            "charge": None,
            "multiplicity": None,
            "accuracy": 1.0,
            "guess": "sad",
            "max_iterations": 250,
            "mixer_damping": 0.4,
            "electric_field": None,
            "spin_polarization": None,
            "electronic_temperature": 300.0,
            "cache_api": True,
            "verbosity": 1,
        }
    )

    @property
    def calculator(self) -> Calculator:
        """GFN-xTB / TBLite calculator."""
        from tblite.ase import TBLite

        return TBLite(atoms=None, **self.calculator_kwargs)


@dataclass
class GFNxTBStaticMaker(GFNxTBRelaxMaker):
    """
    Single-point GFNn-xTB calculation.

    If you use TBLite in your work, consider citing:
    H. Neugebauer, B. Bädorf, S. Ehlert, A. Hansen, and S. Grimme,
    J. Comput. Chem. 44, 2120 (2023).

    If you use GFN1-xTB, consider citing:
    S. Grimme, C. Bannwarth, and P. Shushkov,
    J. Chem. Theory Comput. 13, 1989 (2017).

    If you use GFN2-xTB, consider citing:
    C. Bannwarth, S. Ehlert, and S. Grimme
    J. Chem. Theory Comput. 15, 1652 (2019)

    Parameters
    ----------
    name : str
        The job name.
    relax_cell : bool = True
        Whether to allow the cell shape/volume to change during relaxation.
    fix_symmetry : bool = False
        Whether to fix the symmetry during relaxation.
        Refines the symmetry of the initial structure.
    symprec : float = 1e-2
        Tolerance for symmetry finding in case of fix_symmetry.
    steps : int
        Maximum number of ionic steps allowed during relaxation.
    relax_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.AseTaskDocument()`.
    """

    name: str = "GFN-xTB static"
    steps: int = 1
