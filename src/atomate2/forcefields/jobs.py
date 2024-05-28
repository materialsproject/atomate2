"""Job to relax a structure using a force field (aka an interatomic potential)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ase.units import GPa as _GPa_to_eV_per_A3
from jobflow import Maker, job
from pymatgen.core.trajectory import Trajectory

from atomate2.forcefields import MLFF
from atomate2.forcefields.schemas import ForceFieldTaskDocument
from atomate2.forcefields.utils import Relaxer, ase_calculator, revert_default_dtype

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Callable

    from ase.calculators.calculator import Calculator
    from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)

_FORCEFIELD_DATA_OBJECTS = [Trajectory]


def forcefield_job(method: Callable) -> job:
    """
    Decorate the ``make`` method of forcefield job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.Job` that configures common
    settings for all forcefield jobs. For example, it ensures that large data objects
    (currently only trajectories) are all stored in the atomate2 data store.
    It also configures the output schema to be a ForceFieldTaskDocument :obj:`.TaskDoc`.

    Any makers that return forcefield jobs (not flows) should decorate the
    ``make`` method with @forcefield_job. For example:

    .. code-block:: python

        class MyForcefieldMaker(Maker):
            @forcefield_job
            def make(structure):
                # code to run forcefield job.
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
    return job(
        method, data=_FORCEFIELD_DATA_OBJECTS, output_schema=ForceFieldTaskDocument
    )


@dataclass
class ForceFieldRelaxMaker(Maker):
    """
    Base Maker to calculate forces and stresses using any force field.

    Should be subclassed to use a specific force field. By default,
    the code attempts to use the `self.force_field_name` attr to look
    up a predefined forcefield. To overwrite this behavior, redefine `self.calculator`.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str
        The name of the force field.
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
        Keyword arguments that will get passed to :obj:`Relaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = "Force field relax"
    force_field_name: str = f"{MLFF.Forcefield}"
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    @forcefield_job
    def make(
        self, structure: Structure, prev_dir: str | Path | None = None
    ) -> ForceFieldTaskDocument:
        """
        Perform a relaxation of a structure using a force field.

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

        with revert_default_dtype():
            relaxer = Relaxer(
                self.calculator,
                relax_cell=self.relax_cell,
                fix_symmetry=self.fix_symmetry,
                symprec=self.symprec,
                **self.optimizer_kwargs,
            )
            result = relaxer.relax(structure, steps=self.steps, **self.relax_kwargs)

        return ForceFieldTaskDocument.from_ase_compatible_result(
            self.force_field_name,
            result,
            self.relax_cell,
            self.steps,
            self.relax_kwargs,
            self.optimizer_kwargs,
            self.fix_symmetry,
            self.symprec,
            **self.task_document_kwargs,
        )

    @property
    def calculator(self) -> Calculator:
        """ASE calculator, can be overwritten by user."""
        return ase_calculator(self.force_field_name, **self.calculator_kwargs)


@dataclass
class ForceFieldStaticMaker(ForceFieldRelaxMaker):
    """
    Maker to calculate forces and stresses using any force field.

    Note that while `steps = 1` by default, the user could override
    this setting along with cell shape relaxation (`relax_cell = False`
    by default).

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str
        The name of the force field.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = "Force field static"
    force_field_name: str = "Force field"
    relax_cell: bool = False
    steps: int = 1
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)


@dataclass
class CHGNetRelaxMaker(ForceFieldRelaxMaker):
    """
    Maker to perform a relaxation using the CHGNet universal ML force field.

    Parameters
    ----------
    force_field_name : str
        The name of the force field.
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
        Keyword arguments that will get passed to :obj:`Relaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.CHGNet} relax"
    force_field_name: str = f"{MLFF.CHGNet}"
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(
        default_factory=lambda: {"stress_weight": _GPa_to_eV_per_A3}
    )


@dataclass
class CHGNetStaticMaker(ForceFieldStaticMaker):
    """
    Maker to calculate forces and stresses using the CHGNet force field.

    Parameters
    ----------
    name : str
        The job name.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.CHGNet} static"
    force_field_name: str = f"{MLFF.CHGNet}"
    task_document_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(
        default_factory=lambda: {"stress_weight": _GPa_to_eV_per_A3}
    )


@dataclass
class M3GNetRelaxMaker(ForceFieldRelaxMaker):
    """
    Maker to perform a relaxation using the M3GNet universal ML force field.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str
        The name of the force field.
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
        Keyword arguments that will get passed to :obj:`Relaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.M3GNet} relax"
    force_field_name: str = f"{MLFF.M3GNet}"
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(
        default_factory=lambda: {"stress_weight": _GPa_to_eV_per_A3}
    )


@dataclass
class NequipRelaxMaker(ForceFieldRelaxMaker):
    """
    Maker to perform a relaxation using a Nequip force field.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str
        The name of the force field.
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
        Keyword arguments that will get passed to :obj:`Relaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.Nequip} relax"
    force_field_name: str = f"{MLFF.Nequip}"
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)


@dataclass
class NequipStaticMaker(ForceFieldStaticMaker):
    """
    Maker to calculate energies, forces and stresses using a nequip force field.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str
        The name of the force field.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.Nequip} static"
    force_field_name: str = f"{MLFF.Nequip}"
    task_document_kwargs: dict = field(default_factory=dict)


@dataclass
class M3GNetStaticMaker(ForceFieldStaticMaker):
    """
    Maker to calculate forces and stresses using the M3GNet force field.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str
        The name of the force field.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.M3GNet} static"
    force_field_name: str = f"{MLFF.M3GNet}"
    task_document_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(
        default_factory=lambda: {"stress_weight": _GPa_to_eV_per_A3}
    )


@dataclass
class MACERelaxMaker(ForceFieldRelaxMaker):
    """
    Base Maker to calculate forces and stresses using a MACE potential.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str
        The name of the force field.
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
        Keyword arguments that will get passed to :obj:`Relaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator. E.g. the "model"
        key configures which checkpoint to load with mace.calculators.MACECalculator().
        Can be a URL starting with https://. If not set, loads the universal MACE-MP
        trained for Matbench Discovery on the MPtrj dataset available at
        https://figshare.com/articles/dataset/22715158.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.MACE} relax"
    force_field_name: str = f"{MLFF.MACE}"
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)


@dataclass
class MACEStaticMaker(ForceFieldStaticMaker):
    """
    Base Maker to calculate forces and stresses using a MACE potential.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str
        The name of the force field.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator. E.g. the "model"
        key configures which checkpoint to load with mace.calculators.MACECalculator().
        Can be a URL starting with https://. If not set, loads the universal MACE-MP
        trained for Matbench Discovery on the MPtrj dataset available at
        https://figshare.com/articles/dataset/22715158.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.MACE} static"
    force_field_name: str = f"{MLFF.MACE}"
    task_document_kwargs: dict = field(default_factory=dict)


@dataclass
class GAPRelaxMaker(ForceFieldRelaxMaker):
    """
    Base Maker to calculate forces and stresses using a GAP potential.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str
        The name of the force field.
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
        Keyword arguments that will get passed to :obj:`Relaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.GAP} relax"
    force_field_name: str = f"{MLFF.GAP}"
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(
        default_factory=lambda: {
            "args_str": "IP GAP",
            "param_filename": "gap.xml",
        }
    )
    task_document_kwargs: dict = field(default_factory=dict)


@dataclass
class GAPStaticMaker(ForceFieldStaticMaker):
    """
    Base Maker to calculate forces and stresses using a GAP potential.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str
        The name of the force field.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.GAP} static"
    force_field_name: str = f"{MLFF.GAP}"
    task_document_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(
        default_factory=lambda: {
            "args_str": "IP GAP",
            "param_filename": "gap.xml",
        }
    )
