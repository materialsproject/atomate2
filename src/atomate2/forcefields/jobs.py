"""Job to relax a structure using a force field (aka an interatomic potential)."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ase.io import Trajectory as AseTrajectory
from ase.units import GPa as _GPa_to_eV_per_A3
from jobflow import job
from monty.dev import deprecated
from pymatgen.core.trajectory import Trajectory as PmgTrajectory

from atomate2.ase.jobs import AseRelaxMaker
from atomate2.forcefields import MLFF, _get_formatted_ff_name
from atomate2.forcefields.schemas import ForceFieldTaskDocument
from atomate2.forcefields.utils import ase_calculator, revert_default_dtype

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ase.calculators.calculator import Calculator
    from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)

_FORCEFIELD_DATA_OBJECTS = [PmgTrajectory, AseTrajectory, "ionic_steps"]

_DEFAULT_CALCULATOR_KWARGS = {
    MLFF.CHGNet: {"stress_weight": _GPa_to_eV_per_A3},
    MLFF.M3GNet: {"stress_weight": _GPa_to_eV_per_A3},
    MLFF.NEP: {"model_filename": "nep.txt"},
    MLFF.GAP: {"args_str": "IP GAP", "param_filename": "gap.xml"},
}


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
class ForceFieldRelaxMaker(AseRelaxMaker):
    """
    Base Maker to calculate forces and stresses using any force field.

    Should be subclassed to use a specific force field. By default,
    the code attempts to use the `self.force_field_name` attr to look
    up a predefined forcefield. To overwrite this behavior, redefine `self.calculator`.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str or .MLFF
        The name of the force field.
    relax_cell : bool = True
        Whether to allow the cell shape/volume to change during relaxation.
    fix_symmetry : bool = False
        Whether to fix the symmetry during relaxation.
        Refines the symmetry of the initial structure.
    symprec : float | None = 1e-2
        Tolerance for symmetry finding in case of fix_symmetry.
    steps : int
        Maximum number of ionic steps allowed during relaxation.
    relax_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    ionic_step_data : tuple[str,...] or None
        Quantities to store in the TaskDocument ionic_steps.
        Possible options are "struct_or_mol", "energy",
        "forces", "stress", and "magmoms".
        "structure" and "molecule" are aliases for "struct_or_mol".
    store_trajectory : emmet .StoreTrajectoryOption = "no"
        Whether to store trajectory information ("no") or complete trajectories
        ("partial" or "full", which are identical).
    tags : list[str] or None
        A list of tags for the task.
    task_document_kwargs : dict (deprecated)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = "Force field relax"
    force_field_name: str | MLFF = MLFF.Forcefield
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float | None = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure that force_field_name is correctly assigned as str."""
        self.force_field_name = _get_formatted_ff_name(self.force_field_name)

        # Pad calculator_kwargs with default values, but permit user to override them
        self.calculator_kwargs = {
            **_DEFAULT_CALCULATOR_KWARGS.get(
                MLFF(self.force_field_name.split("MLFF.")[-1]), {}
            ),
            **self.calculator_kwargs,
        }

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
        with revert_default_dtype():
            ase_result = self.run_ase(structure, prev_dir=prev_dir)

        if len(self.task_document_kwargs) > 0:
            warnings.warn(
                "`task_document_kwargs` is now deprecated, please use the top-level "
                "attributes `ionic_step_data` and `store_trajectory`",
                category=DeprecationWarning,
                stacklevel=1,
            )

        return ForceFieldTaskDocument.from_ase_compatible_result(
            str(self.force_field_name),  # make mypy happy
            ase_result,
            self.steps,
            relax_kwargs=self.relax_kwargs,
            optimizer_kwargs=self.optimizer_kwargs,
            relax_cell=self.relax_cell,
            fix_symmetry=self.fix_symmetry,
            symprec=self.symprec if self.fix_symmetry else None,
            ionic_step_data=self.ionic_step_data,
            store_trajectory=self.store_trajectory,
            tags=self.tags,
            **self.task_document_kwargs,
        )

    @property
    def calculator(self) -> Calculator:
        """ASE calculator, can be overwritten by user."""
        return ase_calculator(
            str(self.force_field_name),  # make mypy happy
            **self.calculator_kwargs,
        )


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
    force_field_name : str or .MLFF
        The name of the force field.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict (deprecated)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = "Force field static"
    force_field_name: str | MLFF = MLFF.Forcefield
    relax_cell: bool = False
    steps: int = 1
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)


@deprecated(
    replacement=ForceFieldRelaxMaker,
    deadline=(2025, 1, 1),
    message="To use CHGNet, set `force_field_name = 'CHGNet'` in ForceFieldRelaxMaker.",
)
@dataclass
class CHGNetRelaxMaker(ForceFieldRelaxMaker):
    """
    Maker to perform a relaxation using the CHGNet ML force field.

    Parameters
    ----------
    force_field_name : str or .MLFF
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
        Keyword arguments that will get passed to :obj:`AseRelaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict (deprecated)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.CHGNet} relax"
    force_field_name: str | MLFF = MLFF.CHGNet
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(
        default_factory=lambda: _DEFAULT_CALCULATOR_KWARGS[MLFF.CHGNet]
    )


@deprecated(
    replacement=ForceFieldStaticMaker,
    deadline=(2025, 1, 1),
    message=(
        "To use CHGNet, set `force_field_name = 'CHGNet'` in ForceFieldStaticMaker."
    ),
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
    task_document_kwargs : dict (deprecated)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.CHGNet} static"
    force_field_name: str | MLFF = MLFF.CHGNet
    task_document_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(
        default_factory=lambda: _DEFAULT_CALCULATOR_KWARGS[MLFF.CHGNet]
    )


@deprecated(
    replacement=ForceFieldRelaxMaker,
    deadline=(2025, 1, 1),
    message="To use M3GNet, set `force_field_name = 'M3GNet'` in ForceFieldRelaxMaker.",
)
@dataclass
class M3GNetRelaxMaker(ForceFieldRelaxMaker):
    """
    Maker to perform a relaxation using the M3GNet ML force field.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str or .MLFF
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
        Keyword arguments that will get passed to :obj:`AseRelaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict (deprecated)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.M3GNet} relax"
    force_field_name: str | MLFF = MLFF.M3GNet
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(
        default_factory=lambda: _DEFAULT_CALCULATOR_KWARGS[MLFF.M3GNet]
    )


@deprecated(
    replacement=ForceFieldStaticMaker,
    deadline=(2025, 1, 1),
    message=(
        "To use M3GNet, set `force_field_name = 'M3GNet'` in ForceFieldStaticMaker."
    ),
)
@dataclass
class M3GNetStaticMaker(ForceFieldStaticMaker):
    """
    Maker to calculate forces and stresses using the M3GNet force field.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str or .MLFF
        The name of the force field.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict (deprecated)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.M3GNet} static"
    force_field_name: str | MLFF = MLFF.M3GNet
    task_document_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(
        default_factory=lambda: _DEFAULT_CALCULATOR_KWARGS[MLFF.M3GNet]
    )


@deprecated(
    replacement=ForceFieldRelaxMaker,
    deadline=(2025, 1, 1),
    message="To use NEP, set `force_field_name = 'NEP'` in ForceFieldRelaxMaker.",
)
@dataclass
class NEPRelaxMaker(ForceFieldRelaxMaker):
    """
    Base Maker to calculate forces and stresses using a NEP potential.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str or .MLFF
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
        Keyword arguments that will get passed to :obj:`AseRelaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict (deprecated)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.NEP} relax"
    force_field_name: str | MLFF = MLFF.NEP
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(
        default_factory=lambda: _DEFAULT_CALCULATOR_KWARGS[MLFF.NEP]
    )
    task_document_kwargs: dict = field(default_factory=dict)


@deprecated(
    replacement=ForceFieldStaticMaker,
    deadline=(2025, 1, 1),
    message="To use NEP, set `force_field_name = 'NEP'` in ForceFieldStaticMaker.",
)
@dataclass
class NEPStaticMaker(ForceFieldStaticMaker):
    """
    Base Maker to calculate forces and stresses using a NEP potential.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str or .MLFF
        The name of the force field.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict (deprecated)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.NEP} static"
    force_field_name: str | MLFF = MLFF.NEP
    task_document_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(
        default_factory=lambda: _DEFAULT_CALCULATOR_KWARGS[MLFF.NEP]
    )


@deprecated(
    replacement=ForceFieldRelaxMaker,
    deadline=(2025, 1, 1),
    message="To use Nequip, set `force_field_name = 'Nequip'` in ForceFieldRelaxMaker.",
)
@dataclass
class NequipRelaxMaker(ForceFieldRelaxMaker):
    """
    Maker to perform a relaxation using a Nequip force field.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str or .MLFF
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
        Keyword arguments that will get passed to :obj:`AseRelaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict (deprecated)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.Nequip} relax"
    force_field_name: str | MLFF = MLFF.Nequip
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)


@deprecated(
    replacement=ForceFieldStaticMaker,
    deadline=(2025, 1, 1),
    message=(
        "To use Nequip, set `force_field_name = 'Nequip'` in ForceFieldStaticMaker."
    ),
)
@dataclass
class NequipStaticMaker(ForceFieldStaticMaker):
    """
    Maker to calculate energies, forces and stresses using a nequip force field.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str or .MLFF
        The name of the force field.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict (deprecated)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.Nequip} static"
    force_field_name: str | MLFF = MLFF.Nequip
    task_document_kwargs: dict = field(default_factory=dict)


@deprecated(
    replacement=ForceFieldRelaxMaker,
    deadline=(2025, 1, 1),
    message="To use MACE, set `force_field_name = 'MACE'` in ForceFieldRelaxMaker.",
)
@dataclass
class MACERelaxMaker(ForceFieldRelaxMaker):
    """
    Maker to perform a relaxation using the MACE ML force field.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str or .MLFF
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
        Keyword arguments that will get passed to :obj:`AseRelaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator. E.g. the "model"
        key configures which checkpoint to load with mace.calculators.MACECalculator().
        Can be a URL starting with https://. If not set, loads the universal MACE-MP
        trained for Matbench Discovery on the MPtrj dataset available at
        https://figshare.com/articles/dataset/22715158.
    task_document_kwargs : dict (deprecated)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.MACE} relax"
    force_field_name: str | MLFF = MLFF.MACE
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)


@deprecated(
    replacement=ForceFieldStaticMaker,
    deadline=(2025, 1, 1),
    message="To use MACE, set `force_field_name = 'MACE'` in ForceFieldStaticMaker.",
)
@dataclass
class MACEStaticMaker(ForceFieldStaticMaker):
    """
    Maker to calculate forces and stresses using the MACE force field.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str or .MLFF
        The name of the force field.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator. E.g. the "model"
        key configures which checkpoint to load with mace.calculators.MACECalculator().
        Can be a URL starting with https://. If not set, loads the universal MACE-MP
        trained for Matbench Discovery on the MPtrj dataset available at
        https://figshare.com/articles/dataset/22715158.
    task_document_kwargs : dict (deprecated)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.MACE} static"
    force_field_name: str | MLFF = MLFF.MACE
    task_document_kwargs: dict = field(default_factory=dict)


@deprecated(
    replacement=ForceFieldRelaxMaker,
    deadline=(2025, 1, 1),
    message=(
        "To use SevenNet, set `force_field_name = 'SevenNet'` in ForceFieldRelaxMaker."
    ),
)
@dataclass
class SevenNetRelaxMaker(ForceFieldRelaxMaker):
    """
    Maker to perform a relaxation using the SevenNet ML force field.

    Published in https://pubs.acs.org/doi/10.1021/acs.jctc.4c00190.
    pip install git+https://github.com/MDIL-SNU/SevenNet

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str or .MLFF
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
        Keyword arguments that will get passed to :obj:`AseRelaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator. E.g. the "model"
        key configures which checkpoint to load with mace.calculators.MACECalculator().
        Can be a URL starting with https://. If not set, loads the universal MACE-MP
        trained for Matbench Discovery on the MPtrj dataset available at
        https://figshare.com/articles/dataset/22715158.
    task_document_kwargs : dict (deprecated)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.SevenNet} relax"
    force_field_name: str | MLFF = MLFF.SevenNet
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)


@deprecated(
    replacement=ForceFieldStaticMaker,
    deadline=(2025, 1, 1),
    message=(
        "To use SevenNet, set `force_field_name = 'SevenNet'` in ForceFieldStaticMaker."
    ),
)
@dataclass
class SevenNetStaticMaker(ForceFieldStaticMaker):
    """
    Maker to calculate forces and stresses using the SevenNet force field.

    Published in https://pubs.acs.org/doi/10.1021/acs.jctc.4c00190.
    pip install git+https://github.com/MDIL-SNU/SevenNet

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str or .MLFF
        The name of the force field.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator. E.g. the "model"
        key configures which checkpoint to load with mace.calculators.MACECalculator().
        Can be a URL starting with https://. If not set, loads the universal MACE-MP
        trained for Matbench Discovery on the MPtrj dataset available at
        https://figshare.com/articles/dataset/22715158.
    task_document_kwargs : dict (deprecated)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.SevenNet} static"
    force_field_name: str | MLFF = MLFF.SevenNet
    task_document_kwargs: dict = field(default_factory=dict)


@deprecated(
    replacement=ForceFieldRelaxMaker,
    deadline=(2025, 1, 1),
    message="To use GAP, set `force_field_name = 'GAP'` in ForceFieldRelaxMaker.",
)
@dataclass
class GAPRelaxMaker(ForceFieldRelaxMaker):
    """
    Base Maker to calculate forces and stresses using a GAP potential.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str or .MLFF
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
        Keyword arguments that will get passed to :obj:`AseRelaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`AseRelaxer()`.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict (deprecated)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.GAP} relax"
    force_field_name: str | MLFF = MLFF.GAP
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(
        default_factory=lambda: _DEFAULT_CALCULATOR_KWARGS[MLFF.GAP]
    )
    task_document_kwargs: dict = field(default_factory=dict)


@deprecated(
    replacement=ForceFieldStaticMaker,
    deadline=(2025, 1, 1),
    message="To use GAP, set `force_field_name = 'GAP'` in ForceFieldStaticMaker.",
)
@dataclass
class GAPStaticMaker(ForceFieldStaticMaker):
    """
    Base Maker to calculate forces and stresses using a GAP potential.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str or .MLFF
        The name of the force field.
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs : dict (deprecated)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.GAP} static"
    force_field_name: str | MLFF = MLFF.GAP
    task_document_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(
        default_factory=lambda: _DEFAULT_CALCULATOR_KWARGS[MLFF.GAP]
    )
