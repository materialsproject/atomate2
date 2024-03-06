"""Job to relax a structure using a force field (aka an interatomic potential)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Maker, job

from atomate2.forcefields import MLFF
from atomate2.forcefields.schemas import ForceFieldTaskDocument
from atomate2.forcefields.utils import Relaxer, revert_default_dtype

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)


@dataclass
class ForceFieldRelaxMaker(Maker):
    """
    Base Maker to calculate forces and stresses using any force field.

    Should be subclassed to use a specific force field.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str
        The name of the force field.
    relax_cell : bool = True
        Whether to allow the cell shape/volume to change during relaxation.
    steps : int
        Maximum number of ionic steps allowed during relaxation.
    relax_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer()`.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = "Force field relax"
    force_field_name: str = "Force field"
    relax_cell: bool = True
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    @job(output_schema=ForceFieldTaskDocument)
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

        result = self._relax(structure)

        return ForceFieldTaskDocument.from_ase_compatible_result(
            self.force_field_name,
            result,
            self.relax_cell,
            self.steps,
            self.relax_kwargs,
            self.optimizer_kwargs,
            **self.task_document_kwargs,
        )

    def _relax(self, structure: Structure) -> dict:
        raise NotImplementedError


@dataclass
class ForceFieldStaticMaker(ForceFieldRelaxMaker):
    """
    Maker to calculate forces and stresses using the CHGNet force field.

    Parameters
    ----------
    name : str
        The job name.
    force_field_name : str
        The name of the force field.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = "Force field static"
    force_field_name: str = "Force field"
    task_document_kwargs: dict = field(default_factory=dict)

    @job(output_schema=ForceFieldTaskDocument)
    def make(
        self, structure: Structure, prev_dir: str | Path | None = None
    ) -> ForceFieldTaskDocument:
        """
        Perform a static evaluation using a force field.

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

        result = self._evaluate_static(structure)

        return ForceFieldTaskDocument.from_ase_compatible_result(
            self.force_field_name,
            result,
            relax_cell=False,
            steps=1,
            relax_kwargs=None,
            optimizer_kwargs=None,
            **self.task_document_kwargs,
        )

    def _evaluate_static(self, structure: Structure) -> dict:
        raise NotImplementedError


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
    steps : int
        Maximum number of ionic steps allowed during relaxation.
    relax_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer()`.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.CHGNet} relax"
    force_field_name = f"{MLFF.CHGNet}"
    relax_cell: bool = True
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    def _relax(self, structure: Structure) -> dict:
        from chgnet.model import StructOptimizer

        relaxer = StructOptimizer(**self.optimizer_kwargs)
        return relaxer.relax(
            structure, relax_cell=self.relax_cell, steps=self.steps, **self.relax_kwargs
        )


@dataclass
class CHGNetStaticMaker(ForceFieldStaticMaker):
    """
    Maker to calculate forces and stresses using the CHGNet force field.

    Parameters
    ----------
    name : str
        The job name.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.CHGNet} static"
    force_field_name = f"{MLFF.CHGNet}"
    task_document_kwargs: dict = field(default_factory=dict)

    def _evaluate_static(self, structure: Structure) -> dict:
        from chgnet.model import StructOptimizer

        relaxer = StructOptimizer()
        return relaxer.relax(structure, steps=1)


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
    steps : int
        Maximum number of ionic steps allowed during relaxation.
    relax_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer()`.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.M3GNet} relax"
    force_field_name: str = f"{MLFF.M3GNet}"
    relax_cell: bool = True
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    def _relax(self, structure: Structure) -> dict:
        import matgl
        from matgl.ext.ase import Relaxer

        # Note: the below code was taken from the matgl repo examples.
        # Load pre-trained M3GNet model (currently uses the MP-2021.2.8 database)
        potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")

        relaxer = Relaxer(
            potential=potential, relax_cell=self.relax_cell, **self.optimizer_kwargs
        )

        return relaxer.relax(structure, steps=self.steps, **self.relax_kwargs)


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
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    """

    name: str = f"{MLFF.M3GNet} static"
    force_field_name: str = f"{MLFF.M3GNet}"
    task_document_kwargs: dict = field(default_factory=dict)

    def _evaluate_static(self, structure: Structure) -> dict:
        import matgl
        from matgl.ext.ase import Relaxer

        # Note: the below code was taken from the matgl repo examples.
        # Load pre-trained M3GNet model (currently uses the MP-2021.2.8 database)
        potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")

        relaxer = Relaxer(potential=potential, relax_cell=False)

        return relaxer.relax(structure, steps=1)


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
    steps : int
        Maximum number of ionic steps allowed during relaxation.
    relax_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer()`.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    model: str | Path | None
        Checkpoint to load with :obj:`mace.calculators.MACECalculator()'`. Can be a URL
        starting with https://. If None, loads the universal MACE trained for Matbench
        Discovery on the MPtrj dataset available at
        https://figshare.com/articles/dataset/22715158.
    model_kwargs: dict[str, Any]
        Further keywords (e.g. device, default_dtype, model) for
            :obj:`mace.calculators.MACECalculator()'`.
    """

    name: str = f"{MLFF.MACE} relax"
    force_field_name: str = f"{MLFF.MACE}"
    relax_cell: bool = True
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
    model: str | Path | Sequence[str | Path] | None = None
    model_kwargs: dict = field(default_factory=dict)

    def _relax(self, structure: Structure) -> dict:
        from mace.calculators import mace_mp

        with revert_default_dtype():
            calculator = mace_mp(model=self.model, **self.model_kwargs)
            relaxer = Relaxer(
                calculator, relax_cell=self.relax_cell, **self.optimizer_kwargs
            )
            return relaxer.relax(structure, steps=self.steps, **self.relax_kwargs)


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
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    model: str | Path | None
        Checkpoint to load with :obj:`mace.calculators.MACECalculator()'`. Can be a URL
        starting with https://. If None, loads the universal MACE trained for Matbench
        Discovery on the MPtrj dataset available at
        https://figshare.com/articles/dataset/22715158.
    model_kwargs: dict[str, Any]
        Further keywords (e.g. device, default_dtype, model) for
            :obj:`mace.calculators.MACECalculator()'`.
    """

    name: str = f"{MLFF.MACE} static"
    force_field_name: str = f"{MLFF.MACE}"
    task_document_kwargs: dict = field(default_factory=dict)
    model: str | Path | Sequence[str | Path] | None = None
    model_kwargs: dict = field(default_factory=dict)

    def _evaluate_static(self, structure: Structure) -> dict:
        from mace.calculators import mace_mp

        with revert_default_dtype():
            calculator = mace_mp(model=self.model, **self.model_kwargs)
            relaxer = Relaxer(calculator, relax_cell=False)
            return relaxer.relax(structure, steps=1)


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
    steps : int
        Maximum number of ionic steps allowed during relaxation.
    relax_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer.relax`.
    optimizer_kwargs : dict
        Keyword arguments that will get passed to :obj:`Relaxer()`.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    potential_args_str: str
        args_str for :obj:`quippy.potential.Potential()'`.
    potential_param_file_name: str | Path
        param_file_name for :obj:`quippy.potential.Potential()'`.
    potential_kwargs: dict
        Further keywords for :obj:`quippy.potential.Potential()'`.
    """

    name: str = f"{MLFF.GAP} relax"
    force_field_name: str = f"{MLFF.GAP}"
    relax_cell: bool = True
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
    potential_args_str: str | Path = "IP GAP"
    potential_param_file_name: str = "gap.xml"
    potential_kwargs: dict = field(default_factory=dict)

    def _relax(self, structure: Structure) -> dict:
        from quippy.potential import Potential

        calculator = Potential(
            args_str=self.potential_args_str,
            param_filename=str(self.potential_param_file_name),
            **self.potential_kwargs,
        )
        relaxer = Relaxer(
            calculator, **self.optimizer_kwargs, relax_cell=self.relax_cell
        )
        return relaxer.relax(structure, steps=self.steps, **self.relax_kwargs)


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
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()`.
    potential_args_str: str
        args_str for :obj:`quippy.potential.Potential()'`.
    potential_param_file_name: str | Path
        param_file_name for :obj:`quippy.potential.Potential()'`.
    potential_kwargs: dict
        Further keywords for :obj:`quippy.potential.Potential()'`.
    """

    name: str = f"{MLFF.GAP} static"
    force_field_name: str = f"{MLFF.GAP}"
    task_document_kwargs: dict = field(default_factory=dict)
    potential_args_str: str = "IP GAP"
    potential_param_file_name: str | Path = "gap.xml"
    potential_kwargs: dict = field(default_factory=dict)

    def _evaluate_static(self, structure: Structure) -> dict:
        from quippy.potential import Potential

        calculator = Potential(
            args_str=self.potential_args_str,
            param_filename=str(self.potential_param_file_name),
            **self.potential_kwargs,
        )
        relaxer = Relaxer(calculator, relax_cell=False)
        return relaxer.relax(structure, steps=1)
