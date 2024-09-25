"""Define general ASE-calculator jobs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ase.io import Trajectory as AseTrajectory
from emmet.core.vasp.calculation import StoreTrajectoryOption
from jobflow import Maker, job
from pymatgen.core.trajectory import Trajectory as PmgTrajectory

from atomate2.ase.schemas import AseResult, AseTaskDoc
from atomate2.ase.utils import AseRelaxer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from ase.calculators.calculator import Calculator
    from pymatgen.core import Molecule, Structure

    from atomate2.ase.schemas import AseMoleculeTaskDoc, AseStructureTaskDoc

_ASE_DATA_OBJECTS = [PmgTrajectory, AseTrajectory]


@dataclass
class AseMaker(Maker):
    """
    Define basic template of ASE-based jobs.

    This class defines two functions relevant attributes
    for the ASE TaskDoc schemas, as well as two methods
    that must be implemented in subclasses:
        1. `calculator`: the ASE .Calculator object
        2. `run_ase`: which actually makes the call to ASE.

    Parameters
    ----------
    name: str
        The name of the job
    calculator_kwargs : dict
        Keyword arguments that will get passed to the ASE calculator.
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
    """

    name: str = "ASE maker"
    calculator_kwargs: dict = field(default_factory=dict)
    ionic_step_data: tuple[str, ...] | None = (
        "energy",
        "forces",
        "magmoms",
        "stress",
        "mol_or_struct",
    )
    store_trajectory: StoreTrajectoryOption = StoreTrajectoryOption.NO
    tags: list[str] | None = None

    def run_ase(
        self,
        mol_or_struct: Structure | Molecule,
        prev_dir: str | Path | None = None,
    ) -> AseResult:
        """
        Run ASE, method to be implemented in subclasses.

        This method exists to permit subclasses to redefine `make`
        for different output schemas.

        Parameters
        ----------
        mol_or_struct: .Molecule or .Structure
            pymatgen molecule or structure
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from. Unused, just
                added to match the method signature of other makers.
        """
        raise NotImplementedError

    @property
    def calculator(self) -> Calculator:
        """ASE calculator, method to be implemented in subclasses."""
        raise NotImplementedError


@dataclass
class AseRelaxMaker(AseMaker):
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
    """

    name: str = "ASE relaxation"
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float | None = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)

    @job(data=_ASE_DATA_OBJECTS)
    def make(
        self,
        mol_or_struct: Molecule | Structure,
        prev_dir: str | Path | None = None,
    ) -> AseStructureTaskDoc | AseMoleculeTaskDoc:
        """
        Relax a structure or molecule using ASE as a job.

        Parameters
        ----------
        mol_or_struct: .Molecule or .Structure
            pymatgen molecule or structure
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from. Unused, just
                added to match the method signature of other makers.

        Returns
        -------
        AseStructureTaskDoc or AseMoleculeTaskDoc
        """
        return AseTaskDoc.to_mol_or_struct_metadata_doc(
            getattr(self.calculator, "name", type(self.calculator).__name__),
            self.run_ase(mol_or_struct, prev_dir=prev_dir),
            self.steps,
            relax_kwargs=self.relax_kwargs,
            optimizer_kwargs=self.optimizer_kwargs,
            relax_cell=self.relax_cell,
            fix_symmetry=self.fix_symmetry,
            symprec=self.symprec if self.fix_symmetry else None,
            ionic_step_data=self.ionic_step_data,
            store_trajectory=self.store_trajectory,
            tags=self.tags,
        )

    def run_ase(
        self,
        mol_or_struct: Structure | Molecule,
        prev_dir: str | Path | None = None,
    ) -> AseResult:
        """
        Relax a structure or molecule using ASE, not as a job.

        Parameters
        ----------
        mol_or_struct: .Molecule or .Structure
            pymatgen molecule or structure
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from. Unused, just
                added to match the method signature of other makers.
        """
        if self.steps < 0:
            logger.warning(
                "WARNING: A negative number of steps is not possible. "
                "Behavior may vary..."
            )

        relaxer = AseRelaxer(
            self.calculator,
            relax_cell=self.relax_cell,
            fix_symmetry=self.fix_symmetry,
            symprec=self.symprec,
            **self.optimizer_kwargs,
        )
        return relaxer.relax(mol_or_struct, steps=self.steps, **self.relax_kwargs)


@dataclass
class LennardJonesRelaxMaker(AseRelaxMaker):
    """
    Relax a structure with a Lennard-Jones 6-12 potential.

    This serves mostly as an example of how to create atomate2
    jobs with existing ASE calculators, and test purposes.

    See `atomate2.ase.AseRelaxMaker` for further documentation.
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

    See `atomate2.ase.AseRelaxMaker` for further documentation.
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

    See `atomate2.ase.AseRelaxMaker` for further documentation.
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
        try:
            from tblite.ase import TBLite
        except ImportError:
            raise ImportError(
                "TBLite must be installed; please install TBLite using\n"
                "`pip install -c conda-forge tblite-python`"
            ) from None

        return TBLite(atoms=None, **self.calculator_kwargs)


@dataclass
class GFNxTBStaticMaker(GFNxTBRelaxMaker):
    """
    Single-point GFNn-xTB calculation.

    See `atomate2.ase.{AseRelaxMaker, GFNxTBRelaxMaker}` for further documentation.
    """

    name: str = "GFN-xTB static"
    steps: int = 1
