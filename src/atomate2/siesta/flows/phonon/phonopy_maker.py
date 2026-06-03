"""Phonon calculation workflows for SIESTA."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.jobs.phonon.phonopy import PhonopyMaker
from atomate2.siesta.sets.core import StaticSetGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


@dataclass
class SiestaPhononFlowMaker(PhonopyMaker):
    """
    SIESTA-specific phonon calculation workflow using phonopy.

    This is a convenience class that sets up sensible defaults for SIESTA
    phonon calculations. It uses:
    - Variable-cell relaxation for initial structure optimization
    - High-quality static calculations for forces
    - Automatic supercell generation

    Parameters
    ----------
    name : str
        Name of the workflow
    min_length : float
        Minimum supercell length in Angstroms (default: 6.0)
    displacement : float
        Atomic displacement in Angstroms (default: 0.01)
    relax_maker : Maker | None
        Maker for structure relaxation. Default uses variable-cell relaxation
        with tight convergence.
    static_maker : Maker
        Maker for force calculations. Default uses DZP basis and 300 Ry cutoff.
    kpts : list[int] | None
        K-point mesh for force calculations. If None, uses automatic generation.
    mesh : tuple[int, int, int]
        Q-point mesh for phonon DOS (default: (50, 50, 50))

    Examples
    --------
    >>> from atomate2.siesta.flows.phonon import SiestaPhononFlowMaker
    >>> from pymatgen.core import Structure
    >>> structure = Structure.from_file("POSCAR")
    >>> maker = SiestaPhononFlowMaker(min_length=6.0)
    >>> flow = maker.make(structure)
    """

    name: str = "siesta phonopy"
    min_length: float = 6.0  # 12.0
    displacement: float = 0.01
    relax_maker: Maker | None = field(
        default_factory=lambda: RelaxMaker.variable_cell_relaxation(
            user_params={
                "MD.MaxForceTol": "0.01 eV/Ang",
                "MD.MaxStressTol": "0.1 GPa",
            }
        )
    )
    static_maker: Maker | None = None
    kpts: list[int] | None = None
    mesh: tuple[int, int, int] = (50, 50, 50)

    def __post_init__(self):
        """Initialize static maker with proper settings and apply kpts to both makers."""
        from collections import OrderedDict

        # Create default static maker if not provided
        if self.static_maker is None:
            user_params = OrderedDict(
                {
                    "PAO.BasisSize": "DZP",
                    "Mesh.Cutoff": "300 Ry",  # Use "Mesh.Cutoff" with dot
                }
            )

            # Add k-points to user_params if provided
            if self.kpts is not None:
                user_params["a2s_kpts"] = self.kpts

            self.static_maker = StaticMaker(
                input_set_generator=StaticSetGenerator(
                    user_params=user_params,
                )
            )
        elif self.kpts is not None:
            # Update existing static maker's kpts in user_params
            if hasattr(self.static_maker, "input_set_generator"):
                if self.static_maker.input_set_generator.user_params is None:
                    self.static_maker.input_set_generator.user_params = OrderedDict()
                self.static_maker.input_set_generator.user_params[
                    "a2s_kpts"
                ] = self.kpts

        # Also apply k-points to relax_maker if provided
        if self.kpts is not None and self.relax_maker is not None:
            if hasattr(self.relax_maker, "input_set_generator"):
                if self.relax_maker.input_set_generator.user_params is None:
                    self.relax_maker.input_set_generator.user_params = OrderedDict()
                self.relax_maker.input_set_generator.user_params["a2s_kpts"] = self.kpts

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
        supercell_matrix: list[list[int]] | None = None,
    ) -> Flow:
        """
        Create phonon calculation workflow.

        Parameters
        ----------
        structure : Structure
            Input structure (should be relaxed or will be relaxed if relax_maker provided)
        prev_dir : str | Path | None
            Previous directory for reusing files
        supercell_matrix : list[list[int]] | None
            Override the supercell matrix for this specific calculation

        Returns
        -------
        Flow
            Phonon calculation workflow
        """
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        # Call parent implementation
        return super().make(
            structure, prev_dir=prev_dir, supercell_matrix=supercell_matrix
        )


@dataclass
class PhononConvergenceFlowMaker(BaseSiestaFlowMaker):
    """
    Test phonon convergence with respect to supercell size and displacement.

    This workflow runs multiple phonon calculations with different parameters
    to determine optimal settings for converged phonon properties.

    Parameters
    ----------
    name : str
        Name of the workflow
    supercell_sizes : list[list[list[int]]]
        List of supercell matrices to test
    displacement_values : list[float]
        List of displacement values to test
    base_phonon_maker : PhonopyMaker
        Base phonon maker to use (settings will be updated for each test)

    Examples
    --------
    >>> from atomate2.siesta.flows.phonon import PhononConvergenceFlowMaker
    >>> maker = PhononConvergenceFlowMaker(
    ...     supercell_sizes=[
    ...         [[2,0,0],[0,2,0],[0,0,2]],
    ...         [[3,0,0],[0,3,0],[0,0,3]],
    ...     ],
    ...     displacement_values=[0.005, 0.01, 0.02]
    ... )
    >>> flow = maker.make(structure)
    """

    name: str = "phonon convergence"
    supercell_sizes: list[list[list[int]]] = field(
        default_factory=lambda: [
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        ]
    )
    displacement_values: list[float] = field(default_factory=lambda: [0.01])
    base_phonon_maker: PhonopyMaker = field(default_factory=SiestaPhononFlowMaker)

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """
        Create phonon convergence testing workflow.

        Parameters
        ----------
        structure : Structure
            Input structure
        prev_dir : str | Path | None
            Previous directory

        Returns
        -------
        Flow
            Convergence testing workflow
        """
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        jobs = []

        for i, supercell_matrix in enumerate(self.supercell_sizes):
            for j, displacement in enumerate(self.displacement_values):
                # Create a phonon maker with specific parameters
                phonon_maker = PhonopyMaker(
                    name=f"{self.name}_sc{i + 1}_disp{j + 1}",
                    supercell_matrix=supercell_matrix,
                    displacement=displacement,
                    relax_maker=self.base_phonon_maker.relax_maker,
                    static_maker=self.base_phonon_maker.static_maker,
                    use_symmetry=self.base_phonon_maker.use_symmetry,
                    create_thermal_properties=False,  # Skip thermal for convergence
                )

                # Create flow
                phonon_flow = phonon_maker.make(structure, prev_dir=prev_dir)
                phonon_flow.name = (
                    f"{self.name}_supercell_{supercell_matrix[0][0]}x"
                    f"{supercell_matrix[1][1]}x{supercell_matrix[2][2]}_"
                    f"displacement_{displacement}"
                )
                jobs.append(phonon_flow)

        # Future enhancement: Add convergence analysis job to automatically
        # compare frequencies and determine optimal parameters
        return Flow(jobs, name=self.name)
