"""Grüneisen parameter calculations for SIESTA.

This module implements workflows to calculate mode-dependent Grüneisen parameters,
which characterize the effect of volume changes on phonon frequencies and are
essential for understanding thermal expansion and anharmonicity.

The Grüneisen parameter for mode i is defined as:
    γᵢ = -V/ωᵢ · ∂ωᵢ/∂V

where V is the volume and ωᵢ is the frequency of phonon mode i.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.common.flows.gruneisen import BaseGruneisenMaker
from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.flows.phonon.phonopy_maker import PhonopyMaker
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker

if TYPE_CHECKING:
    from jobflow import Flow
    from pymatgen.core.structure import Structure


logger = logging.getLogger(__name__)

__all__ = ["SiestaGruneisenFlowMaker"]


@dataclass
class SiestaGruneisenFlowMaker(BaseSiestaFlowMaker, BaseGruneisenMaker):
    """
    Maker to calculate Grüneisen parameters using SIESTA.

    This workflow calculates mode-dependent Grüneisen parameters by computing
    phonon frequencies at three different volumes (V₀, V₀(1+δ), V₀(1-δ)) and
    using finite differences to approximate the derivatives.

    Parameters
    ----------
    name : str
        Name of the Grüneisen workflow. Defaults to "siesta gruneisen".
    structure_optimizer : RelaxMaker or None
        Maker to optimize the structure before phonon calculations.
        If None, no optimization is performed.
    phonon_maker : PhononMaker
        Maker to use for phonon frequency calculations at different volumes.
    perc_vol : float
        Percentage volume change for finite difference calculations.
        Default is 0.01 (1% volume change).
    use_symmetry : bool
        Whether to use symmetry in phonon calculations. Default is True.
    symprec : float
        Symmetry precision for spglib. Default is 1e-4.
    compute_gruneisen_param_kwargs : dict
        Additional keyword arguments passed to the Grüneisen parameter
        calculation in phonopy.
    generate_frequencies_eigenvectors_kwargs : dict
        Additional keyword arguments for frequency/eigenvector generation.

    Examples
    --------
    Calculate Grüneisen parameters for silicon:

    >>> from pymatgen.core import Structure
    >>> from atomate2.siesta.flows.phonon import SiestaGruneisenFlowMaker
    >>> from jobflow import run_locally
    >>>
    >>> si = Structure(
    ...     lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
    ...     species=["Si", "Si"],
    ...     coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    ... )
    >>>
    >>> maker = SiestaGruneisenFlowMaker()
    >>> flow = maker.make(si)
    >>> run_locally(flow)

    Using custom volume changes and phonon settings:

    >>> maker = SiestaGruneisenFlowMaker(
    ...     perc_vol=0.02,  # 2% volume change
    ...     phonon_maker=PhononMaker(supercell_matrix=[2, 2, 2], mesh_density=100.0),
    ... )
    """

    name: str = "siesta gruneisen"
    structure_optimizer: RelaxMaker | None = field(default_factory=RelaxMaker)
    phonon_maker: PhonopyMaker = field(default_factory=PhonopyMaker)
    perc_vol: float = 0.01
    use_symmetry: bool = True
    symprec: float = 1e-4
    compute_gruneisen_param_kwargs: dict = field(default_factory=dict)
    generate_frequencies_eigenvectors_kwargs: dict = field(default_factory=dict)

    # Note: dry_run, use_custodian, and tier support inherited from BaseSiestaFlowMaker

    def __post_init__(self) -> None:
        """Propagate dry_run/custodian/tier via the parent ``__post_init__``."""
        super().__post_init__()

    @property
    def prev_calc_dir_argname(self) -> str:
        """Name of argument to pass previous calculation directory.

        Returns
        -------
        str
            The argument name for previous calculation directory in SIESTA,
            which is "prev_dir".
        """
        return "prev_dir"

    def make(self, structure: Structure, prev_dir: str | None = None) -> Flow:
        """
        Create the Grüneisen parameter calculation workflow.

        This method creates a workflow that:
        1. Optionally optimizes the input structure
        2. Creates three structures with volumes V₀, V₀(1+δ), V₀(1-δ)
        3. Calculates phonons for each volume
        4. Computes Grüneisen parameters from frequency changes

        Parameters
        ----------
        structure : Structure
            Input structure for which to calculate Grüneisen parameters.
        prev_dir : str, optional
            Previous calculation directory for restarting.

        Returns
        -------
        Flow
            A flow containing the complete Grüneisen parameter workflow.

        Notes
        -----
        The Grüneisen parameter characterizes the change in phonon frequency
        with volume and is crucial for:
        - Thermal expansion coefficient calculations
        - Understanding anharmonicity
        - Predicting temperature-dependent properties
        """
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        # Log workflow initiation
        logger.info(
            f"Starting Grüneisen parameter calculation for {structure.composition}"
        )
        logger.info(f"Volume change percentage: {self.perc_vol * 100:.1f}%")

        # Map SIESTA-specific attributes to base class attributes
        self.bulk_relax_maker = self.structure_optimizer
        # For constant volume relaxation, use StaticMaker (no relaxation)
        self.const_vol_relax_maker = self._get_static_maker()
        self.code = "siesta"

        # Call parent implementation
        return super().make(structure=structure, prev_dir=prev_dir)

    def _get_static_maker(self) -> StaticMaker:
        """
        Get a static calculation maker for SIESTA.

        This is used internally for constant-volume calculations on the
        expanded/contracted structures. Inherits settings from the phonon maker's
        static_maker if available.

        Returns
        -------
        StaticMaker
            A SIESTA static calculation maker.
        """
        # Try to inherit settings from phonon_maker's static_maker
        if (
            hasattr(self.phonon_maker, "static_maker")
            and self.phonon_maker.static_maker is not None
        ):
            # Copy the input_set_generator to preserve k-points and other settings
            return StaticMaker(
                input_set_generator=self.phonon_maker.static_maker.input_set_generator
            )
        return StaticMaker()

    def _log_calculation_details(self, structure: Structure) -> None:
        """
        Log details about the Grüneisen calculation.

        Parameters
        ----------
        structure : Structure
            The structure being calculated.
        """
        logger.info(
            f"Grüneisen calculation details:\n"
            f"  Structure: {structure.composition.reduced_formula}\n"
            f"  Space group: {structure.get_space_group_info()[0]}\n"
            f"  Volume: {structure.volume:.2f} Å³\n"
            f"  Volume changes: ±{self.perc_vol * 100:.1f}%\n"
            f"  Supercell: {getattr(self.phonon_maker, 'supercell_matrix', 'auto')}\n"
            f"  Symmetry: {self.use_symmetry}"
        )
