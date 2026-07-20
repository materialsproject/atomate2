"""Quasi-harmonic approximation (QHA) calculations for SIESTA.

This module implements workflows for calculating temperature-dependent
thermodynamic properties using the quasi-harmonic approximation, which
combines equation of state (EOS) calculations with phonon frequencies
at different volumes.

The QHA enables calculation of:
- Thermal expansion coefficient α(T)
- Heat capacity at constant volume Cv(T)
- Heat capacity at constant pressure Cp(T)
- Entropy S(T)
- Gibbs free energy G(T,P)
- Bulk modulus as a function of temperature B(T)
"""  # noqa: RUF002

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from atomate2.common.flows.qha import CommonQhaMaker
from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.flows.eos import SiestaEosFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.jobs.phonon.phonopy import PhonopyMaker

if TYPE_CHECKING:
    from jobflow import Flow
    from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)

__all__ = ["SiestaQhaFlowMaker"]


@dataclass
class SiestaQhaFlowMaker(BaseSiestaFlowMaker, CommonQhaMaker):
    """
    Maker to calculate quasi-harmonic properties using SIESTA.

    This workflow combines equation of state (EOS) and phonon calculations
    to obtain temperature-dependent thermodynamic properties within the
    quasi-harmonic approximation (QHA).

    The workflow:
    1. Performs an EOS fit to get equilibrium volume and bulk modulus
    2. Calculates phonons at multiple volumes around equilibrium
    3. Constructs the Gibbs free energy surface G(V,T)
    4. Extracts temperature-dependent properties

    Parameters
    ----------
    name : str
        Name of the QHA workflow. Defaults to "siesta qha".
    structure_optimizer : RelaxMaker or None
        Maker to optimize the structure before calculations.
        If None, no optimization is performed.
    eos_maker : SiestaEosFlowMaker or None
        Maker for equation of state calculations. If None, uses default
        SiestaEosFlowMaker with appropriate settings.
    phonon_maker : PhonopyMaker
        Maker for phonon calculations at different volumes.
    number_of_frames : int
        Number of volumes at which to calculate phonons. Default is 5.
        Must be at least 3 for meaningful fitting.
    ignore_imaginary_modes : bool
        Whether to ignore imaginary phonon modes. Default is False.
        Set to True for metals or systems with soft modes.
    eos_type : str
        Type of EOS to fit. Options: "vinet", "birch_murnaghan", "murnaghan".
        Default is "vinet".
    pressure : float or list[float]
        Pressure(s) in GPa at which to evaluate properties. Default is 0.0.
    temperature : float or list[float]
        Temperature(s) in K at which to evaluate properties.
        Default is [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000].
    skip_analysis : bool
        Whether to skip the QHA analysis step. Default is False.
    volume_factor : float
        Factor to determine volume range for phonon calculations.
        Volumes will span [V₀/volume_factor, V₀*volume_factor].
        Default is 0.95 (±5% volume range).

    Examples
    --------
    Basic QHA calculation for silicon:

    >>> from pymatgen.core import Structure
    >>> from atomate2.siesta.flows.phonon import SiestaQhaFlowMaker
    >>> from jobflow import run_locally
    >>>
    >>> si = Structure(
    ...     lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
    ...     species=["Si", "Si"],
    ...     coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    ... )
    >>>
    >>> maker = SiestaQhaFlowMaker()
    >>> flow = maker.make(si)
    >>> run_locally(flow)

    Custom temperature range and more phonon calculations:

    >>> maker = SiestaQhaFlowMaker(
    ...     number_of_frames=7,  # More volume points
    ...     temperature=[300, 600, 900, 1200],  # Specific temperatures
    ...     phonon_maker=PhonopyMaker(supercell_matrix=[3, 3, 3], mesh_density=100.0),
    ... )

    For metals with soft modes:

    >>> maker = SiestaQhaFlowMaker(
    ...     ignore_imaginary_modes=True, eos_type="birch_murnaghan"
    ... )
    """

    name: str = "siesta qha"
    structure_optimizer: RelaxMaker | None = field(default_factory=RelaxMaker)
    eos_maker: SiestaEosFlowMaker | None = None
    phonon_maker: PhonopyMaker = field(default_factory=PhonopyMaker)
    number_of_frames: int = 5
    ignore_imaginary_modes: bool = False
    eos_type: Literal["vinet", "birch_murnaghan", "murnaghan"] = "vinet"
    pressure: float | list[float] = 0.0
    temperature: float | list[float] = field(
        default_factory=lambda: [
            0.0,
            100.0,
            200.0,
            300.0,
            400.0,
            500.0,
            600.0,
            700.0,
            800.0,
            900.0,
            1000.0,
        ]
    )
    skip_analysis: bool = False
    volume_factor: float = 0.95

    # Note: dry_run, use_custodian, and tier support inherited from BaseSiestaFlowMaker

    def __post_init__(self) -> None:
        """Initialize the QHA maker with appropriate EOS settings.

        BaseSiestaFlowMaker's __post_init__ handles dry_run/custodian/tier
        propagation.
        """
        # Map structure_optimizer to parent class parameters
        # CommonQhaMaker expects initial_relax_maker and eos_relax_maker
        self.initial_relax_maker = self.structure_optimizer

        # Create eos_relax_maker with same user_params as structure_optimizer
        # to preserve k-points and other settings
        if self.structure_optimizer is not None and hasattr(
            self.structure_optimizer, "input_set_generator"
        ):
            user_params = getattr(
                self.structure_optimizer.input_set_generator, "user_params", {}
            )
            self.eos_relax_maker = RelaxMaker.fixed_cell_relaxation(
                user_params=user_params
            )
        else:
            self.eos_relax_maker = RelaxMaker.fixed_cell_relaxation()

        # Create default EOS maker if not provided
        if self.eos_maker is None:
            self.eos_maker = SiestaEosFlowMaker(
                name="EOS for QHA",
                initial_relax_maker=self.structure_optimizer,
                eos_relax_maker=self.eos_relax_maker,
                number_of_frames=9,  # Use more frames for initial EOS
            )
            logger.info(
                f"Created default EOS maker (eos_type will be used during "
                f"fitting: {self.eos_type})"
            )

        # Validate number of frames
        if self.number_of_frames < 3:
            raise ValueError(
                f"number_of_frames must be at least 3, got {self.number_of_frames}"
            )

        # Ensure temperature and pressure are lists
        if not isinstance(self.temperature, list):
            self.temperature = [self.temperature]
        if not isinstance(self.pressure, list):
            self.pressure = [self.pressure]

        # Add compatibility attributes for CommonQhaMaker.__post_init__
        # PhonopyMaker uses different attribute names than VASP's BasePhononMaker
        if not hasattr(self.phonon_maker, "bulk_relax_maker"):
            self.phonon_maker.bulk_relax_maker = getattr(
                self.phonon_maker, "relax_maker", None
            )
        if not hasattr(self.phonon_maker, "static_energy_maker"):
            self.phonon_maker.static_energy_maker = getattr(
                self.phonon_maker, "static_maker", None
            )

        # Call parent's __post_init__ to run validation checks AND propagate
        # dry_run/custodian/tier. This will propagate to all child makers
        # (structure_optimizer, eos_maker, phonon_maker, etc.)
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

    def make(self, structure: Structure) -> Flow:
        """
        Create the quasi-harmonic approximation workflow.

        This method creates a workflow that:
        1. Performs structure optimization (optional)
        2. Calculates equation of state to find equilibrium volume
        3. Generates volumes for phonon calculations
        4. Calculates phonons at each volume
        5. Performs QHA analysis to extract T-dependent properties

        Parameters
        ----------
        structure : Structure
            Input structure for QHA calculations.

        Returns
        -------
        Flow
            A flow containing the complete QHA workflow.

        Notes
        -----
        The quasi-harmonic approximation assumes that phonons are harmonic
        at each volume but allows the frequencies to change with volume.
        This captures thermal expansion effects while maintaining computational
        efficiency compared to fully anharmonic methods.

        The resulting properties include:
        - Thermal expansion coefficient α(T)
        - Heat capacities Cv(T) and Cp(T)
        - Gibbs free energy G(T,P)
        - Entropy S(T)
        - Bulk modulus B(T)
        """  # noqa: RUF002
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        # Log workflow details
        logger.info(f"Starting QHA calculation for {structure.composition}")
        # At this point, temperature is guaranteed to be a list due to __post_init__
        temp_list = (
            self.temperature
            if isinstance(self.temperature, list)
            else [self.temperature]
        )
        logger.info(
            f"QHA settings:\n"
            f"  Number of volumes: {self.number_of_frames}\n"
            f"  Temperature range: {min(temp_list)}-{max(temp_list)} K\n"
            f"  Pressure: {self.pressure} GPa\n"
            f"  EOS type: {self.eos_type}\n"
            f"  Volume range: ±{(1 - self.volume_factor) * 100:.1f}%"
        )

        # Check for magnetic systems
        if (
            hasattr(structure, "site_properties")
            and "magmom" in structure.site_properties
        ):
            logger.warning(
                "Magnetic moments detected. QHA may not be accurate for "
                "magnetic phase transitions."
            )

        # Extract supercell_matrix from phonon_maker if user specified it
        # This allows the user to control supercell size via PhonopyMaker
        supercell_matrix = None
        if (
            hasattr(self.phonon_maker, "supercell_matrix")
            and self.phonon_maker.supercell_matrix is not None
        ):
            supercell_matrix = self.phonon_maker.supercell_matrix
            logger.info(f"Using user-specified supercell matrix: {supercell_matrix}")
        # If not specified, CommonQhaMaker will auto-generate based on min_length
        elif hasattr(self, "min_length") and self.min_length is not None:
            logger.info(
                f"Auto-generating supercell based on min_length={self.min_length} Å"
            )
        else:
            logger.warning(
                "No supercell_matrix or min_length specified, "
                "using CommonQhaMaker defaults"
            )

        # Call parent implementation with supercell_matrix if available
        return super().make(structure=structure, supercell_matrix=supercell_matrix)

    def _validate_structure(self, structure: Structure) -> None:
        """
        Validate the input structure for QHA calculations.

        Parameters
        ----------
        structure : Structure
            Structure to validate.

        Raises
        ------
        ValueError
            If the structure is not suitable for QHA.
        """
        # Check for very large structures
        if len(structure) > 100:
            logger.warning(
                f"Large structure with {len(structure)} atoms. "
                "Phonon calculations may be expensive."
            )

        # Check for low-dimensional systems
        lattice = structure.lattice
        if min(lattice.abc) / max(lattice.abc) < 0.1:
            logger.warning(
                "Structure appears to be low-dimensional. "
                "QHA may not be appropriate for 2D/1D systems."
            )

    def _get_volume_range(self, equilibrium_volume: float) -> list[float]:
        """
        Generate volumes for phonon calculations.

        Parameters
        ----------
        equilibrium_volume : float
            Equilibrium volume from EOS fit in Å³.

        Returns
        -------
        list[float]
            List of volumes for phonon calculations.
        """
        import numpy as np

        # Generate volumes around equilibrium
        v_min = equilibrium_volume * self.volume_factor
        v_max = equilibrium_volume / self.volume_factor

        volumes = np.linspace(v_min, v_max, self.number_of_frames)

        logger.info(
            f"Generated {self.number_of_frames} volumes: {v_min:.2f} to {v_max:.2f} Å³"
        )

        return volumes.tolist()
