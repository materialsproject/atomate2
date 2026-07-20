"""Base class for defect finite-size correction schemes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pymatgen.core import Structure


class CorrectionResult(BaseModel):
    """
    Result from a finite-size correction calculation.

    Contains the correction energy and all metadata needed for analysis
    and validation.
    """

    correction_energy: float = Field(description="Finite-size correction energy in eV")

    scheme_name: str = Field(description="Name of the correction scheme used")

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional correction-specific metadata",
    )

    alignment_energy: float | None = Field(
        None,
        description="Potential alignment correction (eV), if applicable",
    )

    alignment_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata for potential alignment",
    )

    charge_model: str = Field(description="Charge model used (point, gaussian, etc.)")

    converged: bool = Field(
        True,  # noqa: FBT003
        description="Whether the correction calculation converged",
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="Any warnings generated during correction",
    )


class CorrectionScheme(ABC):
    """
    Abstract base class for defect finite-size correction schemes.

    All correction schemes must implement the calculate_correction method
    and provide metadata about the scheme.

    Attributes
    ----------
    name : str
        Name of the correction scheme
    charge_model : str
        Type of charge model (point, gaussian, etc.)
    requires_dielectric : bool
        Whether the scheme requires dielectric constant
    supports_anisotropic : bool
        Whether the scheme supports anisotropic dielectrics
    """

    def __init__(
        self,
        epsilon_static: float | None = None,
        epsilon_ionic: float | None = None,
        epsilon_tensor: list[list[float]] | None = None,
    ) -> None:
        """
        Initialize correction scheme.

        Parameters
        ----------
        epsilon_static : float, optional
            Static dielectric constant (scalar)
        epsilon_ionic : float, optional
            Ionic contribution to dielectric constant
        epsilon_tensor : list[list[float]], optional
            Full dielectric tensor (3x3) for anisotropic corrections
        """
        self.epsilon_static = epsilon_static
        self.epsilon_ionic = epsilon_ionic
        self.epsilon_tensor = epsilon_tensor

        # Validate required parameters
        if self.requires_dielectric and epsilon_static is None:
            raise ValueError(f"{self.name} requires epsilon_static to be provided")

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the correction scheme."""

    @property
    @abstractmethod
    def charge_model(self) -> str:
        """Type of charge model used."""

    @property
    def requires_dielectric(self) -> bool:
        """Whether the scheme requires dielectric constant."""
        return True

    @property
    def supports_anisotropic(self) -> bool:
        """Whether the scheme supports anisotropic dielectrics."""
        return False

    @property
    def requires_potential_data(self) -> bool:
        """Whether the scheme requires electrostatic potential data."""
        return False

    @abstractmethod
    def calculate_correction(
        self,
        defect_structure: Structure,
        host_structure: Structure,
        charge_state: int,
        defect_energy: float,
        host_energy: float,
        defect_site: list[float] | None = None,
        **kwargs,
    ) -> CorrectionResult:
        """
        Calculate the finite-size correction.

        Parameters
        ----------
        defect_structure : Structure
            Relaxed defect supercell structure
        host_structure : Structure
            Pristine host supercell structure
        charge_state : int
            Charge state of the defect (e.g., +2, 0, -1)
        defect_energy : float
            Total energy of defect supercell (eV)
        host_energy : float
            Total energy of host supercell (eV)
        defect_site : list[float], optional
            Fractional coordinates [x, y, z] of defect site
        **kwargs
            Additional scheme-specific parameters

        Returns
        -------
        CorrectionResult
            Correction result with energy and metadata
        """

    def validate_inputs(
        self,
        defect_structure: Structure,
        host_structure: Structure,
        charge_state: int,
    ) -> None:
        """
        Validate input structures and parameters.

        Parameters
        ----------
        defect_structure : Structure
            Defect structure to validate
        host_structure : Structure
            Host structure to validate
        charge_state : int
            Charge state to validate

        Raises
        ------
        ValueError
            If inputs are invalid
        """
        # Check that structures have same lattice (approximately)
        # Compare lattice volumes (should be identical for same supercell)
        vol_diff = abs(defect_structure.lattice.volume - host_structure.lattice.volume)
        if vol_diff / host_structure.lattice.volume > 0.01:  # 1% tolerance
            raise ValueError(
                f"Defect and host structures must have similar lattices. "
                f"Volume difference: {vol_diff:.2f} Å³"
            )

        # Check charge state is non-zero (no correction needed for neutral)
        if charge_state == 0:
            raise ValueError("No finite-size correction needed for neutral defects")

        # Scheme-specific validation can be added in subclasses

    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata about the correction scheme.

        Returns
        -------
        dict
            Dictionary with scheme information
        """
        return {
            "scheme_name": self.name,
            "charge_model": self.charge_model,
            "requires_dielectric": self.requires_dielectric,
            "supports_anisotropic": self.supports_anisotropic,
            "requires_potential_data": self.requires_potential_data,
            "epsilon_static": self.epsilon_static,
            "epsilon_ionic": self.epsilon_ionic,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(epsilon_static={self.epsilon_static})"
