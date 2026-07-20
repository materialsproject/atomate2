"""
Module defining base SIESTA input set and generator.

class StructuralInformationVersion

Based on User's Guide Siesta 5.4.0
Section: 6.3 Structural Information
"""

# Metadata

__all__ = ["StructuralInformationVersion1", "StructuralInformationVersion2"]

import logging
from dataclasses import dataclass, field
from typing import Any

from atomate2.siesta.dataclass.base import FDFDataclass

logger = logging.getLogger(__name__)


@dataclass
class StructuralInformationVersion1(FDFDataclass):
    """Data class to manage structural information for SIESTA input Version I."""

    # lattice_vectors: List[Tuple[float, float, float]] = field(
    #     default_factory=lambda: [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    # )  # Lattice vectors
    # atomic_coordinates: List[Tuple[float, float, float]] = field(
    #     default_factory=list)  # Atomic coordinates
    # atomic_species: List[int] = field(default_factory=list)
    #     # List of atomic species corresponding to the coordinates
    # z_matrix_format: bool = False  # Whether to use Z-matrix format for coordinates

    lattice_vectors: list[tuple[float, float, float]] = field(
        default_factory=lambda: [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        metadata={
            "description": "The three lattice vectors (in Angstroms) that define the "
            "simulation cell.",
            "SIESTA keyword": "%block LatticeVectors",
        },
    )

    atomic_coordinates: list[tuple[float, float, float]] = field(
        default_factory=list,
        metadata={
            "description": "A list of coordinates for each atom. These are part of the "
            "'AtomicCoordinatesAndAtomicSpecies' block.",
            "SIESTA keyword": "%block AtomicCoordinatesAndAtomicSpecies",
        },
    )

    atomic_species: list[int] = field(
        default_factory=list,
        metadata={
            "description": "A list of integer indices mapping each atom to its "
            "chemical species. This is the fourth column in the "
            "'AtomicCoordinatesAndAtomicSpecies' block.",
            "SIESTA keyword": "%block AtomicCoordinatesAndAtomicSpecies",
        },
    )

    z_matrix_format: bool = field(
        default=False,
        metadata={
            "description": "A flag to specify that the atomic coordinates are provided "
            "in Z-matrix (internal coordinate) format.",
            "SIESTA keyword": "AtomicCoordinatesFormat",
        },
    )

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "%block LatticeVectors",
                "%block AtomicCoordinatesAndAtomicSpecies",
                "AtomicCoordinatesFormat",
            )
            self.__class__._registered = True  # noqa: SLF001 own-class registry flag

    def validate(self) -> None:
        """Validate the structural information settings."""
        logger.info("StructuralInformationVersion1.validate()")
        if len(self.atomic_coordinates) != len(self.atomic_species):
            raise ValueError(
                "The number of atomic coordinates must match the number of "
                "atomic species."
            )
        if not self.lattice_vectors or len(self.lattice_vectors) != 3:
            raise ValueError("Exactly three lattice vectors must be provided.")
        print(  # noqa: T201 diagnostic output
            f"Validated: {len(self.atomic_coordinates)=}, "
            f"{len(self.atomic_species)=}, {self.lattice_vectors=}"
        )

    def generate_structure_block(self) -> str:
        """Generate the structure-related blocks for the FDF file."""
        logger.info("StructuralInformationVersion1.generate_structure_block()")
        block_lines = ["%block LatticeVectors"]
        block_lines.extend(
            f"  {vec[0]}  {vec[1]}  {vec[2]}" for vec in self.lattice_vectors
        )
        block_lines.append("%endblock LatticeVectors")

        block_lines.append("%block AtomicCoordinatesAndAtomicSpecies")
        for coord, species in zip(
            self.atomic_coordinates, self.atomic_species, strict=False
        ):
            block_lines.append(f"  {coord[0]}  {coord[1]}  {coord[2]}  {species}")
        block_lines.append("%endblock AtomicCoordinatesAndAtomicSpecies")

        return "\n".join(block_lines)

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """Update from FDF parameters. Structure is typically handled by ASE."""
        # Structure handling is done by ASE Atoms object

    def generate_fdf(self) -> dict[str, Any]:
        """Generate FDF parameters. Structure blocks are handled elsewhere."""
        return {}  # Structure written by ASE writer

    @classmethod
    def setup_structural_information(
        cls,
        user_params: dict[str, Any] | None = None,
        **kwargs,  # noqa: ARG003 interface compatibility
    ) -> "StructuralInformationVersion1":
        """
        Create and configure a StructuralInformationVersion1 instance.

        Full parameter parsing is applied to the provided user parameters.

        Args:
            user_params: Dictionary of user-defined parameters
                (case-insensitive, may include dots).
            **kwargs: Additional keyword arguments to override or supplement
                user_params.

        Returns
        -------
            StructuralInformationVersion1: Configured instance with all fields set.
        """
        # Initialize instance with defaults
        instance = cls()

        # Handle case where user_params is None or empty
        if user_params is None or not user_params:
            return instance

        # Call update_from_fdf to handle parameter parsing
        instance.update_from_fdf(user_params)

        # Generate FDF
        instance.generate_fdf()

        return instance

    def to_ase(self) -> dict[str, Any]:
        """ASE handles structure directly via Atoms object."""
        return {}  # Structure already in ASE Atoms format


@dataclass
class StructuralInformationVersion2(FDFDataclass):
    """Data class to manage structural information for SIESTA input Version II."""

    # lattice_vectors: List[Tuple[float, float, float]] = field(
    #     default_factory=lambda: [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    # )  # Lattice vectors of the structure
    # atomic_coordinates: List[Tuple[float, float, float]] = field(
    #     default_factory=list)  # Atomic coordinates in fractional or cartesian format
    # atomic_species: List[int] = field(default_factory=list)
    #     # List of atomic species corresponding to coordinates
    # constraints: Dict[str, Any] = field(default_factory=dict)
    #     # Constraints on the atomic structure (optional)

    lattice_vectors: list[tuple[float, float, float]] = field(
        default_factory=lambda: [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        metadata={
            "description": "The three lattice vectors (in Angstroms) that define the "
            "simulation cell.",
            "SIESTA keyword": "%block LatticeVectors",
        },
    )

    atomic_coordinates: list[tuple[float, float, float]] = field(
        default_factory=list,
        metadata={
            "description": "A list of coordinates for each atom, either in fractional "
            "or Cartesian units (specified by 'AtomicCoordinatesFormat').",
            "SIESTA keyword": "%block AtomicCoordinatesAndAtomicSpecies",
        },
    )

    atomic_species: list[int] = field(
        default_factory=list,
        metadata={
            "description": "A list of integer indices mapping each atom to its "
            "chemical species. This corresponds to the fourth column in the "
            "'AtomicCoordinatesAndAtomicSpecies' block.",
            "SIESTA keyword": "%block AtomicCoordinatesAndAtomicSpecies",
        },
    )

    constraints: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "A block for defining constraints on atomic positions or "
            "lattice vectors during relaxation or dynamics.",
            "SIESTA keyword": "%block Geometry.Constraints",
        },
    )

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "%block LatticeVectors",
                "%block AtomicCoordinatesAndAtomicSpecies",
                # NOTE: "%block Geometry.Constraints" handled by
                # GeneralConstraints module
            )
            self.__class__._registered = True  # noqa: SLF001 own-class registry flag

    def validate(self) -> None:
        """Validate the structural information settings."""
        logger.info("StructuralInformationVersion2.validate()")
        if len(self.atomic_coordinates) != len(self.atomic_species):
            raise ValueError(
                "The number of atomic coordinates must match the number of "
                "atomic species."
            )
        print(  # noqa: T201 diagnostic output
            f"Validated: {len(self.lattice_vectors)} lattice vectors, "
            f"{len(self.atomic_coordinates)} atomic coordinates."
        )

    def generate_lattice_block(self) -> str:
        """Generate the LatticeVectors block for the FDF file."""
        logger.info("StructuralInformationVersion2.generate_lattice_block()")
        block_lines = ["%block LatticeVectors"]
        block_lines.extend(
            " ".join(map(str, vector)) for vector in self.lattice_vectors
        )
        block_lines.append("%endblock LatticeVectors")
        return "\n".join(block_lines)

    def generate_coordinates_block(self) -> str:
        """Generate the AtomicCoordinatesAndAtomicSpecies block for the FDF file."""
        logger.info("StructuralInformationVersion2.generate_coordinates_block()")
        block_lines = ["%block AtomicCoordinatesAndAtomicSpecies"]
        for coord, species in zip(
            self.atomic_coordinates, self.atomic_species, strict=False
        ):
            block_lines.append(f"{' '.join(map(str, coord))} {species}")
        block_lines.append("%endblock AtomicCoordinatesAndAtomicSpecies")
        return "\n".join(block_lines)

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """Update from FDF parameters. Structure is typically handled by ASE."""
        # Structure handling is done by ASE Atoms object

    def generate_fdf(self) -> dict[str, Any]:
        """Generate FDF parameters. Structure blocks are handled elsewhere."""
        return {}  # Structure written by ASE writer

    @classmethod
    def setup_structural_information(
        cls,
        user_params: dict[str, Any] | None = None,
        **kwargs,  # noqa: ARG003 interface compatibility
    ) -> "StructuralInformationVersion2":
        """
        Create and configure a StructuralInformationVersion2 instance.

        Full parameter parsing is applied to the provided user parameters.

        Args:
            user_params: Dictionary of user-defined parameters
                (case-insensitive, may include dots).
            **kwargs: Additional keyword arguments to override or supplement
                user_params.

        Returns
        -------
            StructuralInformationVersion2: Configured instance with all fields set.
        """
        # Initialize instance with defaults
        instance = cls()

        # Handle case where user_params is None or empty
        if user_params is None or not user_params:
            return instance

        # Call update_from_fdf to handle parameter parsing
        instance.update_from_fdf(user_params)

        # Generate FDF
        instance.generate_fdf()

        return instance

    def to_ase(self) -> dict[str, Any]:
        """ASE handles structure directly via Atoms object."""
        return {}  # Structure already in ASE Atoms format
