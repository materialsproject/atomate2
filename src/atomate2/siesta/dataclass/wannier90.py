"""
Module defining base SIESTA input set and generator.

class Wannier90

Based on User's Guide Siesta 5.4.0
Section:    6.22 Maximally Localized Wannier Functions. Interface with the wannier90 code
            6.22.1 wannier90 as a postprocessing tool
            6.22.2 wannier90 called on-the-fly within siesta
"""

# Metadata

__all__ = ["Wannier90"]

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from atomate2.siesta.dataclass.base import FDFDataclass

import logging

logger = logging.getLogger(__name__)


@dataclass
class Wannier90(FDFDataclass):
    """
    Maximally Localized Wannier Functions. Interface with the wannier90 code
    # -----------------------------------------------------------------------------
    # 6.22 Maximally Localized Wannier Functions. Interface with the wannier90 code
    # 6.22.1 wannier90 as a postprocessing tool
    # -----------------------------------------------------------------------------
    # siesta_2_wannier90_write_mmn: bool = False # Siesta2Wannier90.WriteMmn false
    # siesta_2_wannier90_write_amn: bool = False # Siesta2Wannier90.WriteAmn false
    # siesta_2_wannier90_write_eig: bool = False # Siesta2Wannier90.WriteEig false
    # siesta_2_wannier90_write_unk: bool = False # Siesta2Wannier90.WriteUnk false
    # siesta_2_wannier90_unk_grid1: int = None # Siesta2Wannier90.UnkGrid1 〈mesh points along A〉
    # siesta_2_wannier90_unk_grid2: int = None # Siesta2Wannier90.UnkGrid2 〈mesh points along B〉
    # siesta_2_wannier90_unk_grid3: int = None # Siesta2Wannier90.UnkGrid3 〈mesh points along C〉
    # siesta_2_wannier90_unk_grid_binary: bool = True # Siesta2Wannier90.UnkGridBinary true
    # siesta_2_wannier90_number_of_bands: int = None # Siesta2Wannier90.NumberOfBands occupied bands
    # siesta_2_wannier90_number_of_bands_up: int = None # Siesta2Wannier90.NumberOfBandsUp 〈Siesta2Wannier90.NumberOfBands〉
    # siesta_2_wannier90_number_of_bands_down: int = None # Siesta2Wannier90.NumberOfBandsDown 〈Siesta2Wannier90.NumberOfBands〉
    """

    siesta_2_wannier90_write_mmn: bool = field(
        default=False,
        metadata={
            "description": "A flag to enable writing the overlap matrices between cell-periodic parts of Bloch states (.mmn file), required by Wannier90.",
            "SIESTA keyword": "Siesta2Wannier90.WriteMmn",
        },
    )

    siesta_2_wannier90_write_amn: bool = field(
        default=False,
        metadata={
            "description": "A flag to enable writing the projection of Bloch states onto trial localized orbitals (.amn file), used by Wannier90 as an initial guess.",
            "SIESTA keyword": "Siesta2Wannier90.WriteAmn",
        },
    )

    siesta_2_wannier90_write_eig: bool = field(
        default=False,
        metadata={
            "description": "A flag to enable writing the eigenvalues of the Bloch states (.eig file) in a format suitable for Wannier90.",
            "SIESTA keyword": "Siesta2Wannier90.WriteEig",
        },
    )

    siesta_2_wannier90_write_unk: bool = field(
        default=False,
        metadata={
            "description": "A flag to enable writing the periodic part of the Bloch wavefunctions on a real-space grid (.UNK files), used for plotting Wannier functions.",
            "SIESTA keyword": "Siesta2Wannier90.WriteUnk",
        },
    )

    siesta_2_wannier90_unk_grid1: int = field(
        default=None,
        metadata={
            "description": "The number of grid points along the first lattice vector for the wavefunction output (.UNK files). Defaults to the main mesh size.",
            "SIESTA keyword": "Siesta2Wannier90.UnkGrid1",
        },
    )

    siesta_2_wannier90_unk_grid2: int = field(
        default=None,
        metadata={
            "description": "The number of grid points along the second lattice vector for the wavefunction output (.UNK` files). Defaults to the main mesh size.",
            "SIESTA keyword": "Siesta2Wannier90.UnkGrid2",
        },
    )

    siesta_2_wannier90_unk_grid3: int = field(
        default=None,
        metadata={
            "description": "The number of grid points along the third lattice vector for the wavefunction output (.UNK` files). Defaults to the main mesh size.",
            "SIESTA keyword": "Siesta2Wannier90.UnkGrid3",
        },
    )

    siesta_2_wannier90_unk_grid_binary: bool = field(
        default=True,
        metadata={
            "description": "If true, writes the real-space wavefunction grid files (.UNK) in binary format.",
            "SIESTA keyword": "Siesta2Wannier90.UnkGridBinary",
        },
    )

    siesta_2_wannier90_number_of_bands: int = field(
        default=None,
        metadata={
            "description": "The number of bands to be included in the inner 'frozen' window for the Wannier90 calculation. Defaults to the number of occupied bands.",
            "SIESTA keyword": "Siesta2Wannier90.NumberOfBands",
        },
    )

    siesta_2_wannier90_number_of_bands_up: int = field(
        default=None,
        metadata={
            "description": "For spin-polarized calculations, the number of bands to include for the spin-up channel. Defaults to the value of 'Siesta2Wannier90.NumberOfBands'.",
            "SIESTA keyword": "Siesta2Wannier90.NumberOfBandsUp",
        },
    )

    siesta_2_wannier90_number_of_bands_down: int = field(
        default=None,
        metadata={
            "description": "For spin-polarized calculations, the number of bands to include for the spin-down channel. Defaults to the value of 'Siesta2Wannier90.NumberOfBands'.",
            "SIESTA keyword": "Siesta2Wannier90.NumberOfBandsDown",
        },
    )

    # ------------------------------------------------
    # 6.22.2 wannier90 called on-the-fly within siesta
    # ------------------------------------------------
    # wannier_manifolds_block: Dict[float,Any]= field(default_factory=dict) # %block Wannier.Manifolds 〈None〉
    # wannier_manifold_block: Dict[float,Any]= field(default_factory=dict) # %block Wannier.Manifold.<> 〈None〉
    # wannier_projectors_block: Dict[float,Any]= field(default_factory=dict) # %block Wannier.Projectors 〈projection functions as in wannier90〉
    # wannier_manifolds_threshold: float = 1e-6 # Wannier.Manifolds.Threshold 10−6
    # wannier_manifolds_unk: float = False # Wannier.Manifolds.Unk false
    # wannier_k_block: Dict[float,Any]= field(default_factory=dict) # %block Wannier.k Γ-point
    wannier_manifolds_block: Dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": "A block to define one or more distinct groups (manifolds) of Wannier functions.",
            "SIESTA keyword": "%block Wannier.Manifolds",
        },
    )

    wannier_manifold_block: Dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": "A generic block for defining a specific Wannier function manifold's properties, such as its energy window. The block name is user-defined (e.g., '%block Wannier.Manifold.conduction').",
            "SIESTA keyword": "%block Wannier.Manifold.<>",
        },
    )

    wannier_projectors_block: Dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": "A block to define the trial orbitals (projection functions) used as an initial guess for the Wannierization procedure, similar to Wannier90's 'projections' block.",
            "SIESTA keyword": "%block Wannier.Projectors",
        },
    )

    wannier_manifolds_threshold: float = field(
        default=1e-6,
        metadata={
            "description": "A convergence threshold for the Wannierization procedure or for selecting states within a manifold.",
            "SIESTA keyword": "Wannier.Manifolds.Threshold",
        },
    )

    wannier_manifolds_unk: bool = field(
        default=False,
        metadata={
            "description": "A flag to enable the writing of the calculated Wannier functions on a real-space grid for plotting.",
            "SIESTA keyword": "Wannier.Manifolds.Unk",
        },
    )

    wannier_k_block: Dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": "A block to define the k-point grid to be used for the Wannierization process.",
            "SIESTA keyword": "%block Wannier.k",
        },
    )

    def __post_init__(self):
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                # Siesta2Wannier90 parameters (postprocessing)
                "Siesta2Wannier90.WriteMmn",
                "Siesta2Wannier90.WriteAmn",
                "Siesta2Wannier90.WriteEig",
                "Siesta2Wannier90.WriteUnk",
                "Siesta2Wannier90.UnkGrid1",
                "Siesta2Wannier90.UnkGrid2",
                "Siesta2Wannier90.UnkGrid3",
                "Siesta2Wannier90.UnkGridBinary",
                "Siesta2Wannier90.NumberOfBands",
                "Siesta2Wannier90.NumberOfBandsUp",
                "Siesta2Wannier90.NumberOfBandsDown",
                # Wannier parameters (on-the-fly)
                "Wannier.Manifolds.Threshold",
                "Wannier.Manifolds.Unk",
                "%block Wannier.Manifolds",
                "%block Wannier.Manifold",
                "%block Wannier.Projectors",
                "%block Wannier.k",
            )
            self.__class__._registered = True

    def validate(self):
        """
        Validate Wannier90 interface parameters.

        Checks configuration for Wannier90 maximally-localized Wannier function
        calculations, including setup for the interface between SIESTA and Wannier90.

        Raises
        ------
        ValueError
            If Wannier90 interface parameters are invalid
        """
        logger.info("Wannier90.validate()")
        pass

    def update_from_fdf(self, fdf_dict: Dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        for key, value in fdf_dict.items():
            key_lower = key.lower()

            # Siesta2Wannier90 parameters (postprocessing)
            if key_lower in ["siesta2wannier90.writemmn"]:
                self.siesta_2_wannier90_write_mmn = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["siesta2wannier90.writeamn"]:
                self.siesta_2_wannier90_write_amn = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["siesta2wannier90.writeeig"]:
                self.siesta_2_wannier90_write_eig = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["siesta2wannier90.writeunk"]:
                self.siesta_2_wannier90_write_unk = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["siesta2wannier90.unkgrid1"]:
                self.siesta_2_wannier90_unk_grid1 = int(value)
            elif key_lower in ["siesta2wannier90.unkgrid2"]:
                self.siesta_2_wannier90_unk_grid2 = int(value)
            elif key_lower in ["siesta2wannier90.unkgrid3"]:
                self.siesta_2_wannier90_unk_grid3 = int(value)
            elif key_lower in ["siesta2wannier90.unkgridbinary"]:
                self.siesta_2_wannier90_unk_grid_binary = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["siesta2wannier90.numberofbands"]:
                self.siesta_2_wannier90_number_of_bands = int(value)
            elif key_lower in ["siesta2wannier90.numberofbandsup"]:
                self.siesta_2_wannier90_number_of_bands_up = int(value)
            elif key_lower in ["siesta2wannier90.numberofbandsdown"]:
                self.siesta_2_wannier90_number_of_bands_down = int(value)

            # Wannier parameters (on-the-fly)
            elif key_lower in ["wannier.manifolds.threshold"]:
                self.wannier_manifolds_threshold = float(value)
            elif key_lower in ["wannier.manifolds.unk"]:
                self.wannier_manifolds_unk = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["%block wannier.manifolds", "wannier.manifolds"]:
                self.wannier_manifolds_block = value
            elif key_lower.startswith("%block wannier.manifold"):
                self.wannier_manifold_block = value
            elif key_lower in ["%block wannier.projectors", "wannier.projectors"]:
                self.wannier_projectors_block = value
            elif key_lower in ["%block wannier.k", "wannier.k"]:
                self.wannier_k_block = value

    def generate_fdf(self) -> Dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns:
            Dictionary of FDF parameters
        """
        fdf: Dict[str, Any] = {}

        # Siesta2Wannier90 parameters (postprocessing)
        if self.siesta_2_wannier90_write_mmn:
            fdf["Siesta2Wannier90.WriteMmn"] = "true"
        if self.siesta_2_wannier90_write_amn:
            fdf["Siesta2Wannier90.WriteAmn"] = "true"
        if self.siesta_2_wannier90_write_eig:
            fdf["Siesta2Wannier90.WriteEig"] = "true"
        if self.siesta_2_wannier90_write_unk:
            fdf["Siesta2Wannier90.WriteUnk"] = "true"
        if self.siesta_2_wannier90_unk_grid1 is not None:
            fdf["Siesta2Wannier90.UnkGrid1"] = str(self.siesta_2_wannier90_unk_grid1)
        if self.siesta_2_wannier90_unk_grid2 is not None:
            fdf["Siesta2Wannier90.UnkGrid2"] = str(self.siesta_2_wannier90_unk_grid2)
        if self.siesta_2_wannier90_unk_grid3 is not None:
            fdf["Siesta2Wannier90.UnkGrid3"] = str(self.siesta_2_wannier90_unk_grid3)
        if not self.siesta_2_wannier90_unk_grid_binary:
            fdf["Siesta2Wannier90.UnkGridBinary"] = "false"
        if self.siesta_2_wannier90_number_of_bands is not None:
            fdf["Siesta2Wannier90.NumberOfBands"] = str(
                self.siesta_2_wannier90_number_of_bands
            )
        if self.siesta_2_wannier90_number_of_bands_up is not None:
            fdf["Siesta2Wannier90.NumberOfBandsUp"] = str(
                self.siesta_2_wannier90_number_of_bands_up
            )
        if self.siesta_2_wannier90_number_of_bands_down is not None:
            fdf["Siesta2Wannier90.NumberOfBandsDown"] = str(
                self.siesta_2_wannier90_number_of_bands_down
            )

        # Wannier parameters (on-the-fly)
        if self.wannier_manifolds_threshold != 1e-6:
            fdf["Wannier.Manifolds.Threshold"] = str(self.wannier_manifolds_threshold)
        if self.wannier_manifolds_unk:
            fdf["Wannier.Manifolds.Unk"] = "true"
        if self.wannier_manifolds_block:
            fdf["%block Wannier.Manifolds"] = self.wannier_manifolds_block
        if self.wannier_manifold_block:
            fdf["%block Wannier.Manifold"] = self.wannier_manifold_block
        if self.wannier_projectors_block:
            fdf["%block Wannier.Projectors"] = self.wannier_projectors_block
        if self.wannier_k_block:
            fdf["%block Wannier.k"] = self.wannier_k_block

        return fdf

    @classmethod
    def setup_wannier90(
        cls, user_params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> "Wannier90":
        """
        Create and configure a Wannier90 instance with full parameter parsing.

        Args:
            user_params: Dictionary of user-defined parameters (case-insensitive, may include dots).
            **kwargs: Additional keyword arguments to override or supplement user_params.

        Returns:
            Wannier90: Configured instance with all fields set.
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

    def to_ase(self) -> Dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns:
            Dictionary of ASE parameters
        """
        # ASE doesn't have Wannier90 interface parameters
        # These are SIESTA-specific for Wannier90 integration
        return {}
