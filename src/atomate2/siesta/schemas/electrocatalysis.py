"""Pydantic schemas for electrocatalysis workflows.

This module defines data models for storing and validating results from
electrocatalysis calculations including gas-phase molecular calculations,
reaction pathways, and thermodynamic analysis.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from pymatgen.core import Molecule, Structure


class GasPhaseMoleculeDocument(BaseModel):
    """Results from gas-phase molecular calculation.

    This document stores the output of a GasPhaseMoleculeMaker workflow,
    including energy, geometry, and spin configuration.

    Attributes
    ----------
    formula : str
        Chemical formula (e.g., "O2", "H2O", "CO2")
    total_energy : float
        Total DFT energy (eV)
    spin_polarized : bool
        Whether spin-polarized DFT was used
    spin_type : str
        SIESTA spin type: "polarized" or "non-polarized"
    structure : Structure
        Optimized structure (molecule in box)
    molecule : Molecule | None
        Pymatgen Molecule object (if available)
    spin_config : dict | None
        Spin configuration details from auto-detection
    corrections : dict
        Energy corrections (ZPE, entropy, etc.) - for future use
    """

    formula: str = Field(..., description="Chemical formula")
    total_energy: float = Field(..., description="Total energy (eV)")
    spin_polarized: bool = Field(
        False, description="Whether spin-polarized DFT was used"
    )
    spin_type: str = Field(
        "non-polarized",
        description="SIESTA spin type: 'polarized', 'non-polarized', etc.",
    )
    structure: Structure = Field(
        ..., description="Optimized structure (molecule in box)"
    )
    molecule: Molecule | None = Field(
        None, description="Pymatgen Molecule object (if available)"
    )
    spin_config: dict[str, Any] | None = Field(
        None, description="Spin configuration details from auto-detection"
    )
    corrections: dict[str, float] = Field(
        default_factory=dict, description="Energy corrections (ZPE, entropy, etc.)"
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True  # Allow pymatgen Structure/Molecule types


class ReactionStep(BaseModel):
    """Single step in a reaction pathway.

    Represents one intermediate in an electrochemical reaction pathway,
    such as O2*, OOH*, O*, OH* in the ORR pathway.

    Attributes
    ----------
    label : str
        Step label (e.g., "O2_ads", "OOH_ads", "clean")
    species : str | None
        Adsorbed species formula (None for clean surface)
    site_coords : tuple[float, float] | None
        Adsorption site (x, y) fractional coordinates
    height : float | None
        Height above surface (Å)
    energy : float | None
        Total DFT energy for this step (eV)
    structure : Structure | None
        Optimized structure for this intermediate
    """

    label: str = Field(..., description="Step label (e.g., 'O2_ads')")
    species: str | None = Field(None, description="Added species")
    site_coords: tuple[float, float] | None = Field(
        None, description="Adsorption site (x, y) fractional coordinates"
    )
    height: float | None = Field(None, description="Height above surface (Å)")
    energy: float | None = Field(None, description="Total energy (eV)")
    structure: Structure | None = Field(
        None, description="Optimized structure for this intermediate"
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class ReactionPathwayDocument(BaseModel):
    """Complete reaction pathway analysis results.

    Stores the full analysis of an electrochemical reaction pathway including
    energies, free energies, overpotentials, and rate-limiting steps.

    This is the primary output document for ORR/OER/HER/CO2RR workflows.

    Attributes
    ----------
    surface_name : str
        Name/description of the surface catalyst
    pathway_type : str
        Reaction type: "orr", "oer", "her", "co2rr", etc.
    steps : list[ReactionStep]
        List of reaction intermediates in pathway order
    energies : list[float]
        Absolute DFT energies for each step (eV)
    delta_E : list[float]
        Energy differences between steps (eV)
    delta_G : list[float]
        Free energy differences (eV) - includes ZPE, entropy, voltage
    overpotential_orr : float
        ORR overpotential (V) - only for ORR pathways
    overpotential_oer : float
        OER overpotential (V) - only for OER pathways
    overpotential_gap : float
        Total ORR + OER gap (V)
    rate_limiting_step : int | str
        Index or label of rate-limiting step
    temperature : float
        Temperature for free energy calculation (K)
    pressure : float
        Pressure for gas-phase species (Pa)
    """

    surface_name: str = Field(..., description="Surface catalyst name")
    pathway_type: str = Field(..., description="Reaction type (orr, oer, her, etc.)")
    steps: list[ReactionStep] = Field(..., description="Reaction intermediates")

    # Energies
    energies: list[float] = Field(..., description="Absolute energies (eV)")
    delta_E: list[float] = Field(..., description="Energy differences (eV)")
    delta_G: list[float] = Field(..., description="Free energy differences (eV)")

    # Overpotentials
    overpotential_orr: float = Field(0.0, description="ORR overpotential (V)")
    overpotential_oer: float = Field(0.0, description="OER overpotential (V)")
    overpotential_gap: float = Field(0.0, description="Total gap (V)")

    # Analysis
    rate_limiting_step: int | str = Field(..., description="RLS index or label")

    # Metadata
    temperature: float = Field(298.15, description="Temperature (K)")
    pressure: float = Field(101325.0, description="Pressure (Pa)")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
