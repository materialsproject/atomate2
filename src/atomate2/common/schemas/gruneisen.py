"""General schemas for Gruneisen parameter workflow outputs."""

import logging
from typing import Optional

from emmet.core.structure import StructureMetadata
from pydantic import BaseModel, Field
from pymatgen.phonon.gruneisen import (
    GruneisenParameter,
    GruneisenPhononBandStructureSymmLine,
)

logger = logging.getLogger(__name__)


class GruneisenInputDirs(BaseModel):
    """Collection with all input directories relevant for the Gruneisen run."""

    ground: Optional[str] = Field(
        None, description="The directory with ground state structure phonopy yaml"
    )
    plus: Optional[str] = Field(
        None, description="The directory with expanded structure phonopy yaml"
    )
    minus: Optional[str] = Field(
        None, description="The directory with contracted structure phonopy yaml"
    )


class PhononRunsImaginaryModes(BaseModel):
    """Collection with information whether structure has imaginary modes.

    Information extracted from phonon run for ground, expanded and contracted structures
    """

    ground: Optional[bool] = Field(
        None, description="if true, ground state structure has imaginary modes"
    )
    plus: Optional[bool] = Field(
        None, description="if true, expanded structure has imaginary modes"
    )
    minus: Optional[bool] = Field(
        None, description="if true, contracted structure has imaginary modes"
    )


class GruneisenDerivedProperties(BaseModel):
    """Collection of data derived from the gruneisen workflow."""

    average_gruneisen: Optional[float] = Field(
        None, description="The average Gruneisen parameter"
    )
    thermal_conductivity_slack: Optional[float] = Field(
        None,
        description="The thermal conductivity at the acoustic "
        "Debye temperature with the Slack formula.",
    )


class GruneisenParameterDocument(StructureMetadata):
    """Collection to data from the gruneisen computation."""

    gruneisen_parameter_inputs: GruneisenInputDirs = Field(
        None, description="The directories where the phonon jobs were run."
    )
    phonon_runs_has_imaginary_modes: Optional[PhononRunsImaginaryModes] = Field(
        None,
        description="Collection indicating whether the structures from the "
        "phonon runs have imaginary modes",
    )
    gruneisen_parameter: Optional[GruneisenParameter] = Field(
        None, description="Gruneisen parameter object"
    )
    gruneisen_band_structure: Optional[GruneisenPhononBandStructureSymmLine] = Field(
        None, description="Gruneisen phonon band structure symmetry line object"
    )
    derived_properties: Optional[GruneisenDerivedProperties] = Field(
        None, description="Properties derived from the Gruneisen parameter."
    )
