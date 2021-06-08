"""Core definition of Structure metadata."""

from __future__ import annotations

from typing import List, Optional, Type, TypeVar

from jobflow import Schema
from pydantic import Field
from pymatgen.core import Composition, Structure
from pymatgen.core.periodic_table import Element

from atomate2.common.schemas.symmetry import SymmetryData

T = TypeVar("T", bound="StructureMetadata")


class StructureMetadata(Schema):
    """Mix-in class for structure metadata."""

    # Structure metadata
    nsites: int = Field(None, description="Total number of sites in the structure")
    elements: List[Element] = Field(
        None, description="List of elements in the material"
    )
    nelements: int = Field(None, title="Number of Elements")
    composition: Composition = Field(
        None, description="Full composition for the material"
    )
    composition_reduced: Composition = Field(
        None,
        title="Reduced Composition",
        description="Simplified representation of the composition",
    )
    formula_pretty: str = Field(
        None,
        title="Pretty Formula",
        description="Cleaned representation of the formula",
    )
    formula_anonymous: str = Field(
        None,
        title="Anonymous Formula",
        description="Anonymized representation of the formula",
    )
    chemsys: str = Field(
        None,
        title="Chemical System",
        description="dash-delimited string of elements in the material",
    )
    volume: float = Field(
        None,
        title="Volume",
        description="Total volume for this structure in Angstroms^3",
    )

    density: float = Field(
        None, title="Density", description="Density in grams per cm^3"
    )

    density_atomic: float = Field(
        None,
        title="Packing Density",
        description="The atomic packing density in atoms per cm^3",
    )

    symmetry: SymmetryData = Field(None, description="Symmetry data for this material")

    @classmethod
    def from_composition(
        cls: Type[T],
        composition: Composition,
        fields: Optional[List[str]] = None,
        **kwargs
    ) -> T:
        """
        Create a StructureMetadata model from a composition.

        Parameters
        ----------
        composition
            A pymatgen composition.
        fields
            Composition fields to include.
        kwargs
            Keyword arguements that are passed to the model constructor.

        Returns
        -------
        StructureMetadata
            A structure metadata model.
        """
        fields = (
            [
                "elements",
                "nelements",
                "composition",
                "composition_reduced",
                "formula_pretty",
                "formula_anonymous",
                "chemsys",
            ]
            if fields is None
            else fields
        )
        elsyms = sorted(set([e.symbol for e in composition.elements]))

        data = {
            "elements": elsyms,
            "nelements": len(elsyms),
            "composition": composition,
            "composition_reduced": composition.reduced_composition,
            "formula_pretty": composition.reduced_formula,
            "formula_anonymous": composition.anonymized_formula,
            "chemsys": "-".join(elsyms),
        }

        return cls(**{k: v for k, v in data.items() if k in fields}, **kwargs)

    @classmethod
    def from_structure(
        cls: Type[T],
        structure: Structure,
        fields: Optional[List[str]] = None,
        include_structure: bool = False,
        **kwargs
    ) -> T:
        """
        Create schema from a structure.

        Parameters
        ----------
        structure
            A pymatgen structure.
        fields
            Structure fields to include.
        include_structure
            Whether to include the structure itself in the schema.
        kwargs
            Keyword args that are passed to the Schema constructor.

        Returns
        -------
        StructureMetadata
            A structure metadata model.
        """
        fields = (
            [
                "nsites",
                "elements",
                "nelements",
                "composition",
                "composition_reduced",
                "formula_pretty",
                "formula_anonymous",
                "chemsys",
                "volume",
                "density",
                "density_atomic",
                "symmetry",
            ]
            if fields is None
            else fields
        )
        comp = structure.composition
        elsyms = sorted(set([e.symbol for e in comp.elements]))
        symmetry = SymmetryData.from_structure(structure)

        data = {
            "nsites": structure.num_sites,
            "elements": elsyms,
            "nelements": len(elsyms),
            "composition": comp,
            "composition_reduced": comp.reduced_composition,
            "formula_pretty": comp.reduced_formula,
            "formula_anonymous": comp.anonymized_formula,
            "chemsys": "-".join(elsyms),
            "volume": structure.volume,
            "density": structure.density,
            "density_atomic": structure.volume / structure.num_sites,
            "symmetry": symmetry,
        }

        if include_structure:
            kwargs.update({"structure": structure})

        return cls(**{k: v for k, v in data.items() if k in fields}, **kwargs)
