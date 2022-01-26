"""Core definition of Molecule metadata."""

from typing import List, Optional, Type, TypeVar

from pydantic import BaseModel, Field
from pymatgen.core import Composition, Molecule
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

__all__ = ["MoleculeMetadata"]

T = TypeVar("T", bound="MoleculeMetadata")


class MoleculeMetadata(BaseModel):
    """Mix-in class for molecule metadata."""

    # Molecule metadata
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
    point_group: str = Field(
        None, title="Point group", description="Point group for the molecule"
    )
    charge: int = Field(
        None, title="Charge", description="Total (net) charge of the molecule"
    )
    spin_multiplicity: int = Field(
        None,
        title="Spin multiplicity",
        description="The spin multiplicity (2*S+1) for the molecule",
    )
    nelectrons: int = Field(
        None,
        title="Number of electrons",
        description="The total number of electrons for the molecule",
    )

    @classmethod
    def from_composition(
        cls: Type[T],
        composition: Composition,
        fields: Optional[List[str]] = None,
        **kwargs
    ) -> T:
        """
        Create a MoleculeMetadata model from a composition.

        Parameters
        ----------
        composition : .Composition
            A pymatgen composition.
        fields : list of str or None
            Composition fields to include.
        **kwargs
            Keyword arguements that are passed to the model constructor.

        Returns
        -------
        T
            A molecule metadata model.
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
        elsyms = sorted({e.symbol for e in composition.elements})

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
    def from_molecule(
        cls: Type[T],
        molecule: Molecule,
        fields: Optional[List[str]] = None,
        include_molecule: bool = False,
        **kwargs
    ) -> T:
        """
        Create schema from a molecule.

        Parameters
        ----------
        molecule : .Molecule
            A pymatgen molecule.
        fields : list of str or None
            Molecule fields to include.
        include_molecule : bool
            Whether to include the molecule itself in the schema.
        **kwargs
            Keyword args that are passed to the Schema constructor.

        Returns
        -------
        T
            A molecule metadata model.
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
                "point_group",
                "charge",
                "spin_multiplicity",
                "nelectrons",
            ]
            if fields is None
            else fields
        )
        comp = molecule.composition
        elsyms = sorted({e.symbol for e in comp.elements})
        point_group = PointGroupAnalyzer(molecule).sch_symbol

        data = {
            "nsites": molecule.num_sites,
            "elements": elsyms,
            "nelements": len(elsyms),
            "composition": comp,
            "composition_reduced": comp.reduced_composition,
            "formula_pretty": comp.reduced_formula,
            "formula_anonymous": comp.anonymized_formula,
            "chemsys": "-".join(elsyms),
            "point_group": point_group,
            "charge": int(molecule.charge),
            "spin_multiplicity": molecule.spin_multiplicity,
            "nelectrons": int(molecule.nelectrons),
        }

        if include_molecule:
            kwargs.update({"molecule": molecule})

        return cls(**{k: v for k, v in data.items() if k in fields}, **kwargs)
