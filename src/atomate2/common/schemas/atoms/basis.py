"""Schemas for basis set repreesntations"""

from typing import Dict, Sequence
from pydantic import BaseModel, Field
from jobflow.utils import ValueEnum
from pymatgen.core import Element

class BasisType(ValueEnum):

    GTO = "Gaussian Type Orbitals"
    LCAO = "Linear Combination of Atomic Orbitals"
    PW = "Plane Waves"
    SLATER = "Slater Type Orbitals"

class BasisCardinality(ValueEnum):

    SZV = "SZV"
    SZVP = "SZVP"
    DZV = "DZV"
    DZVP = "DZVP"
    TZV = "TZV"
    TZVP = "TZVP"
    TZV2P = "TZV2P"
    QZVP = "QZVP"
    QZV2P = "QZV2P"

class BasisSet(BaseModel):

    element: Element = Field(None, description="Element for this basis set (if local)")
    basis_type: BasisType = Field(None, description="Type of basis set")
    name: str = Field(None, description="Name of this basis set")
    alias_names: Sequence = Field(None, description="Optional aliases for this basis set")
    filename: str = Field(None, description="Name of the file containing this basis")
    version: str = Field(None, description="Version for this basis set")

class GaussianTypeOrbitalBasisSet(BasisSet):

    basis_type = BasisType.GTO
    cardinality: BasisCardinality = Field(None, description="Cardinality of this basis") 
    nset: int = Field(None, description="Number of exponent sets")
    n: int = Field(None, description="Principle quantum number")
    lmax: int = Field(None, description="Maximum angular momentum quantum number")
    lmin: int = Field(None, description="Minimum angular momentum quantum number")
    nshell: Sequence = Field(None, description="Number of shells for angular momentum quantum number from lmin to lmax")
    exponents: Sequence = Field(None, description="Exponents")
    coefficients: Dict[int, Dict[int, Dict[int, float]]] = Field(None, description="Contraction coefficients Dict[exp->l->shell]")

    class Config:
        arbitrary_types_allowed = True

    @property
    def nexp(self):
        return len(self.exponents)
