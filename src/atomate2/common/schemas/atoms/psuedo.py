from typing import Dict, Sequence
from pydantic import BaseModel, Field
from jobflow.utils import ValueEnum
from pymatgen.electronic_structure.core import Orbital, OrbitalType

class PseudoPotentialType(ValueEnum):

    PAW = "Projector Augmented Wave"
    NC = "Norm Conserving"
    US = "Ultra Soft"
    GTH = "Goedecker-Teter-Hutter Norm Conserving"

class PseudoPotential(BaseModel):

    pseudopotential_type: PseudoPotentialType = Field(None, description="Type of pseudopotential")
    name: str = Field(None, description="Name of this pseudopotential")
    filename: str = Field(None, description="Name of the file containing this pseudopotential")
    functional: str = Field(None, description="Specific functional for which this pseudopotential was optimized")
    version: str = Field(None, description="Version for this pseudopotential")

class GthPseudopotential(PseudoPotential):

    pseudopotential_type = PseudoPotentialType.GTH

    n_elecs: Dict[int, int] = Field(None, description="Number of electrons for each quantum number n")
    r_loc: float = Field(None, description="Radius for the local part defined by the Gaussian function exponent alpha_erf")
    nexp_ppl: int = Field(None, description="Number of the local pseudopotential functions")
    c_exp_ppl: Sequence = Field(None, description="Coefficients of the local pseudopotential functions")
    r: float = Field(None, description="Radius of the nonlocal part for angular momentum quantum number l defined by the Gaussian function exponents alpha_prj_ppnl")
    nprj_ppnl: Dict[int, int] = Field(None, description="Number of the non-local projectors for the angular momentum quantum number l")
    hprj_ppnl: Dict[int, Sequence] = Field(None, description="Coefficients of the non-local projector functions")
    