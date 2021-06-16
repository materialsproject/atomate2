"""Schemas for elastic tensor fitting and related properties."""

from typing import List, Optional

from pydantic import BaseModel, Field
from pymatgen.analysis.elasticity import (
    Deformation,
    ElasticTensor,
    ElasticTensorExpansion,
    Strain,
    Stress,
)
from pymatgen.core import Structure

from atomate2.common.schemas.math import Matrix3D, MatrixVoigt


class DerivedProperties(BaseModel):
    """Properties derived from an elastic tensor."""

    k_voigt: float = Field(None, description="Voigt average of the bulk modulus.")
    k_reuss: float = Field(None, description="Reuss average of the bulk modulus.")
    k_vrh: float = Field(
        None, description="Voigt-Reuss-Hill average of the bulk modulus."
    )
    g_voigt: float = Field(None, description="Voigt average of the shear modulus.")
    g_reuss: float = Field(None, description="Reuss average of the shear modulus.")
    g_vrh: float = Field(
        None, description="Voigt-Reuss-Hill average of the shear modulus."
    )
    universal_anisotropy: float = Field(
        None, description="Universal elastic anisotropy."
    )
    homogeneous_poisson: float = Field(None, description="Homogeneous poisson ratio.")
    y_mod: float = Field(
        None,
        description="Young's modulus (SI units) from the Voight-Reuss-Hill averages of the bulk and shear moduli.",
    )
    trans_v: float = Field(
        None,
        description="Transverse sound velocity (SI units) obtained from the Voigt-Reuss-Hill average bulk modulus.",
    )
    long_v: float = Field(
        None,
        description="Longitudinal sound velocity (SI units) obtained from the Voigt-Reuss-Hill average bulk modulus.",
    )
    snyder_ac: float = Field(
        None, description="Synder's acoustic sound velocity (SI units)."
    )
    snyder_opt: float = Field(
        None, description="Synder's optical sound velocity (SI units)."
    )
    snyder_total: float = Field(
        None, description="Synder's total sound velocity (SI units)."
    )
    clark_thermalcond: float = Field(
        None, description="Clarke's thermal conductivity (SI units)."
    )
    cahill_thermalcond: float = Field(
        None, description="Cahill's thermal conductivity (SI units)."
    )
    debye_temperature: float = Field(
        None,
        description="Debye temperature from longitudinal and transverse sound velocities (SI units).",
    )


class FittingData(BaseModel):
    """Data used to fit elastic tensors."""

    cauchy_stresses: List[Stress] = Field(
        None, description="The Cauchy stresses used to fit the elastic tensor."
    )
    strains: List[Strain] = Field(
        None, description="The strains used to fit the elastic tensor."
    )
    pk_stresses: List[Strain] = Field(
        None, description="The Piola–Kirchoff stresses used to fit the elastic tensor."
    )
    deformations: List[Deformation] = Field(
        None, description="The deformations corresponding to each strain state."
    )


class ElasticTensorDocument(BaseModel):
    """Raw and standardized elastic tensors."""

    raw: MatrixVoigt = Field(None, description="Raw elastic tensor.")
    ieee_format: MatrixVoigt = Field(None, description="Elastic tensor in IEEE format.")


class ElasticDocument(BaseModel):
    """Document containing elastic tensor information and related properties."""

    structure: Structure = Field(
        None, description="The structure for which the elastic data is calculated."
    )
    elastic_tensor: ElasticTensorDocument = Field(
        None, description="Fitted elastic tensor."
    )
    eq_stress: Stress = Field(
        None, description="The equilibrium stress of the structure."
    )
    derived_properties: DerivedProperties = Field(
        None, description="Properties derived from the elastic tensor."
    )
    formula_pretty: str = Field(
        None,
        description="Cleaned representation of the formula",
    )
    fitting_data: FittingData = Field(
        None, description="Data used to fit the elastic tensor."
    )
    fitting_method: str = Field(
        None, description="Method used to fit the elastic tensor."
    )
    order: int = Field(
        None, description="Order of the expansion of the elastic tensor."
    )

    @classmethod
    def from_stresses(
        cls,
        structure: Structure,
        stresses: List[Stress],
        deformations: List[Deformation],
        fitting_method: str,
        order: int,
        equilibrium_stress: Optional[Matrix3D] = None,
    ):
        """
        Create an elastic document from strains and stresses.

        Parameters
        ----------
        structure
            The structure for which strains and stresses were calculated.
        stresses
            A list of corresponding stresses.
        deformations
            A list of corresponding deformations.
        fitting_method
            The method used to fit the elastic tensor. See pymatgen for more details on
            the methods themselves. The options are:
            - "finite_difference" (note this is required if fitting a 3rd order tensor)
            - "independent"
            - "pseudoinverse"
        order
            Order of the tensor expansion to be fitted. Can be either 2 or 3.
        equilibrium_stress
            The stress on the equilibrium (relaxed) structure.

        """
        eq_stress = None
        if equilibrium_stress:
            eq_stress = -0.1 * Stress(equilibrium_stress)

        strains = [d.green_lagrange_strain for d in deformations]
        stresses = [-0.1 * s for s in stresses]
        pk_stresses = [s.piola_kirchoff_2(d) for s, d in zip(stresses, deformations)]

        if order > 2 or fitting_method == "finite_difference":
            # force finite diff if order > 2
            result = ElasticTensorExpansion.from_diff_fit(
                strains, pk_stresses, eq_stress=eq_stress, order=order
            )
            if order == 2:
                result = ElasticTensor(result[0])
        elif fitting_method == "pseudoinverse":
            result = ElasticTensor.from_pseudoinverse(strains, pk_stresses)
        elif fitting_method == "independent":
            result = ElasticTensor.from_independent_strains(
                strains, pk_stresses, eq_stress=eq_stress
            )
        else:
            raise ValueError(f"Unsupported elastic fitting method {fitting_method}")

        ieee = result.convert_to_ieee(structure)
        property_tensor = ieee if order == 2 else ElasticTensor(ieee[0])
        property_dict = property_tensor.get_structure_property_dict(structure)
        derived_properties = DerivedProperties(**property_dict)

        return cls(
            structure=structure,
            eq_stress=eq_stress,
            derived_properties=derived_properties,
            formula_pretty=structure.composition.reduced_formula,
            fitting_method=fitting_method,
            order=order,
            elastic_tensor=ElasticTensorDocument(
                raw=result.voigt, ieee_format=ieee.voigt
            ),
            fitting_data=FittingData(
                cauchy_stresses=stresses,
                strains=strains,
                pk_stresses=pk_stresses,
                deformations=deformations,
            ),
        )
