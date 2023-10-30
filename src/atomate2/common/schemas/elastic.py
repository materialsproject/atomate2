"""Schemas for elastic tensor fitting and related properties."""
from copy import deepcopy
from typing import Optional

import numpy as np
from emmet.core.math import Matrix3D, MatrixVoigt
from emmet.core.structure import StructureMetadata
from pydantic import BaseModel, Field
from pymatgen.analysis.elasticity import (
    Deformation,
    ElasticTensor,
    ElasticTensorExpansion,
    Strain,
    Stress,
)
from pymatgen.core import Structure
from pymatgen.core.tensors import TensorMapping
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2 import SETTINGS


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
        description="Young's modulus (SI units) from the Voight-Reuss-Hill averages of "
        "the bulk and shear moduli.",
    )
    trans_v: float = Field(
        None,
        description="Transverse sound velocity (SI units) obtained from the "
        "Voigt-Reuss-Hill average bulk modulus.",
    )
    long_v: float = Field(
        None,
        description="Longitudinal sound velocity (SI units) obtained from the "
        "Voigt-Reuss-Hill average bulk modulus.",
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
    clark_thermalcond: Optional[float] = Field(
        None, description="Clarke's thermal conductivity (SI units)."
    )
    cahill_thermalcond: float = Field(
        None, description="Cahill's thermal conductivity (SI units)."
    )
    debye_temperature: float = Field(
        None,
        description="Debye temperature from longitudinal and transverse sound "
        "velocities (SI units).",
    )


class FittingData(BaseModel):
    """Data used to fit elastic tensors."""

    cauchy_stresses: list[Matrix3D] = Field(
        None, description="The Cauchy stresses used to fit the elastic tensor."
    )
    strains: list[Matrix3D] = Field(
        None, description="The strains used to fit the elastic tensor."
    )
    pk_stresses: list[Matrix3D] = Field(
        None, description="The Piola-Kirchoff stresses used to fit the elastic tensor."
    )
    deformations: list[Matrix3D] = Field(
        None, description="The deformations corresponding to each strain state."
    )
    uuids: list[str] = Field(None, description="The uuids of the deformation jobs.")
    job_dirs: list[Optional[str]] = Field(
        None, description="The directories where the deformation jobs were run."
    )


class ElasticTensorDocument(BaseModel):
    """Raw and standardized elastic tensors."""

    raw: MatrixVoigt = Field(None, description="Raw elastic tensor.")
    ieee_format: MatrixVoigt = Field(None, description="Elastic tensor in IEEE format.")


class ElasticDocument(StructureMetadata):
    """Document containing elastic tensor information and related properties."""

    structure: Structure = Field(
        None, description="The structure for which the elastic data is calculated."
    )
    elastic_tensor: ElasticTensorDocument = Field(
        None, description="Fitted elastic tensor."
    )
    eq_stress: Optional[Matrix3D] = Field(
        None, description="The equilibrium stress of the structure."
    )
    derived_properties: DerivedProperties = Field(
        None, description="Properties derived from the elastic tensor."
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
        stresses: list[Stress],
        deformations: list[Deformation],
        uuids: list[str],
        job_dirs: list[str],
        fitting_method: str = SETTINGS.ELASTIC_FITTING_METHOD,
        order: Optional[int] = None,
        equilibrium_stress: Optional[Matrix3D] = None,
        symprec: float = SETTINGS.SYMPREC,
        allow_elastically_unstable_structs: bool = True,
    ) -> "ElasticDocument":
        """
        Create an elastic document from strains and stresses.

        Parameters
        ----------
        structure : .Structure
            The structure for which strains and stresses were calculated.
        stresses : list of Stress
            A list of corresponding stresses.
        deformations : list of Deformation
            A list of corresponding deformations.
        uuids: list of str
            A list of uuids, one for each deformation calculation.
        job_dirs : list of str
            A list of job directories, one for each deformation calculation.
        fitting_method : str
            The method used to fit the elastic tensor. See pymatgen for more details on
            the methods themselves. The options are:
            - "finite_difference" (note this is required if fitting a 3rd order tensor)
            - "independent"
            - "pseudoinverse"
        order : int or None
            Order of the tensor expansion to be fitted. Can be either 2 or 3.
        equilibrium_stress : list of list of float
            The stress on the equilibrium (relaxed) structure.
        symprec : float
            Symmetry precision for deriving symmetry equivalent deformations. If
            ``symprec=None``, then no symmetry operations will be applied.
        allow_elastically_unstable_structs : bool
            Whether to allow the ElasticDocument to still complete in the event that
            the structure is elastically unstable.
        """
        strains = [d.green_lagrange_strain for d in deformations]

        if symprec is not None:
            strains, stresses, uuids, job_dirs = _expand_strains(
                structure, strains, stresses, uuids, job_dirs, symprec
            )

        deformations = [s.get_deformation_matrix() for s in strains]

        # -0.1 to convert units from kBar to GPa and stress direction
        stresses = [-0.1 * s for s in stresses]
        eq_stress = None
        if equilibrium_stress:
            eq_stress = -0.1 * Stress(equilibrium_stress)

        pk_stresses = [s.piola_kirchoff_2(d) for s, d in zip(stresses, deformations)]

        if order is None:
            order = 2 if len(stresses) < 70 else 3  # TODO: Figure this out better

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
            raise ValueError(f"Unsupported elastic {fitting_method=}")

        ieee = result.convert_to_ieee(structure)
        property_tensor = ieee if order == 2 else ElasticTensor(ieee[0])

        ignore_errors = bool(allow_elastically_unstable_structs)
        property_dict = property_tensor.get_structure_property_dict(
            structure, ignore_errors=ignore_errors
        )

        derived_properties = DerivedProperties(**property_dict)

        eq_stress = eq_stress.tolist() if eq_stress is not None else eq_stress

        return cls.from_structure(
            structure=structure,
            meta_structure=structure,
            eq_stress=eq_stress,
            derived_properties=derived_properties,
            fitting_method=fitting_method,
            order=order,
            elastic_tensor=ElasticTensorDocument(
                raw=result.voigt.tolist(), ieee_format=ieee.voigt.tolist()
            ),
            fitting_data=FittingData(
                cauchy_stresses=[s.tolist() for s in stresses],
                strains=[s.tolist() for s in strains],
                pk_stresses=[p.tolist() for p in pk_stresses],
                deformations=[d.tolist() for d in deformations],
                uuids=uuids,
                job_dirs=job_dirs,
            ),
        )


def _expand_strains(
    structure: Structure,
    strains: list[Strain],
    stresses: list[Stress],
    uuids: list[str],
    job_dirs: list[str],
    symprec: float,
    tol: float = 1e-3,
) -> tuple[list, list, list[str], list[str]]:
    """
    Use symmetry to expand strains.

    Args:
         tol: tolerance to determine if a strain component is zero. This should be
            smaller than the smallest strain magnitude used to deform the structure.

    Warning:
        This function assumes that each deformed structure is generated from strain
        state with only one non-zero component. If this is not the case, the expanded
        strains will not contain the ones with other strain states. Also see:
        `generate_elastic_deformations()`.
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    symmops = sga.get_symmetry_operations(cartesian=True)

    full_strains = deepcopy(strains)
    full_stresses = deepcopy(stresses)
    full_uuids = deepcopy(uuids)
    full_job_dirs = deepcopy(job_dirs)

    mapping = TensorMapping(full_strains, [True for _ in full_strains])
    for i, strain in enumerate(strains):
        for symmop in symmops:
            rotated_strain = strain.transform(symmop)

            # check if we have more than one perturbed strain component
            if sum(np.abs(rotated_strain.voigt) > tol) > 1:
                continue

            # check if we have seen it before
            if rotated_strain in mapping:
                continue

            # store the rotated strain so we know we've seen it
            mapping[rotated_strain] = True

            # expand the other properties
            full_strains.append(rotated_strain)
            full_stresses.append(stresses[i].transform(symmop))
            full_uuids.append(uuids[i])
            full_job_dirs.append(job_dirs[i])

    return full_strains, full_stresses, full_uuids, full_job_dirs
