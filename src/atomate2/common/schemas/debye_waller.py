"""Schemas for the Debye-Waller factor workflow outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from emmet.core.structure import StructureMetadata
from phonopy import Phonopy
from pydantic import Field
from pymatgen.analysis.diffraction import core as diffraction_core
from pymatgen.analysis.diffraction.core import DiffractionPattern
from pymatgen.analysis.diffraction.neutron import NDCalculator
from pymatgen.analysis.diffraction.tem import TEMCalculator
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.phonon.thermal_displacements import ThermalDisplacementMatrices
from typing_extensions import TypedDict

from atomate2 import SETTINGS
from atomate2.common.schemas.phonons import ThermalDisplacementData

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import Self

    from atomate2.common.jobs.gruneisen import PhononDoc


def check_pymatgen() -> None:
    """Check that pymatgen has anisotropic Debye-Waller factors.

    Released pymatgen ignores the U11_cif, ..., U12_cif site properties, so its
    patterns would have no Debye-Waller factors.
    """
    if not hasattr(diffraction_core, "get_anisotropic_debye_waller_factors"):
        raise ImportError(
            "This pymatgen version has no anisotropic Debye-Waller factors. See the "
            "Debye-Waller workflow section of the VASP page in the atomate2 docs to "
            "install one."
        )


class TEMSpot(TypedDict):
    """Spot of an electron diffraction pattern from pymatgen's TEMCalculator."""

    position: list[float]
    hkl: list[int]
    intensity: float


class DebyeWallerDocument(StructureMetadata):
    """Thermal displacements and diffraction patterns with Debye-Waller factors."""

    structure: Structure | None = Field(
        None,
        description="Primitive cell of the phonon calculation. The sites are in the "
        "order of the thermal displacement matrices. The hkl indices of the patterns "
        "and the TEM beam direction refer to this cell.",
    )
    xrd_kwargs: dict[str, Any] | None = Field(
        None, description="Keyword arguments of pymatgen's XRDCalculator."
    )
    nd_kwargs: dict[str, Any] | None = Field(
        None, description="Keyword arguments of pymatgen's NDCalculator."
    )
    tem_kwargs: dict[str, Any] | None = Field(
        None, description="Keyword arguments of pymatgen's TEMCalculator."
    )
    thermal_displacement_data: ThermalDisplacementData | None = Field(
        None,
        description="Cartesian and CIF thermal displacement matrices U of each "
        "site at each temperature, in Angstrom^2, from the phonon flow.",
    )
    xrd_pattern_static: DiffractionPattern | None = Field(
        None,
        description="X-ray diffraction pattern without Debye-Waller factors, with "
        "x in degrees 2 theta. The intensities are not scaled.",
    )
    xrd_patterns: list[DiffractionPattern] | None = Field(
        None,
        description="X-ray diffraction pattern at each temperature, with the "
        "Debye-Waller factor exp(-2 pi^2 g^T U g) of each site. The intensities "
        "are not scaled.",
    )
    nd_pattern_static: DiffractionPattern | None = Field(
        None,
        description="Neutron diffraction pattern without Debye-Waller factors, with "
        "x in degrees 2 theta. The intensities are not scaled.",
    )
    nd_patterns: list[DiffractionPattern] | None = Field(
        None,
        description="Neutron diffraction pattern at each temperature, with the "
        "Debye-Waller factor of each site. The intensities are not scaled.",
    )
    tem_pattern_static: list[TEMSpot] | None = Field(
        None,
        description="Electron diffraction spots without Debye-Waller factors, with "
        "the position, hkl and intensity of each spot from pymatgen's "
        "TEMCalculator.get_pattern. The intensities are normalized to the "
        "strongest spot.",
    )
    tem_patterns: list[list[TEMSpot]] | None = Field(
        None,
        description="Electron diffraction spots at each temperature, with the "
        "Debye-Waller factor of each site. The intensities are normalized to the "
        "strongest spot of each pattern.",
    )

    @classmethod
    def from_phonon_doc(
        cls,
        phonon_doc: PhononDoc,
        xrd_kwargs: dict | None = None,
        nd_kwargs: dict | None = None,
        tem_kwargs: dict | None = None,
        symprec: float = SETTINGS.PHONON_SYMPREC,
    ) -> Self:
        """
        Compute the diffraction patterns of a phonon run with Debye-Waller factors.

        Parameters
        ----------
        phonon_doc: PhononDoc
            Output of a phonon flow, with the thermal displacement matrices.
        xrd_kwargs: dict or None
            Keyword arguments of pymatgen's XRDCalculator.
        nd_kwargs: dict or None
            Keyword arguments of pymatgen's NDCalculator.
        tem_kwargs: dict or None
            Keyword arguments of pymatgen's TEMCalculator.
        symprec: float
            Symmetry precision of the phonon flow.

        Returns
        -------
        DebyeWallerDocument
        """
        check_pymatgen()
        data = phonon_doc.thermal_displacement_data
        if data is None:
            raise ValueError(
                "The phonon document has no thermal displacement matrices. Run the "
                "phonon flow with create_thermal_displacements=True. No matrices are "
                "stored if its q-point mesh has imaginary modes."
            )
        # the matrices are those of phonopy's primitive cell
        phonon = Phonopy(
            get_phonopy_structure(phonon_doc.structure),
            supercell_matrix=phonon_doc.supercell_matrix,
            primitive_matrix=phonon_doc.primitive_matrix,
            symprec=symprec,
        )
        structure = get_pmg_structure(phonon.primitive)

        # the first structure has no U, for the patterns without the factors
        structures = [structure]
        for u in data.thermal_displacement_matrix:
            tdm = ThermalDisplacementMatrices(
                ThermalDisplacementMatrices.get_reduced_matrix(np.array(u)),
                structure,
                temperature=None,
            )
            structures.append(tdm.to_structure_with_site_properties_Ucif())
        xrd = XRDCalculator(**(xrd_kwargs or {}))
        nd = NDCalculator(**(nd_kwargs or {}))
        tem = TEMCalculator(**(tem_kwargs or {}))
        xrd_patterns = [xrd.get_pattern(s, scaled=False) for s in structures]
        nd_patterns = [nd.get_pattern(s, scaled=False) for s in structures]
        tem_patterns = [_tem_rows(tem.get_pattern(s)) for s in structures]

        return cls.from_structure(
            meta_structure=structure,
            structure=structure,
            xrd_kwargs=xrd_kwargs,
            nd_kwargs=nd_kwargs,
            tem_kwargs=tem_kwargs,
            thermal_displacement_data=data.model_dump(),
            xrd_pattern_static=xrd_patterns[0],
            xrd_patterns=xrd_patterns[1:],
            nd_pattern_static=nd_patterns[0],
            nd_patterns=nd_patterns[1:],
            tem_pattern_static=tem_patterns[0],
            tem_patterns=tem_patterns[1:],
        )


def _tem_rows(pattern: pd.DataFrame) -> list[TEMSpot]:
    """Convert pymatgen's TEM data frame into rows of plain values."""
    return [
        {
            "position": np.asarray(row["Position"]).tolist(),
            "hkl": [int(i) for i in row["(hkl)"]],
            "intensity": float(row["Intensity (norm)"]),
        }
        for _, row in pattern.iterrows()
    ]
