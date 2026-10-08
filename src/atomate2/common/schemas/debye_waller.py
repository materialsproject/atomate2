"""Schemas for the Debye-Waller factor workflow outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from emmet.core.phonon import ThermalDisplacementData
from emmet.core.structure import StructureMetadata
from phonopy import Phonopy
from phonopy.units import Bohr, Hartree
from pydantic import Field
from pymatgen.analysis.diffraction import core as diffraction_core
from pymatgen.analysis.diffraction.core import DiffractionPattern
from pymatgen.analysis.diffraction.neutron import NDCalculator
from pymatgen.analysis.diffraction.tem import TEMCalculator
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.io.vasp import Kpoints
from pymatgen.phonon.thermal_displacements import ThermalDisplacementMatrices
from typing_extensions import TypedDict

from atomate2 import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def _get_thermal_displacement_matrices(
    phonon: Phonopy,
    temperatures: Sequence[float],
    freq_min: float = 0.01,
    include_imaginary_modes: bool = False,
) -> np.ndarray:
    """
    Get the thermal displacement matrices of the primitive cell with phonopy.

    phonopy's mesh must be set with eigenvectors and without mesh symmetry.

    Parameters
    ----------
    phonon: Phonopy
        Phonopy object with force constants and a mesh.
    temperatures: Sequence[float]
        Temperatures in K.
    freq_min: float
        Modes with a frequency of magnitude below this value in THz are left
        out. The three acoustic modes at Gamma are always left out.
    include_imaginary_modes: bool
        Also include the modes with frequencies below -freq_min, as if their
        frequency were real with the same magnitude. Each term of phonopy's sum
        is proportional to (n + 1/2) / f. It is even in f, since
        n(-f) = -1 - n(f). So a second run over these modes gives their terms
        with abs(f). phonopy sets n to zero at T <= 1 K. The term is then odd in
        f, so its sign is flipped there.

    Returns
    -------
    np.ndarray
        Cartesian thermal displacement matrices in Angstrom^2, shape
        (temperatures, primitive sites, 3, 3).
    """
    temps = np.asarray(temperatures, dtype=float)
    phonon.run_thermal_displacement_matrices(
        temperatures=temps, freq_min=freq_min, exclude_gamma_acoustic=True
    )
    matrices = phonon.thermal_displacement_matrices.thermal_displacement_matrices
    if include_imaginary_modes:
        phonon.run_thermal_displacement_matrices(
            temperatures=temps,
            freq_min=-np.inf,
            freq_max=-freq_min,
            exclude_gamma_acoustic=True,
        )
        imaginary = phonon.thermal_displacement_matrices.thermal_displacement_matrices
        imaginary[temps <= 1] *= -1
        matrices = matrices + imaginary
    return np.asarray(matrices)


class DebyeWallerDocument(StructureMetadata):
    """Thermal displacements and diffraction patterns with Debye-Waller factors."""

    structure: Structure | None = Field(
        None,
        description="Primitive cell of the phonon calculation. The sites are in the "
        "order of the thermal displacement matrices. The hkl indices of the patterns "
        "and the TEM beam direction refer to this cell.",
    )
    mesh: tuple[int, int, int] | None = Field(
        None,
        description="Gamma-centered q-point mesh of the thermal displacement matrices.",
    )
    include_imaginary_modes: bool | None = Field(
        None,
        description="Whether the modes below -freq_min are included, as if their "
        "frequency were real with the same magnitude.",
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
        "site at each temperature, in Angstrom^2.",
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
        temperatures: Sequence[float],
        mesh: tuple[int, int, int] | float,
        freq_min: float = 0.01,
        include_imaginary_modes: bool = False,
        xrd_kwargs: dict | None = None,
        nd_kwargs: dict | None = None,
        tem_kwargs: dict | None = None,
        symprec: float = SETTINGS.PHONON_SYMPREC,
    ) -> Self:
        """
        Compute the thermal displacements and diffraction patterns of a phonon run.

        Parameters
        ----------
        phonon_doc: PhononDoc
            Output of a phonon flow, with the force constants stored.
        temperatures: Sequence[float]
            Temperatures in K, not negative.
        mesh: tuple[int, int, int] | float
            q-point mesh, or a q-point density used as kppa in pymatgen's
            Kpoints.automatic_density for the primitive cell.
        freq_min: float
            Modes with a frequency of magnitude below this value in THz are left
            out. The three acoustic modes at Gamma are always left out.
        include_imaginary_modes: bool
            Also include the modes below -freq_min, as if their frequency were
            real with the same magnitude.
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
        if min(temperatures) < 0:
            raise ValueError("The temperatures must not be negative.")
        if phonon_doc.force_constants is None:
            raise ValueError(
                "The phonon document has no force constants. Run the phonon flow "
                "with store_force_constants=True."
            )
        phonon = Phonopy(
            get_phonopy_structure(phonon_doc.structure),
            supercell_matrix=phonon_doc.supercell_matrix,
            primitive_matrix=phonon_doc.primitive_matrix,
            symprec=symprec,
        )
        # atomate2 stores ForceConstants, emmet a plain list
        force_constants = phonon_doc.force_constants
        phonon.force_constants = np.array(
            getattr(force_constants, "force_constants", force_constants)
        )
        # _set_nac_params needs the Born charges of the unit cell. The phonon
        # document stores those of the primitive cell, in phonopy's atom order,
        # so NAC is set here. As in _set_nac_params, there is no NAC for zero
        # charges or FHI-aims.
        born = phonon_doc.born
        if (
            born is not None
            and not np.allclose(born, 0.0)
            and phonon_doc.code != "aims"
        ):
            phonon.nac_params = {
                "born": np.array(born),
                "dielectric": np.array(phonon_doc.epsilon_static),
                "factor": Hartree * Bohr,
            }
        structure = get_pmg_structure(phonon.primitive)
        if isinstance(mesh, int | float | np.number):
            kpoints = Kpoints.automatic_density(
                structure=structure, kppa=float(mesh), force_gamma=True
            )
            mesh_numbers = tuple(int(m) for m in kpoints.kpts[0])
        else:
            mesh_numbers = tuple(int(m) for m in mesh)
        # without mesh symmetry, a shifted mesh can break the site symmetry of U
        phonon.run_mesh(
            mesh_numbers,
            with_eigenvectors=True,
            is_mesh_symmetry=False,
            is_gamma_center=True,
        )
        matrices = _get_thermal_displacement_matrices(
            phonon, temperatures, freq_min, include_imaginary_modes
        )

        # the first structure has no U, for the patterns without the factors
        structures = [structure]
        matrices_cif = []
        for u in matrices:
            tdm = ThermalDisplacementMatrices(
                ThermalDisplacementMatrices.get_reduced_matrix(u),
                structure,
                temperature=None,
            )
            matrices_cif.append(tdm.Ucif.tolist())
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
            mesh=mesh_numbers,
            include_imaginary_modes=include_imaginary_modes,
            xrd_kwargs=xrd_kwargs,
            nd_kwargs=nd_kwargs,
            tem_kwargs=tem_kwargs,
            thermal_displacement_data=ThermalDisplacementData(
                freq_min_thermal_displacements=freq_min,
                thermal_displacement_matrix=matrices.tolist(),
                thermal_displacement_matrix_cif=matrices_cif,
                temperatures_thermal_displacements=list(temperatures),
            ),
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
