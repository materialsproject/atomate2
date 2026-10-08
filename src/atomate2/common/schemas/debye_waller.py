"""Schemas for the Debye-Waller factor workflow outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from emmet.core.math import Matrix3D
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
from pymatgen.io.vasp import Kpoints

from atomate2.common.schemas.phonons import _set_nac_params

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from atomate2.common.jobs.gruneisen import PhononDoc


def get_thermal_displacement_matrices(
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
        Modes below this frequency in THz are left out. This removes the
        acoustic modes at Gamma.
    include_imaginary_modes: bool
        Also include the modes with frequencies below -freq_min, as if their
        frequency were real with the same magnitude. phonopy's sums are odd in
        the frequency, so a second run over these modes gives their terms.
        phonopy sets the phonon population to zero at T <= 1 K, where it is -1
        for a negative frequency, so the sign of those terms is flipped there.

    Returns
    -------
    np.ndarray
        Cartesian thermal displacement matrices in Angstrom^2, shape
        (temperatures, primitive sites, 3, 3).
    """
    temps = np.asarray(temperatures, dtype=float)
    phonon.run_thermal_displacement_matrices(temperatures=temps, freq_min=freq_min)
    matrices = phonon.thermal_displacement_matrices.thermal_displacement_matrices
    if include_imaginary_modes:
        phonon.run_thermal_displacement_matrices(
            temperatures=temps, freq_min=-np.inf, freq_max=-freq_min
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
        "order of the thermal displacement matrices.",
    )
    temperatures: list[float] | None = Field(None, description="Temperatures in K.")
    mesh: tuple[int, int, int] | None = Field(
        None,
        description="Gamma-centered q-point mesh of the thermal displacement matrices.",
    )
    freq_min: float | None = Field(
        None, description="Modes below this frequency in THz are left out."
    )
    include_imaginary_modes: bool | None = Field(
        None,
        description="Whether the modes below -freq_min are included, as if their "
        "frequency were real with the same magnitude.",
    )
    has_imaginary_modes: bool | None = Field(
        None, description="Whether a frequency on the mesh is below -freq_min."
    )
    thermal_displacement_matrices: list[list[Matrix3D]] | None = Field(
        None,
        description="Cartesian thermal displacement matrices U in Angstrom^2 of "
        "each site at each temperature.",
    )
    xrd_pattern_static: DiffractionPattern | None = Field(
        None, description="X-ray diffraction pattern without Debye-Waller factors."
    )
    xrd_patterns: list[DiffractionPattern] | None = Field(
        None,
        description="X-ray diffraction pattern at each temperature, with the "
        "Debye-Waller factor exp(-2 pi^2 g^T U g) of each site.",
    )
    nd_pattern_static: DiffractionPattern | None = Field(
        None, description="Neutron diffraction pattern without Debye-Waller factors."
    )
    nd_patterns: list[DiffractionPattern] | None = Field(
        None,
        description="Neutron diffraction pattern at each temperature, with the "
        "Debye-Waller factor of each site.",
    )
    tem_pattern_static: list[dict[str, Any]] | None = Field(
        None,
        description="Electron diffraction spots without Debye-Waller factors, one "
        "row of pymatgen's TEMCalculator.get_pattern per spot.",
    )
    tem_patterns: list[list[dict[str, Any]]] | None = Field(
        None,
        description="Electron diffraction spots at each temperature, with the "
        "Debye-Waller factor of each site.",
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
        symprec: float = 1e-4,
    ) -> Self:
        """
        Compute the thermal displacements and diffraction patterns of a phonon run.

        Parameters
        ----------
        phonon_doc: PhononDoc
            Output of a phonopy or pheasy phonon flow, with the force constants
            stored.
        temperatures: Sequence[float]
            Temperatures in K.
        mesh: tuple[int, int, int] | float
            q-point mesh, or a q-point density used as kppa in pymatgen's
            Kpoints.automatic_density for the primitive cell.
        freq_min: float
            Modes below this frequency in THz are left out.
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
            Symmetry precision for the Born charges.

        Returns
        -------
        DebyeWallerDocument
        """
        # released pymatgen ignores the thermal_displacement_matrix site property
        if not hasattr(diffraction_core, "get_anisotropic_debye_waller_factors"):
            raise ImportError(
                "This pymatgen version has no anisotropic Debye-Waller factors. "
                "Install the debye-waller dependency group of atomate2."
            )
        if phonon_doc.force_constants is None:
            raise ValueError(
                "The phonon document has no force constants. Run the phonon flow "
                "with store_force_constants=True."
            )
        phonon = Phonopy(
            get_phonopy_structure(phonon_doc.structure),
            supercell_matrix=phonon_doc.supercell_matrix,
            primitive_matrix=phonon_doc.primitive_matrix,
        )
        # atomate2 stores ForceConstants, emmet a plain list
        force_constants = phonon_doc.force_constants
        phonon.force_constants = np.array(
            getattr(force_constants, "force_constants", force_constants)
        )
        _set_nac_params(
            phonon,
            phonon_doc.born,
            phonon_doc.epsilon_static,
            symprec,
            phonon_doc.code,
        )
        structure = get_pmg_structure(phonon.primitive)
        if isinstance(mesh, int | float | np.number):
            kpoints = Kpoints.automatic_density(
                structure=structure, kppa=float(mesh), force_gamma=True
            )
            mesh_numbers = tuple(int(m) for m in kpoints.kpts[0])
        else:
            mesh_numbers = tuple(int(m) for m in mesh)
        # a shifted mesh breaks the symmetry of non-cubic reciprocal cells
        phonon.run_mesh(
            mesh_numbers,
            with_eigenvectors=True,
            is_mesh_symmetry=False,
            is_gamma_center=True,
        )
        has_imaginary_modes = bool(phonon.mesh.frequencies.min() < -freq_min)
        matrices = get_thermal_displacement_matrices(
            phonon, temperatures, freq_min, include_imaginary_modes
        )

        calculators = {
            "xrd": XRDCalculator(**(xrd_kwargs or {})),
            "nd": NDCalculator(**(nd_kwargs or {})),
            "tem": TEMCalculator(**(tem_kwargs or {})),
        }
        patterns: dict[str, list] = {name: [] for name in calculators}
        for u in [None, *matrices]:
            displaced = structure.copy()
            if u is not None:
                displaced.add_site_property("thermal_displacement_matrix", list(u))
            for name, calculator in calculators.items():
                if name == "tem":
                    pattern = calculator.get_pattern(displaced)
                    patterns[name].append(_tem_rows(pattern))
                else:
                    patterns[name].append(
                        calculator.get_pattern(displaced, scaled=False)
                    )

        return cls.from_structure(
            meta_structure=structure,
            structure=structure,
            temperatures=list(temperatures),
            mesh=mesh_numbers,
            freq_min=freq_min,
            include_imaginary_modes=include_imaginary_modes,
            has_imaginary_modes=has_imaginary_modes,
            thermal_displacement_matrices=matrices.tolist(),
            xrd_pattern_static=patterns["xrd"][0],
            xrd_patterns=patterns["xrd"][1:],
            nd_pattern_static=patterns["nd"][0],
            nd_patterns=patterns["nd"][1:],
            tem_pattern_static=patterns["tem"][0],
            tem_patterns=patterns["tem"][1:],
        )


def _tem_rows(pattern: Any) -> list[dict[str, Any]]:
    """Convert pymatgen's TEM data frame into rows of plain values."""
    return [
        {
            "position": np.asarray(row["Position"]).tolist(),
            "hkl": [int(i) for i in row["(hkl)"]],
            "intensity": float(row["Intensity (norm)"]),
            "film_radius": float(row["Film radius"]),
            "interplanar_spacing": float(row["Interplanar Spacing"]),
        }
        for _, row in pattern.iterrows()
    ]
