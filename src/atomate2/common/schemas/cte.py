"""Schemas for the thermal expansion workflow outputs."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from emmet.core.math import Matrix3D, MatrixVoigt
from emmet.core.structure import StructureMetadata
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_pmg_structure
from pymatgen.io.vasp import Kpoints
from scipy.constants import Boltzmann, Planck

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from phonopy import Phonopy
    from typing_extensions import Self

# Voigt order of the symmetric 3x3 tensor components, as in pymatgen
_VOIGT_INDICES = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def get_cte(
    frequencies: np.ndarray,
    gruneisen_tensors: np.ndarray,
    weights: np.ndarray,
    elastic_tensor: np.ndarray,
    volume: float,
    temperatures: Sequence[float],
    min_frequency: float = 1e-3,
) -> tuple[np.ndarray, list[np.ndarray | None]]:
    """
    Get the thermal expansion tensor from mode Grueneisen tensors.

    The thermal stress of each mode is its heat capacity times its Grueneisen
    tensor. The strain that relaxes the summed thermal stress is the thermal
    expansion, alpha = S sum(c * gamma) / (N_q * V). Here S is the elastic
    compliance, c are the modal heat capacities, N_q is the sum of the q-point
    weights and V is the cell volume. The mode Grueneisen tensors are
    symmetrized first, since the strain is symmetric. Modes below
    min_frequency, including the acoustic modes at Gamma, are left out.

    Parameters
    ----------
    frequencies: np.ndarray
        Phonon frequencies in THz, with shape (n_qpoints, n_bands).
    gruneisen_tensors: np.ndarray
        Mode Grueneisen tensors, with shape (n_qpoints, n_bands, 3, 3).
    weights: np.ndarray
        Weight of each q-point, with shape (n_qpoints,).
    elastic_tensor: np.ndarray
        Elastic tensor in GPa and Voigt notation, in the same Cartesian frame as
        the Grueneisen tensors.
    volume: float
        Volume of the cell used for the phonons, in Angstrom^3.
    temperatures: Sequence[float]
        Temperatures in K.
    min_frequency: float
        Modes below this frequency in THz are left out.

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray | None]]
        The thermal expansion tensors in 1/K, with shape (n_temperatures, 3, 3),
        and the heat-capacity weighted mean Grueneisen tensor at each
        temperature. The mean is None where the heat capacity is zero.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    gruneisen_tensors = np.asarray(gruneisen_tensors, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # S in 1/Pa and V in m^3
    compliance = np.linalg.inv(np.asarray(elastic_tensor, dtype=float) * 1e9)
    volume_m3 = volume * 1e-30

    # the Grueneisen tensors of the left-out modes can be large or undefined,
    # so zero them
    kept = frequencies > min_frequency
    gruneisen_tensors = np.where(kept[..., None, None], gruneisen_tensors, 0.0)
    gruneisen_tensors = (gruneisen_tensors + np.swapaxes(gruneisen_tensors, -1, -2)) / 2
    gruneisen_voigt = np.stack(
        [gruneisen_tensors[..., i, j] for i, j in _VOIGT_INDICES], axis=-1
    )
    energies = Planck * frequencies * 1e12  # J

    alphas, mean_gruneisen = [], []
    for temperature in temperatures:
        if temperature <= 0:
            heat_capacities = np.zeros_like(frequencies)
        else:
            x = np.where(kept, energies / (Boltzmann * temperature), 1.0)
            # x^2 e^x / (e^x - 1)^2, written with e^-x so that it cannot overflow
            heat_capacities = np.where(
                kept, Boltzmann * x**2 * np.exp(-x) / np.expm1(-x) ** 2, 0.0
            )  # J/K
        weighted = heat_capacities * weights[:, None]
        thermal_stress = np.einsum("qb,qbv->v", weighted, gruneisen_voigt)
        thermal_stress /= weights.sum() * volume_m3  # Pa/K

        # the compliance gives engineering shear strains, twice the tensor ones
        alpha_voigt = compliance @ thermal_stress
        alpha = np.empty((3, 3))
        for k, (i, j) in enumerate(_VOIGT_INDICES):
            alpha[i, j] = alpha[j, i] = alpha_voigt[k] if k < 3 else alpha_voigt[k] / 2
        alphas.append(alpha)

        total_heat_capacity = weighted.sum()
        if total_heat_capacity > 0:
            mean = np.einsum("qb,qbij->ij", weighted, gruneisen_tensors)
            mean_gruneisen.append(mean / total_heat_capacity)
        else:
            mean_gruneisen.append(None)

    return np.array(alphas), mean_gruneisen


def _expand_born_to_unitcell(phonon: Phonopy) -> np.ndarray:
    """Map the Born charges of the phonopy primitive cell onto the unit cell atoms."""
    primitive = phonon.primitive
    born = np.asarray(phonon.nac_params["born"])
    if len(primitive) == len(phonon.unitcell):
        return born
    primitive_indices = [
        primitive.p2p_map[primitive.s2p_map[s]] for s in phonon.supercell.u2s_map
    ]
    return born[primitive_indices]


class CTEResult(BaseModel):
    """Thermal expansion from one set of second- and third-order force constants."""

    fit_method: str = Field(
        description='Anharmonic fit that gave the force constants, "cocktail" or '
        '"one-shot".'
    )
    lowest_frequency: float = Field(
        description="Lowest phonon frequency on the sampling mesh in THz. Imaginary "
        "frequencies are negative."
    )
    has_imaginary_modes: bool = Field(
        description="Whether a frequency on the sampling mesh lies below "
        "-tol_imaginary_modes. The thermal expansion is not computed in that case."
    )
    thermal_expansion: list[Matrix3D] | None = Field(
        None,
        description="Thermal expansion tensor in 1/K at each temperature, in the "
        "Cartesian frame of the structure.",
    )
    volumetric_thermal_expansion: list[float] | None = Field(
        None,
        description="Volumetric thermal expansion in 1/K at each temperature, the "
        "trace of the thermal expansion tensor.",
    )
    average_gruneisen: list[Matrix3D | None] | None = Field(
        None,
        description="Mode Grueneisen tensor averaged with the mode heat capacities "
        "at each temperature. None where the heat capacity is zero.",
    )


class CTEDocument(StructureMetadata):
    """Thermal expansion from mode Grueneisen tensors and the elastic tensor."""

    structure: Structure | None = Field(
        None, description="Structure used for the phonon and elastic calculations."
    )
    temperatures: list[float] | None = Field(None, description="Temperatures in K.")
    mesh: tuple[int, int, int] | None = Field(
        None, description="q-point mesh used for the mode Grueneisen tensors."
    )
    elastic_tensor: MatrixVoigt | None = Field(
        None,
        description="Elastic tensor in GPa and Voigt notation, in the Cartesian "
        "frame of the structure.",
    )
    min_frequency: float | None = Field(
        None,
        description="Modes below this frequency in THz are left out of the thermal "
        "expansion.",
    )
    tol_imaginary_modes: float | None = Field(
        None,
        description="The thermal expansion of a fit is not computed if a frequency "
        "on the mesh is below -tol_imaginary_modes in THz.",
    )
    phonon_job_dir: str | None = Field(
        None, description="Directory of the pheasy fit that wrote the force constants."
    )
    results: list[CTEResult] | None = Field(
        None, description="Thermal expansion for each anharmonic fit method."
    )

    @classmethod
    def from_force_constants(
        cls,
        phonopy_yaml: str | Path,
        force_constant_files: Mapping[str, tuple[str | Path, str | Path]],
        elastic_tensor: MatrixVoigt,
        structure: Structure,
        temperatures: Sequence[float],
        mesh: tuple[int, int, int] | float,
        tol_imaginary_modes: float,
        min_frequency: float,
        symprec: float,
    ) -> Self:
        """
        Compute the thermal expansion from second- and third-order force constants.

        phono3py gives the frequencies and mode Grueneisen tensors on the q-point
        mesh. Only time reversal symmetry is used to reduce the mesh. The
        non-analytical term correction is applied to the dynamical matrix when
        the phonopy yaml file holds the Born charges and the dielectric tensor.
        Only the third-order force constants enter the strain derivative of the
        dynamical matrix. The results of each fit are written to
        gruneisen_<method>.hdf5 in the current directory.

        Parameters
        ----------
        phonopy_yaml: str or Path
            phonopy.yaml of the pheasy fit, with the unit cell, the supercell
            matrix and the Born charges.
        force_constant_files: Mapping
            For each fit method, "cocktail" or "one-shot", the files with the
            second- and third-order force constants.
        elastic_tensor: MatrixVoigt
            Elastic tensor in GPa and Voigt notation.
        structure: Structure
            Structure of the elastic calculation. Its lattice must match the
            unit cell in phonopy_yaml, so that both tensors are in the same frame.
        temperatures: Sequence[float]
            Temperatures in K, not negative.
        mesh: tuple[int, int, int] or float
            q-point mesh, or a q-point density used as kppa in pymatgen's
            Kpoints.automatic_density for the unit cell.
        tol_imaginary_modes: float
            If a frequency on the mesh is below -tol_imaginary_modes in THz, a
            warning is raised and the thermal expansion of that fit is not
            computed.
        min_frequency: float
            Modes below this frequency in THz are left out.
        symprec: float
            Symmetry precision passed to phono3py.

        Returns
        -------
        CTEDocument
        """
        import h5py
        import phonopy
        from phono3py import Phono3py
        from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5
        from phono3py.phonon3.gruneisen import Gruneisen
        from phonopy.file_IO import parse_FORCE_CONSTANTS

        if not force_constant_files:
            raise ValueError("No force constant files were given.")
        if min(temperatures) < 0:
            raise ValueError("The temperatures must not be negative.")

        phonon = phonopy.load(phonopy_yaml, produce_fc=False, log_level=0)
        if not np.allclose(phonon.unitcell.cell, structure.lattice.matrix, atol=1e-5):
            raise ValueError(
                "The lattice of the elastic calculation differs from the unit cell "
                "of the phonon calculation, so the two tensors are not in the same "
                "frame."
            )
        if list(phonon.unitcell.symbols) != [site.specie.symbol for site in structure]:
            raise ValueError(
                "The atoms of the elastic calculation differ from the unit cell of "
                "the phonon calculation."
            )

        # pheasy writes compact force constants for the unit cell in POSCAR, so
        # the unit cell is also the primitive cell here. "P" keeps it as it is.
        ph3 = Phono3py(
            phonon.unitcell,
            supercell_matrix=phonon.supercell_matrix,
            primitive_matrix="P",
            symprec=symprec,
        )
        nac_params = None
        if phonon.nac_params is not None:
            nac_params = {
                **phonon.nac_params,
                "born": _expand_born_to_unitcell(phonon),
            }

        if isinstance(mesh, int | float | np.number):
            kpoints = Kpoints.automatic_density(
                structure=get_pmg_structure(ph3.primitive),
                kppa=float(mesh),
                force_gamma=True,
            )
            mesh_numbers = tuple(int(m) for m in kpoints.kpts[0])
        else:
            mesh_numbers = tuple(int(m) for m in mesh)

        results = []
        for method, (fc2_file, fc3_file) in force_constant_files.items():
            if Path(fc2_file).suffix == ".hdf5":
                fc2 = read_fc2_from_hdf5(fc2_file)
            else:
                fc2 = parse_FORCE_CONSTANTS(filename=fc2_file)
            fc3 = read_fc3_from_hdf5(fc3_file)

            gruneisen = Gruneisen(
                fc2, fc3, ph3.supercell, ph3.primitive, nac_params=nac_params
            )
            gruneisen.set_sampling_mesh(mesh_numbers, primitive_symmetry=None)
            gruneisen.run()
            filename = f"gruneisen_{method.replace('-', '_')}"
            gruneisen.write(filename=filename)
            # the full third-order force constants can take several GB
            del gruneisen, fc3
            with h5py.File(f"{filename}.hdf5") as file:
                frequencies = file["frequency"][:]
                gruneisen_tensors = file["gruneisen_tensor"][:]
                weights = file["weight"][:]

            lowest = float(frequencies.min())
            has_imaginary_modes = lowest < -tol_imaginary_modes
            result = {
                "fit_method": method,
                "lowest_frequency": lowest,
                "has_imaginary_modes": has_imaginary_modes,
            }
            if has_imaginary_modes:
                warnings.warn(
                    f"The {method} force constants give a frequency of "
                    f"{lowest:.3f} THz, below -{tol_imaginary_modes} THz. The "
                    "thermal expansion is not computed for this fit.",
                    stacklevel=2,
                )
            else:
                alphas, mean_gruneisen = get_cte(
                    frequencies,
                    gruneisen_tensors,
                    weights,
                    np.asarray(elastic_tensor),
                    ph3.primitive.volume,
                    temperatures,
                    min_frequency=min_frequency,
                )
                result["thermal_expansion"] = alphas.tolist()
                result["volumetric_thermal_expansion"] = np.trace(
                    alphas, axis1=1, axis2=2
                ).tolist()
                result["average_gruneisen"] = [
                    None if mean is None else mean.tolist() for mean in mean_gruneisen
                ]
            results.append(CTEResult(**result))

        return cls.from_structure(
            meta_structure=structure,
            structure=structure,
            temperatures=list(temperatures),
            mesh=mesh_numbers,
            elastic_tensor=np.asarray(elastic_tensor).tolist(),
            min_frequency=min_frequency,
            tol_imaginary_modes=tol_imaginary_modes,
            phonon_job_dir=str(Path(phonopy_yaml).parent),
            results=results,
        )
