"""Schemas for electron-phonon renormalisation documents."""

import logging

import numpy as np
from emmet.core.structure import StructureMetadata
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.core import Spin
from typing_extensions import Self

logger = logging.getLogger(__name__)


class RawElectronicData(BaseModel):
    """Raw data used to fit electron-phonon renormalisation."""

    displacement_uuids: list[str] = Field(
        None, description="UUIDs of the displacement band structure calculations"
    )
    displacement_dirs: list[str] = Field(
        None, description="Directories of the displacement band structure calculations"
    )
    displacement_structures: list[Structure] = Field(
        None, description="The electron-phonon displaced structures at each temperature"
    )
    displacement_cbms: list[list[float]] = Field(
        None,
        description="Conduction band minima of the displaced structures, as an"
        "array with the shape (ntemps, ncbms)",
    )
    displacement_vbms: list[list[float]] = Field(
        None,
        description="Valence band maxima of the displaced structures, as an"
        "array with the shape (ntemps, nvbms)",
    )
    bulk_uuid: str = Field(None, description="UUID of the bulk (supercell) calculation")
    bulk_dir: str = Field(
        None, description="Directory of the bulk (supercell) calculation"
    )
    bulk_structure: Structure = Field(
        None, description="The bulk (supercell) structure"
    )
    bulk_cbm: float = Field(
        None, description="Conduction band minimum of the bulk (supercell) structure"
    )
    bulk_vbm: float = Field(
        None, description="Valence band maximum of the bulk (supercell) structure"
    )
    bulk_vbm_band_indices: dict[str, list[int]] = Field(
        None,
        description="Indices of bands that are degenerate at the valence band "
        "maximum (zero indexed) in the bulk (supercell) structure",
    )
    bulk_cbm_band_indices: dict[str, list[int]] = Field(
        None,
        description="Indices of bands that are degenerate at the conduction band "
        "minimum (zero indexed) in the bulk (supercell) structure",
    )
    elph_uuid: str = Field(
        None,
        description="UUID of the electron-phonon calculation that generated the "
        "displaced supercells",
    )
    elph_dir: str = Field(
        None,
        description="Directory of the electron-phonon calculation that generated "
        "the displaced supercells",
    )


class ElectronPhononRenormalisationDoc(StructureMetadata):
    """Electron-phonon band gap renormalisation document."""

    structure: Structure = Field(
        None,
        description="The primitive structure for which the electron-phonon was"
        " calculated",
    )
    temperatures: list[float] = Field(
        None, description="Temperatures at which electron-phonon coupling was obtained"
    )
    band_gaps: list[float] = Field(
        None, description="Temperature renormalised band gaps"
    )
    vbms: list[float] = Field(
        None, description="Temperature renormalised valence band maxima"
    )
    cbms: list[float] = Field(
        None, description="Temperature renormalised conduction band minima"
    )
    delta_band_gaps: list[float] = Field(
        None, description="Change in band gap relative to the bulk structure"
    )
    bulk_band_gap: float = Field(
        None, description="Band gap of the bulk (supercell structure)"
    )
    raw_data: RawElectronicData = Field(
        None,
        description="Raw electronic and structure data used to obtain the "
        "electron-phonon coupling",
    )

    @classmethod
    def from_band_structures(
        cls,
        temperatures: list[float],
        displacement_band_structures: list[BandStructure],
        displacement_structures: list[Structure],
        displacement_uuids: list[str],
        displacement_dirs: list[str],
        bulk_band_structure: BandStructure,
        bulk_structure: Structure,
        bulk_uuid: str,
        bulk_dir: str,
        elph_uuid: str,
        elph_dir: str,
        original_structure: Structure,
    ) -> Self:
        """
        Calculate an electron-phonon renormalisation document from band structures.

        Parameters
        ----------
        temperatures : list of float
            The temperatures at which electron phonon properties were calculated.
        displacement_band_structures : list of BandStructure
            The electron-phonon displaced band structures.
        displacement_structures : list of Structure
            The electron-phonon displaced structures.
        displacement_uuids : list of str
            The UUIDs of the electron-phonon displaced band structure calculations.
        displacement_dirs : list of str
            The calculation directories of the electron-phonon displaced band structure
            calculations.
        bulk_band_structure : BandStructure
            The band structure of the bulk undisplaced supercell calculation.
        bulk_structure : Structure
            The structure of the bulk undisplaced supercell.
        bulk_uuid : str
            The UUID of the bulk undisplaced supercell band structure calculation.
        bulk_dir : str
            The directory of the bulk undisplaced supercell band structure calculation.
        elph_uuid : str
            The UUID of the electron-phonon calculation that generated the displaced
            structures.
        elph_dir : str
            The directory of electron-phonon calculation that generated the displaced
            structures.
        original_structure : Structure
            The original primitive structure for which electron-phonon calculations
            were performed.

        Returns
        -------
        ElectronPhononRenormalisationDoc
            An electron-phonon renormalisation document.
        """
        if bulk_band_structure.is_metal():
            raise ValueError(
                "Bulk band structure is metallic. Cannot calculate band gap "
                "renormalisation"
            )

        if len({band.is_spin_polarized for band in displacement_band_structures}) != 1:
            raise ValueError(
                "Some displacement bands structures are spin polarized and some are "
                "spin paired. Cannot continue."
            )

        # check all displacement calculations match magnetism of bulk
        if (
            bulk_band_structure.is_spin_polarized
            != displacement_band_structures[0].is_spin_polarized
        ):
            raise ValueError(
                "Spin polarization of bulk structure does not match polarization of "
                "displacement band structures"
            )

        # discard metallic displacement calculations and log the issue
        keep = []
        for idx, band_structure in enumerate(displacement_band_structures):
            if band_structure.is_metal():
                temp = temperatures[idx]
                logger.warning(f"T = {temp} K band structure is metallic... skipping")
            else:
                keep.append(idx)

        temperatures = [temperatures[idx] for idx in keep]
        displacement_band_structures = [
            displacement_band_structures[idx] for idx in keep
        ]
        displacement_structures = [displacement_structures[idx] for idx in keep]
        displacement_dirs = [displacement_dirs[idx] for idx in keep]
        displacement_uuids = [displacement_uuids[idx] for idx in keep]

        vbm_band_indices, cbm_band_indices = _get_band_edge_indices(bulk_band_structure)
        bulk_vbm = bulk_band_structure.get_vbm()["energy"]
        bulk_cbm = bulk_band_structure.get_cbm()["energy"]
        bulk_band_gap = bulk_cbm - bulk_vbm

        displacement_cbms = _get_displacement_band_edges(
            displacement_band_structures, cbm_band_indices, cbm=True
        )
        displacement_vbms = _get_displacement_band_edges(
            displacement_band_structures, vbm_band_indices, cbm=False
        )
        cbms = np.mean(displacement_cbms, axis=1)
        vbms = np.mean(displacement_vbms, axis=1)
        band_gaps = cbms - vbms
        delta_band_gaps = band_gaps - bulk_band_gap

        return cls.from_structure(
            structure=original_structure,
            meta_structure=original_structure,
            temperatures=temperatures,
            band_gaps=band_gaps.tolist(),
            vbms=vbms.tolist(),
            cbms=cbms.tolist(),
            delta_band_gaps=delta_band_gaps.tolist(),
            bulk_band_gap=bulk_band_gap,
            raw_data=RawElectronicData(
                displacement_uuids=displacement_uuids,
                displacement_dirs=displacement_dirs,
                displacement_structures=displacement_structures,
                displacement_cbms=displacement_cbms.tolist(),
                displacement_vbms=displacement_vbms.tolist(),
                bulk_uuid=bulk_uuid,
                bulk_dir=bulk_dir,
                bulk_structure=bulk_structure,
                bulk_cbm=bulk_cbm,
                bulk_vbm=bulk_vbm,
                bulk_vbm_band_indices={s.name: i for s, i in vbm_band_indices.items()},
                bulk_cbm_band_indices={s.name: i for s, i in cbm_band_indices.items()},
                elph_uuid=elph_uuid,
                elph_dir=elph_dir,
            ),
        )


def _get_displacement_band_edges(
    band_structures: list[BandStructure],
    band_indices: dict[Spin, list[int]],
    cbm: bool = True,
) -> np.ndarray:
    """Extract band edge energies based on a band structure and band indices."""
    band_edges = []
    for band_structure in band_structures:
        spin_edges = []
        for spin, spin_indices in band_indices.items():
            eigs = band_structure.bands[spin][spin_indices]
            if cbm:
                spin_edges.extend(eigs.min(axis=1).tolist())
            else:
                spin_edges.extend(eigs.max(axis=1).tolist())
        band_edges.append(spin_edges)

    return np.array(band_edges)


def _get_band_edge_indices(
    band_structure: BandStructure,
    tol: float = 0.005,
) -> tuple[dict[Spin, list[int]], dict[Spin, list[int]]]:
    """
    Get indices of degenerate band edge states, within a tolerance.

    Parameters
    ----------
    band_structure : BandStructure
        A band structure.
    tol : float
        Degeneracy tolerance in meV.
    """
    vbm_energy = band_structure.get_vbm()["energy"]
    cbm_energy = band_structure.get_cbm()["energy"]

    vbm_band_indices = {}
    cbm_band_indices = {}
    for spin, spin_energies in band_structure.bands.items():
        vb_idxs = np.where(
            np.any(
                (spin_energies > vbm_energy - tol)
                & (spin_energies < band_structure.efermi),
                axis=1,
            )
        )[0]
        cb_idxs = np.where(
            np.any(
                (spin_energies < cbm_energy + tol)
                & (spin_energies > band_structure.efermi),
                axis=1,
            )
        )[0]
        vbm_band_indices[spin] = vb_idxs.tolist()
        cbm_band_indices[spin] = cb_idxs.tolist()

    return vbm_band_indices, cbm_band_indices
