"""Some utilities of dealing with bands.

Copied from GIMS as of now; should be in its own dedicated FHI-aims python package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from pymatgen.symmetry.bandstructure import HighSymmKpath

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure


class _SegmentDict(TypedDict):
    coords: list[list[float]]
    labels: list[str]
    length: int


def prepare_band_input(structure: Structure, density: float = 20) -> list:
    """Prepare the band information needed for the FHI-aims control.in file.

    Parameters
    ----------
    structure: Structure
        The structure for which the band path is calculated
    density: float
        Number of kpoints per Angstrom.
    """
    bp = HighSymmKpath(structure)
    points, labels = bp.get_kpoints(line_density=density, coords_are_cartesian=False)
    lines_and_labels: list[_SegmentDict] = []
    current_segment: _SegmentDict | None = None
    for label_, coords in zip(labels, points, strict=True):
        # rename the Gamma point label
        label = "G" if label_ in ("GAMMA", "\\Gamma", "Γ") else label_
        if label:
            if current_segment is None:
                current_segment = _SegmentDict(
                    coords=[coords], labels=[label], length=1
                )
            else:
                current_segment["coords"].append(coords)
                current_segment["labels"].append(label)
                current_segment["length"] += 1
                lines_and_labels.append(current_segment)
                current_segment = None
        else:
            current_segment["length"] += 1

    bands = []
    for segment in lines_and_labels:
        start, end = segment["coords"]
        lstart, lend = segment["labels"]
        bands.append(
            f"band {start[0]:9.5f}{start[1]:9.5f}{start[2]:9.5f} "
            f"{end[0]:9.5f}{end[1]:9.5f}{end[2]:9.5f} {segment['length']:4d} "
            f"{lstart:3}{lend:3}"
        )
    return bands
