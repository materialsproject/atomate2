"""Some utilities of dealing with bands.

Copied from GIMS as of now; should be in its own dedicated FHI-aims python package.
"""

import numpy as np
from ase.cell import Cell
from ase.dft.kpoints import kpoint_convert, resolve_kpt_path_string

# TODO add the same procedures but using pymatgen routines


def prepare_band_input(cell: Cell, density: float = 20):
    """Prepare the band information needed for the FHI-aims control.in file.

    Parameters
    ----------
    cell: Cell
        The lattive parameters of the material
    density: float
        Number of kpoints per Angstrom.
    """
    bp = cell.bandpath()
    r_kpts = resolve_kpt_path_string(bp.path, bp.special_points)

    lines_and_labels = []
    for labels, coords in zip(*r_kpts):
        dists = coords[1:] - coords[:-1]
        lengths = [np.linalg.norm(d) for d in kpoint_convert(cell, skpts_kc=dists)]
        points = np.int_(np.round(np.asarray(lengths) * density))
        # I store it here for now. Might be needed to get global info.
        lines_and_labels.append(
            [points, labels[:-1], labels[1:], coords[:-1], coords[1:]]
        )

    bands = []
    for segment in lines_and_labels:
        for points, lstart, lend, start, end in zip(*segment):
            bands.append(
                "band {:9.5f}{:9.5f}{:9.5f} {:9.5f}{:9.5f}{:9.5f} {:4} {:3}{:3}".format(
                    *start, *end, points, lstart, lend
                )
            )

    return bands
