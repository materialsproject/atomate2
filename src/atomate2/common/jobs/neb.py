"""
Define tools for analyzing NEB runs.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from typing import Any, Sequence

    from pymatgen.core import Structure

class NEBInterpolation(Enum):
    """Methods for interpolating NEB images."""
    LINEAR = "linear"
    IDPP = "IDPP"

def get_images_from_endpoints(
    endpoints : tuple[Structure,Structure] | list[Structure],
    num_images : int,
    interpolation_method : NEBInterpolation = NEBInterpolation.LINEAR,
    **interpolation_kwargs,
) -> list[Structure]:
    """
    Interpolate between two endpoints.
    
    Parameters
    -----------
    endpoints : tuple[Structure,Structure] or list[Structure]
        A set of two endpoints to interpolate NEB images from.
    num_images : int
        The number of images to include in the interpolation.
    prev_dir : str or Path or None (default)
        A previous directory to copy outputs from.
    interpolation_method : .NEBInterpolation
        The method to use to interpolate between images.
    **interpolation_kwargs
        kwargs to pass to the interpolation function.
    """

    if interpolation_method == NEBInterpolation.LINEAR:
        return endpoints[0].interpolate(endpoints[1], nimages=num_images, **interpolation_kwargs)
    elif interpolation_method == NEBInterpolation.IDPP:
        try:
            from pymatgen.analysis.diffusion.neb.pathfinder import IDPPSolver
        except ImportError:
            raise ImportError("You must pip install `pymatgen-analysis-diffusion` to generate images with IDPP.")
        return IDPPSolver.from_endpoints(endpoints,nimages=num_images, **interpolation_kwargs)
    
    raise ValueError(f"Unknown {interpolation_method=}")

def neb_spline_fit(
    energies : Sequence[float],
    spline_kwargs : dict | None = None,
    frame_match_tol : float = 1.e-6
) -> dict[str,Any]:
    """
    Define basic NEB analysis tools.

    Parameters
    -----------
    energies : Sequence[float]
        The energies sorted by increasing frame index. Must include endpoints.
    frame_match_tol : float = 1.e-6
        The tolerance for matching the transition state frame index to the
        input frame indices.
    """

    analysis = {
        "energies": list(energies),
        "frame_index": list(
            frame_idx := np.linspace(0., 1., len(energies))
        )
    }
    energies = np.array(energies)

    spline_kwargs = spline_kwargs or {"bc_type": "clamped"}
    spline_fit = CubicSpline(frame_idx,energies,**spline_kwargs)
    analysis["cubic_spline_pars"] = list(spline_fit.c)

    crit_points = spline_fit.derivative().roots()
    analysis["ts_frame_index"] = -1
    analysis["ts_energy"] = -np.inf
    for crit_point in crit_points:
        if (
            (energy := spline_fit(crit_point)) > analysis["ts_energy"]
            and spline_fit(crit_point,2) <= 0.
        ):
            analysis["ts_frame_index"] = crit_point
            analysis["ts_energy"] = energy

    analysis["ts_in_frames"] = any(
        abs(analysis["ts_frame_index"] - frame_idx) < frame_match_tol*max(frame_idx,frame_match_tol)
        for frame_idx in frame_idx
    )
    analysis["forward_barrier"] = analysis["ts_energy"] - energies[0]
    analysis["reverse_barrier"] = analysis["ts_energy"] - energies[-1]

    return analysis