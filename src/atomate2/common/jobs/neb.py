"""Define tools for analyzing NEB runs."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from jobflow import job

if TYPE_CHECKING:
    from pymatgen.core import Structure


class NebInterpolation(Enum):
    """Methods for interpolating NEB images."""

    LINEAR = "linear"
    IDPP = "IDPP"


@job
def get_images_from_endpoints(
    endpoints: tuple[Structure, Structure] | list[Structure],
    num_images: int,
    interpolation_method: NebInterpolation = NebInterpolation.LINEAR,
    **interpolation_kwargs,
) -> list[Structure]:
    """
    Interpolate between two endpoints as a job.

    Parameters
    ----------
    endpoints : tuple[Structure,Structure] or list[Structure]
        A set of two endpoints to interpolate NEB images from.
    num_images : int
        The number of images to include in the interpolation.
    prev_dir : str or Path or None (default)
        A previous directory to copy outputs from.
    interpolation_method : .NebInterpolation
        The method to use to interpolate between images.
    **interpolation_kwargs
        kwargs to pass to the interpolation function.
    """
    return _get_images_from_endpoints(
        endpoints,
        num_images,
        interpolation_method=NebInterpolation(interpolation_method),
        **interpolation_kwargs,
    )


def _get_images_from_endpoints(
    endpoints: tuple[Structure, Structure] | list[Structure],
    num_images: int,
    interpolation_method: NebInterpolation = NebInterpolation.LINEAR,
    **interpolation_kwargs,
) -> list[Structure]:
    """
    Interpolate between two endpoints.

    Parameters
    ----------
    endpoints : tuple[Structure,Structure] or list[Structure]
        A set of two endpoints to interpolate NEB images from.
    num_images : int
        The number of images to include in the interpolation.
    prev_dir : str or Path or None (default)
        A previous directory to copy outputs from.
    interpolation_method : .NebInterpolation
        The method to use to interpolate between images.
    **interpolation_kwargs
        kwargs to pass to the interpolation function.
    """
    if interpolation_method == NebInterpolation.LINEAR:
        return endpoints[0].interpolate(
            endpoints[1], nimages=num_images, **interpolation_kwargs
        )
    if interpolation_method == NebInterpolation.IDPP:
        try:
            from pymatgen.analysis.diffusion.neb.pathfinder import IDPPSolver
        except ImportError as exc:
            raise ImportError(
                "You must pip install `pymatgen-analysis-diffusion` "
                "to generate images with IDPP."
            ) from exc
        return IDPPSolver.from_endpoints(
            endpoints, nimages=num_images, **interpolation_kwargs
        )

    raise ValueError(f"Unknown {interpolation_method=}")
