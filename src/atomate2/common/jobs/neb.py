"""Define tools for analyzing NEB runs."""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

from jobflow import job

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


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

    Note that `num_images` specifies the number of intermediate images
    between two endpoints. Thus, specifying `num_images = 5` will return
    the endpoints and 5 intermediate images.

    Parameters
    ----------
    endpoints : tuple[Structure,Structure] or list[Structure]
        A set of two endpoints to interpolate NEB images from.
    num_images : int
        The number of images to include in the interpolation.
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

    Note that `num_images` specifies the number of intermediate images
    between two endpoints. Thus, specifying `num_images = 5` will return
    the endpoints and 5 intermediate images.

    Parameters
    ----------
    endpoints : tuple[Structure,Structure] or list[Structure]
        A set of two endpoints to interpolate NEB images from.
    num_images : int
        The number of images to include in the interpolation.
    interpolation_method : .NebInterpolation
        The method to use to interpolate between images.
    **interpolation_kwargs
        kwargs to pass to the interpolation function.
    """
    if interpolation_method == NebInterpolation.LINEAR:
        return endpoints[0].interpolate(
            endpoints[1], nimages=num_images + 1, **interpolation_kwargs
        )
    if interpolation_method == NebInterpolation.IDPP:
        try:
            from pymatgen.analysis.diffusion.neb.pathfinder import IDPPSolver
        except ImportError as exc:
            raise ImportError(
                "You must pip install `pymatgen-analysis-diffusion` "
                "to generate images with IDPP."
            ) from exc

        constructor_keys = {"sort_tol", "interpolate_lattices"}
        constructor_kwargs = {
            k: v for k, v in interpolation_kwargs.items() if k in constructor_keys
        }

        return IDPPSolver.from_endpoints(
            endpoints, nimages=num_images, **constructor_kwargs
        ).run(
            **{
                k: v
                for k, v in interpolation_kwargs.items()
                if k not in constructor_keys
            }
        )

    raise ValueError(f"Unknown {interpolation_method=}")
