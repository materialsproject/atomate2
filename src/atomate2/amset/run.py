"""Module defining functions to run amset."""

from __future__ import annotations

import logging
import subprocess

import numpy as np
from pydash import get

logger = logging.getLogger(__name__)
_CONVERGENCE_PROPERTIES = ("mobility.overall", "seebeck")


def run_amset() -> None:
    """Run amset in the current directory."""
    # Run AMSET using the command line as calling from python can cause issues
    # with multiprocessing
    with open("std_out.log", "w") as f_std, open("std_err.log", "w") as f_err:
        subprocess.call(["amset", "run"], stdout=f_std, stderr=f_err)  # noqa: S607


def check_converged(
    new_transport: dict,
    old_transport: dict,
    properties: tuple[str, ...] = _CONVERGENCE_PROPERTIES,
    tolerance: float = 0.1,
) -> bool:
    """
    Check if all transport properties (averaged) are converged within the tol.

    Parameters
    ----------
    new_transport : dict
        The new transport data.
    old_transport : dict
        The old transport data.
    properties : tuple of str
        List of properties for which convergence is assessed. The calculation is only
        flagged as converged if all properties pass the convergence checks. Options are:
        "conductivity", "seebeck", "mobility.overall", "electronic thermal conductivity.
    tolerance : float
        Relative convergence tolerance. Default is ``0.1`` (i.e. 10 %).

    Returns
    -------
    bool
        Whether the new transport data is converged.
    """
    converged = True
    for prop in properties:
        new_prop = get(new_transport, prop, None)
        old_prop = get(old_transport, prop, None)
        if new_prop is None or old_prop is None:
            logger.info(f"'{prop}' not in new or old transport data, skipping...")
            continue

        new_avg = tensor_average(new_prop)
        old_avg = tensor_average(old_prop)
        diff = np.abs((new_avg - old_avg) / new_avg)
        diff[~np.isfinite(diff)] = 0

        # don't check convergence of very small numbers due to numerical noise
        less_than_one = (np.abs(new_avg) < 1) & (np.abs(old_avg) < 1)
        element_converged = less_than_one | (diff <= tolerance)
        if not np.all(element_converged):
            logger.info(f"{prop} is not converged - max diff: {np.max(diff) * 100} %")
            converged = False

    if converged:
        logger.info("amset calculation is converged.")

    return converged


def tensor_average(tensor: list | np.ndarray) -> float | np.ndarray:
    """Calculate the average of the tensor eigenvalues.

    Parameters
    ----------
    tensor : list or numpy array
        A tensor

    Returns
    -------
    float or numpy array
        The average of the eigenvalues.
    """
    return np.average(np.linalg.eigvalsh(tensor), axis=-1)
