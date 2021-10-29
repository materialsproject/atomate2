"""Functions for writing/reading input sets."""

from __future__ import annotations

import logging

from pymatgen.core.structure import Structure

from atomate2.settings import settings
from atomate2.vasp.sets.base import VaspInputSetGenerator

__all__ = ["write_vasp_input_set"]

logger = logging.getLogger(__name__)


def write_vasp_input_set(
    structure: Structure,
    input_set_generator: VaspInputSetGenerator,
    from_prev: bool = False,
    apply_incar_updates: bool = True,
    **kwargs
):
    """
    Write VASP input set.

    Parameters
    ----------
    structure
        A structure.
    input_set_generator
        A VASP input set generator.
    from_prev
        Whether to initialize the input set from a previous calculation.
    apply_incar_updates
        Whether to apply incar updates given in the ~/.atomate2.yaml settings file.
    **kwargs
        Keyword arguments that will be passed to :obj:`.VaspInputSet.write_input`.
    """
    prev_dir = "." if from_prev else None
    vis = input_set_generator.get_input_set(structure, prev_dir=prev_dir)

    if apply_incar_updates:
        vis.incar.update(settings.VASP_INCAR_UPDATES)

    logger.info("Writing VASP input set.")
    vis.write_input(".", **kwargs)
