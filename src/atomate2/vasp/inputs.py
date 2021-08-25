"""Functions for writing/reading input sets."""

from __future__ import annotations

import logging

from pymatgen.core.structure import Structure

from atomate2.vasp.sets.core import VaspInputSetGenerator

__all__ = ["write_vasp_input_set"]

logger = logging.getLogger(__name__)


def write_vasp_input_set(
    structure: Structure,
    input_set_generator: VaspInputSetGenerator,
    from_prev: bool = False,
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
    **kwargs
        Keyword arguments that will be passed to :obj:`.VaspInputSet.write_input`.
    """
    prev_dir = "." if from_prev else None
    vis = input_set_generator.get_input_set(structure, prev_dir=prev_dir)

    logger.info("Writing VASP input set.")
    vis.write_input(".", **kwargs)
