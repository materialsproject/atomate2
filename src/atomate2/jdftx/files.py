"""File operations and default JDFTx file names"""
import logging

# if TYPE_CHECKING:
from pathlib import Path

from pymatgen.core import Structure

from atomate2.jdftx.sets.base import JdftxInputGenerator

logger = logging.getLogger(__name__)


def write_jdftx_input_set(
    structure: Structure,
    input_set_generator: JdftxInputGenerator,
    directory: str | Path = ".",
    **kwargs,
) -> None:
    """
    Write JDFTx input set.

    Parameters
    ----------
    structure : .Structure
        A structure.
    input_set_generator : .JdftxInputGenerator
        A JDFTx input set generator.
    directory : str or Path
        The directory to write the input files to.
    **kwargs
        Keyword arguments to pass to :obj:`.JdftxInputSet.write_input`.
    """
    cis = input_set_generator.get_input_set(structure)

    logger.info("Writing JDFTx input set.")
    cis.write_input(directory, **kwargs)
