"""Core definition of a cclib-generated task document."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, TypeVar, Union

from monty.dev import requires
from monty.json import jsanitize
from pydantic import Field
from pymatgen.core import Molecule
from pymatgen.core.periodic_table import Element

from atomate2 import __version__
from atomate2.common.schemas.molecule import MoleculeMetadata
from atomate2.utils.datetime import datetime_str
from atomate2.utils.path import find_recent_logfile, get_uri

try:
    import cclib
except ImportError:
    cclib = None

__all__ = ["TaskDocument"]

logger = logging.getLogger(__name__)
_T = TypeVar("_T", bound="TaskDocument")


class TaskDocument(MoleculeMetadata):
    """
    Definition of a cclib-generated task document.

    This can be used as a general task document for molecular DFT codes.
    For the list of supported packages, see https://cclib.github.io
    """

    molecule: Molecule = Field(None, description="Final output molecule from the task")
    energy: float = Field(None, description="Final total energy")
    dir_name: str = Field(None, description="Directory where the output is parsed")
    logfile: str = Field(
        None, description="Path to the log file used in the post-processing analysis"
    )
    attributes: Dict = Field(
        None, description="Computed properties and calculation outputs"
    )
    metadata: Dict = Field(
        None,
        description="Calculation metadata, including input parameters and runtime statistics",
    )
    task_label: str = Field(None, description="A description of the task")
    tags: List[str] = Field(None, description="Optional tags for this task document")
    last_updated: str = Field(
        default_factory=datetime_str,
        description="Timestamp for this task document was last updated",
    )
    _schema: str = Field(
        __version__,
        description="Version of atomate2 used to create the document",
        alias="schema",
    )

    @classmethod
    @requires(cclib, "The cclib TaskDocument requires cclib to be installed.")
    def from_logfile(
        cls: Type[_T],
        dir_name: Union[str, Path],
        logfile_extensions: Union[str, List[str]],
        store_trajectory: bool = False,
        store_input_orientation: bool = False,
        additional_fields: Dict[str, Any] = None,
        analysis: Union[str, List[str]] = None,
        proatom_dir: Union[Path, str] = None,
    ) -> _T:
        """
        Create a TaskDocument from a log file.

        For a full description of each field, see https://cclib.github.io/data.html.

        Parameters
        ----------
        dir_name
            The path to the folder containing the calculation outputs.
        logfile_extensions
            Possible extensions of the log file (e.g. ".log", ".out", ".txt", ".chk").
            Note that only a partial match is needed. For instance, `.log` will match
            `.log.gz` and `.log.1.gz`. If multiple files with this extension are found,
            the one with the most recent change time will be used. For an exact match
            only, put in the full file name.
        store_trajectory
            Whether to store the molecule objects along the course of the relaxation
            trajectory.
        store_input_orientation
            Whether to store the molecule object as specified in the input file. Note that
            the initial molecule object is already stored, but it may be re-oriented
            compared to the input file if the code reorients the input geometry.
        additional_fields
            Dictionary of additional fields to add to TaskDocument.
        analysis
            The name(s) of any cclib post-processing analysis to run. Note that for bader,
            ddec6, and hirshfeld, a cube file (.cube, .cub) must be in dir_name.
            Supports: cpsa, mpa, lpa, bickelhaupt, density, mbo, bader, ddec6, hirshfeld.
        proatom_dir
            The path to the proatom directory if ddec6 or hirshfeld analysis are
            requested. See https://cclib.github.io/methods.html for details. If None, the
            PROATOM_DIR environment variable must point to the proatom directory.

        Returns
        -------
        TaskDocument
            A TaskDocument object summarizing the inputs/outputs of the log file.
        """
        from cclib.io import ccread

        logger.info(
            f"Searching for the most recent log file with extensions {logfile_extensions}"
        )

        # Find the most recent log file with the given extension in the
        # specified directory.
        logfile = find_recent_logfile(dir_name, logfile_extensions)
        if not logfile:
            raise FileNotFoundError(
                f"Could not find file with extension {logfile_extensions} in {dir_name}"
            )

        logger.info(f"Getting task doc from {logfile}")

        additional_fields = {} if additional_fields is None else additional_fields

        # Let's parse the log file with cclib
        # str conversion due to cclib bug: https://github.com/cclib/cclib/issues/1096
        cclib_obj = ccread(str(logfile), logging.ERROR)
        if not cclib_obj:
            raise ValueError(f"Could not parse {logfile}")

        # Fetch all the attributes (i.e. all input/outputs from cclib)
        attributes = jsanitize(cclib_obj.getattributes())

        # Store charge and multiplicity since we use it frequently
        charge = cclib_obj.charge
        mult = cclib_obj.mult

        # Let's move the metadata out of attributes for convenience and
        # store it separately
        attributes.pop("metadata")
        metadata = jsanitize(cclib_obj.metadata)

        # monty datetime bug workaround: https://github.com/materialsvirtuallab/monty/issues/275
        if metadata.get("wall_time", None):
            metadata["wall_time"] = [str(m) for m in metadata["wall_time"]]
        if metadata.get("cpu_time", None):
            metadata["cpu_time"] = [str(m) for m in metadata["cpu_time"]]

        # Get the final energy to store as its own key/value pair
        if cclib_obj.scfenergies is not None:
            energy = cclib_obj.scfenergies[-1]
        else:
            energy = None

        # Now we construct the input molecule. Note that this is not necessarily
        # the same as the initial molecule from the relaxation because the
        # DFT package may have re-oriented the system. We only try to store
        # the input if it is XYZ-formatted though since the Molecule object
        # does not support internal coordinates or Gaussian Z-matrix.
        if (
            store_input_orientation
            and cclib_obj.metadata.get("coord_type", None) == "xyz"
            and cclib_obj.metadata.geet("coords", None) is not None
        ):
            input_species = [Element(e) for e in cclib_obj.metadata["coords"][:, 0]]
            input_coords = cclib_obj.metadata["coords"][:, 1:]
            input_molecule = Molecule(
                input_species,
                input_coords,
                charge=charge,
                spin_multiplicity=mult,
            )
            attributes["molecule_unoriented"] = input_molecule

        # These are duplicates of things made with MoleculeMetadata, so we
        # can just remove them here
        duplicates = ["atomnos", "atomcoords", "charge", "mult", "natom"]
        for duplicate in duplicates:
            attributes.pop(duplicate, None)

        # We will remove duplicates in the metadata too
        metadata_duplicates = ["coords", "coord_type"]
        for metadata_duplicate in metadata_duplicates:
            metadata.pop(metadata_duplicate, None)

        # Construct the Molecule object(s) from the trajectory
        species = [Element.from_Z(z) for z in cclib_obj.atomnos]
        coords = cclib_obj.atomcoords
        molecules = [
            Molecule(
                species,
                coord,
                charge=charge,
                spin_multiplicity=mult,
            )
            for coord in coords
        ]
        initial_molecule = molecules[0]
        final_molecule = molecules[-1]
        attributes["molecule_initial"] = initial_molecule
        attributes["molecule_final"] = final_molecule
        if store_trajectory:
            attributes["trajectory"] = molecules

        # Store the HOMO/LUMO energies for convenience
        if cclib_obj.moenergies is not None and cclib_obj.homos is not None:
            homo_energies, lumo_energies, homo_lumo_gaps = _get_homos_lumos(
                cclib_obj.moenergies, cclib_obj.homos
            )
            attributes["homo_energies"] = homo_energies
            attributes["lumo_energies"] = lumo_energies
            attributes["homo_lumo_gaps"] = homo_lumo_gaps

            # The HOMO-LUMO gap for a spin-polarized system is ill-defined.
            # This is why we report both the alpha and beta channel gaps
            # above. Here, we report min(LUMO_alpha-HOMO_alpha,LUMO_beta-HOMO_beta)
            # in case the user wants to easily query by this too. For restricted
            # systems, this will always be the same as above.
            attributes["min_homo_lumo_gap"] = min(homo_lumo_gaps)

        # Calculate any properties
        if analysis:
            if type(analysis) == str:
                analysis = [analysis]
            analysis = [a.lower() for a in analysis]

            # Look for .cube or .cub files
            cubefile_path = find_recent_logfile(dir_name, [".cube", ".cub"])

            for analysis_name in analysis:
                logger.info(f"Running {analysis_name}")
                calc_attributes = cclib_calculate(
                    cclib_obj, analysis_name, cubefile_path, proatom_dir
                )
                if calc_attributes:
                    attributes[analysis_name] = calc_attributes
                else:
                    attributes[analysis_name] = None

        doc = cls.from_molecule(
            molecule=final_molecule,
            include_molecule=True,
            energy=energy,
            dir_name=get_uri(dir_name),
            logfile=get_uri(logfile),
            attributes=attributes,
            metadata=metadata,
        )
        doc = doc.copy(update=additional_fields)
        return doc


@requires(cclib, "cclib_calculate requires cclib to be installed.")
def cclib_calculate(
    cclib_obj,
    method: str,
    cube_file: Union[Path, str],
    proatom_dir: Union[Path, str],
) -> Dict[str, Any]:
    """
    Run a cclib population analysis.

    Parameters
    ----------
    cclib_obj
        The cclib object to run the population analysis on.
    method
        The population analysis method to use.
    cube_file
        The path to the cube file to use for the population analysis.
        Needed only for Bader, DDEC6, and Hirshfeld
    proatom_dir
        The path to the proatom directory to use for the population analysis.
        Needed only for DDEC6 and Hirshfeld.
    """
    from cclib.method import (
        CSPA,
        DDEC6,
        LPA,
        MBO,
        MPA,
        Bader,
        Bickelhaupt,
        Density,
        Hirshfeld,
        volume,
    )

    method = method.lower()
    cube_methods = ["bader", "ddec6", "hirshfeld"]

    if method in cube_methods and not cube_file:
        raise FileNotFoundError(
            f"A cube file must be provied for {method}. Returning None."
        )
    if method in ["ddec6", "hirshfeld"] and not proatom_dir:
        if "PROATOM_DIR" not in os.environ:
            raise EnvironmentError(
                "PROATOM_DIR environment variable not set. Returning None."
            )
        proatom_dir = os.path.expandvars(os.environ["PROATOM_DIR"])
    if proatom_dir and not os.path.exists(proatom_dir):
        raise FileNotFoundError(
            f"Protatom directory {proatom_dir} does not exist. Returning None."
        )

    if cube_file and method in cube_methods:
        vol = volume.read_from_cube(str(cube_file))

    if method == "bader":
        m = Bader(cclib_obj, vol)
    elif method == "bickelhaupt":
        m = Bickelhaupt(cclib_obj)
    elif method == "cpsa":
        m = CSPA(cclib_obj)
    elif method == "ddec6":
        m = DDEC6(cclib_obj, vol, str(proatom_dir))
    elif method == "density":
        m = Density(cclib_obj)
    elif method == "hirshfeld":
        m = Hirshfeld(cclib_obj, vol, str(proatom_dir))
    elif method == "lpa":
        m = LPA(cclib_obj)
    elif method == "mbo":
        m = MBO(cclib_obj)
    elif method == "mpa":
        m = MPA(cclib_obj)
    else:
        raise ValueError(f"{method} is not supported.")

    try:
        m.calculate()
    except AttributeError:
        return None

    # The list of available attributes after a calculation. This is hardcoded for now
    # until https://github.com/cclib/cclib/issues/1097 is resolved. Once it is, we can
    # delete this and just do `return calc_attributes.getattributes()`.
    avail_attributes = [
        "aoresults",
        "fragresults",
        "fragcharges",
        "density",
        "donations",
        "bdonations",
        "repulsions",
        "matches",
        "refcharges",
    ]
    calc_attributes = {}
    for attribute in avail_attributes:
        if hasattr(m, attribute):
            calc_attributes[attribute] = getattr(m, attribute)
    return calc_attributes


def _get_homos_lumos(
    moenergies: List[List[float]], homo_indices: List[int]
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate the HOMO, LUMO, and HOMO-LUMO gap energies in eV.

    Parameters
    ----------
    moenergies
        List of MO energies. For restricted calculations, List[List[float]] is
        length one. For unrestricted, it is length two.
    homo_indices
        Indices of the HOMOs.

    Returns
    -------
    homo_energies
        The HOMO energies (eV), split by alpha and beta
    lumo_energies
        The LUMO energies (eV), split by alpha and beta
    homo_lumo_gaps
        The HOMO-LUMO gaps (eV), calculated as LUMO_alpha-HOMO_alpha and
        LUMO_beta-HOMO_beta
    """
    homo_energies = [moenergies[i][h] for i, h in enumerate(homo_indices)]
    # Make sure that the HOMO+1 (i.e. LUMO) is in moenergies (sometimes virtual
    # orbitals aren't printed in the output)
    for i, h in enumerate(homo_indices):
        if len(moenergies[i]) < h + 2:
            return homo_energies, None, None
    lumo_energies = [moenergies[i][h + 1] for i, h in enumerate(homo_indices)]
    homo_lumo_gaps = [
        lumo_energies[i] - homo_energies[i] for i in range(len(homo_energies))
    ]
    return homo_energies, lumo_energies, homo_lumo_gaps
