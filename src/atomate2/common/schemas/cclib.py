"""Core definition of a cclib-generated task document."""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

from emmet.core.structure import MoleculeMetadata
from monty.dev import requires
from monty.json import jsanitize
from pydantic import Field
from pymatgen.core import Molecule
from pymatgen.core.periodic_table import Element
from typing_extensions import Self

from atomate2 import __version__
from atomate2.utils.datetime import datetime_str
from atomate2.utils.path import find_recent_logfile, get_uri

try:
    import cclib
except ImportError:
    cclib = None


logger = logging.getLogger(__name__)


class TaskDocument(MoleculeMetadata, extra="allow"):  # type: ignore[call-arg]
    """
    Definition of a cclib-generated task document.

    This can be used as a general task document for molecular DFT codes.
    For the list of supported packages, see https://cclib.github.io
    """

    molecule: Optional[Molecule] = Field(
        None, description="Final output molecule from the task"
    )
    energy: Optional[float] = Field(None, description="Final total energy")
    dir_name: Optional[str] = Field(
        None, description="Directory where the output is parsed"
    )
    logfile: Optional[str] = Field(
        None, description="Path to the log file used in the post-processing analysis"
    )
    attributes: Optional[dict] = Field(
        None, description="Computed properties and calculation outputs"
    )
    metadata: Optional[dict] = Field(
        None,
        description="Calculation metadata, including input parameters and runtime "
        "statistics",
    )
    task_label: Optional[str] = Field(None, description="A description of the task")
    tags: Optional[list[str]] = Field(
        None, description="Optional tags for this task document"
    )
    last_updated: str = Field(
        default_factory=datetime_str,
        description="Timestamp for this task document was last updated",
    )
    schema: str = Field(
        __version__, description="Version of atomate2 used to create the document"
    )

    @classmethod
    @requires(cclib, "The cclib TaskDocument requires cclib to be installed.")
    def from_logfile(
        cls,
        dir_name: Union[str, Path],
        logfile_extensions: Union[str, list[str]],
        store_trajectory: bool = False,
        additional_fields: Optional[dict[str, Any]] = None,
        analysis: Optional[Union[str, list[str]]] = None,
        proatom_dir: Optional[Union[Path, str]] = None,
    ) -> Self:
        """Create a TaskDocument from a log file.

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
        additional_fields
            Dictionary of additional fields to add to TaskDocument.
        analysis
            The name(s) of any cclib post-processing analysis to run. Note that for
            bader, ddec6, and hirshfeld, a cube file (.cube, .cub) must be in dir_name.
            Supports: cpsa, mpa, lpa, bickelhaupt, density, mbo, bader, ddec6,
            hirshfeld.
        proatom_dir
            The path to the proatom directory if ddec6 or hirshfeld analysis are
            requested. See https://cclib.github.io/methods.html for details. If None,
            the PROATOM_DIR environment variable must point to the proatom directory.

        Returns
        -------
        TaskDocument
            A TaskDocument object summarizing the inputs/outputs of the log file.
        """
        from cclib.io import ccread

        logger.info(
            f"Searching for most recent log file with extensions {logfile_extensions}"
        )

        # Find the most recent log file with the given extension in the
        # specified directory.
        logfile = find_recent_logfile(dir_name, logfile_extensions)
        if not logfile:
            raise FileNotFoundError(
                f"Could not find file with extension {logfile_extensions} in {dir_name}"
            )

        logger.info(f"Getting task doc from {logfile}")

        additional_fields = additional_fields or {}

        # Let's parse the log file with cclib
        cclib_obj = ccread(logfile, logging.ERROR)
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

        # monty datetime bug workaround: github.com/materialsvirtuallab/monty/issues/275
        if wall_time := metadata.get("wall_time"):
            metadata["wall_time"] = [*map(str, wall_time)]
        if cpu_time := metadata.get("cpu_time"):
            metadata["cpu_time"] = [*map(str, cpu_time)]

        # Get the final energy to store as its own key/value pair
        energy = (
            cclib_obj.scfenergies[-1] if cclib_obj.scfenergies is not None else None
        )

        # Now we construct the input molecule. Note that this is not necessarily
        # the same as the initial molecule from the relaxation because the
        # DFT package may have re-oriented the system. We only try to store
        # the input if it is XYZ-formatted though since the Molecule object
        # does not support internal coordinates or Gaussian Z-matrix.
        if (
            cclib_obj.metadata.get("coord_type") == "xyz"
            and cclib_obj.metadata.get("coords") is not None
        ):
            coords_obj = cclib_obj.metadata["coords"]
            input_species = [Element(row[0]) for row in coords_obj]
            input_coords = [row[1:] for row in coords_obj]
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
        if store_trajectory:
            attributes["trajectory"] = molecules

        # Store the HOMO/LUMO energies for convenience
        if cclib_obj.moenergies is not None and cclib_obj.homos is not None:
            homo_energies, lumo_energies, homo_lumo_gaps = _get_homos_lumos(
                cclib_obj.moenergies, cclib_obj.homos
            )
            attributes["homo_energies"] = homo_energies
            if lumo_energies:
                attributes["lumo_energies"] = lumo_energies
            if homo_lumo_gaps:
                attributes["homo_lumo_gaps"] = homo_lumo_gaps

                # The HOMO-LUMO gap for a spin-polarized system is ill-defined.
                # This is why we report both the alpha and beta channel gaps
                # above. Here, we report min(LUMO_alpha-HOMO_alpha,LUMO_beta-HOMO_beta)
                # in case the user wants to easily query by this too. For restricted
                # systems, this will always be the same as above.
                attributes["min_homo_lumo_gap"] = min(homo_lumo_gaps)

        # Calculate any properties
        if analysis:
            if isinstance(analysis, str):
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
            final_molecule,
            energy=energy,
            dir_name=get_uri(dir_name),
            logfile=get_uri(logfile),
            attributes=attributes,
            metadata=metadata,
        )
        doc.molecule = final_molecule
        return doc.model_copy(update=additional_fields)


@requires(cclib, "cclib_calculate requires cclib to be installed.")
def cclib_calculate(
    cclib_obj: Any,
    method: str,
    cube_file: Union[Path, str],
    proatom_dir: Union[Path, str],
) -> Optional[dict[str, Any]]:
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
    cube_methods = ("bader", "ddec6", "hirshfeld")

    if method in cube_methods and not cube_file:
        raise FileNotFoundError(f"A cube file must be provided for {method}.")
    if method in ("ddec6", "hirshfeld") and not proatom_dir:
        if os.getenv("PROATOM_DIR") is None:
            raise OSError("PROATOM_DIR environment variable not set.")
        proatom_dir = os.path.expandvars(os.environ["PROATOM_DIR"])
    if proatom_dir and not os.path.isdir(proatom_dir):
        raise FileNotFoundError(f"{proatom_dir=} does not exist.")

    if cube_file and method in cube_methods:
        vol = volume.read_from_cube(str(cube_file))

    if method == "bader":
        _method = Bader(cclib_obj, vol)
    elif method == "bickelhaupt":
        _method = Bickelhaupt(cclib_obj)
    elif method == "cpsa":
        _method = CSPA(cclib_obj)
    elif method == "ddec6":
        _method = DDEC6(cclib_obj, vol, str(proatom_dir))
    elif method == "density":
        _method = Density(cclib_obj)
    elif method == "hirshfeld":
        _method = Hirshfeld(cclib_obj, vol, str(proatom_dir))
    elif method == "lpa":
        _method = LPA(cclib_obj)
    elif method == "mbo":
        _method = MBO(cclib_obj)
    elif method == "mpa":
        _method = MPA(cclib_obj)
    else:
        raise ValueError(f"{method=} is not supported.")

    try:
        _method.calculate()
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
        if hasattr(_method, attribute):
            calc_attributes[attribute] = getattr(_method, attribute)
    return calc_attributes


def _get_homos_lumos(
    mo_energies: list[list[float]], homo_indices: list[int]
) -> tuple[list[float], Optional[list[float]], Optional[list[float]]]:
    """
    Calculate the HOMO, LUMO, and HOMO-LUMO gap energies in eV.

    Parameters
    ----------
    mo_energies
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
    homo_energies = [mo_energies[idx][homo] for idx, homo in enumerate(homo_indices)]
    # Make sure that the HOMO+1 (i.e. LUMO) is in MO energies (sometimes virtual
    # orbitals aren't printed in the output)
    for idx, homo in enumerate(homo_indices):
        if len(mo_energies[idx]) < homo + 2:
            return homo_energies, None, None
    lumo_energies = [
        mo_energies[idx][homo + 1] for idx, homo in enumerate(homo_indices)
    ]
    homo_lumo_gaps = [
        lumo_energies[i] - homo_energies[i] for i in range(len(homo_energies))
    ]
    return homo_energies, lumo_energies, homo_lumo_gaps
