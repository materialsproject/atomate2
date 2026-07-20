"""A definition of a MSON document representing an Siesta task."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from emmet.core.math import Matrix3D, Vector3D
from emmet.core.structure import StructureMetadata
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from typing_extensions import Self

from atomate2.siesta.files import load_siesta_input
from atomate2.siesta.schemas.calculation import Calculation, TaskState
from atomate2.siesta.utils.datetime import datetime_str
from atomate2.siesta.utils.path import get_uri, strip_hostname

logger = logging.getLogger(__name__)


class InputDoc(BaseModel):
    """Summary of the inputs for an SIESTA calculation.

    Parameters
    ----------
    xc_functional: str
        Exchange-correlation functional used if not the default
    xc_authors: str
        Exchange-correlation authors used if not the default
    kpts: list[int] | None
        K-point grid used for the calculation [nx, ny, nz]
    basis_size: str | None
        PAO basis size (e.g., 'DZP', 'SZP', 'TZP')
    mesh_cutoff: float | None
        Real space mesh cutoff energy in Ry
    """

    xc_functional: str | None = Field(
        None, description="Exchange-correlation functional used if not the default"
    )
    xc_authors: str | None = Field(
        None, description="Exchange-correlation authors used if not the default"
    )
    kpts: list[int] | None = Field(None, description="K-point grid [nx, ny, nz]")
    basis_size: str | None = Field(
        None, description="PAO basis size (e.g., 'DZP', 'SZP', 'TZP')"
    )
    mesh_cutoff: float | None = Field(
        None, description="Real space mesh cutoff energy in Ry"
    )

    @classmethod
    def from_siesta_calc_doc(cls, calc_doc: Calculation) -> Self:
        """Create a summary from an SIESTA CalculationDocument.

        Parameters
        ----------
        calc_doc: .Calculation
            An SIESTA calculation document.

        Returns
        -------
        .InputDoc
            The calculation input summary.
        """
        logger.info("InputDoc.from_siesta_calc_doc()")

        # Try to load from siesta_parameters.json first
        try:
            siesta_input = load_siesta_input(calc_doc.dir_name)
        except (FileNotFoundError, NotImplementedError):
            # Check for compressed version
            import gzip
            import json
            from pathlib import Path

            compressed_file = (
                Path(calc_doc.dir_name)
                / "siesta_compressed"
                / "siesta_parameters.json.gz"
            )

            if compressed_file.exists():
                logger.info("Loading from compressed siesta_parameters.json.gz")
                try:
                    with gzip.open(compressed_file, "rt") as f:
                        siesta_input = json.load(f)
                except Exception as e:  # noqa: BLE001  parse fallback
                    logger.error(  # noqa: TRY400
                        f"Failed to load compressed parameters: {e}"
                    )
                    return cls()
            else:
                # Fall back to parsing the FDF file directly
                logger.warning("siesta_parameters.json not found, parsing FDF file")
                fdf_file = Path(calc_doc.dir_name) / "siesta.fdf"
                if not fdf_file.exists():
                    logger.error(  # noqa: TRY400
                        "Neither siesta_parameters.json nor siesta.fdf found "
                        f"in {calc_doc.dir_name}"
                    )
                    return cls()

                # Parse FDF file
                siesta_input = cls._parse_fdf_file(fdf_file)

        # Extract k-points from kgrid_Monkhorst_Pack
        kpts = None
        if "kgrid_Monkhorst_Pack" in siesta_input:
            kgrid = siesta_input["kgrid_Monkhorst_Pack"]
            if isinstance(kgrid, list) and len(kgrid) >= 3:
                # Parse k-points from strings like "1 0 0 0.0"
                try:
                    kpts = [
                        int(kgrid[0].split()[0]),
                        int(kgrid[1].split()[1]),
                        int(kgrid[2].split()[2]),
                    ]
                except (IndexError, ValueError, AttributeError):
                    kpts = None

        # Extract basis size - check both singular and plural forms
        basis_size = siesta_input.get("PAO.BasisSize")
        if basis_size is None:
            basis_sizes = siesta_input.get("PAO.BasisSizes")
            if basis_sizes:
                if isinstance(basis_sizes, str):
                    # Single line format
                    basis_size = basis_sizes
                elif isinstance(basis_sizes, list):
                    # Block format - parse species-specific basis
                    # Example: ["Ti  TZP", "O   DZP"]
                    parsed_basis = []
                    for line in basis_sizes:
                        if isinstance(line, str):
                            parts = line.split()
                            if len(parts) >= 2:
                                species = parts[0]
                                basis = parts[1]
                                parsed_basis.append((species, basis))

                    if parsed_basis:
                        # Check if all species use the same basis
                        unique_basis = {b for _, b in parsed_basis}
                        if len(unique_basis) == 1:
                            # All same - just store the basis name
                            basis_size = unique_basis.pop()
                        else:
                            # Mixed - store as "Species:Basis" pairs
                            basis_size = ", ".join(f"{s}:{b}" for s, b in parsed_basis)

        # If still no basis size, try parsing from FDF file
        if basis_size is None:
            import gzip
            from pathlib import Path

            # Try uncompressed FDF first
            fdf_file = Path(calc_doc.dir_name) / "siesta.fdf"
            if fdf_file.exists():
                try:
                    fdf_params = cls._parse_fdf_file(fdf_file)
                    basis_size = fdf_params.get("PAO.BasisSize")
                except Exception as e:  # noqa: BLE001  FDF parse is best-effort
                    logger.warning(f"Failed to parse FDF file: {e}")

            # Log if we can't find basis size
            if basis_size is None and not fdf_file.exists():
                logger.warning("Could not extract basis size - FDF file not found")

        # Extract mesh cutoff
        mesh_cutoff = siesta_input.get("MeshCutoff")

        return cls(
            xc_functional=siesta_input.get("XC.functional"),
            xc_authors=siesta_input.get("XC.authors"),
            kpts=kpts,
            basis_size=basis_size,
            mesh_cutoff=mesh_cutoff,
        )

    @staticmethod
    def _parse_fdf_file(fdf_path: Path) -> dict[str, str | int | float | list[str]]:
        """Parse FDF file to extract key parameters.

        Parameters
        ----------
        fdf_path: Path
            Path to the siesta.fdf file.

        Returns
        -------
        dict[str, str | int | float | list[str]]
            Dictionary with extracted parameters.
        """
        params: dict[str, str | int | float | list[str]] = {}

        with open(fdf_path) as f:
            in_block = False
            block_name = None
            block_content: list[str] = []

            for raw_line in f:
                line = raw_line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Check for block start
                if line.lower().startswith("%block"):
                    in_block = True
                    block_name = line.split()[1]
                    block_content = []
                    continue

                # Check for block end
                if line.lower().startswith("%endblock"):
                    in_block = False
                    if block_name:
                        params[block_name] = block_content
                    block_name = None
                    continue

                # Inside a block
                if in_block:
                    block_content.append(line)
                    continue

                # Regular key-value pair
                if "\t" in line or " " in line:
                    parts = line.split(None, 1)  # Split on first whitespace
                    if len(parts) >= 2:
                        key = parts[0]
                        value = parts[1].split("#")[0].strip()  # Remove inline comments

                        # Try to convert to appropriate type
                        # Remove units if present
                        value_clean = value.split()[0] if " " in value else value

                        try:
                            # Try int first
                            params[key] = int(value_clean)
                        except ValueError:
                            try:
                                # Try float
                                params[key] = float(value_clean)
                            except ValueError:
                                # Keep as string
                                params[key] = value_clean

        return params


class OutputDoc(BaseModel):
    """Summary of the outputs for an Siesta calculation.

    Parameters
    ----------
    structure: Structure
        The final pymatgen Structure of the final system
    trajectory: Sequence[Structure]
        The trajectory of output structures
    energy: float
        The final total DFT energy for the last calculation
    efermi: float
        The Fermi energy for the last calculation
    bandgap: float
        The DFT bandgap for the last calculation
    cbm: float
        CBM for this calculation
    vbm: float
        VBM for this calculation
    forces: list[Vector3D]
        Forces on atoms from the last calculation
    stress: Matrix3D
        Stress on the unit cell from the last calculation
    run_time: float
        Wall time for the calculation in seconds
    geometry_converged: bool
        Whether geometry optimization converged (for relaxation)
    final_max_force: float
        Maximum force on any atom in the final structure (eV/Ang)
    force_tolerance: float
        Force tolerance used for convergence check (eV/Ang)
    """

    structure: Structure | None = Field(None, description="The output structure object")
    trajectory: Sequence[Structure] | None = Field(
        None, description="The trajectory of output structures"
    )
    energy: float | None = Field(
        None, description="The final total DFT energy for the last calculation"
    )
    efermi: float | None = Field(
        None, description="The Fermi energy for the last calculation in eV"
    )
    bandgap: float | None = Field(
        None, description="The DFT bandgap for the last calculation"
    )
    cbm: float | None = Field(None, description="CBM for this calculation")
    vbm: float | None = Field(None, description="VBM for this calculation")
    forces: list[Vector3D] | None = Field(
        None, description="Forces on atoms from the last calculation"
    )
    stress: Matrix3D | None = Field(
        None, description="Stress on the unit cell from the last calculation"
    )
    run_time: float | None = Field(
        None, description="Wall time for the calculation in seconds"
    )
    geometry_converged: bool | None = Field(
        None,
        description=(
            "Whether geometry optimization converged (for relaxation calculations)"
        ),
    )
    final_max_force: float | None = Field(
        None,
        description="Maximum force on any atom in the final structure (eV/Ang)",
    )
    force_tolerance: float | None = Field(
        None, description="Force tolerance used for convergence check (eV/Ang)"
    )

    @property
    def energy_per_atom(self) -> float | None:
        """Energy per atom.

        Returns
        -------
        float | None
            Energy per atom in eV/atom, or None if energy or structure not available.
        """
        if self.energy is not None and self.structure is not None:
            n_atoms = len(self.structure.sites)  # Number of atoms in structure
            return self.energy / n_atoms
        return None

    @classmethod
    def from_siesta_calc_doc(cls, calc_doc: Calculation) -> Self:
        """Create a summary from an Siesta CalculationDocument.

        Parameters
        ----------
        calc_doc: .Calculation
            An Siesta calculation document.

        Returns
        -------
        .OutputDoc
            The calculation output summary.
        """
        logger.info("OutputDoc.from_siesta_calc_doc()")
        return cls(
            structure=calc_doc.output.structure,
            energy=calc_doc.output.total_energy,
            efermi=calc_doc.output.efermi,
            bandgap=calc_doc.output.bandgap,
            cbm=calc_doc.output.cbm,
            vbm=calc_doc.output.vbm,
            forces=calc_doc.output.forces,
            stress=calc_doc.output.stress,
            run_time=calc_doc.output.run_time,
        )


class SiestaTaskDoc(StructureMetadata):
    """Definition of Siesta task document.

    Parameters
    ----------
    dir_name: str
        The directory for this Siesta task
    last_updated: str
        Timestamp for when this task document was last updated
    completed_at: str
        Timestamp for when this task was completed
    input: .InputDoc
        The input to the first calculation
    output: .OutputDoc
        The output of the final calculation
    structure: Structure
        Final output structure from the task
    state: .TaskState
        State of this task
    included_objects: List[.SiestaObject]
        List of Siesta objects included with this task document
    Siesta_objects: Dict[.SiestaObject, Any]
        Siesta objects associated with this task
    task_label: str
        A description of the task
    tags: List[str]
        Metadata tags for this task document
    author: str
        Author extracted from transformations
    icsd_id: str
        International crystal structure database id of the structure
    calcs_reversed: List[.Calculation]
        The inputs and outputs for all Siesta runs in this task.
    transformations: Dict[str, Any]
        Information on the structural transformations, parsed from a
        transformations.json file
    custodian: Any
        Information on the custodian settings used to run this
        calculation, parsed from a custodian.json file
    additional_json: Dict[str, Any]
        Additional json loaded from the calculation directory
    """

    dir_name: str | None = Field(None, description="The directory for this Siesta task")
    last_updated: str | None = Field(
        default_factory=datetime_str,
        description="Timestamp for when this task document was last updated",
    )
    completed_at: str | None = Field(
        None, description="Timestamp for when this task was completed"
    )
    input: InputDoc | None = Field(
        None, description="The input to the first calculation"
    )
    output: OutputDoc | None = Field(
        None, description="The output of the final calculation"
    )
    structure: Structure | None = Field(
        None, description="Final output atoms from the task"
    )
    state: TaskState | None = Field(None, description="State of this task")

    task_label: str | None = Field(None, description="A description of the task")
    tags: list[str] | None = Field(
        None, description="Metadata tags for this task document"
    )
    author: str | None = Field(
        None, description="Author extracted from transformations"
    )
    icsd_id: str | None = Field(
        None, description="International crystal structure database id of the structure"
    )
    calcs_reversed: list[Calculation] | None = Field(
        None, description="The inputs and outputs for all Siesta runs in this task."
    )
    transformations: dict[str, Any] | None = Field(
        None,
        description="Information on the structural transformations, parsed from a "
        "transformations.json file",
    )
    custodian: Any = Field(
        None,
        description="Information on the custodian settings used to run this "
        "calculation, parsed from a custodian.json file",
    )
    additional_json: dict[str, Any] | None = Field(
        None, description="Additional json loaded from the calculation directory"
    )

    @classmethod
    def from_directory(
        cls,
        dir_name: Path | str,
        additional_fields: dict[str, Any] | None = None,
        **siesta_calculation_kwargs,  # noqa: ARG003  documented passthrough
    ) -> Self:
        """Create a task document from a directory containing SIESTA files.

        Parameters
        ----------
        dir_name: Path or str
            The path to the folder containing the calculation outputs.
        additional_fields: Dict[str, Any]
            Dictionary of additional fields to add to output document.
        **siesta_calculation_kwargs
            Additional parsing options that will be passed to the
            :obj:`.Calculation.from_Siesta_files` function.

        Returns
        -------
        .SiestaTaskDoc
            A task document for the calculation.
        """
        logger.info("SiestaTaskDoc.from_directory()")
        logger.info(f"Getting task doc in: {dir_name}")

        if additional_fields is None:
            additional_fields = {}

        dir_name = Path(dir_name)
        task_files = _find_siesta_files(dir_name)
        if len(task_files) == 0:
            raise FileNotFoundError("No Siesta files found!")

        # Process all SIESTA calculation files in the directory
        # Multiple calculations may exist from restarts or multi-step workflows
        calcs_reversed = []
        for files_name, files in task_files.items():
            logger.debug(f"Processing task files: {files_name=} {files=}")
            try:
                calc_doc = Calculation.from_siesta_files(dir_name)
            except (OSError, RuntimeError, ValueError) as e:
                logger.error(  # noqa: TRY400
                    f"Cannot read calculation document from {files_name}: {e}"
                )
                raise RuntimeError(
                    f"Cannot read calculation document from {files_name}"
                ) from e
            calcs_reversed.append(calc_doc)

        tags = additional_fields.get("tags")

        dir_name = get_uri(dir_name)
        dir_name = strip_hostname(dir_name)

        included_objects = None

        # Extract structure from last calculation (handles restart scenarios)
        # If SIESTA crashed and restarted, use the structure from the last run
        if isinstance(calcs_reversed[-1].output.structure, Structure):
            attr = "from_structure"
            dat = {
                "structure": calcs_reversed[-1].output.structure,
                "meta_structure": calcs_reversed[-1].output.structure,
                "include_structure": True,
            }
        else:
            # Fallback if no valid structure available
            attr = "from_structure"
            dat = {
                "structure": None,
                "meta_structure": None,
                "include_structure": False,
            }

        doc = getattr(cls, attr)(**dat)
        ddict = doc.model_dump()

        data = {
            "calcs_reversed": calcs_reversed,
            "completed_at": calcs_reversed[-1].completed_at,
            "dir_name": dir_name,
            "included_objects": included_objects,
            "input": InputDoc.from_siesta_calc_doc(calcs_reversed[0]),
            "meta_structure": calcs_reversed[-1].output.structure,
            "output": OutputDoc.from_siesta_calc_doc(calcs_reversed[-1]),
            "state": calcs_reversed[-1].has_siesta_completed,
            "structure": calcs_reversed[-1].output.structure,
            "tags": tags,
        }

        doc = cls(**ddict)
        doc = doc.model_copy(update=data)
        return doc.model_copy(update=additional_fields, deep=True)


def _find_siesta_files(
    path: Path | str,
) -> dict[str, Any]:
    """Find SIESTA files in a directory.

    Only files in folders with names matching a task name (or alternatively files
    with the task name as an extension, e.g., Siesta.out) will be returned.

    Siesta files in the current directory will be given the task name "standard".

    Parameters
    ----------
    path: str or Path
        Path to a directory to search.

    Returns
    -------
    dict[str, Any]
        The filenames of the calculation outputs for each Siesta task,
        given as a ordered dictionary of::

            {
                task_name: {
                    "Siesta_output_file": Siesta_output_filename,
    """
    logger.info("_find_siesta_files()")
    files = ["MESSAGES"]
    path = Path(path)
    siesta_files = {}
    for file in files:
        siesta_files[file] = Path(file)
    return siesta_files
