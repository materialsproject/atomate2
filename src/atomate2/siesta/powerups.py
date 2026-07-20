"""Powerups for performing common modifications on SIESTA jobs and flows."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from jobflow import Flow, Job, Maker

from atomate2.siesta.jobs.base import BaseSiestaMaker

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Molecule, Structure

    from atomate2.siesta.cluster_profiles import ClusterProfile

logger = logging.getLogger(__name__)


def update_maker_kwargs(
    class_filter: type[Maker] | None,
    dict_mod_updates: dict,
    flow: Job | Flow | Maker,
    name_filter: str | None,
) -> Job | Flow | Maker:
    """
    Update an object inside a Job, a Flow or a Maker.

    A generic method to be shared for more specific updates that will
    build the dict_mod_updates.

    Parameters
    ----------
    flow : .Job or .Flow or .Maker
        A job, flow or Maker.
    dict_mod_updates : dict
        The updates to apply.
    name_filter : str or None
        A filter for the name of the jobs.
    class_filter : Maker or None
        A filter for the class used to generate the flows. Note the class
        filter will match any subclasses.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input modified flow/job/maker.
    """
    logger.info("update_maker_kwargs.__init__()")
    updated_flow = deepcopy(flow)
    if isinstance(updated_flow, Maker):
        updated_flow = updated_flow.update_kwargs(
            {"_set": dict_mod_updates},
            name_filter=name_filter,
            class_filter=class_filter,
            dict_mod=True,
        )
    else:
        updated_flow.update_maker_kwargs(
            {"_set": dict_mod_updates},
            name_filter=name_filter,
            class_filter=class_filter,
            dict_mod=True,
        )
    return updated_flow


def update_user_siesta_settings(
    flow: Job | Flow | Maker,
    siesta_updates: dict[str, Any] = None,
    name_filter: str | None = None,
    class_filter: type[Maker] | None = BaseSiestaMaker,
    new_fdf_flags: dict[str, Any] = None,
    force_unknown: bool = False,
) -> Job | Flow | Maker:
    """
    Update the user_params of any SiestaInputGenerator in the flow.

    Alternatively, if a Maker is supplied, the user_params of the maker will
    be updated.
    Note, this returns a copy of the original Job/Flow/Maker. I.e., the update does not
    happen in place.

    Parameters
    ----------
    flow : .Job or .Flow or .Maker
        A job, flow, or Maker.
    siesta_updates : dict
        The updates to apply. Existing keys in user_params will not be modified
        unless explicitly specified in `siesta_updates`.
    name_filter : str or None
        A filter for the name of the jobs.
    class_filter : Maker or None
        A filter for the BaseSiestaMaker class used to generate the flows. Note the
        class filter will match any subclasses.
    new_fdf_flags : dict
        New flags to add to fdf_arguments outside of user_params.
    force_unknown : bool
        If True, allow unknown FDF parameters not registered by any dataclass.
        If False, raise ValueError for unknown parameters. Default is False.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input flow/job/maker modified to use the updated siesta fdf.
    """
    logger.info("update_user_siesta_settings.__init__()")
    dict_mod_updates = {}
    if siesta_updates is not None:
        # Pass keys as-is to user_params - let the dataclass handle conversion
        dict_mod_updates = {
            f"input_set_generator->user_params->{k}": v
            for k, v in siesta_updates.items()
        }
    if new_fdf_flags:
        dict_mod_updates.update(
            {
                f"input_set_generator->fdf_arguments->{k.lower().replace('.', '_')}": v
                for k, v in new_fdf_flags.items()
            }
        )

    # Pass force_unknown flag to input generator
    if force_unknown:
        dict_mod_updates["input_set_generator->force_unknown"] = force_unknown

    return update_maker_kwargs(class_filter, dict_mod_updates, flow, name_filter)


def update_fdf_siesta_settings(job: Job, user_settings: dict) -> Job:
    """
    Update a SIESTA job with user-defined FDF settings.

    Maps SIESTA FDF input names (like 'PAO.SplitNorm') to the corresponding
    class attributes and updates them.

    Args:
        job: The SIESTA job to be updated.
        user_settings (dict): A dictionary containing the FDF input settings.

    Returns
    -------
        job: The updated job with the new settings.
    """
    logger.info("update_fdf_siesta_settings.__init__()")
    # Assuming job has a BasisSetsAndProjectors instance or similar accessible
    basis_and_projectors = (
        job.basis_sets_and_projectors  # type: ignore[attr-defined]  # dynamic SIESTA maker attribute
    )  # Get the instance of BasisSetsAndProjectors

    # Get the reverse mapping of FDF input names to class attributes
    reverse_mapping = basis_and_projectors.reverse_fdf_mapping()

    # Update the job settings by translating FDF input names to internal attribute names
    for fdf_key, value in user_settings.items():
        # Translate FDF key to internal attribute name
        if fdf_key in reverse_mapping:
            internal_attr = reverse_mapping[fdf_key]
            # Update the attribute on the job's basis_sets_and_projectors instance
            setattr(basis_and_projectors, internal_attr, value)
        else:
            raise KeyError(f"Unknown FDF key: {fdf_key}")

    return job


def add_metadata(
    flow: Job | Flow | Maker,
    metadata: dict[str, Any],
    name_filter: str | None = None,
    class_filter: Maker | None = None,  # noqa: ARG001 kept for interface parity
) -> Job | Flow | Maker:
    """
    Add metadata to jobs in a Flow, Job, or Maker.

    Parameters
    ----------
    flow : .Job or .Flow or .Maker
        A job, flow or Maker.
    metadata : dict
        Dictionary of metadata to add to jobs.
    name_filter : str or None
        A filter for the name of the jobs.
    class_filter : Maker or None
        A filter for the class used to generate the flows.

    Returns
    -------
    Job or Flow or Maker
        A copy of the input with metadata added to matching jobs.
    """
    logger.info("add_metadata()")
    updated_flow = deepcopy(flow)

    if isinstance(updated_flow, Job):
        if name_filter is None or name_filter in updated_flow.name:
            if updated_flow.metadata is None:
                updated_flow.metadata = {}
            updated_flow.metadata.update(metadata)
    elif isinstance(updated_flow, Flow):
        for job in updated_flow:
            if isinstance(job, Job) and (
                name_filter is None or name_filter in job.name
            ):
                if job.metadata is None:
                    job.metadata = {}
                job.metadata.update(metadata)
    elif isinstance(updated_flow, Maker):
        # For Makers, we can't add metadata directly, so we just return it
        logger.warning(
            "Cannot add metadata directly to Maker objects. Create jobs first."
        )

    return updated_flow


def _read_xsf_to_pymatgen(
    file_path: str, as_molecule: bool = False
) -> Structure | Molecule:
    """
    Read XSF file and convert to pymatgen Structure or Molecule.

    Parameters
    ----------
    file_path : str
        Path to XSF file
    as_molecule : bool
        If True, return Molecule instead of Structure

    Returns
    -------
    pymatgen.core.Structure or pymatgen.core.Molecule
    """
    from ase.io import read as ase_read
    from pymatgen.core import Molecule
    from pymatgen.io.ase import AseAtomsAdaptor

    logger.info(f"Reading XSF file: {file_path}")

    try:
        # Read XSF file with ASE
        atoms = cast("Atoms", ase_read(file_path, format="xsf"))
    except Exception as e:
        error_msg = f"Error reading XSF file {file_path}: {e!s}"
        logger.error(error_msg)  # noqa: TRY400 keep plain error log without traceback
        raise ValueError(error_msg) from e

    if as_molecule:
        # Convert to Molecule (non-periodic)
        species = atoms.get_chemical_symbols()
        coords = atoms.get_positions()
        molecule = Molecule(species=species, coords=coords)
        logger.info(f"Converted XSF to Molecule: {molecule.composition.formula}")
        return molecule
    # Convert to Structure (periodic)
    structure: Structure = AseAtomsAdaptor.get_structure(atoms)
    logger.info(f"Converted XSF to Structure: {structure.composition.reduced_formula}")
    return structure


def _read_cif_to_pymatgen(
    file_path: str, as_molecule: bool = False
) -> Structure | Molecule:
    """
    Read CIF file and convert to pymatgen Structure or Molecule.

    Parameters
    ----------
    file_path : str
        Path to CIF file
    as_molecule : bool
        If True, return Molecule instead of Structure

    Returns
    -------
    pymatgen.core.Structure or pymatgen.core.Molecule
    """
    from pymatgen.core import Molecule, Structure

    logger.info(f"Reading CIF file: {file_path}")

    try:
        # Read CIF file directly with pymatgen
        structure = Structure.from_file(file_path)
    except Exception as e:
        error_msg = f"Error reading CIF file {file_path}: {e!s}"
        logger.error(error_msg)  # noqa: TRY400 keep plain error log without traceback
        raise ValueError(error_msg) from e

    if as_molecule:
        # Convert to Molecule (extract coordinates, remove periodicity)
        species = [str(site.specie) for site in structure]
        coords = [site.coords for site in structure]
        molecule = Molecule(species=species, coords=coords)
        logger.info(f"Converted CIF to Molecule: {molecule.composition.formula}")
        return molecule
    logger.info(f"Read CIF as Structure: {structure.composition.reduced_formula}")
    return structure


def siesta_to_pymatgen(
    file_path: str,
    use_xv: bool = True,
    remove_ghost: bool = True,
    add_site_properties: bool = True,
    as_molecule: bool = False,
) -> Structure | Molecule:
    """
    Convert a SIESTA/XSF/CIF structure file to a pymatgen Structure or Molecule.

    This function reads a structure file in various formats, converts it to
    pymatgen Structure or Molecule format, and optionally removes ghost atoms
    and adds site properties (tags, species labels, atomic numbers).

    Supported formats:
    - SIESTA XV files (.XV)
    - SIESTA FDF files (.fdf)
    - XSF files (.xsf) - XCrySDen format
    - CIF files (.cif) - Crystallographic Information File

    Parameters
    ----------
    file_path : str
        Path to structure file (.fdf, .XV, .xsf, or .cif)
    use_xv : bool, optional
        For FDF files: If True, read geometry from associated XV file.
        If False, read from FDF file directly. Ignored for other formats.
        Default is True.
    remove_ghost : bool, optional
        If True, remove ghost atoms (negative Z or '_ghost' in label).
        Only applicable to SIESTA files. Default is True.
    add_site_properties : bool, optional
        If True, add site properties (tags, species_label, species_Z) to
        the pymatgen Structure. Ignored if as_molecule=True. Default is True.
    as_molecule : bool, optional
        If True, return pymatgen Molecule (non-periodic) instead of Structure.
        Useful for adsorbates. Default is False.

    Returns
    -------
    pymatgen.core.Structure or pymatgen.core.Molecule
        The converted structure/molecule

    Examples
    --------
    >>> # Read SIESTA slab as Structure
    >>> slab = siesta_to_pymatgen("slab.XV")

    >>> # Read XSF adsorbate as Molecule
    >>> adsorbate = siesta_to_pymatgen("molecule.xsf", as_molecule=True)

    >>> # Read CIF structure
    >>> structure = siesta_to_pymatgen("structure.cif")

    >>> # Use in adsorption workflow
    >>> from atomate2.siesta.flows.surface import AdsorptionScanMaker
    >>> slab = siesta_to_pymatgen("slab.XV")
    >>> molecule = siesta_to_pymatgen("CO.xsf", as_molecule=True)
    >>> flow = AdsorptionScanMaker().make(slab, molecule)

    >>> # Read from FDF file directly, keep ghosts
    >>> structure = siesta_to_pymatgen("siesta.fdf", use_xv=False, remove_ghost=False)

    Notes
    -----
    - SIESTA files: Uses sisl to read XV/FDF files
    - XSF files: Uses ASE to read XCrySDen format
    - CIF files: Uses pymatgen to read CIF directly

    Ghost atoms (SIESTA only) are identified by:
    - Negative atomic number (Z < 0)
    - '_ghost' in the species label

    For molecules (as_molecule=True), periodicity is removed and site properties
    are not added (Molecule doesn't support site properties).
    """
    from pathlib import Path

    from ase import Atoms
    from pymatgen.io.ase import AseAtomsAdaptor

    logger.info(f"siesta_to_pymatgen: Reading {file_path}")

    # Detect file format from extension
    file_ext = Path(file_path).suffix.lower()

    # Route to appropriate reader based on file type
    if file_ext == ".xsf":
        return _read_xsf_to_pymatgen(file_path, as_molecule)
    if file_ext == ".cif":
        return _read_cif_to_pymatgen(file_path, as_molecule)
    if file_ext in [".fdf", ".xv", ".XV"]:
        # SIESTA format - use original implementation
        pass
    else:
        raise ValueError(
            f"Unsupported file format: {file_ext}. Supported: .fdf, .XV, .xsf, .cif"
        )

    # Original SIESTA implementation continues below
    import sisl

    logger.info(f"siesta_to_pymatgen: Reading SIESTA file {file_path}")

    # Read geometry using sisl
    try:
        structure_sisl = sisl.get_sile(file_path).read_geometry(output=use_xv)
    except Exception as e:
        error_msg = f"Error reading geometry from {file_path}: {e!s}"
        logger.error(error_msg)  # noqa: TRY400 keep plain error log without traceback
        raise ValueError(error_msg) from e

    # Get species information from ChemicalSpeciesLabel block
    try:
        sile = sisl.get_sile(file_path)
        sile.read()
        species_block = sile.get("ChemicalSpeciesLabel")
    except Exception:  # noqa: BLE001 species info is optional; degrade gracefully
        species_block = None
        logger.warning(
            "Could not read ChemicalSpeciesLabel block, species info may be incomplete"
        )

    # Convert to ASE Atoms
    structure_ase = structure_sisl.to.ase()

    # Get species indices for each atom
    atom_species = structure_sisl.atoms.species

    # Create dictionaries for species info
    species_dict = {}
    species_Z_dict = {}  # noqa: N806 Z is the physical atomic-number symbol
    if species_block:
        for entry in species_block:
            parts = entry.strip().split()
            if len(parts) >= 3:
                index = int(parts[0])  # Species index (1-based in FDF)
                Z = int(parts[1])  # noqa: N806 Z is the atomic number
                label = parts[2]  # Species label
                species_dict[index] = label
                species_Z_dict[index] = Z

    # Assign tags and collect species info
    tags = []
    species_labels = []
    species_Z = []  # noqa: N806 Z is the physical atomic-number symbol
    for specie_idx in atom_species:
        fdf_index = specie_idx + 1 if specie_idx < len(species_dict) else specie_idx
        species_label = species_dict.get(fdf_index, "")
        Z = species_Z_dict.get(  # noqa: N806 Z is the atomic number
            fdf_index, structure_sisl.atoms.atom[specie_idx].Z
        )
        species_labels.append(species_label)
        species_Z.append(Z)
        tags.append(int(specie_idx))

    # Store info in ASE Atoms
    structure_ase.info["species_dict"] = species_dict
    structure_ase.info["species_Z_dict"] = species_Z_dict
    structure_ase.info["species_labels"] = species_labels
    structure_ase.info["species_Z"] = species_Z
    structure_ase.set_tags(tags)

    # Remove ghost atoms if requested
    if remove_ghost:
        non_ghost_indices = [
            i
            for i, label in enumerate(species_labels)
            if "_ghost" not in label and species_Z[i] > 0
        ]

        if len(non_ghost_indices) < len(species_labels):
            logger.info(
                f"Removing {len(species_labels) - len(non_ghost_indices)} ghost atoms"
            )
            positions_no_ghost = structure_ase.get_positions()[non_ghost_indices]
            symbols_no_ghost = [
                structure_ase.get_chemical_symbols()[i] for i in non_ghost_indices
            ]
            tags_no_ghost = [tags[i] for i in non_ghost_indices]
            species_labels_no_ghost = [species_labels[i] for i in non_ghost_indices]
            species_Z_no_ghost = [  # noqa: N806 Z is the atomic number
                species_Z[i] for i in non_ghost_indices
            ]

            structure_ase = Atoms(
                symbols=symbols_no_ghost,
                positions=positions_no_ghost,
                cell=structure_ase.get_cell(),
                pbc=structure_ase.get_pbc(),
                tags=tags_no_ghost,
                info={
                    "species_dict": species_dict,
                    "species_Z_dict": {
                        k: v
                        for k, v in species_Z_dict.items()
                        if v > 0 and "_ghost" not in species_dict.get(k, "")
                    },
                    "species_labels": species_labels_no_ghost,
                    "species_Z": species_Z_no_ghost,
                },
            )
            tags = tags_no_ghost
            species_labels = species_labels_no_ghost
            species_Z = species_Z_no_ghost  # noqa: N806 Z is the atomic number

    # Convert to pymatgen Structure or Molecule
    if as_molecule:
        # Return as Molecule (non-periodic)
        from pymatgen.core import Molecule

        species = [str(symbol) for symbol in structure_ase.get_chemical_symbols()]
        coords = structure_ase.get_positions()

        try:
            molecule = Molecule(species=species, coords=coords)
        except Exception as e:
            error_msg = f"Error converting to pymatgen Molecule: {e!s}"
            logger.error(error_msg)  # noqa: TRY400 keep plain error log, no traceback
            raise ValueError(error_msg) from e
        else:
            logger.info(f"Converted molecule: {molecule.composition.formula}")
            return molecule
    else:
        # Return as Structure (periodic)
        try:
            structure_pymatgen: Structure = AseAtomsAdaptor.get_structure(structure_ase)
        except Exception as e:
            error_msg = f"Error converting to pymatgen Structure: {e!s}"
            logger.error(error_msg)  # noqa: TRY400 keep plain error log, no traceback
            raise ValueError(error_msg) from e

        # Add site properties if requested
        if add_site_properties:
            structure_pymatgen.add_site_property("tags", tags)
            structure_pymatgen.add_site_property("species_label", species_labels)
            structure_pymatgen.add_site_property("species_Z", species_Z)

        logger.info(
            f"Converted structure: {structure_pymatgen.composition.reduced_formula}"
        )
        return structure_pymatgen


def update_jobflow_resources(
    flow: Flow | Job,
    resource_configs: dict[str, dict[str, Any]],
    default_resources: dict[str, Any] | None = None,
    verbose: bool = True,
) -> Flow | Job:
    """
    Update jobflow-remote resources for jobs based on name pattern matching.

    This powerup allows you to set different computational resources
    (cores, memory, time) for different types of jobs in a workflow based on
    their names. Useful for heterogeneous workflows where small molecules need
    fewer resources than large slabs.

    Parameters
    ----------
    flow : Flow or Job
        A jobflow Flow or Job to update
    resource_configs : dict[str, dict[str, Any]]
        Dictionary mapping job name patterns to resource configurations.
        Pattern matching is case-insensitive substring matching.
        Each resource config should contain keys like:
        - "mem_per_cpu": str (e.g., "4G")
        - "nodes": int
        - "ntasks_per_node": int
        - "cpus_per_task": int
        - "time": str (e.g., "24:00:00")
        - "partition": str (optional, e.g., "RES")
        - "account": str (optional, e.g., "icn2100")
    default_resources : dict[str, Any] or None
        Default resources to use for jobs that don't match any pattern.
        If None, non-matching jobs keep their existing config.
    verbose : bool
        If True, print which resources are assigned to each job. Default is True.

    Returns
    -------
    Flow or Job
        The input flow/job with updated job configs for jobflow-remote

    Examples
    --------
    Basic usage with molecule and slab jobs:

    >>> from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker
    >>> from atomate2.siesta.powerups import update_jobflow_resources
    >>>
    >>> # Create workflow
    >>> flow = AdsorptionScanFlowMaker().make(slab, molecule)
    >>>
    >>> # Configure resources: small for molecules, large for slabs
    >>> resource_configs = {
    ...     "molecule": {  # Pattern: jobs with "molecule" in name
    ...         "mem_per_cpu": "4G",
    ...         "nodes": 1,
    ...         "ntasks_per_node": 4,
    ...         "cpus_per_task": 1,
    ...         "time": "24:00:00",
    ...     },
    ...     "adsorbate": {  # Pattern: jobs with "adsorbate" in name
    ...         "mem_per_cpu": "4G",
    ...         "nodes": 1,
    ...         "ntasks_per_node": 4,
    ...         "cpus_per_task": 1,
    ...         "time": "24:00:00",
    ...     },
    ... }
    >>>
    >>> # Default for all other jobs (slabs, etc.)
    >>> default_resources = {
    ...     "mem_per_cpu": "4G",
    ...     "nodes": 1,
    ...     "ntasks_per_node": 24,
    ...     "cpus_per_task": 1,
    ...     "time": "24:00:00",
    ... }
    >>>
    >>> # Apply resource updates
    >>> flow = update_jobflow_resources(
    ...     flow,
    ...     resource_configs=resource_configs,
    ...     default_resources=default_resources,
    ...     verbose=True,
    ... )

    Multiple patterns with priority (first match wins):

    >>> resource_configs = {
    ...     "phonon": {"ntasks_per_node": 8, "time": "48:00:00"},  # Phonon jobs
    ...     "relax": {"ntasks_per_node": 16, "time": "24:00:00"},  # Relaxation jobs
    ...     "static": {"ntasks_per_node": 12, "time": "12:00:00"},  # Static jobs
    ... }

    With HPC-specific parameters:

    >>> resource_configs = {
    ...     "molecule": {
    ...         "mem_per_cpu": "4G",
    ...         "ntasks_per_node": 4,
    ...         "time": "24:00:00",
    ...         "partition": "RES",  # Slurm partition
    ...         "account": "icn2100",  # Account for billing
    ...     }
    ... }

    Notes
    -----
    - Pattern matching is case-insensitive: "Molecule" matches "molecule"
    - First matching pattern wins (order matters in Python 3.7+)
    - Use verbose=True to see which jobs get which resources
    - Works with jobflow Flows and individual Jobs
    """
    logger.info("update_jobflow_resources()")

    updated_flow = deepcopy(flow)

    # Handle single Job
    if isinstance(updated_flow, Job):
        job_name = updated_flow.name
        matched = False

        # Try to match against each pattern
        for pattern, resources in resource_configs.items():
            if pattern.lower() in job_name.lower():
                updated_flow.update_config({"manager_config": {"resources": resources}})
                if verbose:
                    cores = resources.get("ntasks_per_node", "?")
                    print(  # noqa: T201 user-facing verbose output
                        f"  {job_name}: {cores} cores (matched pattern: '{pattern}')"
                    )
                matched = True
                break

        # Apply default if no match
        if not matched and default_resources is not None:
            updated_flow.update_config(
                {"manager_config": {"resources": default_resources}}
            )
            if verbose:
                cores = default_resources.get("ntasks_per_node", "?")
                print(f"  {job_name}: {cores} cores (default)")  # noqa: T201

    # Handle Flow (recursively process nested flows)
    elif isinstance(updated_flow, Flow):
        if verbose:
            print("\nUpdating jobflow-remote resources:")  # noqa: T201

        def _process_item(item: Flow | Job) -> None:
            if isinstance(item, Flow):
                for sub_item in item.jobs:
                    _process_item(sub_item)
                return

            job = item
            job_name = job.name
            matched = False

            # Try to match against each pattern
            for pattern, resources in resource_configs.items():
                if pattern.lower() in job_name.lower():
                    job.update_config({"manager_config": {"resources": resources}})
                    if verbose:
                        cores = resources.get("ntasks_per_node", "?")
                        print(  # noqa: T201 user-facing verbose output
                            f"  {job_name}: {cores} cores "
                            f"(matched pattern: '{pattern}')"
                        )
                    matched = True
                    break

            # Apply default if no match
            if not matched and default_resources is not None:
                job.update_config({"manager_config": {"resources": default_resources}})
                if verbose:
                    cores = default_resources.get("ntasks_per_node", "?")
                    print(f"  {job_name}: {cores} cores (default)")  # noqa: T201

        for item in updated_flow.jobs:
            _process_item(item)

    return updated_flow


def _walltime_to_seconds(walltime: str) -> int:
    """Convert ``"HH:MM:SS"`` walltime string to total seconds.

    Parameters
    ----------
    walltime : str
        Time string in ``HH:MM:SS`` format.

    Returns
    -------
    int
    """
    parts = walltime.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid walltime format '{walltime}', expected HH:MM:SS")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 3600 + m * 60 + s


def _seconds_to_walltime(seconds: int) -> str:
    """Convert total seconds to ``"HH:MM:SS"`` walltime string.

    Parameters
    ----------
    seconds : int
        Total seconds (must be >= 0).

    Returns
    -------
    str
    """
    seconds = max(0, seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _cap_walltime(requested: str, maximum: str) -> str:
    """Return the shorter of two walltime strings.

    Parameters
    ----------
    requested : str
        Requested walltime in ``HH:MM:SS``.
    maximum : str
        Maximum allowed walltime in ``HH:MM:SS``.

    Returns
    -------
    str
        The capped walltime in ``HH:MM:SS``.
    """
    req_s = _walltime_to_seconds(requested)
    max_s = _walltime_to_seconds(maximum)
    return _seconds_to_walltime(min(req_s, max_s))


def _estimate_resources_heuristic(num_atoms: int) -> dict[str, Any]:
    """Pure atom-count heuristic for HPC resource estimation.

    This is the original estimation logic extracted for reuse.

    Parameters
    ----------
    num_atoms : int
        Number of atoms in the structure. Use 0 for post-processing jobs.

    Returns
    -------
    dict[str, Any]
        Resource dictionary with ``ntasks_per_node`` and ``time`` keys.
    """
    if num_atoms <= 0:
        # Post-processing / analysis jobs
        return {"ntasks_per_node": 1, "time": "00:30:00", "cpus_per_task": 1}
    if num_atoms <= 8:
        # Small molecules, small bulk cells
        cores = max(2, (num_atoms // 2) * 2)  # 2 or 4
        return {"ntasks_per_node": cores, "time": "02:00:00", "cpus_per_task": 1}
    if num_atoms <= 32:
        # Small cells, unit cells
        cores = max(4, (num_atoms // 4) * 4)  # 4-32, multiples of 4
        cores = min(cores, 8)
        return {"ntasks_per_node": cores, "time": "04:00:00", "cpus_per_task": 1}
    if num_atoms <= 100:
        # Supercells, slabs
        cores = (num_atoms // 4) * 4  # ~1 core/atom, multiples of 4
        cores = max(cores, 8)
        return {"ntasks_per_node": cores, "time": "12:00:00", "cpus_per_task": 1}
    # Large supercells
    cores = min((num_atoms // 4) * 4, 128)  # ~1 core/atom, cap at 128
    cores = max(cores, 16)
    return {"ntasks_per_node": cores, "time": "24:00:00", "cpus_per_task": 1}


def _estimate_resources(
    num_atoms: int, profile: ClusterProfile | None = None
) -> dict[str, Any]:
    """Estimate HPC resources from atom count, optionally capped by a cluster profile.

    When *profile* is ``None`` the function returns the same result as
    before (pure heuristic).  When a
    :class:`~atomate2.siesta.cluster_profiles.ClusterProfile` is supplied the
    heuristic output is adjusted:

    * **Core capping** — ``ntasks_per_node`` is capped at ``cores_per_node``.
    * **Multi-node spreading** — if ideal cores exceed a single node, ``nodes``
      is computed (capped at ``max_nodes``).
    * **Memory check** — estimated memory (~0.75 GB/atom) is compared to
      per-node RAM; extra nodes are added when necessary.
    * **Walltime capping** — ``time`` is clamped to ``max_walltime``.
    * **mem_per_cpu** — computed from ``memory_per_node_gb / cores_per_node``.
    * **SLURM metadata** — ``partition``, ``account``, ``qos`` injected from profile.
    * **GPU** — ``gres`` set when ``gpu_per_node > 0``.

    Parameters
    ----------
    num_atoms : int
        Number of atoms in the structure. Use 0 for post-processing jobs.
    profile : ClusterProfile or None
        Optional cluster hardware description.

    Returns
    -------
    dict[str, Any]
        Resource dictionary suitable for jobflow-remote ``manager_config``.
    """
    import math

    # Start from the pure heuristic
    resources = _estimate_resources_heuristic(num_atoms)

    if profile is None:
        return resources

    ideal_cores = resources["ntasks_per_node"]
    ideal_time = resources["time"]

    # --- Core capping & multi-node spreading ---
    cpn = profile.cores_per_node
    if ideal_cores <= cpn:
        # Fits in a single node
        resources["ntasks_per_node"] = ideal_cores
    else:
        # Need multiple nodes
        nodes = math.ceil(ideal_cores / cpn)
        nodes = min(nodes, profile.max_nodes)
        resources["ntasks_per_node"] = cpn
        resources["nodes"] = nodes

    # --- Memory check (~0.75 GB per atom) ---
    if num_atoms > 0:
        est_memory_gb = num_atoms * 0.75
        nodes_current = resources.get("nodes", 1)
        total_memory_gb = nodes_current * profile.memory_per_node_gb
        if est_memory_gb > total_memory_gb:
            nodes_needed = math.ceil(est_memory_gb / profile.memory_per_node_gb)
            nodes_needed = min(nodes_needed, profile.max_nodes)
            if nodes_needed > nodes_current:
                resources["nodes"] = nodes_needed

    # --- mem_per_cpu ---
    mem_per_cpu_gb = profile.memory_per_node_gb / cpn
    # Express as integer GB string (e.g. "4G") or with one decimal
    if mem_per_cpu_gb == int(mem_per_cpu_gb):
        resources["mem_per_cpu"] = f"{int(mem_per_cpu_gb)}G"
    else:
        resources["mem_per_cpu"] = f"{mem_per_cpu_gb:.1f}G"

    # --- Walltime capping ---
    resources["time"] = _cap_walltime(ideal_time, profile.max_walltime)

    # --- SLURM metadata ---
    if profile.partition is not None:
        resources["partition"] = profile.partition
    if profile.account is not None:
        resources["account"] = profile.account
    if profile.qos is not None:
        resources["qos"] = profile.qos

    # --- GPU ---
    if profile.gpu_per_node > 0:
        resources["gres"] = f"gpu:{profile.gpu_per_node}"

    return resources


def _get_atom_count_from_job(job: Job) -> int | None:
    """
    Try to extract atom count from a job's function arguments.

    Searches both positional args and keyword args for a Structure/Molecule
    object and returns its site count.

    Parameters
    ----------
    job : Job
        A jobflow Job to inspect.

    Returns
    -------
    int or None
        Number of atoms if a structure was found, None otherwise.
    """

    def _is_structure(obj: Any) -> bool:
        """Return True if obj is a pymatgen Structure or Molecule (not a reference)."""
        from jobflow.core.reference import OutputReference

        if isinstance(obj, OutputReference):
            return False
        return hasattr(obj, "num_sites") and hasattr(obj, "species")

    try:
        # Check positional args for a Structure
        if hasattr(job, "function_args"):
            for arg in job.function_args:
                if _is_structure(arg):
                    return arg.num_sites
    except Exception:  # noqa: BLE001, S110 best-effort structure inspection
        pass

    try:
        # Check keyword args for a Structure (e.g., host_structure=...)
        if hasattr(job, "function_kwargs"):
            for val in job.function_kwargs.values():
                if _is_structure(val):
                    return val.num_sites
            # Also check for explicit n_atoms kwarg (e.g., extract_chemical_potential)
            if "n_atoms" in job.function_kwargs:
                n = job.function_kwargs["n_atoms"]
                if isinstance(n, int):
                    return n
    except Exception:  # noqa: BLE001, S110 best-effort structure inspection
        pass

    return None


def auto_allocate_resources(
    flow: Flow | Job,
    base_resources: dict[str, Any] | None = None,
    cluster_profile: ClusterProfile | dict[str, Any] | None = None,
    verbose: bool = True,
) -> Flow | Job:
    """Auto-allocate HPC resources based on atom count for each job in a workflow.

    This powerup inspects each job's structure to determine atom count, then
    assigns appropriate computational resources (cores, walltime) using
    simple heuristics.  When a *cluster_profile* is provided the heuristic
    output is further constrained by real hardware limits (cores per node,
    memory, walltime cap) and SLURM metadata is injected automatically.

    Resource estimation heuristics (before profile adjustment):

    ========== ======================= ===========
    Atom count Cores (ntasks_per_node) Time
    ========== ======================= ===========
    0 (post)   1                       00:30:00
    1-8        2-4                     02:00:00
    9-32       4-8                     04:00:00
    33-100     8-24 (~1/atom)          12:00:00
    100+       16-128 (~1/atom)        24:00:00
    ========== ======================= ===========

    Merge order: ``{base_resources} ← {auto_resources}`` — auto-detected
    values override the base template so that cores and walltime are always
    set from the heuristic (possibly capped by the profile).

    Parameters
    ----------
    flow : Flow or Job
        A jobflow Flow or Job to update.
    base_resources : dict[str, Any] or None
        Base resource template providing cluster-specific values like
        ``partition``, ``account``, ``mem_per_cpu``.  These are merged with
        auto-detected values.  If None, only auto-detected values are set.
    cluster_profile : ClusterProfile or dict or None
        Optional cluster hardware description.  Accepts a
        :class:`~atomate2.siesta.cluster_profiles.ClusterProfile` instance or
        a plain ``dict`` (converted via ``ClusterProfile.from_dict()``).
        When provided, resources are capped to real hardware limits and
        SLURM metadata (partition, account, qos) is injected.
    verbose : bool
        If True, print resource assignments for each job.  Default is True.

    Returns
    -------
    Flow or Job
        A copy of the input with auto-allocated resources on each job.

    Examples
    --------
    Basic auto-allocation (backward compatible):

    >>> from atomate2.siesta.powerups import auto_allocate_resources
    >>> flow = auto_allocate_resources(flow)

    With a predefined cluster profile:

    >>> from atomate2.siesta.cluster_profiles import ClusterProfile
    >>> flow = auto_allocate_resources(flow, cluster_profile=ClusterProfile.mn5())

    With a dict profile (no extra import needed):

    >>> flow = auto_allocate_resources(
    ...     flow,
    ...     cluster_profile={
    ...         "cores_per_node": 48,
    ...         "memory_per_node_gb": 192,
    ...         "partition": "RES",
    ...         "account": "icn2100",
    ...     },
    ... )

    Combine profile with base_resources (base_resources values override profile
    for keys like ``mem_per_cpu``):

    >>> flow = auto_allocate_resources(
    ...     flow,
    ...     cluster_profile=ClusterProfile.agustina(),
    ...     base_resources={"mem_per_cpu": "8G"},
    ... )

    Notes
    -----
    - This is a non-destructive operation (deep-copies the flow).
    - Jobs whose structure cannot be detected are skipped (keep existing config).
    - Works with any workflow: defects, surfaces, EOS, NEB, etc.
    - For manual per-job control, use ``update_jobflow_resources()`` instead.
    """
    from atomate2.siesta.cluster_profiles import ClusterProfile

    logger.info("auto_allocate_resources()")

    # Convert dict → ClusterProfile if needed
    profile = None
    if cluster_profile is not None:
        if isinstance(cluster_profile, dict):
            profile = ClusterProfile.from_dict(cluster_profile)
        else:
            profile = cluster_profile

    updated_flow = deepcopy(flow)

    if verbose:
        print("\nAuto-allocating HPC resources:")  # noqa: T201
        if profile is not None:
            print(f"  Cluster profile: {profile.summary()}")  # noqa: T201

    # Post-processing jobs that don't run SIESTA (minimal resources)
    _postproc_patterns = ("finalize", "summary", "mu_", "extract_chemical")

    def _process_job(job: Job) -> None:
        """Process a single job for resource allocation."""
        # Post-processing jobs get minimal resources regardless of structure
        job_name_lower = job.name.lower()
        if any(p in job_name_lower for p in _postproc_patterns):
            auto_resources = _estimate_resources(0, profile=profile)
            if base_resources is not None:
                merged = {**base_resources, **auto_resources}
            else:
                merged = auto_resources
            job.update_config({"manager_config": {"resources": merged}})
            if verbose:
                cores = merged.get("ntasks_per_node", "?")
                time_str = merged.get("time", "?")
                print(  # noqa: T201 user-facing verbose output
                    f"  {job.name}: {cores} core, {time_str} (post-processing)"
                )
            return

        num_atoms = _get_atom_count_from_job(job)

        if num_atoms is None:
            if verbose:
                print(f"  {job.name}: skipped (no structure detected)")  # noqa: T201
            return

        # Get auto-detected resources (profile-aware when available)
        auto_resources = _estimate_resources(num_atoms, profile=profile)

        # Merge: base provides template, auto overrides cores/time/SLURM
        if base_resources is not None:
            merged = {**base_resources, **auto_resources}
        else:
            merged = auto_resources

        # Apply to job via update_config
        job.update_config({"manager_config": {"resources": merged}})

        if verbose:
            cores = merged.get("ntasks_per_node", "?")
            time_val = merged.get("time", "?")
            nodes = merged.get("nodes", 1)
            node_str = f", {nodes} nodes" if nodes > 1 else ""
            print(  # noqa: T201 user-facing verbose output
                f"  {job.name}: {cores} cores{node_str}, {time_val} ({num_atoms} atoms)"
            )

    def _process_flow(flow_or_job: Flow | Job) -> None:
        """Recursively process all jobs in a flow."""
        if isinstance(flow_or_job, Job):
            _process_job(flow_or_job)
        elif isinstance(flow_or_job, Flow):
            for item in flow_or_job.jobs:
                _process_flow(item)

    _process_flow(updated_flow)

    return updated_flow


def write_output_json_local(results: dict[str, Any]) -> None:
    """Write JSON output to a local database file."""
    import json
    from datetime import datetime

    from pymatgen.core import Element

    logger.info("write_output_json_local.__init__()")

    # Custom JSON encoder to handle datetime, Element, and other non-serializable types
    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()  # Convert datetime to ISO 8601 string
            if isinstance(obj, Element):
                return obj.symbol  # Convert Element to its symbol (e.g., "Na", "Cl")
            if hasattr(obj, "as_dict"):
                return (
                    obj.as_dict()
                )  # Convert objects with as_dict method (like Structure) to dict
            # Add more custom serialization if needed (e.g., handling Path objects)
            return super().default(obj)

    # Loop through the results to extract the serializable part
    serializable_data: dict[Any, dict[Any, Any]] = {}
    # Loop through the results to extract the serializable part
    for job_id, job_data in results.items():
        for step, response in job_data.items():
            # Extract the SiestaTaskDoc from the response
            task_doc = response.output  # Assuming output contains the SiestaTaskDoc

            # Convert SiestaTaskDoc to a dictionary using dict()
            task_dict = task_doc.dict()  # pydantic BaseModel method

            # Add this to the serializable data
            if job_id not in serializable_data:
                serializable_data[job_id] = {}
            serializable_data[job_id][step] = task_dict

    # Convert the serializable data to JSON format and store it in a file
    with open("calculation_results.json", "w") as outfile:
        json.dump(serializable_data, outfile, indent=4, cls=CustomJSONEncoder)
