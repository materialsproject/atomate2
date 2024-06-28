"""Module defining common jobs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jobflow import Response, job
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2 import SETTINGS

if TYPE_CHECKING:
    from pymatgen.core import Structure


@job
def structure_to_primitive(
    structure: Structure, symprec: float = SETTINGS.SYMPREC
) -> Structure:
    """
    Job that creates a standard primitive structure.

    Parameters
    ----------
    structure: Structure object
        input structure that will be transformed
    symprec: float
        precision to determine symmetry

    Returns
    -------
    .Structure
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    return sga.get_primitive_standard_structure()


@job
def structure_to_conventional(
    structure: Structure, symprec: float = SETTINGS.SYMPREC
) -> Structure:
    """
    Job that creates a standard conventional structure.

    Parameters
    ----------
    structure: Structure object
        input structure that will be transformed
    symprec: float
        precision to determine symmetry

    Returns
    -------
    .Structure
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    return sga.get_conventional_standard_structure()


@job
def retrieve_structure_from_materials_project(
    material_id_or_task_id: str,
    use_task_id: bool = False,
    reset_magnetic_moments: bool = False,
) -> Response[Structure]:
    """
    Retrieve a Structure from Materials Project.

    This job is useful for constructing a Flow that always will retrieve the most
    up-to-date data at the time the Flow runs.

    The retrieved Structure will change between subsequent Materials Project database
    releases as old calculation tasks are removed and new, better, calculation tasks
    (e.g. with more accurate lattice parameters) are added.

    Using this job requires that the system where the job runs has a connection to the
    outside internet. It also requires the Materials Project API key to be set
    appropriately via an environment variable or otherwise. Consult the Materials
    Project API documentation for more information.

    Parameters
    ----------
    material_id_or_task_id : str
        The material_id or the task_id for the data record
        being retrieved.
    use_task_id : bool
        If true, will request the Structure from the specific
        calculation task in Materials Project. Structures
        retrieved in this way should not change even between
        new Materials Project database releases.
    reset_magnetic_moments : bool
        If true, will remove any magnetic moment information
        or magnetic ordering on the Structure. Typically,
        this will mean that the Structure will then be
        initialized as ferromagnetic by any child jobs.

    Returns
    -------
    .Response
        A Response with the Structure object as the output,
        and also the database version and specific task_id
        corresponding to that Structure object also stored
    """
    # inline import to avoid required dependency
    try:
        from mp_api.client import MPRester
    except ImportError:
        raise ImportError(
            "This job requires the Materials Project API client "
            "to be installed, via `pip install mp-api` or similar."
        ) from None

    with MPRester() as mpr:
        if use_task_id:
            doc = mpr.tasks.search(material_id_or_task_id, fields=["structure"])[0]
            task_id = material_id_or_task_id
        else:
            doc = mpr.materials.search(
                material_id_or_task_id, fields=["structure", "origins"]
            )[0]
            origins = {prop.name: prop for prop in doc.origins}
            task_id = str(origins["structure"].task_id)

        database_version = mpr.get_database_version()

    structure = doc.structure

    if reset_magnetic_moments and "magmom" in structure.site_properties:
        # Materials Project stores magnetic moments via the `magmom` site property
        # and we can safely assume that here. In general, since magnetic order
        # can be represented in multiple ways such as Species.spin, the
        # following method would be better:
        # CollinearMagneticStructureAnalyzer.get_nonmagnetic_structure()
        structure.remove_site_property("magmom")

    return Response(
        output=structure,
        stored_data={"task_id": task_id, "database_version": database_version},
    )
