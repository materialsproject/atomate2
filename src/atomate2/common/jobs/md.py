"""Jobs for molecular dynamics that are shared between codes."""

from __future__ import annotations

from pathlib import Path

from ase.io import read as ase_read
from jobflow import job
from monty.os.path import zpath
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from atomate2.utils.path import strip_hostname

# file of the force field MD trajectory, written in the ASE format
ASE_TRAJECTORY_FILE = "md_trajectory.traj"


def _get_site_properties(reference: Structure) -> dict:
    """Get the magnetic moments of the reference, the only site property kept."""
    if "magmom" in reference.site_properties:
        return {"magmom": reference.site_properties["magmom"]}
    return {}


@job
def get_md_restart_structure(
    md_dir: str, md_code: str, reference: Structure
) -> Structure:
    """
    Get the final positions and velocities of an MD run.

    For VASP, they are read from CONTCAR, with the velocities in Angstrom/fs.
    For force fields, they are read from the last frame of the ASE trajectory
    file, with the velocities in ASE units. In both cases, the velocities are a
    site property that the next MD run starts from. The thermostat variables
    are not carried over, so they start again from zero in the next run. The
    magnetic moments of the reference, if any, are added as a site property.

    Parameters
    ----------
    md_dir: str
        Directory of the MD run.
    md_code: str
        Code of the MD, "vasp" or "forcefields".
    reference: Structure
        The undisplaced supercell the MD started from.

    Returns
    -------
    Structure
        The final structure, with a "velocities" site property.
    """
    directory = Path(strip_hostname(md_dir))
    if md_code == "vasp":
        structure = Poscar.from_file(zpath(str(directory / "CONTCAR"))).structure
    else:
        atoms = ase_read(directory / ASE_TRAJECTORY_FILE, index=-1)
        structure = Structure(
            atoms.cell[:],
            atoms.get_chemical_symbols(),
            atoms.get_positions(),
            coords_are_cartesian=True,
            site_properties={"velocities": atoms.get_velocities().tolist()},
        )
    for key, values in _get_site_properties(reference).items():
        structure.add_site_property(key, values)
    return structure
