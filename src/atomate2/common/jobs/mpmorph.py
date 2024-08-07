"""Define get_random_packed_structure function.

This file generalizes the MPMorph workflows of
https://github.com/materialsproject/mpmorph
originally written in atomate for VASP only to a more general
code agnostic form.

For information about the current flows, contact:
- Bryant Li (@BryantLi-BLI)
- Aaron Kaplan (@esoteric-ephemera)
- Max Gallant (@mcgalcode)
"""

from __future__ import annotations

from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
from jobflow import job
from mp_api.client import MPRester
from pymatgen.core import Composition, Molecule, Structure
from pymatgen.io.packmol import PackmolBoxGen

if TYPE_CHECKING:
    from pathlib import Path


def get_average_volume_from_mp(composition: Composition) -> float:
    """
    Get the average volume per atom for a given composition from the Materials Project.

    This function will make API calls to the Materials Project.
    Check Materials Project API documentation for more
    information https://next-gen.materialsproject.org/api.

    Parameters
    ----------
    composition : Composition
        The target composition.

    Returns
    -------
    float
        The average volume per atom for the composition.
    """
    with MPRester() as mpr:
        comp_entries = mpr.get_entries(composition.reduced_formula, inc_structure=True)

    vols = None

    if len(comp_entries) > 0:
        vols = [
            entry.structure.volume / entry.structure.num_sites for entry in comp_entries
        ]

    else:
        # Find all Materials project entries containing the elements in the
        # desired composition to estimate starting volume.
        with MPRester() as mpr:
            _entries = mpr.get_entries_in_chemsys(
                [str(el) for el in composition.elements], inc_structure=True
            )

        # Only take entries with at least two elements in common with target composition
        entries = [
            entry
            for entry in _entries
            if len(set(composition).intersection(set(entry.structure.composition))) > 1
        ]

        vols = [entry.structure.volume / entry.structure.num_sites for entry in entries]

    return np.mean(vols)


def get_random_packed_structure(
    composition: Composition | str,
    target_atoms: int = 100,
    vol_exp: float = 1.0,
    tol: float = 2.0,
    return_as_job: bool = False,
    vol_per_atom_source: float | str = "mp",
    packmol_seed: int = 1,
    packmol_output_dir: str | Path | None = None,
) -> Structure:
    """
    Generate a random packed structure with a target number of atoms.

    Designed to make amorphous/glassy structures.

    Parameters
    ----------
    composition : Composition | str
        The composition of the target structure.
    target_atoms : int
        The target number of atoms in the structure.
    vol_exp : float
        The volume expansion factor to apply to the structure.
    tol : float
        The tolerance to apply to the box size.
    return_as_job : bool
        Whether to return the structure as a jobflow job object.
    vol_per_atom_source : float | str
        If float - the volume per atom used to generate lattice size
        If str - "mp" to use the Materials Project API to estimate volume per atom.
        If str - "icsd" to use the ICSD database to estimate volume per atom.
        (Not yet implemented)
    packmol_seed : int
        The seed to use for the packmol random number generator.
    packmol_output_dir : str | Path | None
        The directory to output the packmol files to. If None, a
        temporary directory is used and will be removed after.

    Returns
    -------
    Structure | Job
        The random packed structure.
    """
    if return_as_job:
        return job(
            get_random_packed_structure(
                composition,
                target_atoms,
                vol_exp=vol_exp,
                tol=tol,
                return_as_job=False,
                vol_per_atom_source=vol_per_atom_source,
                packmol_seed=packmol_seed,
            )
        )
    if isinstance(composition, str):
        composition = Composition(composition)

    if isinstance(vol_per_atom_source, (float, int)):
        vol_per_atom = vol_per_atom_source

    elif vol_per_atom_source == "mp":
        vol_per_atom = get_average_volume_from_mp(composition)

    formula, _ = composition.get_integer_formula_and_factor()
    integer_composition = Composition(formula)
    full_cell_composition = integer_composition * np.ceil(
        target_atoms / integer_composition.num_atoms
    )

    supercell_composition = {
        str(el): int(full_cell_composition.element_composition.get(el))
        for el in full_cell_composition
    }

    with TemporaryDirectory() as tmpdir:
        molecules = []
        for element, num_sites in supercell_composition.items():
            xyz_file = f"{tmpdir}/{element}.xyz"
            with open(xyz_file, "w+") as f:
                f.write("1\ncomment\n" + element + " 0.0 0.0 0.0\n")
            molecules.append({"name": element, "number": num_sites, "coords": xyz_file})

        box_scale = (vol_per_atom * full_cell_composition.num_atoms * vol_exp) ** (
            1 / 3
        )
        box_lower_bound = tol / 2
        box_upper_bound = box_scale - tol / 2

        box_size = 3 * [box_lower_bound] + 3 * [box_upper_bound]

        packmol_set = PackmolBoxGen(seed=packmol_seed).get_input_set(
            molecules=molecules, box=box_size
        )
        packmol_output_dir = str(packmol_output_dir or tmpdir)
        packmol_set.write_input(directory=packmol_output_dir)
        packmol_set.run(path=packmol_output_dir)

        mol = Molecule.from_file(f"{packmol_output_dir}/packmol_out.xyz")

    return Structure(
        [[box_scale if i == j else 0.0 for j in range(3)] for i in range(3)],
        species=mol.species,
        coords=mol.cart_coords,
        coords_are_cartesian=True,
    )
