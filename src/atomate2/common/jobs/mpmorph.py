"""Define utility functions for amorphous structure equilibration.

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

from importlib.resources import files as import_resource_file
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Job
from pymatgen.core import Composition, Molecule, Structure
from pymatgen.io.packmol import PackmolBoxGen

if TYPE_CHECKING:
    from pathlib import Path

_DEFAULT_ICSD_AVG_VOL_FILE = str(
    import_resource_file("atomate2.common.jobs")
    / "ICSD_expt_inorg_ordered_avg_vol.json.gz"
)
_DEFAULT_MP_AVG_VOL_FILE = str(
    import_resource_file("atomate2.common.jobs") / "mp_avg_vol.json.gz"
)


def get_average_volume_from_mp_api(
    composition: Composition, mp_api_key: str | None = None
) -> float:
    """
    Get the average volume per atom for a given composition from the Materials Project.

    This function will make API calls to the Materials Project.
    Check Materials Project API documentation for more
    information https://next-gen.materialsproject.org/api.

    Parameters
    ----------
    composition : Composition
        The target composition.
    mp_api_key : str or None
        The user's MP API key.

    Returns
    -------
    float
        The average volume per atom for the composition.
    """
    from mp_api.client import MPRester

    with MPRester(api_key=mp_api_key) as mpr:
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


def get_average_volume_from_mp_cached(
    composition: Composition,
    cache_file: str | Path | None = None,
) -> float:
    """
    Get the average volume per atom for a given composition from cached MP data.

    This function uses cached MP data to accelerate the volume/atom search.

    Parameters
    ----------
    composition : Composition
        The target composition.
    cache_file : str, .Path, or None
        Path to the cached volume file, should be of the same form as used in
        `get_average_volume_from_icsd`

    Returns
    -------
    float
        The average volume per atom for the composition.
    """
    return get_average_volume_from_icsd(
        composition,
        ignore_oxi_states=True,
        icsd_chem_env_file=cache_file or _DEFAULT_MP_AVG_VOL_FILE,
    )


def get_average_volume_from_mp(
    composition: Composition, use_cached: bool = True, **kwargs
) -> float:
    """
    Get the average volume per atom for a given composition from MP data.

    This function will either make MP API calls or used cached data for
    the search.

    Parameters
    ----------
    composition : Composition
        The target composition.
    use_cached : bool = True
        Whether to use cached MP data (True) or make calls to the MP API (False)
    **kwargs : kwargs to pass to the volume/atom search functions, see
        `get_average_volume_from_mp_cached`,
        `get_average_volume_from_mp_api`
        for specific kwargs.

    Returns
    -------
    float
        The average volume per atom for the composition.
    """
    if use_cached:
        return get_average_volume_from_mp_cached(composition, **kwargs)
    return get_average_volume_from_mp_api(composition, **kwargs)


def _get_chem_env_key_from_composition(
    composition: Composition, ignore_oxi_states: bool = True
) -> str:
    """
    Get chemical environment as a string for ICSD avg volume determination.

    Parameters
    ----------
    composition : .Composition
        Structure composition
    ignore_oxi_states : bool = True
        Whether to ignore oxidation states assigned to sites in the structure,
        both in the input composition and ICSD structures.

        Note that 0+ / 0- oxidation states are treated identically even
        when ignore_oxi_states = False.

    Returns
    -------
    Chemical environment returned as a dunder-separated string,
    such as "Ag+__Cu2+__N5+__O2-"
    """
    comp = composition
    if ignore_oxi_states:
        comp = comp.remove_charges()
    chem_env = "__".join(sorted(set(comp.as_dict())))
    for char in ["+", "-"]:
        chem_env = chem_env.replace(f"0{char}", "")
    return chem_env


def get_average_volume_from_icsd(
    composition: Composition,
    ignore_oxi_states: bool = True,
    icsd_chem_env_file: str | Path | None = None,
) -> float:
    """
    Get average volume for a chemical environment from ICSD data.

    The ICSD data is for "reasonable", ordered, experimental inorganic solids.

    Parameters
    ----------
    composition : .Composition
        Structure composition
    ignore_oxi_states : bool = True
        Whether to ignore oxidation states assigned to sites in the structure,
        both in the input composition and ICSD structures.

        Note that 0+ / 0- oxidation states are treated identically even
        when ignore_oxi_states = False.
    icsd_chem_env_file : str | Path | None = None
        The file path to the ICSD chemical environments file.
        For the user to substitute their own values, this should be a dict of the form:
        ```
        {
            "chem_env": list[str],
            "avg_vol": list[float].
            "count": list[int],
            "with_oxi": list[bool],
        }
        ```

    Returns
    -------
    Average volume as a float
    """
    from itertools import combinations

    from pandas import read_json

    icsd_chem_env_file = icsd_chem_env_file or _DEFAULT_ICSD_AVG_VOL_FILE
    icsd_avg_vols = read_json(icsd_chem_env_file)

    def get_entry_from_dict(chem_env: str) -> dict | None:
        data = icsd_avg_vols[icsd_avg_vols["chem_env"] == chem_env]
        data = data[
            data["with_oxi"]
            if (not ignore_oxi_states and len(data[data["with_oxi"]]) > 0)
            else ~data["with_oxi"]
        ]
        if len(data) > 0:
            return {k: data[k].squeeze() for k in ("avg_vol", "count")}
        return None

    chem_env_key = _get_chem_env_key_from_composition(
        composition, ignore_oxi_states=ignore_oxi_states
    )
    if (avg_vol := get_entry_from_dict(chem_env_key)) is not None:
        return avg_vol["avg_vol"]

    vols = []
    counts = 0
    for ielt in range(2, len(composition)):
        for combo in combinations(composition, ielt):
            chem_env_key = _get_chem_env_key_from_composition(
                Composition({spec: 1 for spec in combo}),
                ignore_oxi_states=ignore_oxi_states,
            )

            if (avg_vol := get_entry_from_dict(chem_env_key)) is not None:
                vols.append(avg_vol["avg_vol"] * avg_vol["count"])
                counts += avg_vol["count"]

    return sum(vols) / counts


def get_random_packed_structure(
    composition: Composition | str,
    target_atoms: int = 100,
    vol_multiply: float = 1.0,
    tol: float = 2.0,
    return_as_job: bool = False,
    vol_per_atom_source: float | str = "mp",
    db_kwargs: dict | None = {"use_cached": True},
    packmol_seed: int = 1,
    packmol_output_dir: str | Path | None = None,
) -> Structure | Job:
    """
    Generate a random packed structure with a target number of atoms.

    Designed to make amorphous/glassy structures.
    Defaults to using cached MP data.

    Parameters
    ----------
    composition : Composition | str
        The composition of the target structure.
    target_atoms : int
        The target number of atoms in the structure.
    vol_multiply : float
        The factor to multiply the strcuture volume by.
    tol : float
        The tolerance to apply to the box size.
    return_as_job : bool
        Whether to return the structure as a jobflow job object.
    vol_per_atom_source : float | str
        If float - the volume per atom used to generate lattice size
        If str - "mp" to use the Materials Project API to estimate volume per atom.
        If str - "icsd" to use the ICSD database to estimate volume per atom.
    db_kwargs : dict | None = None
        kwargs to pass to the volume-determining function.
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
        return Job(
            get_random_packed_structure,
            function_kwargs={
                "composition": composition,
                "target_atoms": target_atoms,
                "vol_multiply": vol_multiply,
                "tol": tol,
                "return_as_job": False,
                "vol_per_atom_source": vol_per_atom_source,
                "packmol_seed": packmol_seed,
            },
        )
    if isinstance(composition, str):
        composition = Composition(composition)

    db_kwargs = db_kwargs or {}
    if isinstance(vol_per_atom_source, (float, int)):
        vol_per_atom = vol_per_atom_source

    elif vol_per_atom_source.lower() == "mp":
        vol_per_atom = get_average_volume_from_mp(composition, **db_kwargs)

    elif vol_per_atom_source.lower() == "icsd":
        vol_per_atom = get_average_volume_from_icsd(composition, **db_kwargs)

    else:
        raise ValueError(f"Unknown volume per atom source: {vol_per_atom_source}.")

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

        box_scale = (vol_per_atom * full_cell_composition.num_atoms * vol_multiply) ** (
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